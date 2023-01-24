import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if not hasattr(self.args, "dropout"):
            self.args.dropout = 0

        if not hasattr(self.args, "alternate_corr"):
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(
                output_dim=128, norm_fn="instance", dropout=args.dropout
            )
            self.cnet = SmallEncoder(
                output_dim=hdim + cdim, norm_fn="none", dropout=args.dropout
            )
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(
                output_dim=256, norm_fn="instance", dropout=args.dropout
            )
            self.cnet = BasicEncoder(
                output_dim=hdim + cdim, norm_fn="batch", dropout=args.dropout
            )
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, C, 8 * H, 8 * W)

    def upsample_flow_einops(self, flow, mask):
        """Upsample flow field [2, H/8, W/8] -> [2, H, W] using convex combination.

        The `mask` contains 8x8x3x3 coefficients for each low resolution (/8) position.
        This represents 3x3 coefficients for each final fine resolution position.
        These 3x3 coefficients are softmaxed and used to re-weight 3x3 patches in the low resolution (/8) flow.
        """
        B, xy, H, W = flow.shape
        import einops

        mask = einops.rearrange(
            mask,
            (
                "B (H_low_patch W_low_patch H_fine_patch W_fine_patch) H_low W_low -> "
                "B 1 (H_low_patch W_low_patch) H_fine_patch W_fine_patch H_low W_low"
            ),
            B=B,
            H_low_patch=3,
            W_low_patch=3,
            H_fine_patch=8,
            W_fine_patch=8,
            H_low=H,
            W_low=W,
        )
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)  # this extracts 3x3 patches
        up_flow = einops.rearrange(
            up_flow,
            (
                "B (xy H_low_patch W_low_patch) (H_low W_low) -> "
                "B xy (H_low_patch W_low_patch) 1 1 H_low W_low"
            ),
            B=B,
            xy=xy,
            H_low_patch=3,
            W_low_patch=3,
            H_low=H,
            W_low=W,
        )

        # for some reason, the following reduce gives very slightly different results, than the original:
        # up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = einops.reduce(
            mask * up_flow,
            (
                "B xy (H_low_patch W_low_patch) H_fine_patch W_fine_patch H_low W_low -> "
                "B xy H_fine_patch W_fine_patch H_low W_low"
            ),
            reduction="sum",
            B=B,
            xy=xy,
            H_low_patch=3,
            W_low_patch=3,
            H_fine_patch=8,
            W_fine_patch=8,
            H_low=H,
            W_low=W,
        )
        up_flow = einops.rearrange(
            up_flow,
            (
                "B xy H_fine_patch W_fine_patch H_low W_low -> "
                "B xy (H_low H_fine_patch) (W_low W_fine_patch)"
            ),
            B=B,
            xy=xy,
            H_fine_patch=8,
            W_fine_patch=8,
            H_low=H,
            W_low=W,
        )
        return up_flow

    def forward(
        self,
        image1,
        image2,
        iters=12,
        flow_init=None,
        upsample=True,
        test_mode=False,
        return_cost_volume=False,
        return_flow_and_cost_volume=False,
    ):
        """Estimate optical flow between pair of frames"""

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        flow_up = None
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions. Save a bit of time and do not upsample in each iteration, unless needed
            if (
                (not return_flow_and_cost_volume)
                and (not return_cost_volume)
                and (not test_mode)
            ):
                if up_mask is None:
                    flow_up = upflow8(coords1 - coords0)
                else:
                    # flow_up_orig = self.upsample_flow(coords1 - coords0, up_mask)
                    flow_up = self.upsample_flow_einops(coords1 - coords0, up_mask)
                    # assert torch.all(torch.abs(flow_up - flow_up_orig) < 1e-8), f"diff max {torch.max(torch.abs(flow_up - flow_up_orig))}"
                flow_predictions.append(flow_up)

        if flow_up is None:
            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                # flow_up_orig = self.upsample_flow((coords1 - coords0), up_mask)
                flow_up = self.upsample_flow_einops((coords1 - coords0), up_mask)
                # assert torch.all(torch.abs(flow_up - flow_up_orig) < 1e-8), f"diff max {torch.max(torch.abs(flow_up - flow_up_orig))}"

        if return_flow_and_cost_volume:
            # flow_up: (batch, xy-delta, H, W)
            # cost_volume: (batch * H1 * W1, 1, H2, W2)
            cost_volume = corr_fn.corr_pyramid[0]
            return flow_up, cost_volume

        if return_cost_volume:
            return corr_fn.corr_pyramid[0]

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
