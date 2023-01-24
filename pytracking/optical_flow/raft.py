import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import einops
import numpy as np
from types import SimpleNamespace
from timeit import default_timer as timer
import logging
logger = logging.getLogger(__name__)

from pytracking.utils.various_utils import SparseExceptionLogger
from pytracking.utils.misc import torch_get_featuremap_coords, get_featuremap_coords
try:
    from pytracking.utils.wraft_gui import wRAFTGUI
except ImportError:
    print('no wRAFT GUI')
    wRAFTGUI = None
from pytracking.utils.caching import load_cached_flow

raft_code_path = (Path(__file__).parent.parent /  # pytracking
                  'external' / 'RAFT').resolve()
sys.path.append(str(raft_code_path))
from raft_core.raft import RAFT
from raft_core.weighted_raft import WeightedRAFT
from raft_core.utils.utils import InputPadder


class RAFTWrapper():
    def __init__(self, config):
        self.C = config
        args = SimpleNamespace()
        args.model = config.class_params.model
        args.small = config.class_params.small
        args.mixed_precision = config.class_params.mixed_precision
        args.alternate_corr = config.class_params.alternate_corr
        args.weight_head_structure = config.class_params.weight_head_structure
        if config.class_params.mask_estimation:
            args.mask_estimation = config.class_params.mask_estimation
            args.mask_head_structure = config.class_params.mask_head_structure
        else:
            args.mask_estimation = False

        # net = torch.nn.DataParallel(RAFT(args), device_ids=[0])
        if self.C.raft_type == 'orig':
            net = torch.nn.DataParallel(RAFT(args))
        elif self.C.raft_type == 'weighted':
            net = torch.nn.DataParallel(WeightedRAFT(args))
        elif self.C.raft_type == 'weighted_masked':
            net = torch.nn.DataParallel(WeightedRAFT(args))
        else:
            raise ValueError(f"Unknown RAFT type {self.C.raft_type}")

        logger.info(f"Loading weights from: {self.C.model}")
        state_dict = torch.load(self.C.model)
        if config.add_module_to_statedict:
            state_dict = {f"module.{key}": value for key, value in state_dict.items()}

        if config.backbone_model:
            # will overwrite backbone later, do not load it now
            # (useful when we want to use weights pre-trained on standard RAFT for small-RAFT).
            def in_backbone(name):
                return ('fnet' in name) or ('cnet' in name) or ('update_block' in name)
            state_dict = {k: v for k, v in state_dict.items() if not in_backbone(k)}
        net.load_state_dict(state_dict, strict=not config.non_strict_loading)
        net = net.module  # unwraps dataparallel???
        net.to('cuda')
        net.requires_grad_(False)
        net.eval()
        self.net = net
        self.sparse_logger = SparseExceptionLogger(logger, extra_starts=['[Errno 2] No such file or directory:'])

    def postprocess_weights(self, flat_weights, fn):
        flow_shape = self.last_flow_shape
        weights = einops.rearrange(flat_weights, 'b (h w) -> b 1 h w',
                                   b=flow_shape['batch'], h=flow_shape['H'], w=flow_shape['W'])
        weights = fn(weights)
        flat = einops.rearrange(weights, 'b 1 h w -> b (h w)')
        return flat

    def compute_flow(self, src_img, dst_img, mode='TC', vis=False, src_img_identifier=None,
                     numpy_out=False, do_sigmoid=False):
        """
        args:
            src_img: (H, W, 3) uint8 BGR opencv image
            dst_img: (H, W, 3) uint8 BGR opencv image
            mode: one of "flow" or "TC"
        """
        assert mode in ['flow', 'TC']
        assert src_img.shape == dst_img.shape

        cached_results = False
        if src_img_identifier is not None:
            try:
                flow, weights = load_cached_flow(src_img, self.C.flow_cache_dir, src_img_identifier)
                flow = einops.rearrange(flow, 'xy H W -> 1 xy H W', xy=2)
                if not numpy_out:
                    flow = torch.from_numpy(flow).to('cuda')

                if weights is not None and weights.size > 1:
                    weights_up = einops.rearrange(weights, 'c H W -> 1 c H W', c=1)
                    if not numpy_out:
                        weights_up = torch.from_numpy(weights_up).to('cuda')
                else:
                    weights_up = None
                cached_results = True
                logger.debug('Using pre-computed flow.')
            except Exception as ex:
                self.sparse_logger("no cached flow", ex)

        if not cached_results:
            # RGB!
            H, W = src_img.shape[:2]
            src_img = src_img[:, :, ::-1].copy()  # convert to RGB and make c-contiguous
            src_img = torch.from_numpy(src_img).permute(2, 0, 1).float()
            src_img = src_img[None].to('cuda')

            dst_img = dst_img[:, :, ::-1].copy()  # convert to RGB and make c-contiguous
            dst_img = torch.from_numpy(dst_img).permute(2, 0, 1).float()
            dst_img = dst_img[None].to('cuda')

            if self.C.padding_mode == 'RAFT':
                padder = InputPadder(dst_img.shape)
            elif self.C.padding_mode == 'Michal':
                padder = MichalPadder(dst_img.shape)
            elif self.C.padding_mode == 'nopad':
                padder = NoPadder(dst_img.shape)
            elif self.C.padding_mode == 'crop':
                padder = CropPadder(dst_img.shape)
            else:
                raise ValueError(f"invalid padding_mode '{self.C.padding_mode}'")
            src_img, dst_img = padder.pad(src_img, dst_img)

            start_time = timer()
            if self.C.raft_type == 'orig':
                flow_low, flow_up = self.net(src_img, dst_img, flow_init=None, iters=self.C.iters, test_mode=True)
                weights_low, weights_up, mask_up = None, None, None
            elif self.C.raft_type == 'weighted':
                flow_low, flow_up, cost_volume, weights_low, weights_up = \
                    self.net(src_img, dst_img, flow_init=None, iters=self.C.iters, test_mode=True)
                mask_up = None
            elif self.C.raft_type == 'weighted_masked':
                flow_low, flow_up, cost_volume, weights_low, weights_up, mask_up = \
                    self.net(src_img, dst_img, flow_init=None, iters=self.C.iters, test_mode=True)
            flow_time = float(timer() - start_time)
            logger.debug(f"flow_time [s]: {flow_time}")

            flow = padder.unpad(flow_up)
            weights_up = padder.unpad(weights_up)
            mask_up = padder.unpad(mask_up)

        if self.C.weights_postprocessing_fn:
            weights_up = self.C.weights_postprocessing_fn(weights_up)
        if vis and (wRAFTGUI is not None):
            gui = wRAFTGUI(src_img, dst_img, flow_up, weights_up, flow_low, weights_low)
            gui.run()

        if do_sigmoid:
            weights_up = sigmoid(weights_up)

        if mode == 'flow':
            flow = einops.rearrange(flow, 'batch delta H W -> (batch delta) H W', batch=1)
            if self.C.raft_type in ['weighted', 'weighted_masked']:
                weights_up = einops.rearrange(
                    weights_up,
                    'batch 1 H W -> (batch 1) H W')
            else:
                weights_up = None

            if self.C.raft_type in ['weighted_masked']:
                mask_up = einops.rearrange(mask_up, 'batch 1 H W -> (batch 1) H W')

            if numpy_out and isinstance(flow, torch.Tensor):
                flow = flow.detach().cpu().numpy()
                if weights_up is not None:
                    weights_up = weights_up.detach().cpu().numpy()
                if mask_up is not None:
                    mask_up = mask_up.detach().cpu().numpy()

            if self.C.raft_type in ['weighted_masked']:
                return flow, weights_up, mask_up
            else:
                return flow, weights_up

        elif mode == 'TC':
            flow_flat = einops.rearrange(flow, 'batch delta H W -> (batch delta) (H W)', batch=1)
            flow_shape = einops.parse_shape(flow, 'batch delta H W')
            self.last_flow_shape = flow_shape
            if isinstance(flow_flat, torch.Tensor):
                src_coords = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']), device=flow_flat.device)
            else:
                src_coords = get_featuremap_coords((flow_shape['H'], flow_shape['W']))
            dst_coords = src_coords + flow_flat
            if self.C.raft_type in ['weighted', 'weighted_masked']:
                weights_up_flat = einops.rearrange(
                    weights_up,
                    'batch 1 H W -> (batch 1) (H W)')
            else:
                weights_up_flat = None

            if self.C.raft_type in ['weighted_masked']:
                mask_up_flat = einops.rearrange(mask_up, 'batch 1 H W -> (batch 1) (H W)')
            else:
                mask_up_flat = None

            if numpy_out and isinstance(src_coords, torch.Tensor):
                src_coords = src_coords.detach().cpu().numpy()
                dst_coords = dst_coords.detach().cpu().numpy()
                if weights_up_flat is not None:
                    weights_up_flat = weights_up_flat.detach().cpu().numpy()

                if mask_up_flat is not None:
                    mask_up_flat = mask_up_flat.detach().cpu().numpy()

            if self.C.raft_type in ['weighted_masked']:
                return src_coords, dst_coords, weights_up_flat, mask_up_flat
            else:
                return src_coords, dst_coords, weights_up_flat


class NoPadder():
    def __init__(self, shape):
        """ do not allow input sizes that are not a multiple of 8 """
        B, C, H, W = shape
        assert H % 8 == 0
        assert W % 8 == 0

    def pad(self, src_img, dst_img):
        return src_img, dst_img

    def unpad(self, flow):
        return flow


class CropPadder():
    def __init__(self, shape):
        """ crop the input size to a multiple of 8 (from right and bottom) """
        B, C, H, W = shape
        self.crop_H, self.crop_W = (H // 8) * 8, (W // 8) * 8

    def pad(self, src_img, dst_img):
        src = src_img[:, :, :self.crop_H, :self.crop_W]
        dst = dst_img[:, :, :self.crop_H, :self.crop_W]
        return src, dst

    def unpad(self, flow):
        return flow


class MichalPadder():
    def __init__(self, shape):
        """ Instead of padding the images, just bilinearly upscale them to multiple of 8 """
        B, C, H, W = shape
        self.h_orig, self.w_orig = H, W

        self.h_new = int(np.ceil(float(H) / 8) * 8)
        self.w_new = int(np.ceil(float(W) / 8) * 8)

    def pad(self, src_img, dst_img):
        src = F.interpolate(src_img, size=(self.h_new, self.w_new), mode='bilinear')
        dst = F.interpolate(dst_img, size=(self.h_new, self.w_new), mode='bilinear')
        return src, dst

    def unpad(self, flow):
        h_old, w_old = flow.shape[2:]
        assert h_old == self.h_new
        assert w_old == self.w_new
        flow_r = F.interpolate(flow, size=(self.h_orig, self.w_orig), mode='bilinear')
        flow_r_u = flow_r[:, 0:1, :, :] * self.w_orig / self.w_new
        flow_r_v = flow_r[:, 1:2, :, :] * self.h_orig / self.h_new
        return torch.cat([flow_r_u, flow_r_v], dim=1)


def sigmoid(xs):
    if xs is None:
        return None
    elif isinstance(xs, torch.Tensor):
        return torch.sigmoid(xs)
    else:
        return 1 / (1 + np.exp(-xs))
