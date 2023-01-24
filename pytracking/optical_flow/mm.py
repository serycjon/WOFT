import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
from types import SimpleNamespace
# from timeit import default_timer as timer
from pytracking.utils.timing import cuda_time_measurer as time_measurer
# from pytracking.utils.timing import time_measurer
import mmflow.apis
import logging
logger = logging.getLogger(__name__)

from pytracking.utils.various_utils import SparseExceptionLogger
from pytracking.utils.misc import torch_get_featuremap_coords, get_featuremap_coords
from pytracking.utils.wraft_gui import wRAFTGUI
from pytracking.utils.caching import load_cached_flow
from pytracking.utils.timing import cuda_time_measurer as time_measurer


class MMFlowWrapper():
    def __init__(self, config):
        self.C = config

        self.model = mmflow.apis.init_model(self.C.mm_config_file, self.C.mm_checkpoint_file, device='cuda')
        self.last_cost_volumes = None

        def hook_fn(module, module_input, module_output):
            cost_volumes = module_output[0]
            self.last_cost_volumes = cost_volumes

        if not self.C.weight_head_level or self.C.weight_head_level == 3:
            self.model.decoder.decoders.level3.NetM.corr_up.register_forward_hook(hook_fn)
        elif self.C.weight_head_level == 4:
            self.model.decoder.decoders.level4.NetM.corr_up.register_forward_hook(hook_fn)
        elif self.C.weight_head_level == 5:
            self.model.decoder.decoders.level5.NetM.corr_up.register_forward_hook(hook_fn)
        elif self.C.weight_head_level == 6:
            self.model.decoder.decoders.level6.NetM.corr_up.register_forward_hook(hook_fn)
        else:
            raise ValueError("Unknown weight head level")
        self.weight_head = WeightHead(self.C.weight_head_structure)

        if self.C.model:
            logger.info(f"Loading weights from: {self.C.model}")
            state_dict = torch.load(self.C.model)
            self.weight_head.load_state_dict(state_dict, strict=True)

        self.weight_head.to('cuda')
        self.weight_head.requires_grad_(False)
        self.weight_head.eval()
        self.model.requires_grad_(False)
        self.model.eval()

    def compute_flow(self, src_img, dst_img, mode='TC', vis=False, src_img_identifier=None,
                     numpy_out=False, do_sigmoid=False):
        """
        args:
            src_img: (H, W, 3) uint8 BGR opencv image
            dst_img: (H, W, 3) uint8 BGR opencv image
            mode: one of "flow" or "TC"
        """
        H, W = src_img.shape[:2]
        # the_flow_timer = time_measurer('ms')
        flow = mmflow.apis.inference_model(self.model, img1s=src_img, img2s=dst_img)
        # logger.debug(f"the_flow_timer(): {the_flow_timer()}ms")
        # weight_timer = time_measurer('ms')
        weights_low = self.weight_head(self.last_cost_volumes)
        # logger.debug(f"weight_timer(): {weight_timer()}ms")
        # weight_up_timer = time_measurer('ms')
        weights_up = F.interpolate(weights_low, size=(H, W), mode='bilinear', align_corners=False)
        # logger.debug(f"weight_up_timer(): {weight_up_timer()}ms")
        if do_sigmoid:
            weights_up = sigmoid(weights_up)

        # flow = torch.from_numpy(flow).to('cuda')
        mask_up = None

        if self.C.weights_postprocessing_fn:
            raise NotImplementedError("weights postprocessing not implemented!")

        if mode == 'flow':
            flow = einops.rearrange(flow, 'H W delta -> delta H W', delta=2)
            if weights_up is not None:
                weights_up = einops.rearrange(
                    weights_up,
                    'batch 1 H W -> (batch 1) H W')

            if mask_up is not None:
                mask_up = einops.rearrange(mask_up, 'batch 1 H W -> (batch 1) H W')

            if numpy_out and isinstance(flow, torch.Tensor):
                flow = flow.detach().cpu().numpy()
                if weights_up is not None:
                    weights_up = weights_up.detach().cpu().numpy()
                if mask_up is not None:
                    mask_up = mask_up.detach().cpu().numpy()

            if mask_up is not None:
                return flow, weights_up, mask_up
            else:
                return flow, weights_up

        elif mode == 'TC':
            flow_flat = einops.rearrange(flow, 'H W delta -> delta (H W)', delta=2)
            flow_shape = einops.parse_shape(flow, 'H W delta')
            self.last_flow_shape = flow_shape
            if isinstance(flow_flat, torch.Tensor):
                src_coords = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']), device=flow_flat.device)
            else:
                src_coords = get_featuremap_coords((flow_shape['H'], flow_shape['W']))
            dst_coords = src_coords + flow_flat
            if weights_up is not None:
                weights_up_flat = einops.rearrange(
                    weights_up,
                    'batch 1 H W -> (batch 1) (H W)')
            else:
                weights_up_flat = None

            if mask_up is not None:
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

            if mask_up_flat is not None:
                return src_coords, dst_coords, weights_up_flat, mask_up_flat
            else:
                return src_coords, dst_coords, weights_up_flat


class WeightHead(nn.Module):
    def __init__(self, net_structure):
        super(WeightHead, self).__init__()

        layers = []
        cur_channels = 1
        for data in net_structure:
            if isinstance(data, (list, tuple)):
                new_channels, kernel = data
                assert kernel % 2 == 1
                padding = kernel // 2
            else:
                new_channels = data
                kernel = 3
                padding = 1
            layers.append(nn.Conv2d(cur_channels, new_channels,
                                    kernel_size=kernel, padding=padding))
            layers.append(nn.ReLU())
            cur_channels = new_channels
        layers.append(nn.Conv2d(cur_channels, 1, kernel_size=1, padding=0))

        self.net = nn.Sequential(*layers)
        self.local_window = 7
        # self.init_bias()

    def init_bias(self):
        last_conv = self.net[-1]
        last_conv.bias.data.fill_(4)
        logger.info(f"Initializing high bias ({float(last_conv.bias.data)}) in the last weight estimation conv layer.")

    def forward(self, cost_volumes):
        """Computes flow weights from a cost_volume.
        The weights are to be used in a weighted least squares solver

        args:
            cost_volumes: ((H_local * W_local), H_feat, W_feat)
        returns:
            weights: (1, H_feat, W_feat) tensor
        """

        # a_timer = time_measurer('ms')
        # logger.debug(f"cost_volumes.shape: {cost_volumes.shape}")
        batched = einops.rearrange(cost_volumes, '(H_local W_local) H_feat W_feat -> (H_feat W_feat) 1 H_local W_local',
                                   H_local=self.local_window, W_local=self.local_window)
        # logger.debug(f"batched.shape: {batched.shape}")
        # logger.debug(f"a_timer(): {a_timer()}ms")

        # b_timer = time_measurer('ms')
        res = self.net(batched)
        # logger.debug(f"b_timer(): {b_timer()}ms")

        # c_timer = time_measurer('ms')
        weight_logits = einops.reduce(res,
                                      '(H_feat W_feat) 1 H_local W_local -> 1 1 H_feat W_feat',
                                      reduction='mean',
                                      **einops.parse_shape(cost_volumes, '_ H_feat W_feat'))
        # logger.debug(f"c_timer(): {c_timer()}ms")
        return weight_logits


def sigmoid(xs):
    if xs is None:
        return None
    elif isinstance(xs, torch.Tensor):
        return torch.sigmoid(xs)
    else:
        return 1 / (1 + np.exp(-xs))
