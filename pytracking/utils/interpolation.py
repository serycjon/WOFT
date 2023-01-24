import numpy as np
import torch
import einops

from scipy.interpolate import RegularGridInterpolator
from pytracking.utils.misc import torch_get_featuremap_coords


def chain_flow(flow_AB, flow_BC):
    """ chain flow by bilinear interpolation """
    if flow_AB is None:
        return flow_BC

    flow_shape = einops.parse_shape(flow_AB, 'batch delta H W')
    coords_A = torch_get_featuremap_coords((flow_shape['H'], flow_shape['W']), device=flow_AB.device,
                                           keep_shape=True)
    coords_B = coords_A + flow_AB

    coords_C = bilinear_interpolate_torch(
        flow_BC,
        einops.rearrange(coords_B, 'xy H W -> xy (H W)', xy=2)[1, :],
        einops.rearrange(coords_B, 'xy H W -> xy (H W)', xy=2)[0, :])
    print(f"coords_C.shape: {coords_C.shape}")


class FlowInterpolator(object):
    def __init__(self, flow, additional_data=None):
        """
            flow: (H, W, 2) array of dx, dy pairs
            additional_data: (H, W, [C]) array.
        """
        H, W, C = flow.shape
        assert C == 2
        flow_grid_ys = np.arange(H)
        flow_grid_xs = np.arange(W)
        if additional_data is None:
            data = flow
        else:
            if len(additional_data.shape) < 3:
                additional_data = additional_data[:, :, np.newaxis]
            data = np.concatenate((flow, additional_data), axis=2)
        self.interp = RegularGridInterpolator((flow_grid_ys, flow_grid_xs), data, bounds_error=False, fill_value=np.nan)

    def __call__(self, positions, method='linear'):
        """
        args:
                positions: (N, 2) array of x, y pairs (possibly non-integer)
        """
        return self.interp(positions[:, ::-1], method=method)  # the scipy interpolator wants y, x coordinates


def interp_flow(current_positions, flow, occlusion_mask):
    """Interpolate flow and occlusion masks from non-integer starting point

    args:
        current_positions: (N, 2) array of y, x pairs (non-integer)
        flow: (H, W, 2) array of dx, dy pairs
        occlusion_mask: (H, W) array. Values > 0 mean that the pixel in left image is occluded in the right image
    """
    H, W, C = flow.shape
    assert C == 2

    flow_grid_ys = np.arange(H)
    flow_grid_xs = np.arange(W)
    interp_flow = RegularGridInterpolator((flow_grid_ys, flow_grid_xs), flow, bounds_error=False, fill_value=np.nan)
    new_flow = interp_flow(current_positions, method='linear')

    interp_occl = RegularGridInterpolator((flow_grid_ys, flow_grid_xs), occlusion_mask, bounds_error=False, fill_value=1)
    new_occl = interp_occl(current_positions, method='linear')

    return new_flow, new_occl


def flow_warp_coords(coords_A, flow_AB):
    """warp (non-integer) coordinates coords_A using optical flow.

    args:
        coords_A: (2, N) tensor with x, y coordinates
        flow_AB: (2, H, W) tensor with optical flow

    returns:
        coords_B: (2, N) tensor with x, y coordinates
    """
    sampled_flow = bilinear_interpolate_torch(
        flow_AB,
        coords_A[0, :],
        coords_A[1, :])
    coords_B = coords_A + sampled_flow
    return coords_B


def bilinear_interpolate_torch(data, x, y):
    """
    Bilinear interpolation of (CxHxW) im tensor

    args:
        data: (C, H, W) tensor with data to be interpolated
        x: (N, ) tensor with x coordinates into data
        y: (N, ) tensor with y coordinates into data
    returns:
        interp: (C, N) tensor with values interpolated from data

    https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
    """
    H, W = data.shape[1], data.shape[2]
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, W - 1)
    x1 = torch.clamp(x1, 0, W - 1)
    y0 = torch.clamp(y0, 0, H - 1)
    y1 = torch.clamp(y1, 0, H - 1)

    data_a = data[:, y0, x0]
    data_b = data[:, y1, x0]
    data_c = data[:, y0, x1]
    data_d = data[:, y1, x1]

    x0 = x0.float()
    x1 = x1.float()
    y0 = y0.float()
    y1 = y1.float()

    w_a = einops.rearrange((x1 - x) * (y1 - y), 'N -> 1 N')
    w_b = einops.rearrange((x1 - x) * (y - y0), 'N -> 1 N')
    w_c = einops.rearrange((x - x0) * (y1 - y), 'N -> 1 N')
    w_d = einops.rearrange((x - x0) * (y - y0), 'N -> 1 N')

    interp = (w_a * data_a) + (w_b * data_b) + (w_c * data_c) + (w_d * data_d)
    return interp
