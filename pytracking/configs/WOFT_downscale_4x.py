from pathlib import Path
import numpy as np
import torch
import einops

from pytracking.tracker.YAOF_tracker_single_control import YAOFTrackerSingleControl
from pytracking.utils.config import Config, load_config
from pytracking.utils.least_squares_H import find_homography_nonhomogeneous_QR, torch_proj_errors

import logging
logger = logging.getLogger(__name__)


def redet_success_fn(H_prewarped2init, template_coords, cur_pw_coords, weights):
    errs = torch_proj_errors(H_prewarped2init,
                             einops.rearrange(cur_pw_coords, 'xy N -> 1 xy N', xy=2),
                             einops.rearrange(template_coords, 'xy N -> 1 xy N', xy=2))
    inliers = errs <= 5
    inlier_frac = torch.mean(inliers.float())
    logger.debug(f"inlier_frac: {inlier_frac}")
    return inlier_frac > 0.2


def find_homography(pts_A, pts_B, weights=None):
    N_pts = pts_A.shape[1]  # pts have (1 N xy) shape
    logger.debug(f"# TCs: {N_pts}")
    res = find_homography_nonhomogeneous_QR(pts_A, pts_B, weights=weights)
    return res


def subsampler(coords_a, coords_b, weights):
    assert coords_a.shape == coords_b.shape
    N_pts = coords_a.shape[1]
    assert weights.shape == (1, N_pts)

    to_draw = 500
    if to_draw >= N_pts:
        return coords_a, coords_b, weights

    subsampling_mask = np.zeros(N_pts) > 0

    # if weights is not None:
    #     reliable_points = einops.rearrange(weights, '1 N -> N') > 0.5
    #     if isinstance(reliable_points, torch.Tensor):
    #         reliable_points = reliable_points.cpu().numpy()
    #     subsampling_mask[reliable_points] = True

    eng = torch.quasirandom.SobolEngine(dimension=1)
    indices = eng.draw(to_draw).cpu().numpy().flatten()
    indices = np.round(N_pts * indices).astype(np.int32)
    subsampling_mask[indices] = True

    return coords_a[:, subsampling_mask], coords_b[:, subsampling_mask], weights[:, subsampling_mask]


def get_config():
    conf = Config()

    conf.tracker_class = YAOFTrackerSingleControl
    repo_root = (Path(__file__).parent.parent.parent).resolve()
    conf.flow_config = load_config(repo_root / 'pytracking/optical_flow/configs/v2_SNOB_large_g05_RAFT.py')
    conf.flow_config.padding_mode = 'RAFT'
    conf.flow_config.weights_postprocessing_fn = None
    conf.flow_numpy_out = False

    conf.H_estimator = find_homography
    conf.redet_success_fn = redet_success_fn
    conf.pw_mask = True
    conf.no_prewarp_after_N = 10
    conf.subsampler_fn = subsampler

    conf.downscale_inputs = 4

    return conf
