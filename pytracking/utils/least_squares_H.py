# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import argparse
import numpy as np
import einops
import cv2
import os
import torch
import torch.nn.functional as F
import kornia
import kornia.geometry.conversions as kgc
import kornia.geometry.homography as kgh
# from kornia.utils import _extract_device_dtype
import pandas as pd
import tqdm
from pathlib import Path
import json
import datetime
import logging
logger = logging.getLogger(__name__)

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

import pytracking.utils.geom_utils as gu
from pytracking.utils.various_utils import with_debugger
from pytracking.utils import vis_utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-lr', '--learning_rate', help='', type=float, default=1e-4)
    parser.add_argument('--softmax', help='use softmax weights', action='store_true')
    parser.add_argument('--sigmoid', help='use sigmoid weights', action='store_true')
    parser.add_argument('--relu', help='use relu on soft weights', action='store_true')
    parser.add_argument('--abs', help='use abs on soft weights', action='store_true')
    parser.add_argument('--N_iter', help='number gradient descent interations', type=int, default=1000)
    parser.add_argument('--inl_frac', help='fraction of inliers', type=float, default=0.5)
    parser.add_argument('--seed', help='random seed', type=int)
    parser.add_argument('--verbose', help='', action='store_true')
    parser.add_argument('--opt', help='optimizer', choices=['adamw', 'sgd'], default='adamw')
    parser.add_argument('--experiment_length', help='number of iterations of the whole experiment', type=int, default=1)
    parser.add_argument('--plot', help='just plot the results', type=Path)
    parser.add_argument('--vis_TC', help='just visualize the TC data', action='store_true')
    parser.add_argument('--sgd_batch', help='SGD TC subsample size', type=int)
    parser.add_argument('--hard_H', help='use the harder Homography', action='store_true')

    return parser.parse_args()


def find_homography_nonhomogeneous(points1, points2, weights=None):
    r"""Compute the homography matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 4 Points algorithm.
    Using a nonhomogenous system of equations (fixing the bottom-right element of H - H_{3,3} = 1.
    Limitation: unable to estimate H with H_{3,3} = 0, which can happen.
    see Hartley, Zisserman; Multiple View Geometry, 2nd edition Sec. 4.1.2, Example 4.1 on page 90.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    if not (len(points1.shape) >= 1 and points1.shape[-1] == 2):
        raise AssertionError(points1.shape)
    if points1.shape[1] < 4:
        raise AssertionError(points1.shape)

    # device, dtype = _extract_device_dtype([points1, points2])

    eps = 1e-8
    points1_norm, transform1 = kornia.geometry.epipolar.normalize_points(points1)
    points2_norm, transform2 = kornia.geometry.epipolar.normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1], dim=-1)  # batch, N, 8
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1], dim=-1)   # batch, N, 8
    # interleave the two parts, such that we get the standard form (each correspondence gives two consecutive rows in A)
    A = einops.rearrange(torch.cat((ax, ay), dim=-1), 'B N (two eight) -> B (N two) eight', two=2, eight=8)
    # A_orig = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])   # batch, 2xN, 8
    # assert torch.all(A == A_orig)

    bx = -y2  # batch, N, 1
    by = x2   # batch, N, 1
    b = einops.rearrange(torch.cat((bx, by), dim=-1), 'B N (two one) -> B (N two) one', two=2, one=1)

    if weights is not None:
        # to really be the least squares weights, we would have to use square root of w
        # (to optimize \sum_i w_i (A x_i - b_i)^2, we should multiply both A and b by a diagonal matrix containing sqrt(w_i))

        # splice the weights to each of the two equations (each weight repeated 2 times)
        w = einops.repeat(weights, 'B N -> B (N repeat) 1', repeat=2)
        A = w * A
        b = w * b

    solution = torch.linalg.pinv(A) @ b

    # res = torch.linalg.lstsq(A, b)
    # solution = res.solution  # batch, 8, 1
    # add the H_{3,3} = 1 element
    solution = torch.cat([solution, torch.ones((solution.shape[0], 1, 1), dtype=solution.dtype, device=solution.device)],
                         dim=1)
    H = einops.rearrange(solution, 'B (rows cols) 1 -> B rows cols',
                         rows=3, cols=3)
    H = transform2.inverse() @ (H @ transform1)
    H_norm = H / (H[..., -1:, -1:] + eps)
    # if weights is None:
    #     # All points are equally important
    #     A = A.transpose(-2, -1) @ A
    # else:
    #     # We should use provided weights
    #     if not (len(weights.shape) == 2 and weights.shape == points1.shape[:2]):
    #         raise AssertionError(weights.shape)
    #     w_diag = torch.diag_embed(weights.unsqueeze(dim=-1).repeat(1, 1, 2).reshape(weights.shape[0], -1))
    #     A = A.transpose(-2, -1) @ w_diag @ A

    # try:
    #     _, _, V = torch.svd(A)
    # except RuntimeError:
    #     warnings.warn('SVD did not converge', RuntimeWarning)
    #     return torch.empty((points1_norm.size(0), 3, 3), device=device, dtype=dtype)

    # H = V[..., -1].view(-1, 3, 3)
    # H = transform2.inverse() @ (H @ transform1)
    # H_norm = H / (H[..., -1:, -1:] + eps)
    return H_norm


def find_homography_nonhomogeneous_QR(points1, points2, weights=None):
    r"""Compute the homography matrix using the DLT formulation.

    The linear system is solved by using the Weighted Least Squares Solution for the 4 Points algorithm.
    Using a nonhomogenous system of equations (fixing the bottom-right element of H - H_{3,3} = 1.
    Limitation: unable to estimate H with H_{3,3} = 0, which can happen.
    see Hartley, Zisserman; Multiple View Geometry, 2nd edition Sec. 4.1.2, Example 4.1 on page 90.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    if not (len(points1.shape) >= 1 and points1.shape[-1] == 2):
        raise AssertionError(points1.shape)
    if points1.shape[1] < 4:
        raise AssertionError(points1.shape)

    # device, dtype = _extract_device_dtype([points1, points2])

    eps = 1e-8
    points1_norm, transform1 = kornia.geometry.epipolar.normalize_points(points1)
    points2_norm, transform2 = kornia.geometry.epipolar.normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1], dim=-1)  # batch, N, 8
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1], dim=-1)   # batch, N, 8
    # interleave the two parts, such that we get the standard form (each correspondence gives two consecutive rows in A)
    A = einops.rearrange(torch.cat((ax, ay), dim=-1), 'B N (two eight) -> B (N two) eight', two=2, eight=8)
    # A_orig = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])   # batch, 2xN, 8
    # assert torch.all(A == A_orig)

    bx = -y2  # batch, N, 1
    by = x2   # batch, N, 1
    b = einops.rearrange(torch.cat((bx, by), dim=-1), 'B N (two one) -> B (N two) one', two=2, one=1)

    if weights is not None:
        # to really be the least squares weights, we would have to use square root of w
        # (to optimize \sum_i w_i (A x_i - b_i)^2, we should multiply both A and b by a diagonal matrix containing sqrt(w_i))

        # splice the weights to each of the two equations (each weight repeated 2 times)
        w = einops.repeat(weights, 'B N -> B (N repeat) 1', repeat=2)
        A = w * A
        b = w * b

    res = torch.linalg.qr(A)
    # now we need to solve Rx = Q^T b
    # res.Q has shape (batch, 2xN, 8)
    lhs = res.R
    rhs = einops.rearrange(res.Q, 'B twoN eight -> B eight twoN', eight=8) @ b

    res = torch.triangular_solve(rhs, lhs)
    solution = res.solution  # batch, 8, 1
    # add the H_{3,3} = 1 element
    solution = torch.cat([solution, torch.ones((solution.shape[0], 1, 1), dtype=solution.dtype, device=solution.device)],
                         dim=1)
    H = einops.rearrange(solution, 'B (rows cols) 1 -> B rows cols',
                         rows=3, cols=3)
    H = transform2.inverse() @ (H @ transform1)
    H_norm = H / (H[..., -1:, -1:] + eps)
    return H_norm


def find_homography_dlt(points1, points2, weights=None):
    r"""Compute the homography matrix using the DLT formulation.

    less RAM hungry version of Kornia find_homography_dlt

    The linear system is solved by using the Weighted Least Squares Solution for the 4 Points algorithm.

    Args:
        points1: A set of points in the first image with a tensor shape :math:`(B, N, 2)`.
        points2: A set of points in the second image with a tensor shape :math:`(B, N, 2)`.
        weights: Tensor containing the weights per point correspondence with a shape of :math:`(B, N)`.

    Returns:
        the computed homography matrix with shape :math:`(B, 3, 3)`.
    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    if not (len(points1.shape) >= 1 and points1.shape[-1] == 2):
        raise AssertionError(points1.shape)
    if points1.shape[1] < 4:
        raise AssertionError(points1.shape)

    # device, dtype = _extract_device_dtype([points1, points2])

    eps: float = 1e-8
    points1_norm, transform1 = kornia.geometry.epipolar.normalize_points(points1)
    points2_norm, transform2 = kornia.geometry.epipolar.normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    # DIAPO 11: https://www.uio.no/studier/emner/matnat/its/nedlagte-emner/UNIK4690/v16/forelesninger/lecture_4_3-estimating-homographies-from-feature-correspondences.pdf  # noqa: E501
    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1, y2], dim=-1)
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1, -x2], dim=-1)
    A = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])

    if weights is None:
        # All points are equally important
        A = A.transpose(-2, -1) @ A
    else:
        # We should use provided weights
        if not (len(weights.shape) == 2 and weights.shape == points1.shape[:2]):
            raise AssertionError(weights.shape)
        w = einops.repeat(weights, 'B N -> B (N repeat) 1', repeat=2)
        A = A.transpose(-2, -1) @ (w * A)

    _, _, V = torch.svd(A)

    H = V[..., -1].view(-1, 3, 3)
    H = transform2.inverse() @ (H @ transform1)
    H_norm = H / (H[..., -1:, -1:] + eps)
    return H_norm


def IRLSq_L1(residuals, eps=1e-8):
    return 1 / (torch.abs(residuals) + eps)


def IRLSq_Huber(residuals, k=1, eps=1e-8):
    """ L2 up to +-k, then L1 """
    abs_res = torch.abs(residuals)
    weights = 1 / (abs_res + eps)
    weights[abs_res < k] = 1
    return weights


def find_homography_IRLSq_QR(points1, points2, weights=None, reweighting_fn=IRLSq_L1, n_iter=5):
    """same as find_homography_QR, but using IRLSq to get m-estimator with different loss.

    With weights and L1 IRLSq, this problem is called Weber problem:
    https://en.wikipedia.org/wiki/Weber_problem#Iterative_solutions_of_the_Fermat,_Weber_and_attraction-repulsion_problems
    """
    if points1.shape != points2.shape:
        raise AssertionError(points1.shape)
    if not (len(points1.shape) >= 1 and points1.shape[-1] == 2):
        raise AssertionError(points1.shape)
    if points1.shape[1] < 4:
        raise AssertionError(points1.shape)
    if not points1.is_cuda:
        raise AssertionError("correspondences should be on GPU")

    eps = 1e-8
    points1_norm, transform1 = kornia.geometry.epipolar.normalize_points(points1)
    points2_norm, transform2 = kornia.geometry.epipolar.normalize_points(points2)

    x1, y1 = torch.chunk(points1_norm, dim=-1, chunks=2)  # BxNx1
    x2, y2 = torch.chunk(points2_norm, dim=-1, chunks=2)  # BxNx1
    ones, zeros = torch.ones_like(x1), torch.zeros_like(x1)

    ax = torch.cat([zeros, zeros, zeros, -x1, -y1, -ones, y2 * x1, y2 * y1], dim=-1)  # batch, N, 8
    ay = torch.cat([x1, y1, ones, zeros, zeros, zeros, -x2 * x1, -x2 * y1], dim=-1)   # batch, N, 8
    # interleave the two parts, such that we get the standard form (each correspondence gives two consecutive rows in A)
    A = einops.rearrange(torch.cat((ax, ay), dim=-1), 'B N (two eight) -> B (N two) eight', two=2, eight=8)
    # A_orig = torch.cat((ax, ay), dim=-1).reshape(ax.shape[0], -1, ax.shape[-1])   # batch, 2xN, 8
    # assert torch.all(A == A_orig)

    bx = -y2  # batch, N, 1
    by = x2   # batch, N, 1
    b = einops.rearrange(torch.cat((bx, by), dim=-1), 'B N (two one) -> B (N two) one', two=2, one=1)

    if weights is not None:
        # to really be the least squares weights, we would have to use square root of w
        # (to optimize \sum_i w_i (A x_i - b_i)^2, we should multiply both A and b by a diagonal matrix containing sqrt(w_i))

        # splice the weights to each of the two equations (each weight repeated 2 times)
        w = einops.repeat(weights, 'B N -> B (N repeat) 1', repeat=2)
        A = w * A
        b = w * b

    rew = torch.ones_like(b)
    for iteration in range(n_iter + 1):
        res = torch.linalg.qr(rew * A)
        # now we need to solve Rx = Q^T b
        # res.Q has shape (batch, 2xN, 8)
        lhs = res.R
        rhs = einops.rearrange(res.Q, 'B twoN eight -> B eight twoN', eight=8) @ (rew * b)

        res = torch.triangular_solve(rhs, lhs)
        solution = res.solution  # batch, 8, 1

        residuum = A @ solution - b

        # weighted least squares obtained by sqrt(w) * A * x = sqrt(w) * b
        rew = torch.sqrt(reweighting_fn(residuum))

    # add the H_{3,3} = 1 element
    solution = torch.cat([solution, torch.ones((solution.shape[0], 1, 1), dtype=solution.dtype, device=solution.device)],
                         dim=1)
    H = einops.rearrange(solution, 'B (rows cols) 1 -> B rows cols',
                         rows=3, cols=3)
    H = transform2.inverse() @ (H @ transform1)
    H_norm = H / (H[..., -1:, -1:] + eps)
    return H_norm


def find_homography_TRS(pts_A, pts_B, weights=None):
    ransac_pts_A = pts_A.detach().cpu().numpy()
    ransac_pts_B = pts_B.detach().cpu().numpy()
    Hs = []
    for b_id in range(ransac_pts_A.shape[0]):
        TRS_A, inliers = cv2.estimateAffinePartial2D(
            einops.rearrange(ransac_pts_A[b_id, ...], 'N xy -> N 1 xy', xy=2),
            einops.rearrange(ransac_pts_B[b_id, ...], 'N xy -> N 1 xy', xy=2),
            ransacReprojThreshold=3, maxIters=10000, confidence=0.999)
        H = np.concatenate((TRS_A, [[0, 0, 1]]), axis=0)
        eps = 1e-8
        H_norm = H / (H[2, 2] + eps)
        Hs.append(H_norm)
    Hs = torch.from_numpy(np.array(Hs)).to(pts_A.device)
    return Hs


def find_homography_cvransac(pts_A, pts_B, weights=None,
                             max_iters=10000, thr=1.4142, conf=0.995):
    N_pts = pts_A.shape[1]
    assert N_pts >= 4, "Not enough correspodences for RANSAC"
    ransac_pts_A = pts_A
    ransac_pts_B = pts_B

    using_torch = False
    if isinstance(pts_A, torch.Tensor):
        ransac_pts_A = ransac_pts_A.detach().cpu().numpy()
        ransac_pts_B = ransac_pts_B.detach().cpu().numpy()
        using_torch = True

    ransac_pts_A = einops.rearrange(ransac_pts_A, 'B N xy -> B N 1 xy', xy=2)
    ransac_pts_B = einops.rearrange(ransac_pts_B, 'B N xy -> B N 1 xy', xy=2)
    Hs = []
    for b_id in range(ransac_pts_A.shape[0]):
        H, _ = cv2.findHomography(ransac_pts_A[b_id, ...],
                                  ransac_pts_B[b_id, ...],
                                  method=cv2.RANSAC, maxIters=max_iters, ransacReprojThreshold=thr,
                                  confidence=conf)
        eps = 1e-8
        H_norm = H / (H[2, 2] + eps)
        Hs.append(H_norm)
    Hs = np.array(Hs)
    if using_torch:
        Hs = torch.from_numpy(Hs).to(pts_A.device)
    return Hs


def from_numpy(x):
    return torch.from_numpy(x).detach().clone()


def torch_reproj_errors(GT_H, est_H, pts_A):
    """
    args:
        GT_H: (B, 3, 3) batch of GT homographies mapping pts_A somewhere
        est_H: (B, 3, 3) batch of estimated homographies mapping pts_A somewhere
        pts_A: (B, 3, N) batch of N points in homogeneous coordinates"""
    # proj forward by GT_H, then backward by est_H, measure L2 errors
    # inv(est_H) * GT_H * pts_A
    # from pytorch docs https://pytorch.org/docs/stable/generated/torch.linalg.inv.html#torch.linalg.inv
    # Consider using torch.linalg.solve() if possible for multiplying a matrix on the left by the inverse, as:
    # torch.linalg.solve(A, B) == A.inv() @ B
    # It is always preferred to use solve() when possible, as it is faster and more numerically stable than computing the inverse explicitly.
    # BUT: for my current combination of pytorch / cuda / stuff, this gives error: RuntimeError: CUDA error: invalid configuration argument
    #      when used on more than approx 500 pts_A
    # reproj_pts = torch.linalg.solve(est_H, torch.matmul(GT_H, kgc.convert_points_to_homogeneous(pts_A.permute(0, 2, 1)).permute(0, 2, 1)))
    reproj_pts = torch.linalg.inv(est_H) @ torch.matmul(GT_H, kgc.convert_points_to_homogeneous(pts_A.permute(0, 2, 1)).permute(0, 2, 1))
    reproj_pts = kgc.convert_points_from_homogeneous(reproj_pts.permute(0, 2, 1)).permute(0, 2, 1)
    L2_err = torch.sqrt(einops.reduce(torch.square(reproj_pts - pts_A),
                                      'B xy N -> B N', xy=2, reduction='sum'))
    return L2_err


def torch_proj_diff_errors(GT_H, est_H, pts_A):
    """
    args:
        GT_H: (B, 3, 3) batch of GT homographies mapping pts_A somewhere
        est_H: (B, 3, 3) batch of estimated homographies mapping pts_A somewhere
        pts_A: (B, 2, N) batch of N points in homogeneous coordinates"""
    # proj forward by GT_H, then forward by est_H, measure L2 errors

    to_proj = kgc.convert_points_to_homogeneous(pts_A.permute(0, 2, 1)).permute(0, 2, 1)
    GT_proj_pts = torch.matmul(GT_H, to_proj)
    GT_proj_pts = kgc.convert_points_from_homogeneous(GT_proj_pts.permute(0, 2, 1)).permute(0, 2, 1)
    est_proj_pts = torch.matmul(est_H, to_proj)
    est_proj_pts = kgc.convert_points_from_homogeneous(est_proj_pts.permute(0, 2, 1)).permute(0, 2, 1)
    L2_err = torch.sqrt(einops.reduce(torch.square(GT_proj_pts - est_proj_pts),
                                      'B xy N -> B N', xy=2, reduction='sum'))
    return L2_err


def torch_e2p(pts):
    """convert points to homogeneous coordinates
    args:
        pts: (B, 2, N)
    returns:
        homo: (B, 3, N)
    """
    homogeneous = kgc.convert_points_to_homogeneous(einops.rearrange(pts, 'B xy N -> B N xy', xy=2))
    homogeneous = einops.rearrange(homogeneous, 'B N xyz -> B xyz N', xyz=3)
    return homogeneous


def torch_p2e(homo):
    """convert points from homogeneous coordinates
    args:
        homo: (B, 3, N)
    returns:
        pts: (B, 2, N)
    """
    pts = kgc.convert_points_from_homogeneous(einops.rearrange(homo, 'B xyz N -> B N xyz', xyz=3))
    pts = einops.rearrange(pts, 'B N xy -> B xy N', xy=2)
    return pts


def torch_H_proj(H, pts):
    """
    args:
        H: (B, 3, 3) batch of homographies
        pts: (B, 2, N) batch of points to be warped by the homographies
    """
    proj_pts = torch_p2e(torch.matmul(H, torch_e2p(pts)))
    return proj_pts


def torch_proj_errors(GT_H, pts_A, pts_B):
    """Compute L2 distance between given correspondences and correspondences created by H-warp

    args:
        GT_H: (B, 3, 3) batch of GT homographies mapping pts_A somewhere
        pts_A: (B, 3, N) batch of N source points
        pts_A: (B, 3, N) batch of N destination points
    returns:
        L2_err: (B, N) L2 distances
    """
    # proj forward pts_A by GT_H, measure L2 errors
    proj_pts = torch.matmul(GT_H, kgc.convert_points_to_homogeneous(pts_A.permute(0, 2, 1)).permute(0, 2, 1))
    proj_pts = kgc.convert_points_from_homogeneous(proj_pts.permute(0, 2, 1)).permute(0, 2, 1)
    L2_err = torch.sqrt(einops.reduce(torch.square(proj_pts - pts_B),
                                      'B xy N -> B N', xy=2, reduction='sum'))
    return L2_err


def reproj_errors(GT_H, est_H, pts_A, mean=True):
    # proj forward by GT_H, then backward by est_H, measure L2 errors
    forward_backward_H = gu.compose_H(GT_H, np.linalg.inv(est_H))

    reproj_pts = gu.H_proj(forward_backward_H, pts_A)
    L2_err = np.sqrt(einops.reduce(np.square(reproj_pts - pts_A),
                                   'xy N -> N', xy=2, reduction='sum'))
    if mean:
        return float(einops.reduce(L2_err, 'N -> 1', reduction='mean'))
    else:
        return L2_err
