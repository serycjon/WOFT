# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import numpy as np
import einops
import torch
import cv2
from pathlib import Path
from types import SimpleNamespace
import logging
from pytracking.utils.geom_utils import compose_H
from pytracking.utils import vis_utils
# from pytracking.utils.timing import cuda_time_measurer as time_measurer
from pytracking.utils.timing import time_measurer


logger = logging.getLogger(__name__)


class YAOFTrackerSingleControl():
    def __init__(self, config):
        self.C = config
        if self.C.subsampler_fn:
            self.C.subsampler_fn = make_forward_compatible(self.C.subsampler_fn)
        self.flower = config.flow_config.of_class(config.flow_config)
        self.device = 'cuda'

    def init(self, img, mask, img_identifier=None):
        if self.C.downscale_inputs:
            img = cv2.resize(img, None, fx=1 / self.C.downscale_inputs, fy=1 / self.C.downscale_inputs)
            mask = cv2.resize(mask, None, fx=1 / self.C.downscale_inputs, fy=1 / self.C.downscale_inputs)
            img_identifier = None  # used for loading cached flow - prevent doing this when scaling inputs

        self.template_img = img
        self.template_mask = torch.from_numpy(mask > 0).to(self.device)
        self.np_template_mask = mask

        template_contours, _ = cv2.findContours(np.uint8(mask > 0), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(template_contours) == 1
        self.template_contour = einops.rearrange(template_contours[0], 'N 1 xy -> xy N', xy=2)

        # tracker state:
        self.prev_H2init = np.eye(3)
        self.last_good_H2init = np.eye(3)
        self.prev_img_identifier = img_identifier
        self.prev_img = img
        self.fast_forward = False
        self.lost = False
        self.N_lost = 0

    def set_fast_meta(self, meta):
        self.fast_forward = True
        self.fast_forward_H2init = meta.estim_H_current2template
        self.fast_forward_meta = meta

        if self.C.downscale_inputs:
            raise NotImplementedError("Fastforward not compatible with input downscaling yet.")

    def track(self, input_img, debug=False, img_identifier=None):
        meta = SimpleNamespace()

        if self.C.downscale_inputs:
            input_img = cv2.resize(input_img, None, fx=1 / self.C.downscale_inputs, fy=1 / self.C.downscale_inputs)

        # fast-forward {{{
        if self.fast_forward:
            H_cur2init = self.fast_forward_H2init
            meta = self.fast_forward_meta
            self.last_good_H2init = H_cur2init
            self.lost = False
            self.N_lost = 0

            self.prev_img_identifier = img_identifier
            self.prev_img = input_img.copy()
            self.prev_H2init = H_cur2init
            self.fast_forward = False
            return H_cur2init, meta
        # }}}

        if self.C.no_prewarp_after_N and self.N_lost > self.C.no_prewarp_after_N:
            self.last_good_H2init = np.eye(3)

        meta.last_good_H2init = self.last_good_H2init.copy()

        ## estimate 'global' flow (0 -> pre-warped T) with pre-warp using the last good homography
        prewarp_timer = time_measurer('ms')
        prewarp_H = self.last_good_H2init
        extra_meta = False
        if extra_meta:
            meta.prewarp_H = prewarp_H
        prewarped_input = cv2.warpPerspective(input_img, prewarp_H,
                                              (input_img.shape[1], input_img.shape[0]),
                                              flags=cv2.INTER_LINEAR)
        pw_mask = cv2.warpPerspective(np.ones(input_img.shape[:2]), prewarp_H,
                                      (input_img.shape[1], input_img.shape[0]),
                                      flags=cv2.INTER_LINEAR)
        pw_mask = torch.from_numpy(pw_mask > 0).to(self.device)
        logger.debug(f"prewarp_timer(): {prewarp_timer()}ms")

        global_TC_timer = time_measurer('ms')
        # global_flow_timer = cuda_time_measurer('ms')
        global_flow_timer = time_measurer('ms')
        template_coords, cur_pw_coords, weights = self.flower.compute_flow(self.template_img, prewarped_input, mode='TC', vis=False,
                                                                           do_sigmoid=True)
        extra_vis = False
        if extra_vis:
            extra_vis_dir = Path('/ssd/export/YAOFT-extra-vis/')
            extra_vis_dir.mkdir(parents=True, exist_ok=True)
            extra_vis_name = f'{img_identifier[1]}--{img_identifier[2]:04d}'
            extra_vis_flow, extra_vis_weights = self.flower.compute_flow(self.template_img, prewarped_input, mode='flow', vis=False, do_sigmoid=True)
            np_weights = (to_numpy(extra_vis_weights) * 255).astype(np.uint8).reshape(input_img.shape[0], input_img.shape[1])
            import os
            import sys
            raft_path = Path(os.path.join(os.path.dirname(__file__), '..', 'external', 'RAFT')).resolve()
            sys.path.append(str(raft_path))

            cv2.imwrite(str(extra_vis_dir / f'{extra_vis_name}--weights.png'), np_weights)

            # from raft_core.utils.flow_viz import flow_to_image
            # np_flow = einops.rearrange(to_numpy(extra_vis_flow), 'xy H W -> H W xy', xy=2)
            # extra_vis_flow = flow_to_image(np_flow, clip_flow=10, convert_to_bgr=True)
            # cv2.imwrite(str(extra_vis_dir / f'{extra_vis_name}--flow.png'), extra_vis_flow)

            if img_identifier[2] == 1:
                cv2.imwrite(str(extra_vis_dir / f'{extra_vis_name}--template.png'), self.template_img)
            cv2.imwrite(str(extra_vis_dir / f'{extra_vis_name}--current.png'), prewarped_input)

        if extra_meta:
            meta.prewarped_flow = (to_numpy(template_coords).copy(), to_numpy(cur_pw_coords).copy(), to_numpy(weights).copy())
        post_hoc_weights = None
        if self.C.post_hoc_weights_postprocessing_fn:
            post_hoc_weights = self.flower.postprocess_weights(weights.clone(), self.C.post_hoc_weights_postprocessing_fn)
        logger.debug(f"global_flow_timer(): {global_flow_timer()}ms")

        mask_coords_timer = time_measurer('ms')
        template_coords, cur_pw_coords, weights, post_hoc_weights, pw_in_mask = self._mask_coords(template_coords, cur_pw_coords,
                                                                                                  weights, post_hoc_weights,
                                                                                                  pw_mask,
                                                                                                  do_pw_mask=not self.C.do_not_mask_TCs_by_prewarped)
        logger.debug(f"mask_coords_timer(): {mask_coords_timer()}ms")
        if extra_meta:
            meta.prewarped_masked_flow = (to_numpy(template_coords).copy(), to_numpy(cur_pw_coords).copy(), to_numpy(weights).copy())
        template_coords = to_float(template_coords)  # 2, N
        if self.C.subsampler_fn:
            subsampling_timer = time_measurer('ms')
            template_coords, cur_pw_coords, weights, post_hoc_weights = self.C.subsampler_fn(template_coords, cur_pw_coords,
                                                                                             weights, post_hoc_weights)
            logger.debug(f"subsampling_timer(): {subsampling_timer()}ms")
        if extra_meta:
            meta.prewarped_masked_subsampled_flow = (to_numpy(template_coords).copy(), to_numpy(cur_pw_coords).copy(), to_numpy(weights).copy())
        logger.debug(f"global_TC_timer(): {global_TC_timer()}ms")

        global_H_timer = time_measurer('ms')
        H_prewarped2init = to_float(
            self.C.H_estimator(einops.rearrange(cur_pw_coords, 'xy N -> 1 N xy', xy=2),
                               einops.rearrange(template_coords, 'xy N -> 1 N xy', xy=2),
                               weights))
        np_H_prewarped2init = to_numpy(H_prewarped2init)[0, :, :]
        H_global_cur2init = compose_H(prewarp_H, np_H_prewarped2init)
        meta.H_global_cur2init = H_global_cur2init.copy()
        logger.debug(f"global_H_timer(): {global_H_timer()}ms")

        global_check_timer = time_measurer('ms')
        global_H_success = self.C.redet_success_fn(H_prewarped2init, template_coords, cur_pw_coords,
                                                   post_hoc_weights if post_hoc_weights is not None else weights)
        logger.debug(f"global_check_timer(): {global_check_timer()}ms")
        logger.debug(f"global_H_success: {global_H_success}")

        if global_H_success:
            H_cur2init = H_global_cur2init
            self.lost = False
            self.N_lost = 0
        else:
            self.lost = True
            self.N_lost += 1

            if self.C.no_local_H:
                H_cur2init = H_global_cur2init
            else:
                ## compute the local flow (T-1 -> T) and estimate a homography.  Will be concatenated with the previous frame H
                local_TC_timer = time_measurer('ms')
                local_flow_timer = time_measurer('ms')
                prev_coords, cur_coords, weights = self.flower.compute_flow(self.prev_img, input_img, mode='TC',
                                                                            # src_img_identifier=self.prev_img_identifier,
                                                                            src_img_identifier=None,
                                                                            do_sigmoid=True, numpy_out=bool(self.C.flow_numpy_out))
                post_hoc_weights = None
                if self.C.post_hoc_weights_postprocessing_fn:
                    post_hoc_weights = self.flower.postprocess_weights(weights.clone(), self.C.post_hoc_weights_postprocessing_fn)
                logger.debug(f"local_flow_timer(): {local_flow_timer()}ms")
                prev_coords, cur_coords, weights, post_hoc_weights = self._mask_coords_flow(prev_coords, cur_coords, weights, post_hoc_weights)
                if self.C.subsampler_fn:
                    prev_coords, cur_coords, weights, post_hoc_weights = self.C.subsampler_fn(prev_coords, cur_coords, weights, post_hoc_weights)
                logger.debug(f"local_TC_timer(): {local_TC_timer()}ms")

                local_H_timer = time_measurer('ms')
                try:
                    H_flow = self.C.H_estimator(
                        einops.rearrange(cur_coords, 'xy N -> 1 N xy', xy=2),
                        einops.rearrange(to_float(prev_coords), 'xy N -> 1 N xy', xy=2),
                        weights)
                    H_flow = einops.rearrange(to_numpy(H_flow), '1 r c -> r c', r=3, c=3)
                    H_local_cur2init = compose_H(H_flow, self.prev_H2init)
                except Exception:
                    logger.warning("local flow RANSAC failed")
                    H_local_cur2init = self.prev_H2init
                meta.H_local_cur2init = H_local_cur2init.copy()
                logger.debug(f"local_H_timer(): {local_H_timer()}ms")
                H_cur2init = H_local_cur2init

        # vis {{{
        if debug:
            vis_timer = time_measurer('ms')

            pw_flow, pw_weights = self.flower.compute_flow(self.template_img, prewarped_input, mode='flow', numpy_out=True, do_sigmoid=True)
            pw_flow = einops.rearrange(to_numpy(pw_flow),
                                       'xy H W -> H W xy', xy=2)
            pw_colors = vis_utils.cv2_colormap(einops.rearrange(pw_weights, '1 H W -> H W'), vmin=0, vmax=1, do_colorbar=False)
            np_pw_in_mask = einops.rearrange(pw_in_mask.detach().cpu().numpy(),
                                             '(H W) -> H W', **einops.parse_shape(pw_colors, 'H W _'))
            pw_colors = np.dstack((pw_colors, np.ones(pw_colors.shape[:2], dtype=pw_colors.dtype)))
            pw_colors[np_pw_in_mask < 1, :] = 0
            # pw_colors[np_pw_in_mask > 0, :] = 255
            vis_pw_flow = vis_utils.vis_flow_align(pw_flow, self.template_img, prewarped_input, grid_sz=5,
                                                   arrow_color=einops.rearrange(pw_colors,
                                                                                'H W C -> (H W) C', C=4),
                                                   pt_radius=3, show_flow=False)
            cv2.imshow("cv: vis_pw_flow", vis_pw_flow)

            local_flow, _ = self.flower.compute_flow(self.prev_img, input_img, mode='flow',
                                                     src_img_identifier=self.prev_img_identifier, numpy_out=True)
            local_flow = einops.rearrange(to_numpy(local_flow),
                                          'xy H W -> H W xy', xy=2)
            vis_local_flow = vis_utils.vis_flow_align(local_flow, self.prev_img, input_img, grid_sz=30)
            cv2.imshow("cv: vis_local_flow", vis_local_flow)

            local_flow = vis_utils.vis_alignment_plain(self.prev_img, input_img)
            prewarp = vis_utils.vis_alignment_plain(self.template_img, prewarped_input)
            afterwarped_input = cv2.warpPerspective(input_img, H_global_cur2init,
                                                    (input_img.shape[1], input_img.shape[0]),
                                                    flags=cv2.INTER_LINEAR)
            afterwarp = vis_utils.vis_alignment_plain(self.template_img, afterwarped_input)

            if True:
                canvas = self.template_img.copy()
                tmp = to_numpy(template_coords)
                N_coords = tmp.shape[1]
                for i in range(N_coords):
                    coord = tuple(tmp[:, i].astype(np.int32).tolist())
                    cv2.circle(canvas, coord, radius=2, color=(255, 255, 255), thickness=-1)
                cv2.imshow("cv: template points", canvas)

            if True:
                composition = vis_utils.tile(
                    vis_utils.griddify(
                        vis_utils.name_fig([local_flow, prewarp, afterwarp],
                                           ['local flow', 'prewarped', 'afterwarped']),
                        cols=1))
                cv2.namedWindow("cv: composition", cv2.WINDOW_NORMAL)
                cv2.imshow("cv: composition", composition)
            else:
                cv2.imshow("cv: local", local_flow)
                cv2.imshow("cv: prewarp", prewarp)
                cv2.imshow("cv: afterwarp", afterwarp)

            logger.debug(f"vis_timer(): {vis_timer()}ms")
        # }}}

        state_update_timer = time_measurer('ms')
        self.prev_img_identifier = img_identifier
        self.prev_img = input_img.copy()
        self.prev_H2init = H_cur2init.copy()

        if not self.lost:
            self.last_good_H2init = H_cur2init.copy()

        meta.lost = self.lost
        meta.N_lost = self.N_lost
        meta.global_H_success = global_H_success
        logger.debug(f"state_update_timer(): {state_update_timer()}ms")

        if self.C.downscale_inputs:
            H_downscale = np.diag([1 / self.C.downscale_inputs, 1 / self.C.downscale_inputs, 1.0])
            H_upscale = np.diag([self.C.downscale_inputs, self.C.downscale_inputs, 1.0])
            H_cur2init = compose_H(H_downscale, H_cur2init, H_upscale)

        return H_cur2init, meta

    def _mask_coords(self, template_coords, cur_coords, weights, post_weights, pw_mask=None,
                     do_pw_mask=True):
        in_template_mask = self.template_mask[template_coords[1, :], template_coords[0, :]]
        if pw_mask is not None:
            H, W = pw_mask.shape
            cur_coords_int = cur_coords.round().long()
            cur_coords_oob = torch.logical_or(
                torch.any(cur_coords < 0, dim=0),
                torch.logical_or(cur_coords_int[0, :] >= W,
                                 cur_coords_int[1, :] >= H))
            in_pw_mask = torch.ones_like(in_template_mask, device=template_coords.device) > 0
            in_pw_mask[cur_coords_oob] = False
            if do_pw_mask:
                tmp = pw_mask[cur_coords_int[1, in_pw_mask], cur_coords_int[0, in_pw_mask]]
                in_pw_mask[in_pw_mask.clone()] = tmp
            in_mask = torch.logical_and(in_template_mask, in_pw_mask)
        else:
            in_mask = in_template_mask

        template_coords = template_coords[:, in_mask]
        cur_coords = cur_coords[:, in_mask]
        if weights is not None:
            weights = weights[:, in_mask]
        if post_weights is not None:
            post_weights = post_weights[:, in_mask]
        return template_coords, cur_coords, weights, post_weights, in_mask

    def _mask_coords_flow(self, prev_coords, cur_coords, weights, post_weights):
        prev_mask = cv2.warpPerspective(self.np_template_mask, np.linalg.inv(self.prev_H2init),
                                        (self.template_mask.shape[1], self.template_mask.shape[0]),
                                        flags=cv2.INTER_NEAREST) > 0
        if isinstance(prev_coords, torch.Tensor) and prev_coords.is_cuda:
            prev_mask = torch.from_numpy(prev_mask).to(prev_coords.device)
        in_mask = prev_mask[prev_coords[1, :], prev_coords[0, :]]
        prev_coords = prev_coords[:, in_mask]
        cur_coords = cur_coords[:, in_mask]
        if weights is not None:
            weights = weights[:, in_mask]
        if post_weights is not None:
            post_weights = post_weights[:, in_mask]
        return prev_coords, cur_coords, weights, post_weights


def to_float(xs):
    if isinstance(xs, torch.Tensor):
        return xs.float()
    else:
        return xs.astype(np.float32)


def to_numpy(xs):
    if isinstance(xs, torch.Tensor):
        return xs.detach().cpu().numpy()
    else:
        return xs


def make_forward_compatible(subsampler_fn):
    """Add a new parameter for post-hoc postprocessed weights."""
    # https://stackoverflow.com/a/41188411/1705970
    from inspect import signature
    orig_signature = signature(subsampler_fn)
    orig_params = orig_signature.parameters
    orig_n_params = len(orig_params)

    if orig_n_params == 3:
        def new_fn(coords_a, coords_b, weights, post_weights):
            if post_weights is not None:
                raise NotImplementedError("Using post-hoc weights post-processing with a subsampler that takes only 3 arguments")

            orig_res = subsampler_fn(coords_a, coords_b, weights)
            return orig_res + (None, )

        return new_fn
    else:
        return subsampler_fn
