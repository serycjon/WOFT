# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import os
import sys
import argparse
from pathlib import Path
import logging
import numpy as np
import cv2
from pytracking.utils.config import load_config
from pytracking.utils import vis_utils as vu
from pytracking.utils import io as io_utils

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('video', help='a path to a video or a directory containing extracted video frames', type=Path)
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--gpu', help='cuda device')
    parser.add_argument('--config', help='path to tracker config file', type=Path, default=Path("pytracking/configs/WOFT.py"))

    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format=format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return args


def run(args):
    config = load_config(args.config)
    tracker_cls = config.tracker_class
    tracker = tracker_cls(config)

    results = None

    video_path = args.video
    cap = io_utils.GeneralVideoCapture(video_path)

    success, frame = cap.read()
    if success is not True:
        print(f"Reading frame from {video_path} failed.")
        exit(-1)

    init_mask = select_rect_mask(frame)
    # initialized by a user-selected bounding box mask.  Feel free to use arbitrary shape init_masks
    # For example:
    # init_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    # init_polygon_xy = np.array([[100, 350], [165, 350], [165, 240]], dtype=np.int32)
    # cv2.fillPoly(init_mask, [init_polygon_xy], 255)

    tracker.init(frame, init_mask)
    print('press q to quit')

    while True:
        ret, frame = cap.read()
        if frame is None:
            return

        last_H = np.eye(3)
        try:
            H_2init, _ = tracker.track(frame)
            last_H = H_2init.copy()
        except Exception:
            logger.exception("Tracker exception")
            H_2init = last_H.copy()

        frame_vis = triv_tracker_vis(frame.copy(), init_mask.copy(), H_2init.copy())
        cv2.imshow("cv: WOFT", frame_vis)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return 0


def select_rect_mask(img):
    canvas = img.copy()

    cv2.putText(canvas, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1.5, (0, 0, 0), 1)

    x, y, w, h = cv2.selectROI("cv: WOFT", canvas, fromCenter=False)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y:y + h + 1,
         x:x + w + 1] = 255
    return mask


def triv_tracker_vis(frame, init_mask, H_2init, frame_i=None, seq_name=None):
    current_mask = cv2.warpPerspective(init_mask,
                                       np.linalg.inv(H_2init),
                                       (frame.shape[1], frame.shape[0]),
                                       flags=cv2.INTER_NEAREST)

    vis = vu.blend_mask(frame, current_mask, color=(0, 255, 0),
                        fill=False, contour_thickness=2)

    if frame_i is not None or seq_name is not None:
        vis = vu.draw_text(vis, f'{seq_name} #{frame_i}', pos='tl',
                           size=1, thickness=2)
    return vis


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
