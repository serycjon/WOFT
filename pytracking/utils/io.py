import glob
import time
import datetime
import itertools
from pathlib import Path
from collections import deque
import cv2
import re
import gzip
import pickle
import io
import numpy as np
import einops
import os


def get_frames(path):
    paths = glob.glob(f'{path}/*.jpg')
    return sorted([Path(path) for path in paths])


def video_seek_frame(time_string, fps=30):
    parsed_time = time.strptime(time_string, "%H:%M:%S")
    delta = datetime.timedelta(hours=parsed_time.tm_hour, minutes=parsed_time.tm_min, seconds=parsed_time.tm_sec)
    time_seconds = int(delta.total_seconds())
    pos = fps * time_seconds
    return pos


def video_seek_frame_name(query_frame_name, frame_paths):
    frame_names = [path.stem for path in frame_paths]
    regexp = re.compile(r'0*' + query_frame_name)
    for i, name in enumerate(frame_names):
        if re.match(regexp, name):
            return i
    raise ValueError(f"Frame {query_frame_name} not found.")


def frames_from_time(directory, time_string, fps=30):
    frames = get_frames(directory)
    start_index = video_seek_frame(time_string, fps)

    for i in range(start_index, len(frames)):
        yield (frames[i], cv2.imread(str(frames[i])))


def frames_from_name(directory, start_name):
    frames = get_frames(directory)
    start_index = video_seek_frame_name(start_name, frames)

    for i in range(start_index, len(frames)):
        yield (frames[i], cv2.imread(str(frames[i])))


class LookaheadIter:

    def __init__(self, it):
        self._iter = iter(it)
        self._ahead = deque()

    def __iter__(self):
        return self

    def __next__(self):
        if self._ahead:
            return self._ahead.popleft()
        else:
            return next(self._iter)

    def lookahead(self):
        for x in self._ahead:
            yield x
        for x in self._iter:
            self._ahead.append(x)
            yield x

    def peek(self, *a):
        return next(iter(self.lookahead()), *a)


def load_maybe_gzipped_pkl(path):
    suffix = Path(path).suffix
    if suffix == '.pklz':
        open_fn = gzip.open
    elif suffix == '.pkl':
        open_fn = open
    else:
        ValueError(f"Unknown pickle file suffix ({suffix}).")

    with open_fn(path, 'rb') as fin:
        data = pickle.load(fin)

    return data


class CPU_Unpickler(pickle.Unpickler):
    """ https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    I have pickled something in meta as a GPU tensor..."""

    def find_class(self, module, name):
        import torch

        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def read_flow_png(path):
    """Read png-compressed flow

    Args:
        path: png flow file path

    Returns:
        flow: blab
        valid: blab
    """
    # to specify not to change the image depth (16bit)
    flow = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    # flow shape (H, W, 2) valid shape (H, W)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 32.0
    return flow, valid


def write_flow_png(path, flow, valid=None):
    """Write a compressed png flow

    Args:
        path: write path
        flow: (H, W, 2) xy-flow
        valid: None, or (H, W) array with validity mask
    """
    flow = 32.0 * flow + 2**15  # compress (resolution step 1/32, maximal flow 1024 (same as Sintel width))
    if valid is None:
        valid = np.ones([flow.shape[0], flow.shape[1], 1])
    else:
        valid = einops.rearrange(valid, 'H W -> H W 1', **einops.parse_shape(flow, 'H W _'))
    data = np.concatenate([flow, valid], axis=2).astype(np.uint16)
    cv2.imwrite(str(path), data[:, :, ::-1])


class GeneralVideoCapture(object):
    """A cv2.VideoCapture replacement, that can also read images in a directory"""

    def __init__(self, path, reverse=False):
        images = Path(path).is_dir()
        self.image_inputs = images
        if images:
            self.path = path
            self.images = sorted([f for f in next(os.walk(path))[2] if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']])
            if reverse:
                self.images = self.images[::-1]
            self.i = 0
        else:
            self.cap = cv2.VideoCapture(str(path))

    def read(self):
        if self.image_inputs:
            if self.i >= len(self.images):
                return False, None
            img_path = os.path.join(self.path,
                                    self.images[self.i])
            self.frame_src = self.images[self.i]
            img = cv2.imread(img_path)
            self.i += 1
            return True, img
        else:
            return self.cap.read()

    def release(self):
        if self.image_inputs:
            return None
        else:
            return self.cap.release()

