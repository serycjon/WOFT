# -*- coding: utf-8 -*-
import sys
import argparse
from pathlib import Path
import cv2
import hashlib
import tqdm
import pickle
import gzip
import numpy as np

from pytracking.utils.config import load_config


def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', help='dataset config', type=Path)
    parser.add_argument('--cache_dir', help='path to cache directory', type=Path, default=Path('~/personal/ssd_export/cache/'))

    return parser.parse_args()


def run(args):
    ds_conf = load_config(args.dataset)
    dataset = ds_conf.dataset

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    table = {}
    for seq in tqdm.tqdm(dataset, desc='sequence'):
        N_frames = len(seq.frames)
        for i in tqdm.tqdm(range(N_frames), desc='frame'):
            to_store = (seq.name, i)
            img = cv2.imread(seq.frames[i])
            img_hash = hashlib.sha256(img.data).digest()
            if img_hash in table:
                raise RuntimeError("Hash conflict...")
            table[img_hash] = to_store
        break

    cache_path = args.cache_dir / f'{ds_conf.dataset_name}.pklz'
    with gzip.open(cache_path, 'wb') as fout:
        pickle.dump(table, fout)
    return 0


def identify_image(img, cache):
    img_hash = hashlib.sha256(img.data).digest()
    return cache.get(img_hash, None)


def load_cached_flow(img_a, flow_cache_dir, img_a_identifier):
    dataset_name, seq_name, frame_i = img_a_identifier
    flow_path = flow_cache_dir / dataset_name / seq_name / f'{frame_i}-{frame_i + 1}.npz'
    data = np.load(flow_path, allow_pickle=True)
    flow = data['half_flow'].astype(np.float32)
    weights = data['half_weights'].astype(np.float32)
    return flow, weights


def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
