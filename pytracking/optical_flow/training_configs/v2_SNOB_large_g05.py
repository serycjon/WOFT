from pathlib import Path

from pytracking.utils.config import Config
from pytracking.optical_flow.raft import RAFTWrapper
from pytracking.utils.least_squares_H import torch_reproj_errors
from pytracking.utils.least_squares_H import find_homography_nonhomogeneous_QR
from pytracking.evaluation.coco_H_synth_dataset import COCOHSynth


def get_config():
    conf = Config()

    conf.flow = Config()
    conf.flow.of_class = RAFTWrapper
    conf.flow.raft_type = 'weighted'

    conf.flow.class_params = Config()
    conf.flow.class_params.small = False
    conf.flow.class_params.mixed_precision = False
    conf.flow.class_params.alternate_corr = False
    conf.flow.class_params.weight_head_structure = [(128, 3), (128, 3), (128, 3)]

    # conf.model_conf.model = '/home.stud/neoramic/repos/raft_new_debug/RAFT/checkpoints/100000_raft-rob-100k.pth'
    conf.flow.model = Path(__file__).absolute().parent.parent.parent / 'external/RAFT/models/raft-sintel.pth'

    conf.flow.iters = 12
    conf.flow.padding_mode = 'nopad'
    ## For starting from pre-trained plain RAFT
    conf.flow.non_strict_loading = True
    conf.flow.add_module_to_statedict = False

    ## For starting from pre-trained wRAFT
    # conf.flow.non_strict_loading = False
    # conf.flow.add_module_to_statedict = True

    conf.train = Config()
    conf.train.epochs = 10
    # the dataset was generated by
    # python prepare_wraft_dataset.py --gpu 2 optical_flow/training_configs/sintel-SNOB-0_large_Clr-6-700k.py
    # python prepare_wraft_dataset.py --gpu 2 optical_flow/training_configs/sintel-SNOB-0_large_Clr-6-700k.py --val
    dataset_path = Path(__file__).absolute().parent.parent.parent / 'synth_dataset'
    conf.train.dataset = COCOHSynth(dataset_path / "train_SNOB")
    conf.train.val_dataset = COCOHSynth(dataset_path / "val_SNOB")
    conf.train.H_estimator = find_homography_nonhomogeneous_QR
    conf.train.loss_fn = torch_reproj_errors
    conf.train.lr = 1e-3
    conf.train.lr_step = 1
    conf.train.lr_gamma = 0.5
    conf.train.max_loss = 100.0

    return conf
