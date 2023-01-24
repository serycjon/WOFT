from pathlib import Path
from pytracking.utils.config import Config
from pytracking.optical_flow.raft import RAFTWrapper


def get_config():
    conf = Config()

    conf.of_class = RAFTWrapper
    conf.raft_type = 'weighted'

    conf.class_params = Config()
    conf.class_params.small = False
    conf.class_params.mixed_precision = False
    conf.class_params.alternate_corr = False
    conf.class_params.weight_head_structure = [(128, 3), (128, 3), (128, 3)]

    weight_dir = Path(__file__).absolute().parent.parent.parent / 'weights'
    conf.model = weight_dir / 'v2_SNOB_large_g05_RAFT/wraft_weights-ep01-end.pth'
    conf.add_module_to_statedict = True
    conf.non_strict_loading = False

    conf.iters = 12
    conf.padding_mode = 'RAFT'

    conf.name = Path(__file__).stem

    return conf
