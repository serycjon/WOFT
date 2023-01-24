from pathlib import Path
from pytracking.utils.config import Config
from pytracking.optical_flow.raft import RAFTWrapper


def get_config():
    conf = Config()

    conf.of_class = RAFTWrapper
    conf.raft_type = 'orig'

    conf.class_params = Config()
    conf.class_params.small = False
    conf.class_params.mixed_precision = False
    conf.class_params.alternate_corr = False

    conf.model = Path(__file__).absolute().parent.parent.parent / 'external' / 'RAFT' / 'models' / 'raft-sintel.pth'

    conf.iters = 24
    conf.padding_mode = 'RAFT'

    return conf
