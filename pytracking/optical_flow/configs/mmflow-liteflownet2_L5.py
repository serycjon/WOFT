from pathlib import Path
from pytracking.utils.config import Config
from pytracking.optical_flow.mm import MMFlowWrapper


def get_config():
    conf = Config()

    conf.of_class = MMFlowWrapper

    config_name = 'liteflownet2_ft_4x1_600k_sintel_kitti_320x768'

    pytracking_dir = Path(__file__).absolute().parent.parent.parent
    conf.mm_config_file = str(pytracking_dir / f'external/mmflow/configs/liteflownet2/{config_name}.py')
    conf.mm_checkpoint_file = f'~/.cache/mim/{config_name}.pth'
    conf.weight_head_structure = [(128, 3), (128, 3), (128, 3)]
    conf.weight_head_level = 5

    weight_dir = Path(__file__).absolute().parent.parent.parent / 'weights'
    conf.model = weight_dir / 'liteflownet2_v2/wraft_weights-ep04-end.pth'

    conf.name = Path(__file__).stem
    return conf
