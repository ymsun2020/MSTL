from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.MSTL.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    yaml_file = os.path.join(prj_dir, 'experiments/MSTL/MSTL.yaml' )
    update_config_from_file(yaml_file)

    params.cfg = cfg

    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(prj_dir,"tracking/networks/STARKLightningXtrt_ep0120.pth.tar")
    params.save_all_boxes = False

    return params
