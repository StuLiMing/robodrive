# Adapted from https://github.com/wayveai/fiery/blob/master/fiery/config.py

import argparse
from fvcore.common.config import CfgNode as _CfgNode


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary."""
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, _CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                'Key {} with value {} is not a valid type; valid types: {}'.format(
                    '.'.join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


class CfgNode(_CfgNode):
    """Remove once https://github.com/rbgirshick/yacs/issues/19 is merged."""

    def convert_to_dict(self):
        return convert_to_dict(self)


CN = CfgNode

_C = CN()

# General settings
_C.DATASET = CN()
_C.DATASET.ORIGINAL_SIZE = CN()
_C.DATASET.ORIGINAL_SIZE.HEIGHT = 900  # Original height of dataset images
_C.DATASET.ORIGINAL_SIZE.WIDTH = 1600  # Original width of dataset images
_C.DATASET.VERSION ='v1.0-mini'

_C.PATH=CN()
_C.PATH.DATAROOT="C:\Users\lm\Desktop/robodrive/v1.0-mini"           # Path to main data folder containing nuScenes or RobotCar
_C.PATH.EVALIMG="./images/eval"
_C.PATH.GTIMG="./images/gt"
_C.PATH.RGBIMG="./images/rgb"
_C.PATH.WORK="./visualization"

_C.GT=CN()
_C.GT.MINDEPTH=0.1
_C.GT.MAXDEPTH=200

def get_parser():
    parser = argparse.ArgumentParser(description='md4all training')
    parser.add_argument('--config', default='', metavar='FILE', help='Path to config file')
    parser.add_argument(
        'opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER,
    )
    return parser


def get_cfg(args=None, cfg_dict=None):
    """ First get default config. Then merge cfg_dict. Then merge according to args. """

    cfg = _C.clone()

    if cfg_dict is not None:
        cfg.merge_from_other_cfg(CfgNode(cfg_dict))

    if args is not None:
        if args.config:
            cfg.merge_from_file(args.config)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
    return cfg

args = get_parser().parse_args()
cfg = get_cfg(args)
