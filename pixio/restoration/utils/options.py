
import yaml
import os
from collections import OrderedDict

def ordered_yaml():
    """Support OrderedDict for yaml.
    """
    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def parse_options(opt_path):
    Loader, _ = ordered_yaml()
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    return opt

def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Returns:
        str: Option string for printing.
    """
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg
