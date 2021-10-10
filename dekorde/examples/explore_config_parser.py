from collections import namedtuple

from dekorde.loaders import load_conf
from dekorde.paths import CONF_JSON


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def explore_config_parser():
    conf = convert(load_conf())
    print(conf)


if __name__ == '__main__':
    explore_config_parser()
