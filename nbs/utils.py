import argparse
from distutils.util import strtobool
from types import SimpleNamespace


def parse_arg(config: SimpleNamespace):
    parser = argparse.ArgumentParser()
    for k, v in config.__dict__.items():
        parser.add_argument(f"--{k}", default=v)
    new_config = parser.parse_args()
    for k, v in config.__dict__.items():
        if isinstance(v, bool):
            print(new_config.__dict__[k])
            new_config.__dict__[k] = bool(strtobool(new_config.__dict__[k]))
    return new_config