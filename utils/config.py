# utils/config.py

import yaml


def load_config(path):

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg