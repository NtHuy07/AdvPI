import os
import numpy as np
import random
import torch
from dataclasses import is_dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def validate_config(config):
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)
        try:
            return OmegaConf.to_object(config)
        except ValueError:
            return config
    elif isinstance(config, (dict, DictConfig)):
        return DictConfig(config)
    elif is_dataclass(config):
        return config
    else:
        try:
            return OmegaConf.load(config)
        except IOError:
            raise IOError(
                "Invalid config type. Must be a path to a yaml, a dict, or dataclass."
            )

def set_seed(seed=None, deterministic=False) -> int:
    if seed is None:
        seed = np.random.default_rng().integers(2**32 - 1)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    return seed

def safe_open_w(path, *args, **kwargs):
    """Open "path" for writing, creating any parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, *args, **kwargs)
