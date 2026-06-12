# -*- coding: utf-8 -*-
"""
Global RNG seeding for reproducible training runs
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seed all RNG layers used during training

    Args:
        seed: Integer seed applied to every random source
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)