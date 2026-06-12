# -*- coding: utf-8 -*-
"""
Configuration loader that composes yaml layers and applies runtime overrides

An experiment lists the component configs it includes, this loader merges them in order, then
the experiment body overrides its components and the command line overrides everything

IMPORTANT: the fully resolved config is frozen into the run directory, so any run reproduces
from that single artifact
"""

import os

import yaml

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(BASE_PATH, "config")


def _read(path: str) -> dict:
    """Read a single yaml file into a dictionary"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, recursing into nested dict blocks instead of replacing them"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve(config_path: str, overrides: dict) -> dict:
    """
    Compose the included layers, merge the experiment body, and apply command line overrides

    Args:
        config_path: Absolute path to the experiment config file
        overrides: Command line values that take precedence, for example the W&B mode

    Returns:
        dict: The fully resolved configuration ready to be frozen into the run directory
    """
    config = _read(config_path)
    merged: dict = {}
    for layer in config.get("includes", []):  # Compose component configs in order
        merged = _deep_merge(merged, _read(os.path.join(CONFIG_DIR, layer)))

    merged = _deep_merge(merged, {k: v for k, v in config.items() if k != "includes"})  # Experiment wins
    merged = _deep_merge(merged, {k: v for k, v in overrides.items() if v is not None})  # Command line wins
    return merged
