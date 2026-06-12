# -*- coding: utf-8 -*-
"""
Registry that maps method names to distillation classes

Methods register themselves at import time, so importing the methods module is enough to populate
the table and build any method from a single configuration key
"""

from typing import Callable, Dict, Type

from kd.base import DistillationMethod

_REGISTRY: Dict[str, Type[DistillationMethod]] = {}


def register(name: str) -> Callable:
    """Return a decorator that records a distillation class under a unique name"""

    def _decorator(cls: Type[DistillationMethod]) -> Type[DistillationMethod]:
        if name in _REGISTRY:
            raise KeyError(f"Duplicate distillation method name {name}")  # Guard shadowing
        _REGISTRY[name] = cls
        return cls

    return _decorator


def build(config: dict) -> DistillationMethod:
    """
    Instantiate the distillation method selected by configuration

    Args:
        config: Method block holding the registered name and its hyperparameters

    Returns:
        DistillationMethod: A ready to use method instance
    """
    import kd.methods  # Trigger the registration side effects

    name = config["name"]
    if name not in _REGISTRY:
        raise KeyError(f"Unknown distillation method {name}")
    return _REGISTRY[name](**config.get("params", {}))
