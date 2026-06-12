# -*- coding: utf-8 -*-
"""
Forward hook extractor for intermediate distillation signals

Logit and geometry signals are read directly from the detector output, the intermediate feature
and attention signals are produced inside the neck and are captured here through forward hooks

A method declares the signals it needs, the engine builds an extractor only when feature or
attention signals are requested, the three logit methods need no hooks at all
"""

from kd.base import SignalSpec

FEATURE_STAGES = ("decoded_stage1", "decoded_stage2", "decoded_stage3", "decoded_stage4")
ATTENTION_STAGES = ("att1", "att2", "att3")


def _signal_modules(model, signals):
    """Map the requested signals to the neck submodules that produce them"""
    modules = []
    neck = model.decoder
    if SignalSpec.FEATURES in signals:
        for i, name in enumerate(FEATURE_STAGES):
            modules.append((f"feat_{i}", getattr(neck, name)))
    if SignalSpec.ATTENTION in signals:
        for i, name in enumerate(ATTENTION_STAGES):
            modules.append((f"att_{i}", getattr(neck, name)))
    return modules


class SignalExtractor:
    """
    Capture intermediate signals from a detector through forward hooks

    The extractor attaches a hook to every neck module that produces a requested signal and stores
    the latest output under a stable name, the captured tensors are read after a forward pass and
    cleared before the next one

    Args:
        model: Detector to instrument
        signals: Iterable of SignalSpec values the method requires
    """

    def __init__(self, model, signals):
        self._handles = []
        self._captured = {}
        for name, module in _signal_modules(model, signals):
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        """Build a forward hook that stores the module output under the given name"""

        def hook(module, inputs, output):
            self._captured[name] = output

        return hook

    @property
    def signals(self) -> dict:
        """The signals captured during the most recent forward pass"""
        return self._captured

    def clear(self):
        """Drop the captured signals before the next forward pass"""
        self._captured = {}

    def remove(self):
        """Detach every hook from the model"""
        for handle in self._handles:
            handle.remove()
        self._handles = []
