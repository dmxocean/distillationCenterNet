# -*- coding: utf-8 -*-
"""
Catalogue of distillation methods in a single comparable file

Every method lives here as a small class, so the full set can be read, added, and compared in one
view

IMPORTANT: each compute relies only on teacher and student outputs, never on ground truth, which
is what makes every method compatible with unlabelled distillation
"""

from kd.base import DistillationMethod, MethodOutput, SignalSpec
from kd.registry import register
from losses import ops
from losses.supervised import FOCAL_ALPHA, FOCAL_BETA, focal_loss


@register("logit_kl")
class LogitKL(DistillationMethod):
    """
    Temperature scaled Kullback Leibler divergence over heatmap logits

    Softening both logits by a temperature compares them as distributions and transfers dark
    knowledge across the full spatial map
    """

    def __init__(self, temperature: float = 6.0, weight: float = 1.0):
        self.temperature = temperature
        self.weight = weight

    def required_signals(self):
        return (SignalSpec.LOGITS,)

    def compute(self, student_out, teacher_out, targets, student_signals=None, teacher_signals=None):
        kl = ops.softened_kl(student_out.logits, teacher_out.logits, self.temperature)
        return MethodOutput(total=self.weight * kl, terms={"kd_kl": kl})


@register("logit_mse")
class LogitMSE(DistillationMethod):
    """
    Mean squared error over teacher peak locations

    Masking the loss to confident teacher centres focuses the transfer on objects and stays robust
    on noisy unlabelled images
    """

    def __init__(self, peak_threshold: float = 0.3, weight: float = 1.0):
        self.peak_threshold = peak_threshold
        self.weight = weight

    def required_signals(self):
        return (SignalSpec.LOGITS,)

    def compute(self, student_out, teacher_out, targets, student_signals=None, teacher_signals=None):
        mask = ops.teacher_peak_mask(teacher_out.prob, self.peak_threshold)  # Centres only
        mse = ops.masked_mse(student_out.logits, teacher_out.logits, mask)
        return MethodOutput(total=self.weight * mse, terms={"kd_mse": mse})


@register("logit_bce")
class LogitBCE(DistillationMethod):
    """
    Binary cross entropy against soft teacher targets

    Softened teacher logits become soft labels and each location is a separate object or background
    decision
    """

    def __init__(self, temperature: float = 6.0, weight: float = 1.0):
        self.temperature = temperature
        self.weight = weight

    def required_signals(self):
        return (SignalSpec.LOGITS,)

    def compute(self, student_out, teacher_out, targets, student_signals=None, teacher_signals=None):
        bce = ops.soft_bce(student_out.logits, teacher_out.logits, self.temperature)
        return MethodOutput(total=self.weight * bce, terms={"kd_bce": bce})


@register("pseudo")
class Pseudo(DistillationMethod):
    """
    Hard pseudo label transfer from the teacher's confident detections

    Confident teacher peaks are re-rendered as clean gaussian targets and scored with the same
    focal loss as real supervision, so the transfer carries the teacher's decisions instead of
    its soft beliefs
    """

    def __init__(self, score_threshold: float = 0.3, weight: float = 1.0):
        self.score_threshold = score_threshold
        self.weight = weight

    def required_signals(self):
        return (SignalSpec.LOGITS, SignalSpec.GEOMETRY)

    def compute(self, student_out, teacher_out, targets, student_signals=None, teacher_signals=None):
        target = ops.pseudo_targets(teacher_out.prob, teacher_out.geometry, self.score_threshold)
        focal = focal_loss(student_out.prob, target, FOCAL_ALPHA, FOCAL_BETA)
        return MethodOutput(total=self.weight * focal, terms={"kd_pseudo": focal})


@register("attention")
class Attention(DistillationMethod):
    """
    Attention transfer between the gated skip features of the necks

    The captured stage features are channel collapsed into spatial saliency maps and unit
    normalised, so the student mimics where the teacher looks rather than what it outputs, and
    the width gap between the networks needs no adapter
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def required_signals(self):
        return (SignalSpec.ATTENTION,)

    def compute(self, student_out, teacher_out, targets, student_signals=None, teacher_signals=None):
        att = ops.attention_transfer(student_signals, teacher_signals)
        return MethodOutput(total=self.weight * att, terms={"kd_att": att})


# Future methods are added in this same file, each declaring the extra signals it needs
# A feature method declares SignalSpec.FEATURES
