# -*- coding: utf-8 -*-
"""
Training steps that compose supervised and distillation losses

The labelled and unlabelled streams are interleaved into a single batch type, so the distillation
term applies on every batch while the supervised term applies only when ground truth is present

The same distill step serves both regimes, the absence of targets on an unlabelled batch is what
switches the supervised term off, the teacher is the only supervision there
"""

from typing import NamedTuple

import torch

from data.streams import Batch, Targets
from losses import ops


class LossWeights(NamedTuple):
    """Relative weights of the hard supervised term and the distillation term"""

    hard: float
    distill: float


def move_batch(batch: Batch, device: str) -> Batch:
    """Move a batch and its optional targets to the target device"""
    images = batch.images.to(device, non_blocking=True)
    if batch.targets is None:
        return Batch(images, None, batch.source)
    targets = Targets(
        batch.targets.heatmap.to(device, non_blocking=True),
        batch.targets.geometry.to(device, non_blocking=True),
        batch.targets.ignore_mask.to(device, non_blocking=True),
    )
    return Batch(images, targets, batch.source)


def supervised_step(batch, model, supervised):
    """Run one supervised step for the teacher or the baseline"""
    out = model(batch.images)
    hard = supervised(out, batch.targets)
    return hard.total, hard, out


def distill_step(batch, student, teacher, method, supervised, weights, geom_weight=0.0, peak_threshold=0.3, extractors=None):
    """
    Run one optimisation step over a mixed labelled or unlabelled batch

    Args:
        batch: Mixed batch holding images, optional targets, and a source tag
        student: Trainable student detector
        teacher: Frozen teacher detector
        method: Distillation method built from the registry
        supervised: Supervised loss module, focal plus geometry
        weights: LossWeights with the hard and distillation weights
        geom_weight: Relative weight of the geometry distillation term, zero disables it
        peak_threshold: Teacher confidence for the geometry transfer mask
        extractors: Optional student and teacher SignalExtractor pair for hooked methods

    Returns:
        The total loss, the distillation output, and the optional supervised output
    """
    if extractors is not None:
        for extractor in extractors:  # Drop the previous batch's captures before the forwards
            extractor.clear()
    student_out = student(batch.images)
    with torch.no_grad():
        teacher_out = teacher(batch.images)  # Frozen pseudo targets for distillation

    if extractors is None:
        distill = method.compute(student_out, teacher_out, batch.targets)
    else:
        student_extractor, teacher_extractor = extractors
        distill = method.compute(student_out, teacher_out, batch.targets, student_signals=student_extractor.signals, teacher_signals=teacher_extractor.signals)
    distill_total = distill.total  # Distillation applies on every batch
    if geom_weight > 0:  # Transfer box size so the unlabelled stream trains the geometry head too
        centre = ops.teacher_peak_mask(teacher_out.prob, peak_threshold).amax(dim=1, keepdim=True)
        geom = ops.masked_geometry_l1(student_out.geometry, teacher_out.geometry, centre.expand_as(teacher_out.geometry))
        distill.terms["kd_geom"] = geom
        distill_total = distill_total + geom_weight * geom
    loss = weights.distill * distill_total

    hard = None
    if batch.targets is not None:  # Supervised term only on labelled data
        hard = supervised(student_out, batch.targets)
        loss = loss + weights.hard * hard.total

    return loss, distill, hard
