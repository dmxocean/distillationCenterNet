# -*- coding: utf-8 -*-
"""
Reusable distillation primitives shared by the methods

These operations compare teacher and student heatmap logits only, never ground truth, so the
same call trains on labelled and unlabelled batches alike

The teacher tensors are detached inside every primitive, the teacher is frozen and its outputs
must not carry gradients into the student update
"""

import numpy as np
import torch
import torch.nn.functional as F

from data.targets import draw_umich_gaussian, gaussian_radius

NMS_KERNEL = 5  # Training peak window for the teacher mask, distinct from the eval nms_kernel in config
PEAK_EPS = 1e-6  # Float tolerance for local maximum equality
LOSS_EPS = 1e-4  # Normalisation guard for masked terms


def softened_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Temperature scaled Kullback Leibler divergence over the flattened heatmap logits

    Both logits are softened by the temperature and read as a spatial distribution per image, the
    divergence is scaled by the squared temperature so the gradient magnitude is preserved

    Args:
        student_logits: Student pre sigmoid heatmap logits
        teacher_logits: Teacher pre sigmoid heatmap logits
        temperature: Softening temperature applied to both logits

    Returns:
        A scalar divergence averaged over the batch
    """
    teacher = teacher_logits.detach().float()
    student_logits = student_logits.float()
    batch = student_logits.shape[0]
    student_logp = F.log_softmax(student_logits.view(batch, -1) / temperature, dim=-1)
    teacher_p = F.softmax(teacher.view(batch, -1) / temperature, dim=-1)
    return F.kl_div(student_logp, teacher_p, reduction="batchmean") * (temperature**2)


def teacher_peak_mask(teacher_prob: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Build a mask of confident teacher centres for peak focused transfer

    A location is kept when it is a local maximum of the teacher heatmap and its confidence
    exceeds the threshold, which focuses the loss on object centres and ignores noisy background

    Args:
        teacher_prob: Teacher sigmoid heatmap
        threshold: Minimum teacher confidence to keep a centre

    Returns:
        A float mask matching the heatmap shape
    """
    prob = teacher_prob.detach().float()
    pooled = F.max_pool2d(prob, kernel_size=NMS_KERNEL, stride=1, padding=NMS_KERNEL // 2)
    is_peak = (prob >= pooled - PEAK_EPS).float()
    return ((prob * is_peak) > threshold).float()


def masked_mse(student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean squared error between student and teacher logits over the masked locations

    Args:
        student_logits: Student pre sigmoid heatmap logits
        teacher_logits: Teacher pre sigmoid heatmap logits
        mask: Float mask selecting the locations to compare

    Returns:
        A scalar error normalised by the number of masked locations
    """
    teacher = teacher_logits.detach().float()
    student_logits = student_logits.float()
    mask = mask.float()
    squared = (student_logits - teacher) ** 2 * mask
    return squared.sum() / (mask.sum() + LOSS_EPS)


def masked_geometry_l1(student_geometry: torch.Tensor, teacher_geometry: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean absolute error between student and teacher geometry over the masked centres

    The geometry head is only meaningful at object centres, so the mask restricts the transfer to the
    confident teacher peaks, this lets the box size head learn from unlabelled images where no ground
    truth geometry exists

    Args:
        student_geometry: Student width and height map
        teacher_geometry: Teacher width and height map
        mask: Float mask selecting the locations to compare

    Returns:
        A scalar error normalised by the number of masked locations
    """
    teacher = teacher_geometry.detach().float()
    student_geometry = student_geometry.float()
    mask = mask.float()
    absolute = torch.abs(student_geometry - teacher) * mask
    return absolute.sum() / (mask.sum() + LOSS_EPS)


def pseudo_targets(teacher_prob: torch.Tensor, teacher_geometry: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Render hard gaussian pseudo targets from the confident teacher peaks

    Each confident teacher centre is minted as a definite detection and redrawn as a clean
    gaussian peak whose radius follows the teacher's predicted box size, so the student trains
    against the teacher's decisions with the same target shape as real supervision

    An empty peak set renders an all zero target, the focal loss then transfers pure background
    suppression, which is itself teacher knowledge

    Args:
        teacher_prob: Teacher sigmoid heatmap
        teacher_geometry: Teacher width and height map in heatmap pixels
        threshold: Minimum teacher confidence to mint a pseudo centre

    Returns:
        A float target heatmap matching the teacher heatmap shape
    """
    mask = teacher_peak_mask(teacher_prob, threshold)
    target = np.zeros(mask.shape, dtype=np.float32)
    peaks = mask.nonzero()
    if peaks.numel():
        geometry = teacher_geometry.detach().float()
        sizes = geometry[peaks[:, 0], :, peaks[:, 2], peaks[:, 3]].cpu().numpy()  # Width and height per peak
        for (b, c, y, x), (w, h) in zip(peaks.cpu().numpy(), sizes):
            radius = max(1, int(gaussian_radius((float(h), float(w)))))
            draw_umich_gaussian(target[b, c], (int(x), int(y)), radius)
    return torch.from_numpy(target).to(teacher_prob.device)


def attention_map(features: torch.Tensor) -> torch.Tensor:
    """
    Collapse a feature map into a normalised spatial attention map

    The channel energies are summed into a single spatial map and unit normalised per image, so
    maps from networks of different widths become directly comparable without adapters

    Args:
        features: Feature map of shape batch, channels, height, width

    Returns:
        A flattened unit norm attention map per image
    """
    energy = features.float().pow(2).sum(dim=1).flatten(1)
    return F.normalize(energy, dim=1)


def attention_transfer(student_signals: dict, teacher_signals: dict) -> torch.Tensor:
    """
    Mean squared error between normalised attention maps across the captured stages

    Each stage is channel collapsed and unit normalised before comparison, the per stage errors
    are averaged so every decoding depth contributes equally to the transfer

    Args:
        student_signals: Captured student stage tensors keyed by stage name
        teacher_signals: Captured teacher stage tensors keyed by stage name

    Returns:
        A scalar transfer loss averaged over the stages
    """
    stage_losses = []
    for name in sorted(student_signals):
        student_map = attention_map(student_signals[name])
        teacher_map = attention_map(teacher_signals[name].detach())
        stage_losses.append((student_map - teacher_map).pow(2).mean())
    return torch.stack(stage_losses).mean()


def soft_bce(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Binary cross entropy of the student logits against soft teacher targets

    The teacher logits are softened into per location soft labels, each location is then a separate
    object or background decision scored against the softened student logits

    Args:
        student_logits: Student pre sigmoid heatmap logits
        teacher_logits: Teacher pre sigmoid heatmap logits
        temperature: Softening temperature applied to both logits

    Returns:
        A scalar binary cross entropy scaled by the squared temperature
    """
    teacher = teacher_logits.detach().float()
    soft_target = torch.sigmoid(teacher / temperature)
    student = student_logits.float() / temperature
    return F.binary_cross_entropy_with_logits(student, soft_target) * (temperature**2)
