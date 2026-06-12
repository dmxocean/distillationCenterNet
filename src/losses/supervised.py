# -*- coding: utf-8 -*-
"""
Supervised loss for labelled batches, focal heatmap plus masked geometry

The focal term handles the extreme foreground background imbalance of centre detection across all
enabled class channels, the geometry term is a masked L1 applied only where a ground truth centre
exists

The supervised loss fires only when a batch carries targets, so it never touches the unlabelled
stream, the distillation term is what supervises unlabelled images
"""

from typing import NamedTuple

import torch
import torch.nn as nn

FOCAL_ALPHA = 2  # Foreground focusing exponent
FOCAL_BETA = 4  # Negative location decay exponent
LOSS_EPS = 1e-4  # Normalisation guard for the masked geometry term


class SupervisedOutput(NamedTuple):
    """Supervised loss with named sub terms for logging"""

    total: torch.Tensor
    focal: torch.Tensor
    reg: torch.Tensor


def focal_loss(prob: torch.Tensor, target: torch.Tensor, alpha: int, beta: int, ignore_mask=None) -> torch.Tensor:
    """
    Distance weighted focal loss over the per class centre heatmap

    Positive locations are the exact gaussian peaks, negative locations are penalised less the
    closer they sit to a peak, the loss is normalised by the number of positive centres, cells under
    the ignore mask are dropped so crowd regions neither reward nor penalise the model

    Args:
        prob: Sigmoid heatmap from the detector
        target: Ground truth gaussian heatmap
        alpha: Foreground focusing exponent
        beta: Negative location decay exponent
        ignore_mask: Optional per class mask with one at the cells to drop from the loss

    Returns:
        A scalar focal loss for the batch
    """
    prob = prob.float()  # AMP independent, the clamp and logs run at full precision
    target = target.float()
    epsilon = torch.finfo(prob.dtype).eps
    prob = prob.clamp(epsilon, 1.0 - epsilon)

    positive = (target == 1).float()
    num_positive = positive.sum()

    pos_loss = (torch.pow(1 - prob, alpha) * torch.log(prob)) * positive
    neg_loss = (torch.pow(1 - target, beta) * torch.pow(prob, alpha) * torch.log(1 - prob)) * (1 - positive)
    if ignore_mask is not None:
        keep = 1.0 - ignore_mask  # Zero the contribution of the crowd ignore cells
        pos_loss = pos_loss * keep
        neg_loss = neg_loss * keep
    return -(pos_loss.sum() + neg_loss.sum()) / torch.clamp(num_positive, min=1.0)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Masked mean absolute error over locations holding a ground truth centre

    Args:
        pred: Predicted geometry map
        target: Ground truth geometry map, nonzero only at centres
        eps: Normalisation guard

    Returns:
        A scalar regression loss normalised by the number of active centres
    """
    pred = pred.float()
    target = target.float()
    mask = (target > 0).float()
    loss = torch.abs(target * mask - pred * mask).sum()
    return loss / (mask.sum() + eps)


class SupervisedLoss(nn.Module):
    """
    Combined focal and geometry loss applied to labelled batches

    Args:
        focal_weight: Weight on the heatmap focal term
        reg_weight: Weight on the masked geometry term
    """

    def __init__(self, focal_weight: float = 1.0, reg_weight: float = 0.1):
        super().__init__()
        self.focal_weight = focal_weight
        self.reg_weight = reg_weight

    def forward(self, out, targets) -> SupervisedOutput:
        """
        Compute the supervised loss from the detector output and the ground truth targets

        Args:
            out: CenterNetOutput holding the probability heatmap and the geometry map
            targets: Targets holding the ground truth heatmap, geometry map, and crowd ignore mask

        Returns:
            A SupervisedOutput with the weighted total and the focal and geometry sub terms
        """
        focal = focal_loss(out.prob, targets.heatmap, FOCAL_ALPHA, FOCAL_BETA, targets.ignore_mask)
        reg = masked_l1(out.geometry, targets.geometry, LOSS_EPS)
        total = self.focal_weight * focal + self.reg_weight * reg
        return SupervisedOutput(total=total, focal=focal, reg=reg)
