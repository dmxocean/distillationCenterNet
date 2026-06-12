# -*- coding: utf-8 -*-
"""
Optimiser and learning rate schedule construction

The schedule warms the learning rate up linearly for the first epochs, then follows a cosine
decay from the base rate down to the configured minimum, the scheduler is stepped once per epoch
"""

import math

import torch


def build_optimizer(model, config):
    """Build an AdamW optimiser over the trainable parameters"""
    return torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4),
    )


def build_scheduler(optimizer, config, last_epoch=-1):
    """
    Build a warmup then cosine learning rate schedule stepped per epoch

    Args:
        optimizer: Optimiser whose learning rate is scheduled
        config: Resolved configuration holding epochs, warmup_epochs, lr, and min_lr
        last_epoch: Index of the last completed epoch, set on resume to restore the schedule

    Returns:
        A LambdaLR scheduler producing the warmup and cosine multipliers
    """
    epochs = config["epochs"]
    warmup = config.get("warmup_epochs", 0)
    base_lr = config["lr"]
    min_ratio = config.get("min_lr", 0.0) / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch):
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup  # Linear warmup toward the base rate
        progress = (epoch - warmup) / max(1, epochs - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay across the run
        return min_ratio + (1 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
