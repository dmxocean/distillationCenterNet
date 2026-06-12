# -*- coding: utf-8 -*-
"""
Loader for the frozen teacher network

Offline distillation keeps the teacher fixed, so this module builds the architecture, restores
the checkpoint trained on labelled data, and freezes every parameter

IMPORTANT: the returned module is in evaluation mode with gradients disabled and must never be
optimised
"""

import os
import torch

from models.centernet import CenterNet


def load_teacher(checkpoint_path: str, num_classes: int, device: str, img_size: int = 640, stride: int = 4) -> CenterNet:
    """
    Build the teacher, restore weights, and return a frozen evaluation model

    Args:
        checkpoint_path: Absolute path to the teacher checkpoint trained on labelled data
        num_classes: Number of enabled detection classes for the heatmap head
        device: Target device identifier for the loaded model
        img_size: Square input resolution the detector is configured for
        stride: Output stride between the input image and the head maps

    Returns:
        CenterNet: Teacher network in evaluation mode with gradients disabled
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found at {checkpoint_path}")

    model = CenterNet(num_classes=num_classes, width=1.0, pretrained=False, img_size=img_size, stride=stride)  # Full capacity teacher
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()  # Disable dropout and batch norm updates
    model.requires_grad_(False)  # Freeze, the teacher is never trained
    return model.to(device)
