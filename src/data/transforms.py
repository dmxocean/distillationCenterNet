# -*- coding: utf-8 -*-
"""
Image augmentation pipelines for training and evaluation

The training pipeline resizes by the longest side, pads to a square, and applies photometric
perturbations that leave geometry untouched, so the rendered targets stay aligned

The validation pipeline keeps only the deterministic resize and pad, so evaluation and teacher
inference see a stable view of every image
"""

import cv2
import albumentations as albu


def build_train_transform(img_size=640):
    """
    Build the stochastic training augmentation pipeline

    Args:
        img_size: Target square resolution after resize and pad

    Returns:
        An albumentations Compose with the full training transform
    """
    return albu.Compose(
        [
            albu.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA, p=1.0),
            albu.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                p=1.0,
            ),
            albu.RandomGamma(gamma_limit=(80, 120), p=1.0),
            albu.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1, p=1.0),
            albu.ImageCompression(quality_range=(50, 100), p=1.0),
        ],
        p=1.0,
    )


def build_val_transform(img_size=640):
    """
    Build the deterministic resize and pad pipeline for evaluation and teacher inference

    Args:
        img_size: Target square resolution after resize and pad

    Returns:
        An albumentations Compose with the validation transform
    """
    return albu.Compose(
        [
            albu.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_AREA, p=1.0),
            albu.PadIfNeeded(
                min_height=img_size,
                min_width=img_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                p=1.0,
            ),
        ],
        p=1.0,
    )
