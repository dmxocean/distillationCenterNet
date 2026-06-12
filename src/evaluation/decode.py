# -*- coding: utf-8 -*-
"""
Decoding dense predictions into bounding boxes

A spatial maximum pool suppresses non peak locations, the surviving centres above the score
threshold are read per class and combined with the geometry map to reconstruct boxes in image
space
"""

import torch
import torch.nn.functional as F


def maxpool_nms(heatmap: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    """Suppress non peak locations by keeping values equal to their local maximum"""
    padding = (kernel - 1) // 2
    pooled = F.max_pool2d(heatmap, kernel_size=kernel, stride=1, padding=padding)
    return heatmap * (heatmap == pooled).float()


def decode_boxes(prob, geometry, threshold=0.3, stride=4, kernel=5, top_k=None):
    """
    Decode one image heatmap and geometry into per class boxes

    Args:
        prob: Heatmap of shape classes, height, width with confidences in the unit range
        geometry: Geometry of shape two, height, width holding width and height per location
        threshold: Minimum confidence to accept a centre
        stride: Downsampling ratio used to map grid coordinates back to image pixels
        kernel: Max pool window for local peak detection
        top_k: Optional cap keeping only the highest scoring detections in the image

    Returns:
        A list of boxes as dicts with class, centre, width, height, and score in image space
    """
    suppressed = maxpool_nms(prob.unsqueeze(0), kernel=kernel)[0]
    boxes = []
    for channel in range(prob.shape[0]):
        mask = suppressed[channel] > threshold
        rows, cols = mask.nonzero(as_tuple=True)
        for row, col in zip(rows.tolist(), cols.tolist()):
            boxes.append(
                {
                    "cls": channel,
                    "cx": col * stride,
                    "cy": row * stride,
                    "w": geometry[0, row, col].item() * stride,
                    "h": geometry[1, row, col].item() * stride,
                    "score": suppressed[channel, row, col].item(),
                }
            )
    if top_k is not None and len(boxes) > top_k:  # Keep the strongest detections per image
        boxes = sorted(boxes, key=lambda b: -b["score"])[:top_k]
    return boxes
