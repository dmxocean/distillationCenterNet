# -*- coding: utf-8 -*-
"""
CenterNet detector built from a backbone, an upsampling neck, and detection heads

The detector represents objects as centre points, the heatmap head locates centres per class and
the geometry head regresses their width and height, which avoids anchor boxes entirely

The same architecture serves the teacher and the students, only the width multiplier and the
loaded checkpoint differ, so distillation compares aligned outputs from networks of different size
"""

from typing import NamedTuple

import torch
import torch.nn as nn

from models.backbone import MobileNetV2Encoder
from models.neck import Decoder
from models.heads import HeatmapHead, GeometryHead


class CenterNetOutput(NamedTuple):
    """
    Structured detector output shared by the teacher and the students

    Attributes:
        prob: Sigmoid heatmap with per class centre probabilities
        logits: Raw pre sigmoid heatmap scores used for loss and distillation
        geometry: Predicted width and height maps for box reconstruction
    """

    prob: torch.Tensor
    logits: torch.Tensor
    geometry: torch.Tensor


class CenterNet(nn.Module):
    """
    Point based detector pairing a width scalable backbone with a fusion neck and heads

    The image is encoded into multi scale features, the neck restores stride four resolution, and
    the heads produce a per class centre heatmap and a width height map

    Args:
        num_classes: Number of enabled detection classes for the heatmap head
        width: Backbone width multiplier, full for the teacher and narrow for the students
        pretrained: Whether to seed the backbone with ImageNet weights, honoured at full width
        img_size: Network input resolution, sets the geometry clamp with the stride
        stride: Downsampling ratio between input and heatmap
    """

    def __init__(self, num_classes: int = 1, width: float = 1.0, pretrained: bool = True, img_size: int = 640, stride: int = 4):
        super().__init__()
        self.encoder = MobileNetV2Encoder(pretrained=pretrained, width=width)
        encoder_channels = self.encoder.out_channels
        decoder_channels = list(encoder_channels)

        geometry_max = img_size // stride  # Largest box side in heatmap pixels, the full padded input
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels)
        self.heatmap_head = HeatmapHead(decoder_channels[3], decoder_channels[3], num_classes)
        self.geometry_head = GeometryHead(decoder_channels[3], decoder_channels[3], geometry_max)

    def forward(self, x: torch.Tensor) -> CenterNetOutput:
        """
        Detect object centres and geometry for a batch of images

        Args:
            x: Image batch in the zero to two hundred fifty five range

        Returns:
            A CenterNetOutput with the probability heatmap, the raw logits, and the geometry map
        """
        stride32_out, stride16_out, stride8_out, stride4_out = self.encoder(x)
        decoder_output = self.decoder(stride32_out, stride16_out, stride8_out, stride4_out)

        prob, logits = self.heatmap_head(decoder_output)
        geometry = self.geometry_head(decoder_output)
        return CenterNetOutput(prob=prob, logits=logits, geometry=geometry)
