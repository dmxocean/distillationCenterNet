# -*- coding: utf-8 -*-
"""
Detection heads producing dense centre heatmaps and geometry maps

The heatmap head estimates the probability of an object centre at every spatial location across
num_classes channels, so the architecture is multilabel by construction

The geometry head regresses the width and height at every location, the two heads share the
stride four feature from the neck so their predictions are spatially aligned
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neck import SeparableBlock


class HeatmapHead(nn.Module):
    """
    Convolutional head predicting per class centre probability heatmaps

    A separable block keeps representational capacity, then a convolution projects to one channel
    per enabled class, the bias starts negative so the network favours low confidence at startup

    Args:
        in_channels: Channels of the incoming neck feature
        n_filters: Intermediate channels inside the head
        num_classes: Number of enabled detection classes, one channel each
    """

    def __init__(self, in_channels: int, n_filters: int, num_classes: int):
        super().__init__()
        self.block = SeparableBlock(in_channels, n_filters)
        self.out = nn.Conv2d(n_filters, num_classes, 3, padding=1, bias=True)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.constant_(self.out.bias, -2.94)  # Favour low confidence at startup

    def forward(self, decoder_output: torch.Tensor):
        """
        Project neck features into activated probabilities and raw logits

        Args:
            decoder_output: Stride four feature map from the neck

        Returns:
            The sigmoid heatmap and the raw logits used for loss computation
        """
        x = self.block(decoder_output)
        out = self.out(x)
        return torch.sigmoid(out), out


class GeometryHead(nn.Module):
    """
    Convolutional head regressing object width and height per location

    A separable block precedes a projection to two channels, the output is passed through softplus
    and clamped so the predicted sizes stay positive and bounded
    """

    def __init__(self, in_channels: int, n_filters: int, geometry_max: float = 160.0):
        super().__init__()
        self.block = SeparableBlock(in_channels, n_filters)
        self.out = nn.Conv2d(n_filters, 2, 3, padding=1, bias=True)
        self.geometry_max = geometry_max  # Largest box side in heatmap pixels, the full padded input
        nn.init.zeros_(self.out.bias)

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """Project neck features into bounded positive width and height maps"""
        return F.softplus(self.out(self.block(decoder_output))).clamp(max=self.geometry_max)
