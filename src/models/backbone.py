# -*- coding: utf-8 -*-
"""
Width scalable MobileNetV2 backbone for hierarchical feature extraction

The backbone turns an input image into four multi scale feature maps at strides four, eight,
sixteen, and thirty two, which the neck later fuses through skip connections

The width multiplier scales every layer so the same architecture serves a full capacity teacher
and a narrow student, ImageNet weights are loaded only at full width to provide a strong prior
"""

import torch
import torch.nn as nn
import torchvision.models as models

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # Channel means for ImageNet normalisation
IMAGENET_STD = (0.229, 0.224, 0.225)  # Channel standard deviations for ImageNet normalisation


class MobileNetV2Encoder(nn.Module):
    """
    Hierarchical feature extractor based on MobileNet version two

    The backbone is split into four stages by spatial stride so the neck can fuse multi scale
    features, the width multiplier scales the channel count and pretrained weights load only at
    full width

    Args:
        pretrained: Whether to load ImageNet weights, honoured only at full width
        width: Width multiplier that scales the number of filters in every layer
    """

    def __init__(self, pretrained: bool = True, width: float = 1.0):
        super().__init__()

        use_pretrained = pretrained and width == 1.0
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if use_pretrained else None
        features = models.mobilenet_v2(weights=weights, width_mult=width).features

        self.stage1 = features[:4]  # Stride four
        self.stage2 = features[4:7]  # Stride eight
        self.stage3 = features[7:14]  # Stride sixteen
        self.stage4 = features[14:17]  # Stride thirty two

        self.use_imagenet_weights = use_pretrained
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).reshape(1, 3, 1, 1))

        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            s4 = self.stage1(dummy)
            s8 = self.stage2(s4)
            s16 = self.stage3(s8)
            s32 = self.stage4(s16)
        self.out_channels = (s32.shape[1], s16.shape[1], s8.shape[1], s4.shape[1])
        self.train()

    def forward(self, x: torch.Tensor):
        """
        Run the image batch through the backbone stages

        Args:
            x: Input image batch in the zero to two hundred fifty five range

        Returns:
            Feature maps ordered from stride thirty two down to stride four
        """
        x = x / 255.0  # Scale pixels into the unit range
        x = (x - self.mean) / self.std  # ImageNet standardisation, identical at train, distillation, and eval

        stride4_out = self.stage1(x)
        stride8_out = self.stage2(stride4_out)
        stride16_out = self.stage3(stride8_out)
        stride32_out = self.stage4(stride16_out)
        return stride32_out, stride16_out, stride8_out, stride4_out

    def train(self, mode: bool = True):
        """Set training mode while freezing batch norm statistics for the pretrained prior"""
        super().train(mode)
        if self.use_imagenet_weights:
            for layer in self.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()  # Keep the ImageNet running statistics fixed
        return self
