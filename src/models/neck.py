# -*- coding: utf-8 -*-
"""
Upsampling neck that recovers spatial resolution with attention gated skip fusion

The neck progressively upsamples the deepest backbone feature and fuses each skip connection
through an attention gate, so fine spatial detail is restored while background noise is suppressed

Depthwise separable blocks keep the decoding path light, the named stage and attention modules
double as the attachment points for intermediate feature and attention distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Spatial attention gate that filters a skip feature with a deeper gating signal

    The gate learns to weight spatial locations by relevance, emphasising object regions in the
    high resolution skip features before they are concatenated into the decoding path

    Args:
        feat_channels: Channels of the high resolution skip feature
        gate_channels: Channels of the lower resolution gating signal
        filters: Intermediate channels used for the projection
    """

    def __init__(self, feat_channels: int, gate_channels: int, filters: int):
        super().__init__()
        self.feat_conv = nn.Conv2d(feat_channels, filters, 1, bias=True)
        self.feat_bn = nn.BatchNorm2d(filters)
        self.gate_conv = nn.Conv2d(gate_channels, filters, 1, bias=True)
        self.gate_bn = nn.BatchNorm2d(filters)
        self.out_conv = nn.Conv2d(filters, 1, 1, bias=True)
        self.out_bn = nn.BatchNorm2d(1)

        nn.init.constant_(self.out_bn.bias, 4.0)  # Start near an open gate so skips pass early

    def forward(self, features: torch.Tensor, gating_signal: torch.Tensor) -> torch.Tensor:
        """
        Emphasise relevant regions of the skip features using the gating signal

        Args:
            features: High resolution skip feature from the backbone
            gating_signal: Lower resolution signal from the decoding path

        Returns:
            The skip feature reweighted by the learned spatial attention mask
        """
        feat_proj = self.feat_bn(self.feat_conv(features))
        gate_proj = self.gate_bn(self.gate_conv(gating_signal))
        mask = torch.sigmoid(self.out_bn(self.out_conv(F.relu(feat_proj + gate_proj))))
        return features * mask


class SeparableBlock(nn.Module):
    """
    Depthwise separable convolution block

    The block factorises a standard convolution into a spatial depthwise stage and a pointwise
    channel fusion stage, which keeps the decoding path computationally light
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform the feature map through the depthwise and pointwise stages"""
        x = F.relu6(self.depthwise_bn(self.depthwise(x)))
        x = F.relu6(self.pointwise_bn(self.pointwise(x)))
        return x


class Decoder(nn.Module):
    """
    Hierarchical upsampling decoder with attention modulated skip fusion

    The decoder restores resolution in four stages, each stage upsamples the running feature,
    gates the matching backbone skip, and fuses them before the next separable block

    Args:
        encoder_channels: Channel depth at each backbone level from stride thirty two to four
        decoder_channels: Filter counts for each decoder stage
    """

    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()

        stride32_channels, stride16_channels, stride8_channels, stride4_channels = encoder_channels
        filters_stage1, filters_stage2, filters_stage3, filters_stage4 = decoder_channels

        self.decoded_stage1 = SeparableBlock(stride32_channels, filters_stage1)
        self.skip1 = SeparableBlock(stride16_channels, filters_stage1)
        self.att1 = AttentionGate(filters_stage1, filters_stage1, filters_stage1)

        self.decoded_stage2 = SeparableBlock(2 * filters_stage1, filters_stage2)
        self.skip2 = SeparableBlock(stride8_channels, filters_stage2)
        self.att2 = AttentionGate(filters_stage2, filters_stage2, filters_stage2)

        self.decoded_stage3 = SeparableBlock(2 * filters_stage2, filters_stage3)
        self.skip3 = SeparableBlock(stride4_channels, filters_stage3)
        self.att3 = AttentionGate(filters_stage3, filters_stage3, filters_stage3)

        self.decoded_stage4 = SeparableBlock(2 * filters_stage3, filters_stage4)

    def forward(self, stride32_out, stride16_out, stride8_out, stride4_out) -> torch.Tensor:
        """
        Upsample and fuse the four backbone features into a high resolution map

        Args:
            stride32_out: Deepest backbone feature
            stride16_out: Stride sixteen backbone feature
            stride8_out: Stride eight backbone feature
            stride4_out: Shallowest backbone feature

        Returns:
            A stride four feature map ready for the detection heads
        """
        decoded_stage1 = self.decoded_stage1(stride32_out)
        upsampled_stage1 = F.interpolate(decoded_stage1, scale_factor=2, mode="nearest")
        concat_stage1 = torch.cat([upsampled_stage1, self.att1(self.skip1(stride16_out), upsampled_stage1)], dim=1)

        decoded_stage2 = self.decoded_stage2(concat_stage1)
        upsampled_stage2 = F.interpolate(decoded_stage2, scale_factor=2, mode="nearest")
        concat_stage2 = torch.cat([upsampled_stage2, self.att2(self.skip2(stride8_out), upsampled_stage2)], dim=1)

        decoded_stage3 = self.decoded_stage3(concat_stage2)
        upsampled_stage3 = F.interpolate(decoded_stage3, scale_factor=2, mode="nearest")
        concat_stage3 = torch.cat([upsampled_stage3, self.att3(self.skip3(stride4_out), upsampled_stage3)], dim=1)

        return self.decoded_stage4(concat_stage3)
