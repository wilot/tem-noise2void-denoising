"""resunet.py

A ResUNet++ implementation for Noise2Void.
"""

import torch
from torch import nn
import numpy as np


class SqueezeExcite(nn.Module):
    """SqueezeExcite block (like a channelwise bottleneck) as used in ResUNet++

    Takes an average over pixels in a batch. Passes those (B, C, 1, 1) shaped averages through a neural net which
    contains a bottleneck. Multiply (channel-wise) the output of the neural net with the original (B, C, H, W) input.
    """

    def __init__(self, channels: int, rate: int = 8):
        assert 1 < rate <= channels // 2
        super(SqueezeExcite, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # For any input, average-reduce height & width dimensions to 1x1
        self.net = nn.Sequential(
            nn.Linear(channels, channels // rate, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // rate, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)  # Flatten the final unit axes (..., 1, 1)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x


class EntryBlock(nn.Module):
    """The entry block for ResUNet.

    Similar to the double-convolution block but removes the initial batch-norm and ReLU from the beginning of the
    block.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(EntryBlock, self).__init__()

        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        )
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv_branch(inputs)
        r = self.res_branch(inputs)
        return x + r


class DoubleConvBlock(nn.Module):
    """A residual double-convolution block from the ResUNet++ architecture.

    Contains batch-norm, relu, conv2d, batch-norm, relu, conv2d, squeeze-excite
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(DoubleConvBlock, self).__init__()
        self.attention = SqueezeExcite(in_channels)
        self.conv_branch = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.res_branch = nn.Sequential(  # The residual carry-over
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.attention(inputs)
        x = self.conv_branch(inputs)
        res = self.res_branch(inputs)
        out = x + res
        return out


class ASPP(nn.Module):
    """Atrous spatial pyrmidal pooling, used at the bottleneck of a number of networks."""

    def __init__(self, in_channels: int, out_channels: int, rates: list[int] = [1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.conv_bn_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=rate, padding=rate),
                nn.BatchNorm2d(out_channels)
            )
            for rate in rates
        ])
        self.point_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [block(x) for block in self.conv_bn_branches]
        x = torch.sum(torch.stack(branches, dim=0), dim=0)  # Stack in a new axis and sum across that axis
        x = self.point_conv(x)
        return x


class AttentionGate(nn.Module):
    """Attention Gate as used in ResUNet++. Initially devised in AttentionUNet.

    Mixes the skip connection with the output from the previous layer, thereby helping the model learn to ignore
    unimportant large-scale information being passed through the skip connection. Outputs the same number of channels
    as the input-channels. Note: the ResUNet++ implementation isn't quite like the original AttentionUNet...
    """

    def __init__(self, in_channels: int, skip_channels: int):
        super(AttentionGate, self).__init__()
        out_channels = in_channels

        self.g_block = nn.Sequential(
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(),
            nn.Conv2d(skip_channels, out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        self.x_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.gx_block = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.g_block(skip)
        x1 = self.x_block(x)
        gx = g + x1
        gx = self.gx_block(gx)
        x = x * gx
        return x


class DecoderBlock(nn.Module):
    """A decoder block as defined in ResUNet++, contianing skip-attention, upsampling, skip-concatenation and a
    residual double-conv block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(DecoderBlock, self).__init__()

        self.attention = AttentionGate(in_channels, skip_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.resconv = DoubleConvBlock(in_channels + skip_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.attention(x, skip)
        out = self.upsample(out)
        out = torch.cat([out, skip], dim=1)
        out = self.resconv(out)
        return out


class ResUNet(nn.Module):
    """ResUNet++ architecture."""

    def __init__(self):
        super(ResUNet, self).__init__()

        self.enc1 = EntryBlock(1, 16, stride=1)
        self.enc2 = DoubleConvBlock(16, 32, stride=2)  # Use stride to half image size
        self.enc3 = DoubleConvBlock(32, 64, stride=2)
        self.enc4 = DoubleConvBlock(64, 128, stride=2)
        self.bottleneck = ASPP(128, 256)
        self.dec3 = DecoderBlock(256, 64, 128)
        self.dec2 = DecoderBlock(128, 32, 64)
        self.dec1 = DecoderBlock(64, 16, 32)
        self.head = nn.Sequential(
            ASPP(32, 16),
            nn.Conv2d(16, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x_skip1 = self.enc1(x)
        x_skip2 = self.enc2(x_skip1)
        x_skip3 = self.enc3(x_skip2)
        x_bottleneck = self.enc4(x_skip3)
        x_bottleneck = self.bottleneck(x_bottleneck)
        x = self.dec3(x_bottleneck, x_skip3)
        x = self.dec2(x, x_skip2)
        x = self.dec1(x, x_skip1)
        x = self.head(x)
        return x

    @property
    def num_params(self) -> int:
        return sum(param.numel() for param in self.parameters())