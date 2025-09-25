"""U-Net

Contains the building blocks of a U-Net CNN architecture. Contains the following building-blocks as PyTorch Modules:

* EncoderBlock - Represents the double convolution, activation and, optionally, pooling found in the encoder path.
* DilatedEncoderBlock - An EncoderBlock with a larger kernel size by default along with dilated convolution.
* DecoderBlock - Represents the double convolution, activation and up-convolution found in U-Net's decoder path.
* HeadBlock - Represents the final layer of a U-Net, with a 1x1 convolution and optional/custom activation.
* U-Net - The U-Net architecture configurable as required.
"""

import torch
import torch.nn as nn
import torch.optim


class MaxBlurPool(nn.Module):
    """A MaxBlurPool operation, as a replacement for MaxPool"""

    def __init__(self, kernel_size: int = 2):
        super(MaxBlurPool, self).__init__()
        self.max = nn.MaxPool2d(kernel_size, stride=1)
        self.blurpool_mode = 'bilinear'  # Corresponds to 'triangle-3'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.max(x)
        x = nn.functional.interpolate(x, scale_factor=0.5, mode=self.blurpool_mode)
        return x


class AvgBlurPool(nn.Module):
    """A MaxBlurPool operation, as a replacement for MaxPool"""

    def __init__(self):
        super(AvgBlurPool, self).__init__()
        self.blurpool_mode = 'bilinear'  # Corresponds to 'triangle-3'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=0.5, mode=self.blurpool_mode)
        return x


class EncoderBlock(nn.Module):
    """U-Net encoder block"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, pool: bool = True):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.GroupNorm(8, out_ch)
        self.bn2 = nn.GroupNorm(8, out_ch)
        # self.dropout = nn.Dropout2d(p=0.2)
        self.activation = nn.LeakyReLU()
        # self.pool: Union[nn.Module, None] = nn.AvgPool2d(2) if pool else None
        self.pool: nn.Module | None = AvgBlurPool() if pool else None

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        # x = self.dropout(x)
        return x


class DilatedEncoderBlock(nn.Module):
    """A U-Net encoder block with dilation on the first convolution. Kernel size of 5 by default."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, pool: bool = True):
        super(DilatedEncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding='same', dilation=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding='same')
        self.bn1 = nn.GroupNorm(8, out_ch)
        self.bn2 = nn.GroupNorm(8, out_ch)
        # self.dropout = nn.Dropout2d(p=0.2)
        self.activation = nn.LeakyReLU()
        # self.pool: Union[nn.Module, None] = nn.AvgPool2d(2) if pool else None
        self.pool: nn.Module | None = AvgBlurPool() if pool else None

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        # x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    """UNet decoder block with skip connections"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super(DecoderBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv1 = nn.Conv2d(
            in_ch + in_ch // 2, out_ch, kernel_size=kernel_size, padding="same"
        )  # Skip has fewer chans
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.GroupNorm(8, out_ch)
        self.bn2 = nn.GroupNorm(8, out_ch)
        # self.dropout = nn.Dropout2d(p=0.2)
        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor, skip_features: torch.Tensor):
        x = self.upsample(x)
        x = torch.cat((skip_features, x), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        # x = self.dropout(x)
        return x


class HeadBlock(nn.Module):
    """U-Net head, converting the output of the decoder into the desired number of channels with 1D convolutions"""

    def __init__(self, in_ch: int, out_ch: int, activation: nn.Module | None):
        super(HeadBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)  # A 1x1 convolution
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNet(nn.Module):
    """An adaptable implementation of the U-Net with skip connections for semantic segmentation or image-to-image
    translation.

    Methods
    -------
    forward(x: torch.Tensor) : torch.Tensor
        Passes `x` forwards through the model.
    predict(x: np.ndarray, device: torch.device, batch_size: int=16) : np.ndarray
        For making predictions from a trained model. Must specify the device the model is currently on.
    save(filename: str) : None
        Saves the parameters of the model as it currently is.
    load(filename: Path)
        Loads pre-trained parameters from file.
    num_params() : int
        Counts the number of parameters in the network.
    """

    def __init__(
        self, input_channels: int, output_channels: int, num_layers: int, first_layer_channels: int,
        first_layer_kernel_size: int, activation: nn.Module | None
    ):
        """
        Parameters
        ----------
        input_channels : int
            The number of channels in the network input
        output_channels : int
            The number of output segmentation maps
        num_layers : int
            The depth of the network i.e. the number of layers in the encoder branch of the network.
        first_layer_channels : int
            The number of channels formed in the first layer of the network. This is doubled in every subsequent layer
            of the encoder. Should be greater than the number of input channels.
        activation : Union[nn.Module, None]
            The activation to apply to the head of the network.
        """

        super(UNet, self).__init__()

        # Here I define the number of channels in each part of the network
        inner_channels = [first_layer_channels * 2**layer for layer in range(num_layers)]

        encoder_block_output_channels = inner_channels
        encoder_block_input_channels = [input_channels,] + inner_channels[:-1]

        bottleneck_block_input_channels = encoder_block_output_channels[-1]
        bottleneck_block_output_channels = bottleneck_block_input_channels * 2

        decoder_block_input_channels = [bottleneck_block_output_channels,] + list(reversed(inner_channels))[:-1]
        decoder_block_output_channels = list(reversed(inner_channels))

        head_block_input_channels = decoder_block_output_channels[-1]
        head_block_output_channels = output_channels

        self.encoder_blocks = nn.ModuleList(
            [DilatedEncoderBlock(
                encoder_block_input_channels[0], encoder_block_output_channels[0], first_layer_kernel_size, pool=False
            )] +
            [
                EncoderBlock(in_ch, out_ch, 3) for in_ch, out_ch
                in zip(encoder_block_input_channels[1:], encoder_block_output_channels[1:])
            ]
        )

        self.bottleneck_block = EncoderBlock(bottleneck_block_input_channels, bottleneck_block_output_channels, 3)

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch) for in_ch, out_ch
            in zip(decoder_block_input_channels[:-1], decoder_block_output_channels[:-1])
        ] + [DecoderBlock(decoder_block_input_channels[-1], decoder_block_output_channels[-1])]
        )

        self.head_block = HeadBlock(head_block_input_channels, head_block_output_channels, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the batch forwards through the network."""

        skip_features: list[torch.Tensor] = list()  # Saves the skip connections
        for skip_num, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            skip_features.append(x)
        x = self.bottleneck_block(x)
        for decoder_block, skip_feature in zip(self.decoder_blocks, reversed(skip_features)):
            x = decoder_block(x, skip_feature)
        x = self.head_block(x)
        return x

    @property
    def num_params(self) -> int:
        return sum(param.numel() for param in self.parameters())
