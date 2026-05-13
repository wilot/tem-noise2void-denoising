"""generators.py

Code to generate the models from config files.
"""

from noise2void.models.unet import UNet
from noise2void.models.resunet import ResUNet
from noise2void.models.swinunet import SwinUNet
from noise2void.models.swinunet import SwinTransformer


def _generate_unet_from_config(config) -> UNet:
    """Generates a UNet with the appropriate configuration"""

    return UNet(
        input_channels=len(config.channels), output_channels=len(config.channels), num_layers=config.model.depth,
        first_layer_channels=config.model.first_layer_channels,
        first_layer_kernel_size=config.model.initial_kernel_size, activation=None
    )


def _generate_resunet_from_config(config) -> ResUNet:
    """Generates a ResUNet from the configuration file"""

    return ResUNet()


def _generate_swinunet_from_config(config) -> SwinUNet:
    """Generates a SwinUNet from configuration file"""

    assert (config.image_size & (config.image_size-1) == 0) and config.image_size > 256
    max_window_size = config.image_size // 4 // 2**(config.model.depth - 1)
    return SwinUNet(
        img_size=config.image_size, patch_size=4, in_chans=len(config.channels), out_chans=len(config.channels),
        window_size=max_window_size, embed_dim=96
    )


def _generate_swintransformer_from_config(config) -> SwinTransformer:
    """Generates a Swin Transformer from configuration file"""

    window_size = 8

    assert (config.image_size & (config.image_size - 1) == 0) and config.image_size >= window_size * 2
    return SwinTransformer(
        img_size=config.image_size, patch_size=4, in_chans=len(config.channels), out_chans=len(config.channels),
        window_size=window_size, embed_dim=96, num_layers=config.model.depth
    )


# A mapping from the string specified in every config file's model section to a method to generate it from that config
model_generators = {
    "UNet": _generate_unet_from_config,
    "ResUNet": _generate_resunet_from_config,
    "SwinUNet": _generate_swinunet_from_config,
    "SwinTransformer": _generate_swintransformer_from_config
}