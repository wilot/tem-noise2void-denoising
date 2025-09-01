"""generators.py

Code to generate the models from config files.
"""

from noise2void.models.unet import UNet


def _generate_unet_from_config(config) -> UNet:
    """Generates a Tungsten dataset with the appropriate configuration"""

    return UNet(
        input_channels=len(config.channels), output_channels=len(config.channels), num_layers=config.model.depth,
        first_layer_channels=config.model.first_layer_channels,
        first_layer_kernel_size=config.model.initial_kernel_size, activation=None
    )


# A mapping from the string specified in every config file's model section to a method to generate it from that config
model_generators = {
    "UNet": _generate_unet_from_config
}