"""generators.py

Code to generate the datasets from config files.
"""

from noise2void.datasets.channels import Channel
from noise2void.datasets.tungsten_dataset import TungstenDataset
from noise2void.datasets.iridium_glc_dataset import IridiumVideoDataset


def _generate_tungsten_from_config(config) -> TungstenDataset:
    """Generates a Tungsten dataset with the appropriate configuration"""

    channels = [Channel(chan) for chan in config.channels]
    return TungstenDataset(
        config.image_size, channels, config.dataset.px_scale, config.dataset.example_index
    )


def _generate_iridium_video_from_config(config) -> IridiumVideoDataset:
    """Generates an IridiumDataset from config"""

    return IridiumVideoDataset(512, config.dataset.example_index)


# A mapping from the string specified in every config file's dataset section to a method to generate it from that config
dataset_generators = {
    "WS2": _generate_tungsten_from_config,
    "IridiumGLCVideo": _generate_iridium_video_from_config,
}