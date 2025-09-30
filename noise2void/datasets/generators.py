"""generators.py

Code to generate the datasets from config files.
"""

from noise2void.datasets.channels import Channel
from noise2void.datasets.tungsten_dataset import TungstenDataset
from noise2void.datasets.iridium_glc_dataset import IridiumVideoDataset


def _generate_tungsten_from_config(config, predict: bool) -> TungstenDataset:
    """Generates a Tungsten dataset with the appropriate configuration"""

    channels = [Channel(chan) for chan in config.channels]
    return TungstenDataset(
        config.image_size, channels, config.dataset.px_scale,
        config.dataset.example_index if config.dataset.example_index != "none" else None
    )


def _generate_iridium_video_from_config(config, predict: bool) -> IridiumVideoDataset:
    """Generates an IridiumDataset from config. If predict is set, the video filter is ignored (e.g. for inference)."""

    try:  # In older config file versions, this was None
        video_filter = config.dataset.video_filter
    except:
        video_filter = None
    if video_filter == "none":
        video_filter = None
    return IridiumVideoDataset(
        config.image_size, config.dataset.example_index, None if predict else video_filter
    )


# A mapping from the string specified in every config file's dataset section to a method to generate it from that config
dataset_generators = {
    "WS2": _generate_tungsten_from_config,
    "IridiumGLCVideo": _generate_iridium_video_from_config,
}