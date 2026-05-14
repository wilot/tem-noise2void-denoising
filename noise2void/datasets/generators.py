"""generators.py

Code to generate the datasets from config files.
"""

from noise2void.datasets.channels import Channel
from noise2void.datasets.iridium_glc_dataset import IridiumVideoDataset


def _generate_iridium_video_from_config(config, predict: bool) -> IridiumVideoDataset:
    """Generates an IridiumDataset from config."""

    # This dataset only has the HAADF channel!
    channels = [Channel(chan) for chan in config.channels]
    if len(channels) > 1 or channels[0] != Channel.HAADF:
        raise ValueError(f"Only HAADF channel is valid for IrVideo dataset, not {channels=}")

    try:  # In older config file versions, this was None
        video_filter = config.dataset.video_filter
    except:
        video_filter = None
    if video_filter == "none":
        video_filter = None
    return IridiumVideoDataset(
        config.image_size, config.dataset.example_index, video_filter
    )


# A mapping from the string specified in every config file's dataset section to a method to generate it from that config
dataset_generators = {
    "IridiumGLCVideo": _generate_iridium_video_from_config,
}
