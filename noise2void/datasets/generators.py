"""generators.py

Code to generate the datasets from config files.
"""

from noise2void.datasets.channels import Channel
from noise2void.datasets.dataset import ExampleVideoDataset


def _generate_video_from_config(config, predict: bool) -> ExampleVideoDataset:
    """Generates a dataset from config. If predict is set, the video filter is ignored (e.g. for inference)."""

    return ExampleVideoDataset(
        config.image_size, config.dataset.example_index
    )


def _generate_au_acetone_from_config(config, predict: bool) -> ExampleVideoDataset:
    """Generates an AuAcetoneDataset from config."""

    return ExampleVideoDataset(
        config.image_size, config.dataset.example_index
    )


# A mapping from the string specified in every config file's dataset section to a method to generate it from that config
dataset_generators = {
    "IridiumGLCVideo": _generate_video_from_config,
    "AuAcetoneDataset": _generate_au_acetone_from_config,
}
