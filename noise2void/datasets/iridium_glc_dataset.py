"""iridium_glc_dataset.py

A dataset abstracting over the iridium experimental data.
"""

import tomllib
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms.v2 as tforms
import numpy as np

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

import hyperspy.api as hs
from tqdm import tqdm

from noise2void.datasets.channels import Channel, MultiChannelMetadata, MultiChannelDataset


class IridiumDataset(Dataset, MultiChannelDataset):
    """Abstraction over the iridium nanoparticle graphene-liquid-cell experimental data.

    With this dataset, only HAADF and BF were collected and all good data is around the 15MX magnification.
    """

    DATA_DIRECTORY = Path("data/GLC-2_Ir/raw")
    BLACKLIST_FILE = Path("data/GLC-2_Ir/iridium_dataset_blacklist.toml")

    def __init__(self, image_size: int, channels: list[Channel], example_index: int | None = None):
        """
        Parameters
        ----------
        image_size: int
            The size of each image in pixels. Outputs will be randomly cropped
        channels: list[Channel]
            The channels that must be included with each image. This is in-order!
        example_index: int | None
            The index of the reserved validation image to hold back. None specifies a random selection.
        """

        # TODO: Implement dataset