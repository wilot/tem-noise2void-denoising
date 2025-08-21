"""tungsten_dataset.py

Contains the PyTorch Dataset for the Tunsten Disulphide experimental data.

The dataset contains data from three experimental sessions, divided into five samples. Each image contains HAADF and
BF channels, and sometimes LAADF too. Some images are of very low magnification, some of an intermediate mag and most
are at high magnification. Most images are 2K or 4K, although a few are 1K.

Note: to delete all generated PNGs, navigate to the Tungsten WS2 data directory and use this command
`find . -type f -regex ".*\.png" -not -name "1777865_LAADF_fft.png" -delete`
"""

import tomllib
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms.v2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

import hyperspy.api as hs
from tqdm import tqdm

from channels import Channel, MultiChannelMetadata


class TungstenDataset:
    """Abstraction over the twisted WS2 dataset"""

    DATA_DIRECTORY = Path("data/Twisted WS2/raw_images")
    BLACKLIST_FILE = Path("noise2void/datasets/tungsten_dataset_blacklist.txt")

    def __init__(self, image_size: int, channels: list[Channel], px_scale: float):
        """
        Parameters
        ----------
        image_size: int
            The size of each image in pixels. Outputs will be randomly cropped
        channels: list[Channel]
            The channels that must be included with each image
        px_scale: float
            The pixel size, in nanometres. Images with similar `px_scale` will be interpolated to match this value
        """
        print("Initialising Dataset")
        self.channels = channels
        self.image_size = image_size
        self.px_scale = px_scale
        print("Finding filepaths")
        self._all_sample_filegroups = self._find_filepaths()
        print("Fetching scales and shapes")
        for meta in self._all_sample_filegroups:
            meta.fetch_scale_shape()

        with open(self.BLACKLIST_FILE, 'rb') as file:
            blacklist = tomllib.load(file)
        assert set(blacklist.keys()) == set(meta.sample for meta in self._all_sample_filegroups)
        self.samples = set(blacklist.keys())

        print("Applying blacklist")
        self._valid_sample_filegroups = list(filter(  # TODO: Bug, this is filtering out everything!
            lambda meta: meta.is_blacklisted(blacklist),
            self._all_sample_filegroups
        ))
        print(f"Post blacklist length: {len(self._valid_sample_filegroups)}")

        print("Filtering by channels")
        self._channel_sample_filegroups = list(filter(
            lambda meta: meta.has_channels(channels),
            self._valid_sample_filegroups
        ))

        print("Filtering by scale")
        self._to_scale_sample_filegroups = list(filter(
            lambda meta: meta.in_scale_range(self.px_scale, 3.),
            self._channel_sample_filegroups
        ))

        self.sample_filegroups = self._to_scale_sample_filegroups

    def _plot_images(self):
        """Converts each multi-channel image to a PNG"""

        with tqdm(total=193, desc="Plotting multi-channel images") as pbar:

            for meta in self._all_sample_filegroups:
                a_chan = next(iter(meta.fpaths.keys()))
                savepath = meta.fpaths[a_chan].parent / \
                    (meta.fpaths[a_chan].stem.strip(a_chan.value + "_") + ".png")
                if savepath.exists():
                    continue

                chans = sorted(meta.fpaths.keys(), key=lambda chan: chan.value)
                assert len(chans) > 0
                gridspec_kw = dict(left=0, right=1, bottom=0, top=0.91, hspace=0, wspace=0)
                fig, axes = plt.subplots(
                    1, len(chans), figsize=(len(chans) * 8, 8 * 1.1), gridspec_kw=gridspec_kw
                )
                for ax_index, chan in enumerate(chans):
                    sig = hs.load(str(meta.fpaths[chan]))
                    axes[ax_index].imshow(sig.data, cmap="inferno", interpolation=None)
                    axes[ax_index].set_title(chan.value)
                    axes[ax_index].axis("off")
                sbar = ScaleBar(
                    meta.px_scale, "nm", location="lower right", color='w', box_color='k', box_alpha=0.7
                )
                axes[ax_index].add_artist(sbar)
                fig.savefig(savepath, dpi=210)
                plt.close(fig)
                pbar.update()

    def print_dataset_stats(self):
        """Prints sample-wise details about the samples' images, including image sizes and channels used."""

        # Count channels and ensure consistency
        print("\n\n#       Channels       #")
        for sample in self.samples:
            print(f"{sample:-^16}")
            sample_channels: set[Channel] | None = None
            for meta in filter(lambda meta: meta.sample == sample, self._all_sample_filegroups):
                chans = set(meta.fpaths.keys())
                if  sample_channels is None:
                    sample_channels = set(chans)
                else:
                    an_fpath = meta.an_fpath()
                    if len(sample_channels) > len(chans):
                        missing_channels = sample_channels - chans
                        print(f"Warning: channel count mismatch for {an_fpath.name}, missing {missing_channels}")
                    elif len(sample_channels) < len(chans):
                        extra_channels = chans - sample_channels
                        print(f"Warning: channel count mismatch for {an_fpath.name}, extra {extra_channels}")
            print(f"Sample channels: {sample_channels}")

        print("\n\n#       Image shapes       #")
        for sample in self.samples:
            print(f"{sample:-^16}")
            image_shapes: list[int] = list()
            for meta in filter(lambda meta: meta.sample == sample, self._all_sample_filegroups):
                im_shape = meta.shape
                image_shapes.append(im_shape)
            im_sizes, size_frequencies = np.unique(image_shapes, return_counts=True)
            print(f"Found image sizes: {im_sizes}")
            print(f"Frequencies: {size_frequencies}")

        print("\n\n    Dataset sizes    ")
        print(f"Valid images: {len(self._valid_sample_filegroups)}")
        print(f"Images satisfying channels {self.channels}: {len(self._channel_sample_filegroups)}")
        print(f"Images satisfying mag constraints: {len(self._to_scale_sample_filegroups)}")
        print(f"Total images used: {len(self.sample_filegroups)}")
        print(f"Effective number of {self.image_size}px images in dataset (after mag interpolation): {self._calculate_effective_length()}")

    def _calculate_effective_length(self) -> float:
        """Calculates the effective length of the dataset, taking into account interpolation and tiled cropping"""

        total = 0.0
        for meta in self.sample_filegroups:
            interp_factor = meta.scale / self.px_scale
            new_shape = interp_factor * meta.shape
            effective_images = (new_shape / self.image_size) ** 2.
            total += effective_images
        return total

    @classmethod
    def _find_filepaths(cls) -> list[MultiChannelMetadata]:
        """Constructs the dictionary of filepaths in the experimental dataset

        Returns
        -------
        dict[str, dict[int, dict[Channel, Path]]]
            Dictionary of filepaths of the form `dict[sample-name][image-index][channel]`
        """

        sample_filedirs = {  # Different glob and pairing for each, yay!
            "Apr25_tWS2_1": cls.DATA_DIRECTORY / "Apr25" / "tWS2_1",
            "Apr25_tWS2_2": cls.DATA_DIRECTORY / "Apr25" / "tWS2_2",
            "May25_tWS2_4": cls.DATA_DIRECTORY / "May25" / "WS2-4",
            "Jul24_tWS2_1": cls.DATA_DIRECTORY / "Jul24" / "ws2_1",
            "Jul24_tWS2_3": cls.DATA_DIRECTORY / "Jul24" / "ws2_3",
        }

        # Using same str-keys, link files with the same ID as channels
        sample_filegroups: dict[str, dict[int, dict[Channel, Path]]] = dict()

        for sample in ("Apr25_tWS2_1", "Apr25_tWS2_2"):  # These two are arranged the same
            sample_filegroups[sample] = dict()
            for fpath in sample_filedirs[sample].glob("*.dm4"):
                try:
                    chan, index = fpath.stem.split("_")
                except:  # This file path is not in an expected format
                    print(f"Unexpected format for file: {fpath}")
                    continue
                chan = Channel[chan]
                index = int(index)
                try:
                    cls._try_insert_filepath(sample_filegroups, sample, chan, index, fpath)
                except ValueError:
                    continue

        sample_filegroups["May25_tWS2_4"] = dict()
        for fpath in sample_filedirs["May25_tWS2_4"].glob("*.dm4"):
            try:
                chan, mag, index = fpath.stem.split("_")
            except:
                print(f"Unexpected format for file: {fpath}")
                continue
            chan = Channel[chan]
            index = int(index)
            if mag[-2:] == "kX" or mag[-2:] == "KX":
                continue  # Too low mag to be useful
            try:
                cls._try_insert_filepath(sample_filegroups, "May25_tWS2_4", chan, index, fpath)
            except ValueError:
                continue

        for sample in ("Jul24_tWS2_1", "Jul24_tWS2_3"):  # These two are organised the same
            sample_filegroups[sample] = dict()
            for fpath in sample_filedirs[sample].glob("*.dm3"):
                try:
                    mag, chan, index = fpath.stem.split("_")
                except:
                    print(f"Unexpected format for file {fpath}")
                    continue
                chan = Channel[chan]
                index = int(index)
                if mag[-2:] == "kX" or mag[-2:] == "KX":
                    continue  # Too low mag to be useful
                if float(mag[:-2]) < 1.8:  # Too low mag to be useful
                    continue
                try:
                    cls._try_insert_filepath(sample_filegroups, sample, chan, index, fpath)
                except ValueError:
                    continue

        sample_filegroups_list = list()
        for sample in sample_filegroups:
            for im_index in sample_filegroups[sample]:
                sample_filegroups_list.append(
                    MultiChannelMetadata(sample_filegroups[sample][im_index], sample, im_index)
                )

        return sample_filegroups_list

    @staticmethod
    def _try_insert_filepath(
        sample_filegroups: dict[str, dict[int, dict[Channel, Path]]], sample: str, chan: Channel, index: int,
        fpath: Path
    ):
        """Attempts to insert an item into the filegroup-dict, raising a ValueError if a duplicate item is already
        present in the structure."""

        if sample_filegroups.get(sample) is None:
            sample_filegroups[sample] = dict()
        if sample_filegroups[sample].get(index) is None:
            sample_filegroups[sample][index] = dict()
        if sample_filegroups[sample][index].get(chan) is None:  # This should not be populated yet
            sample_filegroups[sample][index][chan] = fpath
        else:
            warn = f"An image with index {index} and channel {chan} from {sample} already exists:"
            warn += f"\t{fpath}\n\t{sample_filegroups[sample][index][chan]}"
            raise ValueError(warn)


if __name__ == "__main__":
    # For getting info on the samples and plotting etc.

    dset = TungstenDataset(256, [Channel.HAADF, Channel.BF, Channel.LAADF], 0.007)
    dset.print_dataset_stats()