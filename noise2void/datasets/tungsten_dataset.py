"""tungsten_dataset.py

Contains the PyTorch Dataset for the Tunsten Disulphide experimental data.

The dataset contains data from three experimental sessions, divided into five samples. Each image contains HAADF and
BF channels, and sometimes LAADF too. Some images are of very low magnification, some of an intermediate mag and most
are at high magnification. Most images are 2K or 4K, although a few are 1K.

Note: to delete all generated PNGs, navigate to the Tungsten WS2 data directory and use this command
`find . -type f -regex ".*\\.png" -not -name "1777865_LAADF_fft.png" -delete`
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


class TungstenDataset(Dataset, MultiChannelDataset):
    """Abstraction over the twisted WS2 dataset

    This Dataset handles loading channels from file, stacking them, and applying interpolations and cropping.
    Normalisations are also performed here. A single image is kept back from the dataset to be used as a validation
    example.
    """

    DATA_DIRECTORY = Path("data/Twisted WS2/raw_images")
    BLACKLIST_FILE = Path("noise2void/datasets/tungsten_dataset_blacklist.txt")

    def __init__(
        self, image_size: int, channels: list[Channel], px_scale: float, example_index: int | None = None,
            crop_bounds: int=5
    ):
        """
        Parameters
        ----------
        image_size: int
            The size of each image in pixels. Outputs will be randomly cropped
        channels: list[Channel]
            The channels that must be included with each image. This is in-order!
        px_scale: float
            The pixel size, in nanometres. Images with similar `px_scale` will be interpolated to match this value
        example_index: int | None
            The index of the reserved validation image to hold back. None specifies a random selection.
        crop_bounds: int
            Each image will be cropped to the nearest 2^`crop_bounds` to work with a unet of `crop-bounds` depth
        """
        print("Initialising Dataset")
        assert len(set(channels)) == len(channels)
        self._channels = channels
        self.image_size = image_size
        self.px_scale = px_scale
        self.crop_bounds = crop_bounds
        print("Finding filepaths")
        self._all_sample_filegroups = self._find_filepaths(self.channels)
        print("Fetching scales and shapes")
        for meta in self._all_sample_filegroups:
            meta.fetch_scale_shape()

        with open(self.BLACKLIST_FILE, 'rb') as file:
            blacklist = tomllib.load(file)
        assert set(blacklist.keys()) == set(meta.sample for meta in self._all_sample_filegroups)
        self.samples = set(blacklist.keys())

        print("Applying blacklist")
        self._valid_sample_filegroups = list(filter(
            lambda meta: not meta.is_blacklisted(blacklist),
            self._all_sample_filegroups
        ))
        print(f"Post blacklist length: {len(self._valid_sample_filegroups)}")

        print("Filtering by channels")
        self._channel_sample_filegroups = list(filter(
            lambda meta: meta.has_channels(set(channels)),
            self._valid_sample_filegroups
        ))

        print("Filtering by scale")
        self._to_scale_sample_filegroups = list(filter(
            lambda meta: meta.in_scale_range(self.px_scale, 3.),
            self._channel_sample_filegroups
        ))

        self._transform = tforms.Compose([
            tforms.RandomCrop(self.image_size),
            # tforms.RandomHorizontalFlip(),
            # tforms.RandomVerticalFlip(),
        ])

        self._sample_filegroups = self._to_scale_sample_filegroups
        assert len(self.sample_filegroups) > 1
        if example_index is None:
            self._reserved_example_meta = self._sample_filegroups.pop(  # Randomly select an example
                int(torch.randint(0, len(self._sample_filegroups), (1,))[0])
            )
        else:
            self._reserved_example_meta = self._sample_filegroups.pop(example_index)
        self._reserved_example: torch.Tensor | None = None

    def load_interpolate(self, meta: MultiChannelMetadata) -> torch.Tensor:
        """Loads and interpolates the multi-channel image"""

        datum = torch.from_numpy(meta.load_channels(self.channels)).to(torch.float32)
        interp_factor = meta.px_scale / self.px_scale
        interp_shape = int(interp_factor * meta.shape)
        if interp_shape % 2**self.crop_bounds != 0:  # Round this shape to the nearest order of two
            interp_shape += 2**self.crop_bounds - interp_shape % 2**self.crop_bounds
        print(f"load_interpolate interpolating to {interp_shape} for {meta.fpaths[Channel.HAADF]}")
        datum = tforms.functional.resize(datum, interp_shape)  # Interpolate to correct scale
        return datum

    def uninterpolate(self, meta: MultiChannelMetadata, datum: torch.Tensor) -> torch.Tensor:
        """Reverses any interpolation that would be applied to the image to make its magnification conform to the
        dataset as a whole."""

        datum = tforms.functional.resize(datum, meta.shape)
        return datum

    def _load_interpolate_random_crop(self, meta: MultiChannelMetadata) -> torch.Tensor:
        """Loads, interpolates and crops the multi-channel image"""

        datum = self.load_interpolate(meta)
        datum = self._transform(datum)  # Randomly crop
        return datum

    def _load_interpolate_crop(self, meta: MultiChannelMetadata, crop_size: int) -> torch.Tensor:
        """Loads, interpolates and deterministically crops"""

        datum = self.load_interpolate(meta)
        datum = tforms.functional.center_crop(datum, crop_size)
        return datum

    @property
    def reserved_example(self) -> torch.Tensor:
        """A cropped and normalised example held back from the training set for validations."""

        if self._reserved_example is None:
            datum = self._load_interpolate_crop(self._reserved_example_meta, 1024)
            self._reserved_example = self.normalise(datum)  # The example for validations
        return self._reserved_example

    @property
    def sample_filegroups(self):
        """Multichannel metadata for the file-froups"""

        return self._sample_filegroups

    @property
    def channels(self) -> list[Channel]:
        """The order in which each channel is stored in the arrays"""

        return self._channels

    @staticmethod
    def normalise(datum: torch.Tensor) -> torch.Tensor:
        """Normalises the image between zero and one"""

        datum -= torch.mean(datum, dim=(-2, -1), keepdim=True)
        datum /= torch.std(datum, dim=(-2, -1), keepdim=True) * 5
        datum = torch.tanh(datum)

        # datum -= torch.amin(datum, dim=(-2, -1), keepdim=True)
        # datum /= torch.amax(datum, dim=(-2, -1), keepdim=True)
        return datum

    @classmethod
    def get_savename(cls, meta: MultiChannelMetadata) -> Path:
        """Returns the savepath, according to this dataset's saving convention, for this datum. This savepath is
        has any channel in the filename removed."""

        if meta.sample in ("Apr25_tWS2_1", "Apr25_tWS2_2"):
            savename = meta.fpaths[Channel.HAADF].name.replace("HAADF_", '')
        elif meta.sample in ("May25_tWS2_4"):
            chan, mag, index = meta.fpaths[Channel.HAADF].stem.split("_")
            savename = index + meta.fpaths[Channel.HAADF].suffix
        elif meta.sample in ("Jul24_tWS2_1", "Jul24_tWS2_3"):
            mag, chan, index = meta.fpaths[Channel.HAADF].stem.split("_")
            savename = index + meta.fpaths[Channel.HAADF].suffix
        else:
            raise ValueError(f"Unknown sample {meta.sample}")
        savepath = meta.fpaths[Channel.HAADF].parent / savename  # This is the path relative to project root
        savepath = savepath.relative_to(cls.DATA_DIRECTORY)  # This is relative to this dataset's data directory
        return savepath

    def __len__(self) -> int:
        """The number of images in the dataset. Note random crops are applied, so the effective number is larger."""

        return len(self.sample_filegroups)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Loads, interpolates and randomly crops from the specified multi-channel image"""

        datum = self._load_interpolate_random_crop(self.sample_filegroups[index])[0]
        datum = self.normalise(datum)
        return datum

    @classmethod
    def _find_filepaths(cls, channel_order: list[Channel]) -> list[MultiChannelMetadata]:
        """Constructs the dictionary of filepaths in the experimental dataset. Each channel in the multi-channel
        images is saved in a separate file. Each file only contains a single image (there are no videos).

        Parameters
        ----------
        channel_order: list[Channel]
            The order in which to store the channels in the MultiChannelMetadata. This will match the way the channels
            are arranged in the tensors!

        Returns
        -------
        list[MultiChannelMetadata]
            The metadata discovered.
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
                sample_filegroups_list.append(  # This implicitly drops channels not in the `channel_order`!
                    MultiChannelMetadata(sample_filegroups[sample][im_index], channel_order, sample, im_index, None)
                )

        return sample_filegroups_list

    def _print_dataset_stats(self):
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
                image_shapes.append(meta.shape)
            im_sizes, size_frequencies = np.unique(image_shapes, return_counts=True)
            print(f"Found image sizes: {im_sizes}")
            print(f"Frequencies: {size_frequencies}")

        print("\n\n    Dataset sizes    ")
        print(f"Valid images: {len(self._valid_sample_filegroups)}")
        print(f"Images satisfying channels {self.channels}: {len(self._channel_sample_filegroups)}")
        print(f"Images satisfying mag constraints: {len(self._to_scale_sample_filegroups)}")
        print(f"Total images used: {len(self.sample_filegroups)}")
        print(f"Effective number of {self.image_size}px images in dataset (after mag interpolation): {self._calculate_effective_length():.1E}")

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

    def _calculate_effective_length(self) -> float:
        """Calculates the effective length of the dataset, taking into account interpolation and tiled cropping"""

        total = 0.0
        for meta in self.sample_filegroups:
            interp_factor = meta.px_scale / self.px_scale
            new_shape = interp_factor * meta.shape
            effective_images = (new_shape / self.image_size) ** 2.
            total += effective_images
        return total

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

    dset = TungstenDataset(256, [Channel.HAADF, Channel.BF], 0.007, None)
    # dset._print_dataset_stats()

    # Plot example images
    fig, axes = plt.subplots(len(dset.channels), 4, sharex=True, sharey=True)
    for col in axes.T:
        index = torch.randint(0, len(dset), (1,))[0]
        datum = dset[index]
        for chan_index in range(len(dset.channels)):
            col[chan_index].imshow(datum[chan_index], cmap="inferno", interpolation=None)
    axes[0, 0].set_ylabel("HAADF")
    axes[1, 0].set_ylabel("BF")

    # Plot example histograms
    fig, axes = plt.subplots(len(dset.channels), 4, sharex=True, sharey=True)
    for col in axes.T:
        index = torch.randint(0, len(dset), (1,))[0]
        datum = dset[index]
        for chan_index in range(len(dset.channels)):
            col[chan_index].hist(datum[chan_index].flatten(), bins=64)
    axes[0, 0].set_ylabel("HAADF")
    axes[1, 0].set_ylabel("BF")
    plt.show(block=True)