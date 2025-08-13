"""tungsten_dataset.py

Contains the PyTorch Dataset for the Tunsten Disulphide experimental data.

Note: to delete all generated PNGs, navigate to the Tungsten WS2 data directory and use this command
`find . -type f -regex ".*\.png" -not -name "1777865_LAADF_fft.png" -delete`
"""

import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms.v2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

import hyperspy.api as hs
from tqdm import tqdm

from channels import Channel



class TungstenDataset:
    """Abstraction over the twisted WS2 dataset"""

    DATA_DIRECTORY = Path("data/Twisted WS2/raw_images")

    def __init__(self, image_size: int, channels: list[Channel]):
        self.channels = channels
        self.image_size = image_size
        self.sample_filegroups = self._find_filepaths()

    def plot_images(self):
        """Converts each multi-channel image to a PNG"""

        with tqdm(total=193, desc="Plotting multi-channel images") as pbar:

            for sample in self.sample_filegroups:
                for im_index in self.sample_filegroups[sample]:

                    a_chan = next(iter(self.sample_filegroups[sample][im_index].keys()))
                    savepath = self.sample_filegroups[sample][im_index][a_chan].parent / \
                               (self.sample_filegroups[sample][im_index][a_chan].stem.strip(a_chan.value + "_") + ".png")
                    if savepath.exists():  # Already plotted...
                        continue

                    chans = sorted(self.sample_filegroups[sample][im_index].keys(), key=lambda chan: chan.value)
                    assert len(chans) > 0
                    gridspec_kw = dict(left=0, right=1, bottom=0, top=0.91, hspace=0, wspace=0)
                    fig, axes = plt.subplots(
                        1, len(chans), figsize=(len(chans) * 8, 8 * 1.1), gridspec_kw=gridspec_kw
                    )
                    for ax_index, chan in enumerate(chans):
                        sig = hs.load(self.sample_filegroups[sample][im_index][chan])
                        axes[ax_index].imshow(sig.data, cmap="inferno", interpolation=None)
                        axes[ax_index].set_title(chan.value)
                        axes[ax_index].axis("off")
                    px_scale, px_units = sig.axes_manager[-1].scale, sig.axes_manager[-1].units
                    if px_units == "Å":
                        px_scale /= 10.
                        px_units = "nm"
                    sbar = ScaleBar(
                        px_scale, px_units, location="lower right", color='w', box_color='k', box_alpha=0.7
                    )
                    axes[ax_index].add_artist(sbar)
                    fig.savefig(savepath, dpi=210)
                    plt.close(fig)
                    pbar.update()


    def _print_samplegroup_stats(self):
        """Prints sample-wise details about the samples' images, including image sizes and channels used."""

        # Count channels and ensure consistency
        print("\n\n#       Channels       #")
        for sample in self.sample_filegroups:
            print(f"{sample:-^16}")
            sample_channels: set[Channel] | None = None
            for im_index in self.sample_filegroups[sample]:
                chans = set(self.sample_filegroups[sample][im_index].keys())
                if sample_channels is None:
                    sample_channels = set(chans)
                else:
                    fpath = next(iter(self.sample_filegroups[sample][im_index].values()))  # One filepath to help debug
                    if len(sample_channels) > len(chans):
                        missing_channels = sample_channels - chans
                        print(f"Warning: channel count mismatch for {fpath.name}, missing {missing_channels}")
                    elif len(sample_channels) < len(chans):
                        extra_channels = chans - sample_channels
                        print(f"Warning: channel count mismatch for {fpath.name}, extra {extra_channels}")
            print(f"Sample channels: {sample_channels}")

        print("\n\n#       Image shapes       #")
        for sample in self.sample_filegroups:
            print(f"{sample:-^16}")
            image_shapes: list[int] = list()
            for im_index in self.sample_filegroups[sample]:
                an_fpath = next(iter(self.sample_filegroups[sample][im_index].values()))
                sig = hs.load(str(an_fpath))  # Load any image
                im_shape = sig.data.shape
                if len(im_shape) != 2:
                    print(f"Warning: unexpected image shape for {an_fpath.name} {im_shape=}")
                    continue
                if im_shape[0] != im_shape[1]:
                    print(f"Warning: image is not square for {an_fpath.name} {im_shape=}")
                    continue
                image_shapes.append(im_shape[-1])
            im_sizes, size_frequencies = np.unique(image_shapes, return_counts=True)
            print(f"Found image sizes: {im_sizes}")
            print(f"Frequencies: {size_frequencies}")

    @staticmethod
    def _try_insert(
            sample_filegroups: dict[str, dict[int, dict[Channel, Path]]], sample: str, chan: Channel, index: int,
            filepath: Path
    ) -> bool:
        """Inserts the file path into the directory structure, or returns false if there is a problem and prints it"""

        if sample_filegroups[sample].get(index) is None:  # New index entry
            sample_filegroups[sample][index] = {chan: filepath}
        else:
            if sample_filegroups[sample][index].get(chan) is None:  # New channel (expected)
                sample_filegroups[sample][index][chan] = filepath
            else:  # Duplicate image?!
                warn = f"Found duplicate channel for image {filepath} and " + \
                       f"{sample_filegroups[sample][index][chan]}"
                print(warn)
                return False
        return True

    @classmethod
    def _find_filepaths(cls) -> dict[str, dict[int, dict[Channel, Path]]]:
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
                if not cls._try_insert(sample_filegroups, sample, chan, index, fpath):
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
            if not cls._try_insert(sample_filegroups, "May25_tWS2_4", chan, index, fpath):
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
                if not cls._try_insert(sample_filegroups, sample, chan, index, fpath):
                    continue

        return sample_filegroups


if __name__ == "__main__":
    # For getting info on the samples and plotting etc.

    dset = TungstenDataset(256, list())
    print(f"{dset.total_images}")
    dset.plot_images()