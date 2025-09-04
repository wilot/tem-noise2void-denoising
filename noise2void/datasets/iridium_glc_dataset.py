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
import matplotlib.animation as ani
from matplotlib_scalebar.scalebar import ScaleBar

from tqdm import tqdm

from noise2void.datasets.channels import Channel, MultiChannelMetadata, MultiChannelDataset


class IridiumVideoDataset(Dataset, MultiChannelDataset):
    """Abstraction over the iridium nanoparticle videos."""

    DATA_DIRECTORY = Path("data/GLC-2_Ir/raw")
    PLOT_DIRECTORY = Path("data/GLC-2_Ir/plots")
    VIDEO_STACK_PATH = Path("data/GLC-2_Ir/video_stack.pt")
    BLACKLIST_FILE = Path("noise2void/datasets/iridium_glc_blacklist.toml")
    VIDEO_PLOT_FPS = 10

    def __init__(self, image_size: int, example_index: int | None=None):

        assert image_size <= 512
        self.image_size = image_size
        self._channels = Channel.HAADF

        # Find all videos, load their shapes, load the blacklist
        self._all_filegroups = list(filter(  # Only select the videos
            lambda meta: meta.frames is not None, _find_iridium_filepaths(self.DATA_DIRECTORY)
        ))
        with open(self.BLACKLIST_FILE, 'rb') as f:
            self.blacklist = tomllib.load(f)
        assert set(self.blacklist["videos"].keys()) == set(meta.index for meta in self._all_filegroups)
        self._valid_filegroups = list(filter(
            lambda meta: self.blacklist["videos"][meta.index] is not False, self._all_filegroups
        ))
        for meta in self._valid_filegroups:
            meta.fetch_scale_shape()  # The scale metadata is wrong, but the shape is useful!
            if meta.shape != 512 and meta.shape != 1024:  # They're all this
                raise ValueError(f"Video: {meta.fpaths[Channel.HAADF]} has unexpected shape {meta.shape}")
            meta.px_scale = None
            meta.frames = sum(stop - start for start, stop in self.blacklist["videos"][meta.index])  # Count ok frames
        self._sample_filegroups = self._valid_filegroups

        # Calculate the length of the total dataset
        self._len = sum(  # They're all 512 or 1024 px videos
            meta.frames if meta.shape == 512 else meta.frames * 4
            for meta in self.sample_filegroups
        )
        if example_index is None:
            self._reserved_example_index = int(torch.randint(0, self._len, (1,))[0])
        else:
            self._reserved_example_index = example_index
        self._len -= 1  # Hold the reserved example back!
        if not self.VIDEO_STACK_PATH.exists():
            self._create_compressed_dataset()
        self._video_stack: torch.Tensor | None = None

    @property
    def sample_filegroups(self) -> list[MultiChannelMetadata]:
        return self._sample_filegroups

    def load_interpolate(self, meta: MultiChannelMetadata) -> torch.Tensor:
        """Loads the image/video and interpolates to required scale"""

        datum = torch.from_numpy(meta.load_channels(self.channels)).to(torch.float32)

        # Skip interpolation if video, the scale metadata is garbage
        datum = torch.concat(  # Only keep the valid frames
            [datum[start:stop] for start, stop in self.blacklist["videos"][meta.index]]
        )
        return datum

    def uninterpolate(self, meta: MultiChannelMetadata, datum: torch.Tensor) -> torch.Tensor:
        """Reverses any interpolation done when loading the image to the required mag/scale. No interpolation is done
        on the videos, so nothing to do here."""

        return datum

    @staticmethod
    def normalise(datum: torch.Tensor) -> torch.Tensor:
        """Normalises the video"""

        datum -= torch.mean(datum, dim=(-2, -1), keepdim=True)
        datum /= torch.std(datum, dim=(-2, -1), keepdim=True) * 5
        datum = torch.tanh(datum)
        return datum

    @property
    def channels(self) -> list[Channel]:
        """The order in which each channel is stored in the arrays"""

        return [Channel.HAADF]

    @property
    def reserved_example(self) -> torch.Tensor:
        """A cropped and normalised example held back from the training set for validations."""

        datum = self.video_stack[self._reserved_example_index]
        return datum

    @classmethod
    def get_savename(cls, meta: MultiChannelMetadata) -> Path:
        """Generates a savepath, relative to any save directory, for the given datum"""

        savename = meta.fpaths[Channel.HAADF].name.replace("HAADF ", '')
        return Path(savename)

    @property
    def video_stack(self):
        """A cache of all the pre-normalised, cropped/tiled video frames"""

        if self._video_stack is None:
            self._video_stack = torch.load(self.VIDEO_STACK_PATH)
        return self._video_stack

    def __len__(self):
        return self._len

    def __getitem__(self, index: int) -> torch.Tensor:

        if index >= self._reserved_example_index:
            index += 1  # Skip the reserved example!
        return self.video_stack[index]

    def _create_compressed_dataset(self):
        """Saves all the videos into a single HDF5 for much faster loading.

        I can dave this all into one massive file and load it into ram, but this would crash most PCs!
        """

        video_stack = list()
        for meta in self.sample_filegroups:
            video = self.load_interpolate(meta)
            if video.shape[-1] == 1024:  # They're all 1K or 512px
                video_stack.append(video[..., :512, :512])  # Upper left
                video_stack.append(video[..., :512, 512:])  # Upper right
                video_stack.append(video[..., 512:, :512])  # Lower left
                video_stack.append(video[..., 512:, :512])  # Lower right
            else:
                video_stack.append(video)
        # The big array
        video_stack = torch.concat(video_stack)
        video_stack = self.normalise(video_stack)
        torch.save(video_stack, self.VIDEO_STACK_PATH)

    def _plot(self):
        """Plot a visual representation of every item in the dataset"""

        with tqdm(total=len(self._all_filegroups), desc="Plotting") as pbar:
            for meta in self._all_filegroups:

                # Determine savepath
                savename = meta.fpaths[Channel.HAADF].stem.replace("HAADF ", '') + ".mp4"
                savepath = self.PLOT_DIRECTORY / savename
                if savepath.exists():
                    continue  # Already plotted

                gridspec_kw = dict(left=0, right=1, bottom=0, top=0.952, hspace=0, wspace=0)
                fig, axes = plt.subplots(
                    1, len(meta.fpaths), figsize=(len(meta.fpaths) * 8, 8 * 1.05), gridspec_kw=gridspec_kw,
                    squeeze=False
                )
                for ax in axes.flatten():
                    ax.axis("off")

                video = meta.load_channels([Channel.HAADF])
                video = torch.from_numpy(video).to(torch.float32)
                video = self.normalise(video)
                writer = ani.FFMpegWriter(fps=self.VIDEO_PLOT_FPS)
                with writer.saving(fig, savepath, dpi=210):
                    im = axes[0, 0].imshow(video[0, 0], cmap="inferno")
                    frame_count_text = axes[0, 0].text(
                        video.shape[-1] * 0.95, video.shape[-2] * 0.95, f"{0:03d}", color='w'
                    )
                    writer.grab_frame()
                    for frame_index in range(1, video.shape[0]):
                        im.set_data(video[frame_index, 0])
                        frame_count_text.set_text(f"{frame_index:03d}")
                        writer.grab_frame()
                plt.close(fig)
                pbar.update()


class IridiumImageDataset(Dataset, MultiChannelDataset):
    """Abstraction over the iridium nanoparticle graphene-liquid-cell experimental data. Images only!

    With this dataset, only HAADF and BF were collected and all good data is around the 15MX magnification.
    """

    DATA_DIRECTORY = Path("data/GLC-2_Ir/raw")
    PLOT_DIRECTORY = Path("data/GLC-2_Ir/plots")
    BLACKLIST_FILE = Path("noise2void/datasets/iridium_glc_blacklist.toml")

    def __init__(
        self, image_size: int, channels: list[Channel], example_index: int | None = None, crop_bounds: int=4
    ):
        """
        Parameters
        ----------
        image_size: int
            The size of each image in pixels. Outputs will be randomly cropped
        channels: list[Channel]
            The channels that must be included with each image. This is in-order!
        example_index: int | None
            The index of the reserved validation image to hold back. None specifies a random selection.
        crop_bounds: int
            Each image will be cropped to the nearest 2^`crop_bounds` to work with a unet of `crop-bounds` depth
        """

        print("Initialising dataset")
        self.image_size = image_size
        self._channels = channels
        self.px_scale = 0.013  # Pixel size in nanometres. For this experiment, this is the only valid mag
        self.crop_bounds = crop_bounds
        print("Finding filepaths")
        self._all_filegroups = list(filter(  # Filter out the videos
            lambda meta: meta.frames is None, _find_iridium_filepaths(self.DATA_DIRECTORY)
        ))
        print(f"{len(self._all_filegroups)=}")
        print("Fetching scales and shapes", flush=True)
        for meta in tqdm(self._all_filegroups, desc="Reading scales and shapes"):
            meta.fetch_scale_shape()

        # TODO: Filter by blacklist
        with open(self.BLACKLIST_FILE, 'rb') as f:
            self.blacklist = tomllib.load(f)
        self._pruned_sample_filegroups = list(filter(
            lambda meta: meta.index not in self.blacklist["images"]["blacklist"],
            self._all_filegroups
        ))
        print(f"{len(self._pruned_sample_filegroups)=}")

        self._filtered_sample_filegroups = list(filter(  # Filter out images without the required channels
            lambda meta: meta.has_channels(set(self._channels)), self._pruned_sample_filegroups
        ))
        print(f"{len(self._filtered_sample_filegroups)=}")

        print("Filtering by scale")
        self._to_scale_sample_filegroups = list(filter(
            lambda meta: meta.in_scale_range(self.px_scale, 3), self._filtered_sample_filegroups
        ))
        print(f"{len(self._to_scale_sample_filegroups)=}")

        self._sample_filegroups = self._to_scale_sample_filegroups
        assert len(self._sample_filegroups) > 1
        if example_index is None:
            self._reserved_example_meta = self._sample_filegroups.pop(  # Randomly select an example
                int(torch.randint(0, len(self._sample_filegroups), (1,))[0])
            )
        else:
            self._reserved_example_meta = self._sample_filegroups.pop(example_index)
        self._reserved_example: torch.Tensor | None = None

        self._transform = tforms.Compose([
            tforms.RandomCrop(self.image_size),
            # tforms.RandomHorizontalFlip(),
            # tforms.RandomVerticalFlip(),
        ])

    @property
    def sample_filegroups(self) -> list[MultiChannelMetadata]:
        return self._sample_filegroups

    def load_interpolate(self, meta: MultiChannelMetadata) -> torch.Tensor:
        """Loads the image/video and interpolates to required scale"""

        datum = torch.from_numpy(meta.load_channels(self.channels)).to(torch.float32)
        interp_factor = meta.px_scale / self.px_scale
        interp_shape = int(interp_factor * meta.shape)
        if interp_shape % 2**self.crop_bounds != 0:  # Round this shape to the nearest order of two (for UNet)
            interp_shape += 2**self.crop_bounds - interp_shape % 2**self.crop_bounds
        datum = tforms.functional.resize(datum, interp_shape)  # Interpolate to correct scale
        return datum

    def uninterpolate(self, meta: MultiChannelMetadata, datum: torch.Tensor) -> torch.Tensor:
        """Reverses any interpolation done when loading the image to the required mag/scale"""

        datum = tforms.functional.resize(datum, meta.shape)
        return datum

    @staticmethod
    def normalise(datum: torch.Tensor) -> torch.Tensor:
        """Normalises an image or video"""

        datum -= torch.mean(datum, dim=(-2, -1), keepdim=True)
        datum /= torch.std(datum, dim=(-2, -1), keepdim=True) * 5
        datum = torch.tanh(datum)
        return datum

    @property
    def channels(self) -> list[Channel]:
        """The order in which each channel is stored in the arrays"""

        return self._channels

    @property
    def reserved_example(self) -> torch.Tensor:
        """A cropped and normalised example held back from the training set for validations."""

        if self._reserved_example is None:
            datum = self._load_interpolate_crop(self._reserved_example_meta, 1024)
            self._reserved_example = self.normalise(datum)  # The example for validations
        return self._reserved_example

    @classmethod
    def get_savename(cls, meta: MultiChannelMetadata) -> Path:
        """Generates a savepath, relative to any save directory, for the given datum"""

        savename = meta.fpaths[Channel.HAADF].name.replace("HAADF", '').replace("_", '', 1)
        return Path(savename)

    def __len__(self):
        return len(self.sample_filegroups)

    def __getitem__(self, index: int) -> torch.Tensor:

        image = self._load_interpolate_random_crop(self.sample_filegroups[index])
        image = self.normalise(image)
        return image

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

    def _plot(self):
        """Plot a visual representation of every item in the dataset"""

        with tqdm(total=len(self._all_filegroups), desc="Plotting") as pbar:
            for meta in self._all_filegroups:

                # Determine savepath
                savename = meta.fpaths[Channel.HAADF].stem.replace("HAADF", '').replace("_", '', 1) + ".png"
                savepath = self.PLOT_DIRECTORY / savename
                if savepath.exists():
                    continue  # Already plotted

                gridspec_kw = dict(left=0, right=1, bottom=0, top=0.952, hspace=0, wspace=0)
                fig, axes = plt.subplots(
                    1, len(meta.fpaths), figsize=(len(meta.fpaths) * 8, 8 * 1.05), gridspec_kw=gridspec_kw,
                    squeeze=False
                )
                for ax in axes.flatten():
                    ax.axis("off")
                sbar = ScaleBar(
                    meta.px_scale, "nm", location="lower right", color='w', box_color='k', box_alpha=0.7
                )
                axes[-1, -1].add_artist(sbar)

                image = meta.load_channels(self.channels)
                image = torch.from_numpy(image).to(torch.float32)
                image = self.normalise(image)
                for ax_index, chan in enumerate(meta.fpaths.keys()):
                    axes[0, ax_index].imshow(image[0, ax_index], cmap="inferno", interpolation=None)
                    axes[0, ax_index].set_title(chan.value)
                fig.savefig(savepath, dpi=210)
                plt.close(fig)
                pbar.update()

    def _print_dataset_stats(self):
        """Prints details about the images and videos, including image sizes used."""

        print("\n\n#       Image shapes       #")
        image_shapes: list[int] = list()
        for meta in self._all_filegroups:
            image_shapes.append(meta.shape)
        im_sizes, size_frequencies = np.unique(image_shapes, return_counts=True)
        print(f"Found image sizes: {im_sizes}")
        print(f"Frequencies: {size_frequencies}")

        print("\n\n    Dataset sizes    ")
        print(f"Total data: {len(self._all_filegroups)}")
        print(f"Total images: {len([meta for meta in self._all_filegroups if meta.frames is None])}")
        print(f"Total videos: {len([meta for meta in self._all_filegroups if meta.frames is not None])}")
        print(f"Channel filtered data: {len(self._filtered_sample_filegroups)}")
        print(f"Data satisfying mag constraints: {len(self._to_scale_sample_filegroups)}")
        print(f"Total data used: {len(self.sample_filegroups)}")
        print(f"Effective number of {self.image_size}px images in dataset (after mag interpolation): {self._calculate_effective_length():.1E}")

    def _calculate_effective_length(self) -> float:
        """Calculates the effective length of the dataset, taking into account interpolation, cropping and videos"""

        total = 0.0
        for meta in self.sample_filegroups:
            interp_factor = meta.px_scale / self.px_scale
            new_shape = interp_factor * meta.shape
            image_multiplier = (new_shape / self.image_size) ** 2.
            total += image_multiplier
        return total


def _find_iridium_filepaths(data_directory: Path) -> list[MultiChannelMetadata]:
    """Constructs a list of multi-channel datum metadata from the experimental dataset. Each channel in the
    multi-channel images are saved in a separate file. Each file only contains a single image (there are no videos).
    This links them all together.

    Returns
    -------
    list[MultiChannelMetadata]
        The metadata discovered.
    """

    sample_filegroups_images: dict[int, dict[Channel, Path]] = dict()  # Uniquely identified by an ID number
    # All videos are HAADF only in this dataset, and their unique id is a string rather than int
    sample_filegroups_videos: dict[str, Path] = dict()

    for fpath in data_directory.glob("*.dm4"):

        if "Image_movie" in fpath.name:  # This is a video file
            try:
                _, rest = fpath.stem.split(" ")
                index, _ = rest.strip("Image_movie_").split("_")
            except:
                print(f"Unexpected filename format for file: {fpath}")
                continue
            sample_filegroups_videos[index] = fpath
        elif "Image Stack" in fpath.name:
            continue  # Ignore this file!
        else:
            try:
                chan, mag, index = fpath.stem.split("_")
            except:
                print(f"Unexpected filename format for file: {fpath}")
                continue
            _try_insert_filepath(sample_filegroups_images, Channel[chan], index, fpath)

    sample_filegroups_list = list()
    for index in sample_filegroups_videos:
        meta = MultiChannelMetadata(  # Set frames to -1, this should be determined after parsing the blacklist!
            {Channel.HAADF: sample_filegroups_videos[index]}, [Channel.HAADF], "IrGLC",
            index, -1
        )
        sample_filegroups_list.append(meta)
    for index in sample_filegroups_images:
        meta = MultiChannelMetadata(  # Is an image, not video, so frames is None
            sample_filegroups_images[index], [Channel.HAADF, Channel.BF], "IrGLC", index,
            None
        )
        sample_filegroups_list.append(meta)
    return sample_filegroups_list


def _try_insert_filepath(
    sample_filegroups: dict[int | str, dict[Channel, Path]], chan: Channel, index: int | str, fpath: Path
):
    """Attempts to insert an item into the filegroup-dict, raising a ValueError if a duplicate item is already
    present in the structure."""

    if sample_filegroups.get(index) is None:
        sample_filegroups[index] = dict()
    if sample_filegroups[index].get(chan) is None:  # This should not be populated yet
        sample_filegroups[index][chan] = fpath
    else:
        warn = f"An image with index {index} and channel {chan} already exists:"
        warn += f"\t{fpath}\n\t{sample_filegroups[index][chan]}"
        raise ValueError(warn)


if __name__ == "__main__":
    # For getting info on the samples and plotting etc.

    dset = IridiumVideoDataset(512)
    print(f"There are {len(dset)} frames")
    print(f"{dset.video_stack.shape=}")

    # Plot some examples
    fig, axes = plt.subplots(2, 4)
    for ax in axes.flatten():
        rand_index = int(torch.randint(0, len(dset), (1,))[0])
        frame = dset[rand_index]
        ax.imshow(frame[0], cmap="inferno")
        ax.axis("off")
    plt.show(block=True)
