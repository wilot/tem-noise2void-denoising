"""example_dataset.py

For each experiment a new 'Dataset' needs to be written. This is because in practice the organization of file-saving
changes with every experient and microscope operator, institution etc. This module is purely an illustrative example of
what such a Dataset could look like. This is therefore obviously not meant to actually work, but just be a starting
point for writing a new Dataset.
"""

import tomllib
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib_scalebar.scalebar import ScaleBar

from tqdm import tqdm

from noise2void.datasets.channels import Channel, MultiChannelMetadata, MultiChannelDataset


class ExampleVideoDataset(Dataset, MultiChannelDataset):
    """Abstraction over the experimental videos."""

    DATA_DIRECTORY = Path("data/raw")
    PLOT_DIRECTORY = Path("data/plots")
    VIDEO_STACK_PATH = Path("data/video_stack.npy")
    BLACKLIST_FILE = Path("data/blacklist.toml")
    VIDEO_PLOT_FPS = 10

    def __init__(self, image_size: int, example_index: int | None=None):
        """
        Parameters
        ----------
        image_size: int
            The size of the frames to be outputted. Should be either 512px or 256px. Videos will be tiled to match
        example_index: int | None
            The index of the example frame to hold back for example plotting. If None, this will be chosen randomly
        """

        assert image_size == 512 or image_size == 256
        self.image_size = image_size
        self._channels = Channel.HAADF

        # Find all videos, link channels that may be saved in separate files
        self._all_filegroups = list(filter(  # Only select the videos
            lambda meta: meta.frames is not None, self._find_data_filepaths()
        ))

        if self.VIDEO_STACK_PATH.exists():  # The data stack is already cached
            self._video_stack = np.load(self.VIDEO_STACK_PATH)
            self._len = self._video_stack.shape[0]
            if example_index is None:
                self._reserved_example_index = int(torch.randint(2, self._len, (1,))[0])
            else:
                self._reserved_example_index = example_index
            self._len -= 1
            self._sample_filegroups = self._all_filegroups
            return

        # Load a blacklist and filter the data
        with open(self.BLACKLIST_FILE, 'rb') as f:
            self.blacklist = tomllib.load(f)
        self._valid_filegroups = list(filter(
            lambda meta: self.blacklist["videos"][meta.index] is not False, self._all_filegroups
        ))

        # Check size of every datum, extend dataset length by tiling large images if necessary
        for meta in tqdm(self._valid_filegroups, desc="Reading metadata"):
            meta.fetch_scale_shape()  # The scale metadata is wrong, but the shape is useful!
            if meta.shape != 512 and meta.shape != 1024:  # Catch any weird image sizes
                raise ValueError(f"Video: {meta.fpaths[Channel.HAADF]} has unexpected shape {meta.shape}")
            meta.px_scale = None
            meta.frames = sum(stop - start for start, stop in self.blacklist["videos"][meta.index])  # Count ok frames
        self._sample_filegroups = self._valid_filegroups

        self._video_stack = self._create_compressed_dataset(return_array=True)
        self._len = self._video_stack.shape[0]

        if example_index is None:
            self._reserved_example_index = int(torch.randint(2, self._len, (1,))[0])
        else:
            self._reserved_example_index = example_index
        self._len -= 1  # Hold the reserved example back!

    @property
    def sample_filegroups(self) -> list[MultiChannelMetadata]:
        return self._sample_filegroups

    def load_interpolate(self, meta: MultiChannelMetadata) -> torch.Tensor:
        """Loads the image/video and interpolates to required scale. Does not do any H, W cropping"""

        datum = torch.from_numpy(meta.load_channels(self.channels)).to(torch.float32)
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
        # datum = torch.tanh(datum)
        return datum

    @property
    def channels(self) -> list[Channel]:
        """The order in which each channel is stored in the arrays"""

        return [Channel.HAADF, Channel.BF]

    @property
    def reserved_example(self) -> torch.Tensor:
        """A cropped and normalised example held back from the training set for validations."""

        datum = self.video_stack[self._reserved_example_index][None, ...]  # Shape [1, channels, H, W]
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
            saved_video = torch.load(self.VIDEO_STACK_PATH)
            if saved_video.shape[-1] == self.image_size:
                self._video_stack = saved_video
            else:
                self._video_stack = self._create_compressed_dataset(return_array=True)
        return self._video_stack

    def __len__(self):
        return self._len

    def __getitem__(self, index: int) -> torch.Tensor:

        if index >= self._reserved_example_index:
            index += 1  # Skip the reserved example!
        return self.video_stack[index]

    def _create_compressed_dataset(self, return_array: bool=False) -> torch.Tensor | None:
        """Saves all the videos into a single file for much faster loading."""

        assert self.image_size == 512 or self.image_size == 256

        video_stack = list()
        for meta in self.sample_filegroups:
            video = self.load_interpolate(meta)
            video = self.normalise(video)  # Normalise frame-wise
            if torch.any(torch.isnan(video)):
                raise ValueError(f"WARNING: NaNs found in normalised video from {meta.fpaths[Channel.HAADF]}")
            if video.shape[-1] == 1024:  # They're all 1K or 512px, most are 512px
                video_stack.append(video[..., :512, :512])  # Upper left
                video_stack.append(video[..., :512, 512:])  # Upper right
                video_stack.append(video[..., 512:, :512])  # Lower left
                video_stack.append(video[..., 512:, :512])  # Lower right
            else:
                video_stack.append(video)
        # The big array
        video_stack = torch.concat(video_stack)

        # Now fold the stack to 256px if necessary
        if self.image_size == 256:
            unfold = video_stack.unfold(2, 256, 256).unfold(3, 256, 256)
            unfold = torch.permute(unfold, (0, 2, 3, 1, 4, 5)).contiguous().view((-1, video_stack.shape[1], 256, 256))
            video_stack = unfold
            self._len = video_stack.shape[0] - 1
        np.save(self.VIDEO_STACK_PATH, video_stack.numpy())
        if return_array:
            return video_stack
        else:
            return None

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

    @classmethod
    def _find_data_filepaths(cls) -> list[MultiChannelMetadata]:
        """Constructs a list of multi-channel datum metadata from the experimental dataset. Each channel of the
        multi-channel images have been saved in a separate file by the microscope. Each file only contains a single image
        (there are no videos). This links them all together.

        Returns
        -------
        list[MultiChannelMetadata]
            The metadata discovered.
        """

        sample_filegroups: dict[str, dict[Channel, Path]] = dict()  # Uniquely identified by an ID number
        sample_frames: dict[str, int] = dict()

        for fpath in cls.DATA_DIRECTORY.glob("*.dm3"):

            # Filename format: "HAADF STACK(100)-50.dm3" → channel="HAADF", index="50", frames=100
            channel_str = fpath.stem.split()[0]                     # e.g. "HAADF"
            index = fpath.stem.rsplit("-", 1)[1]                    # e.g. "50"
            frames = int(fpath.stem.split("(")[1].split(")")[0])    # e.g. 100
            channel = Channel(channel_str)
            if index not in sample_filegroups:
                sample_filegroups[index] = {channel: fpath}
                sample_frames[index] = frames
            else:
                if channel not in sample_filegroups[index]:
                    sample_filegroups[index][channel] = fpath
                else:
                    raise ValueError(f"File with index {index} and channel {channel_str} already exist")

        sample_filegroups_list = list()
        for index in sample_filegroups:
            meta = MultiChannelMetadata(
                sample_filegroups[index], [Channel.HAADF, Channel.BF], "AuAcetone", index,
                sample_frames[index]
            )
            sample_filegroups_list.append(meta)
        return sample_filegroups_list
