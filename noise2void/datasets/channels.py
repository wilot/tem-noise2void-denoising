"""channels.py

A type for HAADF, LAADF and BF channels in STEM.
"""

from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

import numpy as np
import torch

import hyperspy.api as hs


class Channel(Enum):
    """STEM imaging channel"""

    BF = "BF"
    LAADF = "LAADF"
    HAADF = "HAADF"


class MultiChannelMetadata:
    """Represents the metadata and file-paths of a multi-channel image or video."""

    def __init__(
        self, fpaths: dict[Channel, Path], channel_order: list[Channel], sample: str, index, frames: int | None
    ):
        """
        Parameters
        ----------
        fpaths: dict[Channel, Path]
            A mapping of all the channels and the filepaths containing that channel's data
        channel_order: list[Channel]
            The memory layout of the channels. Any channel in the `fpaths` and not here will be dropped
        sample: str
            A tag identifying the sample
        index
            A unique index of this multi-channel image within the sample
        frames: int | None
            The number of frames, if the files contain each channel of the video, or None if it is an image
        """
        assert len(fpaths) > 0
        self.fpaths: dict[Channel, Path] = {chan: fpaths[chan] for chan in channel_order if chan in fpaths}
        self.sample = sample
        self.index = index
        self.frames = frames  # The number of valid frames, not necessarily acceptable, in the file
        self.px_scale: float | None = None
        self.shape: int | None = None

    def fetch_scale_shape(self):
        an_fpath = self.an_fpath()
        sig = hs.load(str(an_fpath), lazy=True)  # Load any image
        px_scale, px_units = sig.axes_manager[-1].scale, sig.axes_manager[-1].units
        if px_units == "Å":  # Everything in nanometres
            px_scale /= 10.
            px_units = "nm"
        elif px_units == "µm":
            px_scale *= float(1E3)
            px_units = "nm"
        assert px_units == "nm", f"Unexpected units: {px_units}"
        assert sig.data.shape[-2] == sig.data.shape[-1]
        self.px_scale = px_scale
        self.shape = sig.data.shape[-1]

    def has_channels(self, channels: set[Channel]) -> bool:
        return channels.issubset(set(self.fpaths.keys()))

    def is_blacklisted(self, blacklist: dict[str, list]) -> bool:
        """Expects the blacklist to be a dict, indexed by sample-name, with each item containing a list of indices
        that should be excluded."""
        if self.sample in blacklist.keys():
            if self.index in blacklist[self.sample]:
                return True
        return False

    def in_scale_range(self, scale: float, factor: float) -> bool:
        """Scale is pixel-size in nanometres, and the permitted range is from 1/factor to factor times the scale."""
        if scale / factor < self.px_scale < scale * factor:
            return True
        return False

    def an_fpath(self) -> Path:
        return next(iter(self.fpaths.values()))

    def load_channels(self, channels: list[Channel]) -> np.ndarray:
        """Loads the specified channels into an [frames, C, Y, X] array where C are the channels specified, in the
        order provided. Raises an error if the channel is not found. The datatype is not modified"""

        if not all(chan in self.fpaths for chan in channels):
            raise ValueError(f"The channels {channels} are not all found for this image:\n\t{self.fpaths}")
        if self.frames is None:  # Stack images along channel, add a frame dimension (of length one)
            return np.stack(
                [hs.load(self.fpaths[chan]).data for chan in channels],
                axis=0
            )[None, ...]
        else:  # Stack the multi-channel videos
            return np.stack(
                [hs.load(self.fpaths[chan]).data for chan in channels],
                axis=1
            )


class MultiChannelDataset(ABC):
    """A generic MultiChannel dataset"""

    @property
    @abstractmethod
    def sample_filegroups(self) -> list[MultiChannelMetadata]:
        raise NotImplementedError

    @abstractmethod
    def load_interpolate(self, meta: MultiChannelMetadata) -> torch.Tensor:
        """Load and interpolate the image/video, returning a torch `Tensor` of shape [frames, channels, H, W]"""
        raise NotImplementedError

    @abstractmethod
    def uninterpolate(selfself, meta: MultiChannelMetadata, datum: torch.Tensor) -> torch.Tensor:
        """Reverses any interpolation that would be applied to the image to make its magnification conform to the
        dataset"""
        raise NotImplemented

    @staticmethod
    @abstractmethod
    def normalise(datum: torch.Tensor) -> torch.Tensor:
        """Normalise tensors of shape [frames, channels, H, W]"""
        raise NotImplementedError

    @property
    @abstractmethod
    def reserved_example(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def channels(self) -> list[Channel]:
        """The arrangement of channels in the data"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_savename(cls, meta: MultiChannelMetadata) -> Path:
        """Returns an appropriate savename, according to this dataset's conventions, for this datum."""
        raise NotImplementedError