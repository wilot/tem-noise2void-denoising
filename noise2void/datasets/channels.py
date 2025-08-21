"""channels.py

A type for HAADF, LAADF and BF channels in STEM.
"""

from enum import Enum
from pathlib import Path

import hyperspy.api as hs


class Channel(Enum):
    """STEM imaging channel"""

    BF = "BF"
    LAADF = "LAADF"
    HAADF = "HAADF"


class MultiChannelMetadata:

    def __init__(
        self, fpaths: dict[Channel, Path], sample: str, index: int,
    ):
        assert len(fpaths) > 0
        self.fpaths = fpaths
        self.sample = sample
        self.index = index
        self.px_scale = None
        self.shape = None

    def fetch_scale_shape(self):
        an_fpath = self.an_fpath()
        sig = hs.load(str(an_fpath))  # Load any image
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

    def is_blacklisted(self, blacklist: dict[str, list[int]]) -> bool:
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