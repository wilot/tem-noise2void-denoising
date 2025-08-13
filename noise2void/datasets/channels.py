"""channels.py

A type for HAADF, LAADF and BF channels in STEM.
"""

from enum import Enum


class Channel(Enum):
    """STEM imaging channel"""

    BF = "BF"
    LAADF = "LAADF"
    HAADF = "HAADF"