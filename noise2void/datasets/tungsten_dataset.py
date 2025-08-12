"""tungsten_dataset.py

Contains the PyTorch Dataset for the Tunsten Disulphide experimental data.
"""

import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms.v2
import numpy as np

import hyperspy.api as hs


class TungstenDataset(IterableDataset):
