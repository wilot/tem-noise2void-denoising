"""Dataset

Contains Dataset classes, which provide an abstraction layer between the image files used in training, and the Trainer
that actually conducts training on the model being trained.

Since the organisation of experimental data often varies wildly, this will likely need to be adapted for each
experimental dataset used.

Each dataset should be a subclass of `pytorch.utils.data.Dataset`.

Current adaptation is for the AuPd_GLC data.
"""

import glob, re, math
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms, torchvision.transforms.v2, torchvision.transforms.functional
import numpy as np
import numba as nb

import hyperspy.api as hs

EXP_DATA_DIR = Path('data')
PIXEL_SCALE = 0.1  # Size of pixels in nm after


FilenamePair: tuple[Path, Path] = namedtuple('FilenamePair', 'BF ADF')

class SignalType(Enum):
        """An enum representing an image being either bright field or ADF"""
        ADF = 0
        BF = 1


class ExperimentalRecentDataset(IterableDataset):
    """A dataset representing a set of experimental dual-channel videos from the `SAM` data. 
    
    The data contains a set of videos from four experiments with different solvents. These are `ace_d`, `ace_h`,
    `cyc_d` and `cyc_h`. They can be selected individually by passing those names to the initialiser, or otherwise
    all will be used.
    """

    def __init__(self, selected_experiments: tuple[str] | None=None, preload: bool=False, divisor: int=1):
        """
        Parameters
        ----------
        selected_experiments : tuple[str] | None
            The selected experiments to included. If None, includes everything.
        preload: bool
            Load everything into RAM all at once...
        divisor: int
            Reduce the dataset symmetrically across experiments by a fraction
        """

        data_dir = EXP_DATA_DIR / 'SAM'
        selected_experiment_dirs = filter(lambda dir: dir.is_dir(), data_dir.iterdir()) \
                                   if selected_experiments is None else selected_experiments
        
        assert isinstance(divisor, int) and divisor > 0
        self.divisor = divisor
        
        self.data_filenames: list[FilenamePair] = list()
        for exp in selected_experiment_dirs:
            files = exp.glob('*.hspy')
            self.data_filenames.extend(self.pair_filenames(files))

        self.videos = [self.load_filepair(pair, self.divisor) for pair in self.data_filenames] \
                      if preload else [None for _ in self.data_filenames]
        self.video_frames: list[int] = [self.get_frames(index) for index in range(len(self.data_filenames))]
        self._len = sum(self.video_frames)
    
    @staticmethod
    def pair_filenames(filenames: list[Path]) -> list[FilenamePair]:
        """Searches for BF and HAADF videos of matching filenumber, returning the pairs. Raises errors if filenumber
        duplicates are found or if there aren't pairings where there should be."""

        pattern = r'(?:BF|HAADF)_(\d+)\.hspy'  # Extracts filename's filenumber

        bf_filenames: dict[int, str] = dict()
        adf_filenames: dict[int, str] = dict()
        filename_pairs: list[FilenamePair] = list()

        for filename in filenames:
            # Get filenumber
            match = re.search(pattern, filename.name)
            if not match: raise ValueError('Could not identify filenumber from the filename ' + str(filename))
            filenumber = int(match.group(1))
            # Get channel
            if 'BF' in filename.name:
                filenames_dict = bf_filenames
            elif 'HAADF' in filename.name:
                filenames_dict = adf_filenames
            else:
                raise ValueError('Could not determine channel type of ' + str(filename))
            # Check for duplication & record
            if filenumber in filenames_dict:  # Seen the same filenumber twice
                raise ValueError('Duplicate filenumber detected ' + str(filename))
            filenames_dict[filenumber] = filename

        if not bf_filenames.keys() == adf_filenames.keys():
            raise ValueError('No filename pair found for filenumber ' + str(number))
        for number in bf_filenames.keys():
            bf_filename, adf_filename = bf_filenames[number], adf_filenames[number]
            filename_pairs.append(FilenamePair(bf_filename, adf_filename))

        return filename_pairs
    
    def get_frames(self, video_id: int) -> int:
        """Finds the number of frames in the video"""

        if self.videos[video_id] is not None: return len(self.videos[video_id])
        vid_file = self.data_filenames[video_id].BF
        vid = hs.load(str(vid_file))
        if len(vid.data.shape) != 3:
            raise ValueError('Unexpected video shape for ' + vid_file)
        return vid.data[::self.divisor].shape[0]
    
    @staticmethod
    def load_filepair(filepair: FilenamePair, divisor: int) -> torch.Tensor:

        bf_vid = hs.load(filepair.BF).data[::divisor].astype(np.float32)
        adf_vid = hs.load(filepair.ADF).data[::divisor].astype(np.float32)

        bf_vid -= bf_vid.mean()
        bf_vid /= bf_vid.std() * 4
        adf_vid -= adf_vid.mean()
        adf_vid /= adf_vid.std() * 4

        # Datum is of the shape (frame, channel, Y, X)
        datum = torch.from_numpy(np.stack((adf_vid, bf_vid), axis=1))
        return datum
    
    def __getitem__(self, index: int) -> torch.Tensor:

        # Find the appropriate file, and the frame within that file
        current_start = 0
        for filename_index, frames in enumerate(self.video_frames):
            next_start = frames + current_start
            if current_start <= index and index < next_start:
                break
            current_start = next_start
        else: raise IndexError
        frame_in_stack = index - current_start

        if self.videos[filename_index] is None:
            datum = self.load_pair(self.data_filenames[filename_index])
            self.videos[filename_index] = datum
        return self.videos[filename_index][frame_in_stack]
    
    def __len__(self):
        return self._len
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single threaded dataloader
            start_index, end_index = 0, len(self)
        else:
            per_worker = int(math.ceil((len(self)) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start_index = self.start + worker_id * per_worker
            end_index = min(start_index + per_worker, len(self))
        return (self[index] for index in range(start_index, end_index))
    

class MonochannelDataset(ExperimentalRecentDataset):
    """Identical to ExperimentalRecentDataset, except it outputs only a single channel for each frame rather than both.
    This can be chosen with the initialiser."""

    def __init__(self, channel: Literal[0] | Literal[1], selected_experiments: tuple[str] | None=None,
                 preload: bool=False, divisor: int=1):
        """
        Parameters
        ----------
        channel: Literal[0] | Literal[1]
            Which channel to use from the original dataset
        selected_experiments : tuple[str] | None
            The selected experiments to included. If None, includes everything.
        preload: bool
            Load everything into RAM all at once...
        divisor: int
            Reduce the dataset symmetrically across experiments by a fraction
        """

        self.channel = channel
        super().__init__(selected_experiments, preload, divisor)

    def load_filepair(self, filepair: FilenamePair, divisor: int) -> torch.Tensor:
        """The channel that gets loaded depends on which channel was selected in the initialiser"""

        if self.channel == 0:
            vid = hs.load(filepair.ADF).data[::divisor].astype(np.float32)
        else:
            vid = hs.load(filepair.BF).data[::divisor].astype(np.float32)
        vid -= vid.mean()
        vid /= vid.std() * 4
        return torch.from_numpy(vid[:, None, ...])


class ExperimentalStackDataset(Dataset):
    """A dataset representing a set of experimental videos from the AuPd_GLC/stacks data.
    
    Implements a Pytorch `Dataset` for the dataset. Exposes an iterator.

    Attributes
    ----------
    SELECTED_STACKS : tuple[int]
        Used to select which data-files should be included in this dataset. Should be a const!
    
    Methods
    -------
    get_frames(pair: FilenamePair) : int
        Reads the number of frames held within that file
    """

    SELECTED_STACKS = 15, 16, 17#, 18, 20, 21, 22

    def __init__(self):
        stack_dir = EXP_DATA_DIR / 'AuPd_GLC' / 'stacks'
        self.data_filenames = [
            FilenamePair(stack_dir / f'stack_{stack_num}_bf.dm3', stack_dir / f'stack_{stack_num}_haadf.dm3')
            for stack_num in self.SELECTED_STACKS
        ]
        # self.data_filenames = [FilenamePair('data/BF STACK(100)-57.dm3', 'data/HAADF STACK(100)-57.dm3')]
        self.videos: list[torch.Tensor | None] = [None for _ in self.data_filenames]  # Load lazily
        self.stack_frames = [self.get_frames(file_pair) for file_pair in self.data_filenames]  # Frames per stack
        self._len = sum(self.stack_frames)

    @staticmethod
    def get_frames(pair: FilenamePair) -> int:
        """Reads the number of frames in the data stack."""

        vid = hs.load(pair.BF)
        return vid.data.shape[0]
    
    def __getitem__(self, index) -> torch.Tensor:
        """Loads from file when required"""

        assert index < len(self)
        
        # Find the appropriate file, and the frame within that file
        current_start = 0
        for filename_index, frames in enumerate(self.stack_frames):
            next_start = frames + current_start
            if current_start <= index and index < next_start:
                break
            current_start = next_start
        else: raise IndexError

        frame_in_stack = index - current_start
        
        if self.videos[filename_index] is None:  # Lazy load the files

            bf_vid = hs.load(self.data_filenames[filename_index].BF).data.astype(np.float32)
            adf_vid = hs.load(self.data_filenames[filename_index].ADF).data.astype(np.float32)

            # Standardise
            # TODO: This should be its own function. Don't be lazy!
            bf_vid -= bf_vid.mean()
            bf_vid /= bf_vid.std() * 4
            adf_vid -= adf_vid.mean()
            adf_vid /= adf_vid.std() * 4

            # Datum is of the shape (frame, channel, Y, X)
            datum = torch.tensor(np.stack((adf_vid, bf_vid), axis=1).astype(np.float32))
            self.videos[filename_index] = datum

        return self.videos[filename_index][frame_in_stack]
    
    def __len__(self):
        return self._len
    
    def __iter__(self):
        self._current_index = -1  # Not yet started
        return self
    
    def __next__(self):
        self._current_index += 1
        if self._current_index >= len(self):
            raise StopIteration
        return self[self._current_index]



class ExperimentalDataset:
    """A dataset representing a set of experimental images from the AuPd_GLC/stacks data.

    The data glob `EXP_GLOB` should return all BF/HAADF .dm3 files. They will then be linked together using the number
    in their filename.
    
    Implements a Pytorch `Dataset` for the dataset. Exposes an iterator.

    Attributes
    ----------
    EXP_GLOB : pathlib.Path (const)
        A glob for finding the images.
    SELECTED_IMAGES : tuple[int] (const)
        The selected image numbers to be used
    
    Methods
    -------
    get_signal_type(filename: str) : SignalType
        Checks the filename to see if it is a bright-field channel or ADF channel
    get_filenumber(filename: str) : int
        Extracts the image's ID/number from its filename
    get_fname_pairs(data_files: list[pathlib.Path]) : list[FilenamePair]
        Sorts through all the filenames and pairs them up by matching number/ID.
    scale_in_range(pair: FilenamePair) : bool
        Checks whether the image's magnification is within a permitted range.
    get_scales() : list[float]
        Gets the size of pixels, in nanometres, for each image in the dataset.
    """

    EXP_GLOB = EXP_DATA_DIR / 'AuPd_GLC' / '*.dm3'
    SELECTED_IMAGES = 64, 60, 67, 57, 58

    def __init__(self) -> None:        
        data_files = [Path(fname) for fname in glob.glob(str(self.EXP_GLOB))]  # Each contains only a single frame
        self.data_pairs: list[FilenamePair] = self.get_fname_pairs(data_files)
        self.crop = torchvision.transforms.RandomCrop(size=(512, 512))

    @staticmethod
    def get_signal_type(filename: str) -> SignalType:
        """Determines the signal type of the filename. Should contain the name of the file only, excluding any
        preceding path"""

        if 'HAADF' in filename:
            return SignalType.ADF
        elif 'BF' in filename:
            return SignalType.BF
        else:
            raise ValueError(f"This filename's signal type is not understood {filename}.")
    
    @staticmethod
    def get_filenumber(filename: str) -> int:
        """Extracts the numeric ID of the file from the filename. The filename should not include any preceding
        path."""

        fnum = re.findall(r'\d+(?!\d|$)', filename)[0]
        fnum = int(fnum)
        return fnum

    def get_fname_pairs(self, data_files: list[Path]) -> list[FilenamePair]:

        bf_fnames = dict()
        adf_fnames = dict()
        for data_file in data_files:
            data_file_number = self.get_filenumber(data_file.name)
            if data_file_number not in self.SELECTED_IMAGES: continue
            if self.get_signal_type(data_file.name) is SignalType.BF:
                sig_fnames = bf_fnames
            else:
                sig_fnames = adf_fnames
            if data_file_number in sig_fnames.keys():  # Found this twice, but should be unique
                raise ValueError(f"Found {data_file_number} filename number twice! {data_file}")
            sig_fnames[data_file_number] = data_file
        
        if not set(bf_fnames.keys()) == set(adf_fnames.keys()):
            # There is a mismatch between the two keys
            raise ValueError("Couldn't find matching keys for each filename")

        pairs: list[FilenamePair] = [
            FilenamePair(bf_fnames[fnum], adf_fnames[fnum]) for fnum in bf_fnames.keys()
        ]
        pairs = [pair for pair in pairs if self.scale_in_range(pair)]
        
        return pairs
    
    @staticmethod
    def scale_in_range(pair: FilenamePair) -> bool:
        """Checks whether the pixel size of the image-pair is suitable"""

        BF_signal = hs.load(pair.BF)
        image_px_size = BF_signal.original_metadata['ImageList']['TagGroup0']['ImageData']['Calibrations'] \
            ['Dimension']['TagGroup0']['Scale']  # In nanometers
        image_zoom = image_px_size / PIXEL_SCALE  # Factor the image needs to be magnified by to standard scale
        # return 0.25 < image_zoom  and image_zoom < 6
        return True
    
    def get_scales(self) -> list[float]:
        """Loads each datum and extracts the scale (pixel size) in nanometers."""

        pixel_sizes: list[float] = list()
        for index in range(len(self)):
            file_num = index
            BF_signal = hs.load(self.data_pairs[file_num].BF)
            # ADF_signal = hs.load(self.data_pairs[file_num].ADF)
            image_px_size = BF_signal.original_metadata['ImageList']['TagGroup0']['ImageData']['Calibrations'] \
                ['Dimension']['TagGroup0']['Scale']  # In nanometers
            pixel_sizes.append(image_px_size)
        return pixel_sizes
    
    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, index: int) -> torch.Tensor:

        file_num = index
        # print(f'{self.data_pairs[file_num].BF}')
        BF_signal = hs.load(self.data_pairs[file_num].BF)
        ADF_signal = hs.load(self.data_pairs[file_num].ADF)
        image_px_size = BF_signal.original_metadata['ImageList']['TagGroup0']['ImageData']['Calibrations'] \
            ['Dimension']['TagGroup0']['Scale']  # In nanometers
        image_zoom = image_px_size / PIXEL_SCALE  # Factor the image needs to be magnified by to standard scale

        image = torch.tensor(np.stack((ADF_signal.data, BF_signal.data)).astype(np.float32))[None, ...]

        # Interpolate and crop/pad to standardise scale
        image = torch.nn.functional.interpolate(image, scale_factor=(image_zoom, image_zoom), mode='bilinear')
        if image.shape[-1] < 512:
            # Must pad the image. Cannot pad greater than image-size, so tile as much as possible and pad the remainder
            total_padding = 512 - image.shape[-2], 512 - image.shape[-1]  # single sided total padding
            repeats = total_padding[0] // image.shape[-2] + 1, total_padding[1] // image.shape[-1] + 1  # The tiling
            padding = total_padding[0] % image.shape[-2], total_padding[1] % image.shape[-1]  # Padding after tiling
            split_padding = [  # Share the necessary padding evenly between axis start and end
                int(math.floor(padding[0] / 2.)),
                int(math.ceil(padding[0] / 2.)),  # If an odd padding required, have the axis-end padding be larger
                int(math.floor(padding[1] / 2.)),
                int(math.ceil(padding[1] / 2.))
            ]
            split_padding = [split_padding[i] for i in (2, 0, 3, 1)]  # Rearrange for torchvision
            image = torch.tile(image, repeats)
            image = torchvision.transforms.functional.pad(image, split_padding, padding_mode='reflect')
            if any(shape != 512 for shape in image.shape[-2:]):
                raise ValueError
        elif image.shape[-1] > 512:
            image = self.crop.forward(image)

        image = torchvision.transforms.functional.normalize(
            image,
            torch.mean(image, dim=(-2, -1)).squeeze(),
            torch.std(image, dim=(-2, -1)).squeeze() * 5.  # 0->1 range is 5 std.devs
        )

        return image[0]

    def __iter__(self):
        self.current_index = -1  # Not started yet
        return self

    def __next__(self):
        self.current_index += 1
        if self.current_index >= len(self):
            raise StopIteration
        return self[self.current_index]


# TESTING
if __name__ == '__main__':

    dset = ExperimentalStackDataset()
    dset.get_frames(0)
