# Short Usage Notes - Quick guide to the data set up, training and denoising

**1. Place your data**
- Copy raw data files into `data/raw/`
- Delete `data/video_stack.npy` if it exists from a previous run, so it gets rebuilt
```bash
rm data/video_stack.npy
```

**2. Generate and edit the blacklist**
- Open `noise2void/datasets/generate_blacklist.py` and set `FILE_EXTENSION`, `FILENAME_PATTERN`, and `PRIMARY_CHANNEL` to match your filenames
- Run the generator to create a starter blacklist with all frames included:
  ```bash
  python -m noise2void.datasets.generate_blacklist
  ```
- Edit the generated `data/blacklist.toml` to trim bad frames:
  - Restrict to valid ranges: `"50" = [[0, 100]]`  (start inclusive, stop exclusive)
  - Multiple ranges to skip a section: `"50" = [[0, 60], [80, 100]]`
  - Exclude entirely: `"50" = false`
```bash
conda activate Noise2Void
python -m noise2void.datasets.generate_blacklist
```

**3. Edit the config** — `noise2void/config.yaml`
- `run_directory` — set the experiment name, e.g. `"runs/MyExperiment/run_{index}"` (index auto-increments). 
- `image_size` - set the image size
- `channels` - set the channels, e.g. `["HAADF", "BF"]` or `["HAADF"]`
- `dataset.type` — set to the name of your dataset class as registered in `generators.py`
- `dataset.example_index` — frame index to use as the validation preview image
- `trainer.epochs` / `trainer.batch_size` — adjust for your hardware if needed

> **Changing the number of channels?** The `channels` list in `config.yaml` controls the model's
> input/output size, but the dataset class has its own hardcoded channel list that must match.
> Simply changing the config is **not** sufficient — the following must also be updated in your
> dataset class (e.g. `noise2void/datasets/dataset.py`):
>
> - **`channels` property** — must return the same channel list as the config:
>   ```python
>   @property
>   def channels(self) -> list[Channel]:
>       return [Channel.HAADF]   # for a single HAADF-only dataset
>   ```
> - **`_find_data_filepaths`** — the `channel_order` argument passed to `MultiChannelMetadata`
>   must include only the channels present in your data:
>   ```python
>   meta = MultiChannelMetadata(fpaths, [Channel.HAADF], ...)   # not [Channel.HAADF, Channel.BF]
>   ```
> - **`get_savename`** — if your primary/only channel is not `HAADF`, update the hardcoded
>   `Channel.HAADF` key used to derive the output filename.

**4. Train**
```bash
# conda activate Noise2Void
python -m noise2void.train_script
```

**5. Denoise**
```bash
python -m noise2void.predict_script runs/MyExperiment/run_000 outputs/MyExperiment
```
Replace `MyExperiment` and `run_NNN` with your experiment name and the run folder just created.
The output directory (`outputs/MyExperiment/run_000`) must not already exist — delete it first if re-running:
```bash
rm -rf outputs/MyExperiment/run_000
```

**6. Outputs** — saved to `outputs/MyExperiment/run_NNN/`
- `*_DENOISED.hspy` — denoised signal in HyperSpy format
- `*.png` — side-by-side noisy vs. denoised comparison image
- `*.mp4` — side-by-side comparison video
- `* HISTOGRAM.png` — intensity histogram comparing input and output











# Noise2Void Denoising - Full Details

A package for denoising STEM experiments using the Noise2Void technique. Models can be trained with several
backbone architectures (UNet, ResUNet, SwinUNet, SwinTransformer) and training is distributed across all available
GPUs using PyTorch `DistributedDataParallel`.

As new experimental datasets are produced, a new module should be added to the `datasets/` directory to abstract
over the file naming and directory structure of that experiment. The `example_dataset.py` module is a
heavily-commented template for doing this.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Project Layout](#2-project-layout)
3. [Adapting the Code for a New Dataset](#3-adapting-the-code-for-a-new-dataset)
   - [3.1 Create a Dataset class](#31-create-a-dataset-class)
   - [3.2 Generate and edit the blacklist file](#32-generate-and-edit-the-blacklist-file)
   - [3.3 Register the dataset](#33-register-the-dataset)
4. [Configuring a Training Run](#4-configuring-a-training-run)
5. [Training](#5-training)
6. [Monitoring Training](#6-monitoring-training)
7. [Running Inference / Denoising](#7-running-inference--denoising)
8. [Understanding the Outputs](#8-understanding-the-outputs)
9. [Configuration Reference](#9-configuration-reference)

---

## 1. Environment Setup

All dependencies are managed via Conda. Create and activate the environment from the provided file:

```bash
conda env create -f conda_environment.yml
conda activate Noise2Void
```

Key dependencies installed by this environment:

| Package | Purpose |
|---|---|
| `pytorch` / `torchvision` | Neural network training and inference |
| `hyperspy` | Loading `.dm4` / `.hspy` microscopy files |
| `hydra-core` | Config file management for the train script |
| `tensorboard` | Training loss visualisation |
| `numba` | CPU-accelerated Noise2Void pixel masking |
| `matplotlib-scalebar` | Scale bars on output figures |
| `tqdm` | Progress bars |

---

## 2. Project Layout

```
tem-noise2void-denoising/
├── conda_environment.yml       # Conda environment specification
├── data/
│   ├── raw/                    # ← Raw microscopy files go here
│   ├── blacklist.toml          # Frame blacklist (TOML format, generated by generate_blacklist.py)
│   └── video_stack.npy         # Auto-generated cache of all frames (created on first run)
├── noise2void/
│   ├── config.yaml             # ← Main configuration file you edit before each run
│   ├── train_script.py         # Entry point: train a model
│   ├── predict_script.py       # Entry point: denoise data with a trained model
│   ├── trainer.py              # Noise2Void training loop (rarely needs editing)
│   ├── datasets/
│   │   ├── channels.py             # Channel enum (HAADF, LAADF, BF) and abstract base class
│   │   ├── example_dataset.py      # ← Template for writing a new Dataset class
│   │   ├── generate_blacklist.py   # Script to auto-generate data/blacklist.toml
│   │   ├── generators.py           # Maps config strings to Dataset constructors
│   │   └── iridium_glc_dataset.py  # Example of a real, complete dataset implementation
│   └── models/
│       ├── generators.py       # Maps config strings to model constructors
│       ├── unet.py             # U-Net architecture
│       ├── resunet.py          # Residual U-Net architecture
│       └── swinunet.py         # Swin Transformer U-Net architectures
├── runs/                       # Training run directories (auto-created during training)
│   └── <experiment>/
│       └── run_NNN/
│           ├── model_state.pt      # Saved model weights
│           ├── traced_model.pt     # TorchScript model (used for inference)
│           ├── noise2void/         # Snapshot of the code used for this run
│           └── training_log/       # TensorBoard logs and epoch plots
└── outputs/                    # Denoised results (created during inference)
    └── <experiment>/
        └── run_NNN/
```

> **Runs are self-contained.** When training starts, the entire `noise2void/` source directory is
> copied into the run folder. This means the config and code that produced a model are always
> preserved alongside the weights.

---

## 3. Adapting the Code for a New Dataset

The only code you need to write is a single Dataset class. The template in
`noise2void/datasets/example_dataset.py` shows every method that must be implemented.

### 3.1 Create a Dataset class

Create a new file, e.g. `noise2void/datasets/my_dataset.py`. Your class must inherit from both
`torch.utils.data.Dataset` and `noise2void.datasets.channels.MultiChannelDataset`.

The key things the class is responsible for:

**Finding files** — implement `_find_data_filepaths()` (or equivalent) to walk your `DATA_DIRECTORY`
and return a list of `MultiChannelMetadata` objects. Each `MultiChannelMetadata` maps channel
types (`Channel.HAADF`, `Channel.LAADF`, `Channel.BF`) to the file paths that hold them.

```python
# Example of building a MultiChannelMetadata for a single-channel HAADF video
from noise2void.datasets.channels import Channel, MultiChannelMetadata

meta = MultiChannelMetadata(
    fpaths={Channel.HAADF: Path("data/raw/my_video.dm4")},
    channel_order=[Channel.HAADF],
    sample="my_experiment",
    index="my_video",   # A unique identifier used as the blacklist key
    frames=1000,        # Number of frames; set to None for a single image
)
```

**Loading data** — implement `load_interpolate(meta)`. Load the file(s) referenced in `meta` and
return a `torch.Tensor` of shape `[frames, channels, H, W]` (or `[1, channels, H, W]` for a
single image). Apply any spatial interpolation here if different files have different pixel sizes.

```python
def load_interpolate(self, meta: MultiChannelMetadata) -> torch.Tensor:
    datum = torch.from_numpy(meta.load_channels(self.channels)).to(torch.float32)
    # Optionally: interpolate to a common pixel size
    return datum
```

**Normalising data** — implement `normalise(datum)` as a static method. The default used across
the existing datasets is mean/std normalisation per frame:

```python
@staticmethod
def normalise(datum: torch.Tensor) -> torch.Tensor:
    datum -= torch.mean(datum, dim=(-2, -1), keepdim=True)
    datum /= torch.std(datum, dim=(-2, -1), keepdim=True) * 5
    return datum
```

**Dataset indexing** — `torch.utils.data.Dataset` requires `__len__` and `__getitem__`. These
should operate on the cached frame stack. One frame index is reserved for validation and must be
skipped in `__getitem__`:

```python
def __len__(self) -> int:
    return self._len  # total frames minus the reserved example

def __getitem__(self, index: int) -> torch.Tensor:
    if index >= self._reserved_example_index:
        index += 1  # skip over the reserved example frame
    return self.video_stack[index]  # shape [channels, H, W]
```

**Blacklist file path** — declare `BLACKLIST_FILE` as a class attribute pointing to
`data/blacklist.toml`. This is read in `__init__` to filter and trim frames:

```python
BLACKLIST_FILE = Path("data/blacklist.toml")
```

**Caching the dataset** — for large video datasets, implement `_create_compressed_dataset()` to
pre-load, normalise, and tile all frames into a single tensor and save it to `VIDEO_STACK_PATH`
(e.g. `data/video_stack.npy`). The `__init__` method should check whether this cache
file exists and skip re-building it if so. This is important because loading many raw files
on every training epoch would be very slow.

**Image tiling** — if your raw images are larger than `image_size` (e.g. 1024 px), split them
into non-overlapping tiles inside `_create_compressed_dataset()`. For 1024 → 512 px:

```python
if video.shape[-1] == 1024:
    video_stack.append(video[..., :512, :512])  # top-left
    video_stack.append(video[..., :512, 512:])  # top-right
    video_stack.append(video[..., 512:, :512])  # bottom-left
    video_stack.append(video[..., 512:, 512:])  # bottom-right
```

**Providing a validation example** — implement the `reserved_example` property. This returns a
single `[1, channels, H, W]` tensor that is held back from training and used to produce epoch
visualisation plots. Set `example_index` in `config.yaml` to always use the same frame.

**Declaring channels** — implement the `channels` property to return the list of `Channel`
values your dataset loads, in the order they are stacked in the array. **This must match the
`channels` list in `config.yaml` exactly** — the model's input/output size is derived from
`len(config.channels)`, so a mismatch will cause a shape error at the start of training.

```python
@property
def channels(self) -> list[Channel]:
    return [Channel.HAADF, Channel.BF]  # or [Channel.HAADF] for a single-channel dataset
```

Also ensure the `channel_order` passed to every `MultiChannelMetadata` in `_find_data_filepaths`
lists the same channels in the same order.

**Saving filenames** — implement `get_savename(meta)` to return a `Path` (relative) that the
prediction script will use when saving the denoised output. The default implementation in
`example_dataset.py` uses `Channel.HAADF` as the key to look up the filename — if your primary
channel is not HAADF, update this key accordingly.

**Implement `uninterpolate`** — if you rescaled data in `load_interpolate`, reverse that here so
the denoised output matches the original pixel size. If no interpolation was applied, simply
`return datum`.

---

### 3.2 Generate and edit the blacklist file

The blacklist is a TOML file that lets you exclude entire files or specific frame ranges within a
video. This is important for removing frames that contain beam damage, sample drift, or
contamination artefacts.

**Generate a starter blacklist** using `noise2void/datasets/generate_blacklist.py`. Before
running it, open the file and set the three variables at the top to match your filenames:

```python
FILE_EXTENSION   = ".dm3"    # file extension of your raw data
FILENAME_PATTERN = r"^(?P<channel>\w+) STACK\((?P<frames>\d+)\)-(?P<id>\d+)$"
                             # regex with named groups: channel, id, and optionally frames
PRIMARY_CHANNEL  = "HAADF"  # only process files for this channel; None = process all
```

The `frames` group in the regex is optional — if it matches a number in the filename that group
is used directly and the file is not opened. If the group is absent the file is loaded lazily
to count frames.

Then run from the project root:

```bash
python -m noise2void.datasets.generate_blacklist
# or with custom paths:
python -m noise2void.datasets.generate_blacklist \
    --data_dir data/raw \
    --output data/blacklist.toml
```

This writes `data/blacklist.toml` with every discovered video set to `[[0, N]]` (all frames included).
Edit the file afterwards to trim bad frames:

```toml
[videos]
# Each key is the `index` string parsed from the filename by FILENAME_PATTERN.
# The value is either:
#   false              → exclude the entire video
#   [[start, stop]]    → one or more valid frame ranges (Python-style, exclusive stop)
#   [[s1, e1], [s2, e2]]  → multiple valid ranges within the same video

"video_001" = [[5, 980]]          # frames 5–979 are valid
"video_002" = [[0, 300], [320, 800]]  # skip frames 300–319 (e.g. a jitter event)
"video_003" = false               # exclude entirely
```

Load the blacklist inside your Dataset `__init__` using `tomllib` (built into Python 3.11+,
no extra install needed):

```python
import tomllib

with open(self.BLACKLIST_FILE, 'rb') as f:  # must open in binary mode 'rb'
    self.blacklist = tomllib.load(f)
```

Then filter and count valid frames:

```python
self._valid_filegroups = [m for m in all_filegroups
                          if self.blacklist["videos"][m.index] is not False]
for meta in self._valid_filegroups:
    meta.frames = sum(stop - start
                      for start, stop in self.blacklist["videos"][meta.index])
```

When loading a video in `load_interpolate`, concatenate only the valid frame ranges:

```python
datum = torch.concat(
    [datum[start:stop] for start, stop in self.blacklist["videos"][meta.index]]
)
```

---

### 3.3 Register the dataset

Open `noise2void/datasets/generators.py` and add a generator function and a mapping entry:

```python
from noise2void.datasets.my_dataset import MyDataset

def _generate_my_dataset_from_config(config, predict: bool) -> MyDataset:
    return MyDataset(
        image_size=config.image_size,
        example_index=config.dataset.example_index,
    )

dataset_generators = {
    "IridiumGLCVideo": _generate_iridium_video_from_config,  # existing entry
    "MyDataset": _generate_my_dataset_from_config,           # ← add this
}
```

The string key (`"MyDataset"`) is what you will put in `config.yaml`.

---

## 4. Configuring a Training Run

All training parameters live in `noise2void/config.yaml`. Edit this file before each run.

```yaml
# Unique name for this group of runs. "{index}" auto-increments (run_000, run_001, …)
run_directory: "runs/MyExperiment/run_{index}"

# Spatial size of each training patch (pixels). Must match what your Dataset produces.
image_size: 512

# Which detector channels to use. Must match what your Dataset's `channels` property returns
# AND the channel_order in every MultiChannelMetadata built by _find_data_filepaths.
# See section 3.1 for details — changing this value alone is not sufficient.
channels: ["HAADF"]

model:
  name: "UNet"              # Options: UNet | ResUNet | SwinUNet | SwinTransformer
  depth: 5                  # Number of encoder/decoder levels
  first_layer_channels: 32  # Feature maps in the first layer (doubles at each level)
  initial_kernel_size: 5    # Kernel size for the first convolutional layer
  state_savename: "model_state.pt"    # Filename for the raw weight checkpoint
  traced_savename: "traced_model.pt"  # Filename for the TorchScript model

trainer:
  learning_rate: 1e-5   # Adam optimiser learning rate
  batch_size: 6         # Images per GPU per step (reduce if you run out of VRAM)
  epochs: 24            # Total training epochs
  spacing: 23           # Noise2Void grid spacing (pixels). Should be ≥ receptive field radius
  test_fraction: 0.25   # Fraction of frames held back for test-loss evaluation

dataset:
  type: "MyDataset"     # ← Must match the key you added to generators.py
  px_scale: 0.007       # Pixel size in nanometres, used only for scale bars in plots
  video_filter: ["file1.dm3", "file2.dm3"]  # Restrict to these files; set to `none` to use all
  example_index: 800    # Frame index to hold back as the validation/preview image

predicter:
  video_fps: 10         # Frame rate of saved .mp4 output
  device: "cuda:0"      # PyTorch device string for inference
  batch_size: 8         # Frames per batch during inference
```

> **`video_filter`** is useful when you have many videos but only want to train on a subset.
> Set it to `none` (unquoted, no list) or remove the key entirely to use all available data.

> **`example_index`** should be set to a frame that is representative of your data. It is
> displayed as a side-by-side noisy/denoised comparison plot at the end of each epoch.

---

## 5. Training

Make sure you are in the project root and the conda environment is active, then run:

```bash
conda activate Noise2Void
cd /path/to/tem-noise2void-denoising
python -m noise2void.train_script
```

**What happens during training:**

1. A new numbered run directory is created under the path specified by `run_directory`
   (e.g. `runs/MyExperiment/run_000/`).
2. The entire `noise2void/` source directory is copied into the run folder so the exact code
   version is preserved alongside the trained weights.
3. The dataset is initialised. If `data/video_stack.npy` does not yet exist, all raw files are loaded,
   normalised, tiled to `image_size`, and saved to that cache file. **This first-run cache
   creation can take several minutes** depending on the number and size of your files.
   **Delete `data/video_stack.npy` before retraining if you have changed the `channels` config,
   added or removed data files, or edited the blacklist** — otherwise the stale cache will be used.
4. Training runs across all available GPUs using `DistributedDataParallel`. Progress and loss
   values are written to TensorBoard logs and a per-epoch side-by-side validation image is saved.
5. On completion, two model files are written to the run directory:
   - `model_state.pt` — raw PyTorch state dict (useful for fine-tuning or inspection).
   - `traced_model.pt` — a TorchScript / traced module (used by the prediction script).

> **Single-GPU or CPU machines:** training will still work. `DistributedDataParallel` with a
> single GPU degrades gracefully; with no GPU the process will be slow but functional.

---

## 6. Monitoring Training

TensorBoard logs are written to `<run_directory>/training_log/`. Launch TensorBoard to watch
training and test loss in real time:

```bash
tensorboard --logdir runs/
```

Then open `http://localhost:6006` in a browser.

Per-epoch validation images (noisy input vs. denoised output for the reserved example frame) are
saved to `<run_directory>/training_log/epoch_plots/epoch_NNN.png`. These are useful for a quick
sanity-check without needing TensorBoard.

---

## 7. Running Inference / Denoising

Once training is complete, use `predict_script.py` to denoise the full dataset:

```bash
python -m noise2void.predict_script <run_path> <output_path> [--dataset_name <name>]
```

| Argument | Description |
|---|---|
| `run_path` | Path to the completed run directory, e.g. `runs/MyExperiment/run_000` |
| `output_path` | Root directory where results will be saved, e.g. `outputs/MyExperiment` |
| `--dataset_name` | *(Optional)* Name of a **different** dataset to apply the model to. If omitted, the dataset from the training config is used. |

**Example — denoise the training data:**
```bash
python -m noise2void.predict_script runs/MyExperiment/run_000 outputs/MyExperiment
```

**Example — apply the model to a different dataset:**
```bash
python -m noise2void.predict_script runs/MyExperiment/run_000 outputs/MyExperiment \
    --dataset_name OtherDataset
```

**What happens during inference:**

1. The `config.yaml` and `traced_model.pt` from inside the run folder are loaded.
2. The dataset is re-initialised in prediction mode (the `video_filter` is ignored so all files
   are denoised, not just the training subset).
3. Each file in the dataset is loaded, normalised, and passed through the model in batches
   (controlled by `predicter.batch_size` in the config).
4. The denoised output is written to `<output_path>/<run_name>/`.

> **Output directory must not exist:** the script will refuse to run if
> `<output_path>/<run_name>/` already exists, to prevent accidentally overwriting results.
> Delete it first if re-running: `rm -rf outputs/MyExperiment/run_000`

> **Memory tip:** if inference crashes with an out-of-memory error, lower `predicter.batch_size`
> in the config inside the run folder (`<run_path>/noise2void/config.yaml`).

---

## 8. Understanding the Outputs

For every input file the prediction script produces three output files in the output directory:

| File | Description |
|---|---|
| `<filename>_DENOISED.hspy` | Denoised data in HyperSpy format, preserving the original signal metadata (axes, units, etc.). Can be opened directly in HyperSpy or Digital Micrograph. |
| `<filename>.png` | Side-by-side comparison image: top row = noisy input, bottom row = denoised output. One column per detector channel. A scale bar is added if `px_scale` is set. |
| `<filename>.mp4` | (Video datasets only) Side-by-side comparison video at `predicter.video_fps` fps. |
| `<filename> HISTOGRAM.png` | (Video datasets only) Pixel intensity histograms comparing noisy input and denoised output distributions channel-wise. |

The `.hspy` files are the primary scientific output. They can be loaded back into Python with:

```python
import hyperspy.api as hs
sig = hs.load("outputs/MyExperiment/run_000/my_video_DENOISED.hspy")
sig.plot()
```

---

## 9. Configuration Reference

### `model.name` — available architectures

| Value | Description |
|---|---|
| `UNet` | Standard U-Net with skip connections. The default and most well-tested option. |
| `ResUNet` | Residual U-Net. |
| `SwinUNet` | Swin Transformer-based U-Net. Requires `image_size` to be a power of 2 greater than 256. |
| `SwinTransformer` | Pure Swin Transformer encoder–decoder. Same size constraints as SwinUNet. |

### `trainer.spacing` — Noise2Void grid spacing

The Noise2Void technique masks a sparse grid of pixels in the input and trains the network to
predict those masked pixels from their surroundings. `spacing` controls the distance between
masked pixels. It should be at least as large as the effective receptive field radius of the
network so that no masked pixel falls within the receptive field of another. For a 5-level UNet
with `first_layer_channels: 32`, a spacing of ~23 px is appropriate.

### `dataset.px_scale` — pixel size

Used only for drawing scale bars on the diagnostic plots. It does not affect training or the
numerical values in the denoised `.hspy` output. Set to `null` to suppress scale bars.

### `dataset.video_filter` — restricting training data

To train on only a subset of your files (e.g. to keep some held-out as a true test set), list
their filenames here. The prediction script ignores this filter when denoising — it always
processes every file in the dataset.
