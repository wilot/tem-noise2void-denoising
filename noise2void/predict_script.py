"""predict_script.py

Once a model has been trained, this script be used to apply the model to the entire dataset and plot the denoised
results in their native format and also image/video formats, for visualisation.

The script can be used to apply a trained model to the data that the model was trained on, or to a different
dataset.
"""

import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

import tqdm
import hyperspy.api as hs
from matplotlib_scalebar.scalebar import ScaleBar
import omegaconf

from noise2void.datasets.channels import MultiChannelDataset, MultiChannelMetadata, Channel
from noise2void.datasets.generators import dataset_generators

CPU = torch.device("cpu")
VIDEO_FPS = 10  # Defaults in case these aren't specified in the config
MAX_PREDICT_SIZE = 8  # The maximum number of images to process on GPU at once (to prevent crashes)


def validate_args(args, model_file: Path, config_file: Path):
    """Validates the CLI arguments, checks that the necessary files exist."""

    if not args.run_path.exists():
        print(f"The run folder {args.run_path} does not exist.")
        sys.exit(2)
    if args.dataset_name is not None:
        if args.dataset_name not in dataset_generators:
            print(f"The dataset {args.dataset_name} is not recognised.")
            sys.exit(2)

    if not config_file.exists():
        print(f"The training config file has not been found at {config_file}")
        sys.exit(1)
    if not model_file.exists():
        print(f"The model weights have not been found at {model_file}")
        sys.exit(1)
    return


def save_plot_image(meta: MultiChannelMetadata, image: np.ndarray, noisy_image: np.ndarray, savepath: Path):
    """Plots and saves a single image. Image must be an [1, C, Y, X] array for C channels"""

    plot_savepath = savepath.with_suffix(".png")
    vmin, vmax = np.min(noisy_image, axis=(0, 2, 3)), np.max(noisy_image, axis=(0, 2, 3))

    gridspec_kw = dict(bottom=0, top=1, left=0, right=1, wspace=0, hspace=0)
    if image.shape[1] > 1:
        fig, axes = plt.subplots(
            2, image.shape[1], figsize=(8 * image.shape[1], 2 * 8), gridspec_kw=gridspec_kw, squeeze=False
        )
    else:  # Plot horizontal if only one channel, looks nicer
        fig, axes = plt.subplots(
            image.shape[1], 2, figsize=(2 * 8, 8), gridspec_kw=gridspec_kw, squeeze=False
        )
        axes = axes.T  # This trick means the rest of the code works without changing anything
    for chan_index in range(image.shape[1]):
        axes[0, chan_index].imshow(noisy_image[0, chan_index], cmap="inferno", vmin=vmin[chan_index], vmax=vmax[chan_index])
        axes[1, chan_index].imshow(image[0, chan_index, ...], cmap="inferno", vmin=vmin[chan_index], vmax=vmax[chan_index])
        axes[0, chan_index].axis('off')
        axes[1, chan_index].axis("off")
    if meta.px_scale is not None:
        sbar = ScaleBar(meta.px_scale, "nm", location="lower right", color='w', box_color='k', box_alpha=0.7)
        axes[-1, -1].add_artist(sbar)
    fig.savefig(plot_savepath, dpi=210)
    plt.close(fig)


def save_plot_video(meta: MultiChannelMetadata, video: np.ndarray, noisy_video: np.ndarray, fps: int, savepath: Path):
    """Plots and saves a video. Must be in the shape [frames, C, H, W] for channels C, height H and width W. Also
    plots a histogram (channelwise)."""

    assert video.shape[0] > 1
    vid_savepath = savepath.with_suffix(".mp4")
    hist_savepath = savepath.parent / (savepath.stem + " HISTOGRAM.png")

    noisy_vmin, noisy_vmax = np.min(noisy_video, axis=(0, 2, 3)), np.max(noisy_video, axis=(0, 2, 3))
    denoised_vmin, denoised_vmax = np.min(video, axis=(0, 2, 3)), np.max(video, axis=(0, 2, 3))

    gridspec_kw = dict(bottom=0, top=1, left=0, right=1, wspace=0, hspace=0)
    if video.shape[1] > 3:
        fig, axes = plt.subplots(
            2, video.shape[1], figsize=(8 * video.shape[1], 2 * 8), gridspec_kw=gridspec_kw, squeeze=False
        )
    else:  # Plot the other way around for single channel videos, looks nicer
        fig, axes = plt.subplots(
            video.shape[1], 2, figsize=(2 * 8, 8), gridspec_kw=gridspec_kw, squeeze=False
        )
        axes = axes.T  # This trick means the rest of the code can stay the same
    for ax in axes.flatten():
        ax.axis("off")
    if meta.px_scale is not None:
        sbar = ScaleBar(meta.px_scale, "nm", location="lower right", color='w', box_color='k', box_alpha=0.7)
        axes[-1, -1].add_artist(sbar)
    writer = ani.FFMpegWriter(fps=fps, codec="h264_nvenc", extra_args=["-cq:v", "19"])  # Use Nvidia GPU encoder
    with writer.saving(fig, vid_savepath, dpi=210):
        noisy_chan_ims = [  # Initialise the plot
            axes[0, chan_index].imshow(
                noisy_video[0, chan_index], cmap="inferno", vmin=noisy_vmin[chan_index], vmax=noisy_vmax[chan_index]
            )
            for chan_index in range(video.shape[1])
        ]
        denoised_chan_ims = [
            axes[1, chan_index].imshow(
                video[0, chan_index], cmap="inferno", vmin=denoised_vmin[chan_index], vmax=denoised_vmax[chan_index]
            )
            for chan_index in range(video.shape[1])
        ]
        writer.grab_frame()
        for frame_index in range(1, video.shape[0]):
            for chan_index in range(video.shape[1]):
                noisy_chan_ims[chan_index].set_data(noisy_video[frame_index, chan_index, ...])
                denoised_chan_ims[chan_index].set_data(video[frame_index, chan_index, ...])
            writer.grab_frame()
    plt.close(fig)

    # Plot histogram
    fig, axes = plt.subplots(video.shape[1], 2, sharex=True, sharey=False, squeeze=False)
    for chan in range(video.shape[1]):
        axes[chan, 0].hist(noisy_video.flatten(), bins=128)
        axes[chan, 1].hist(video.flatten(), bins=128)
    axes[0, 0].set_title("Noisy input histogram")
    axes[0, 1].set_title("Output histogram")
    fig.savefig(hist_savepath, dpi=210)
    plt.close(fig)


def save_hspy(meta: MultiChannelMetadata, datum: np.ndarray, channel_list: list[Channel], savepath: Path):
    """Saves the image or video to a hyperspy file, channel-wise, in the same directory structure as the source data"""

    data_savepath = savepath.with_suffix(".hspy")
    for chan_index in range(datum.shape[1]):
        chan_savepath = data_savepath.parent / (meta.fpaths[channel_list[chan_index]].stem + "_DENOISED.hspy")
        sig = hs.load(str(meta.fpaths[channel_list[chan_index]]))
        sig.data = datum[:, chan_index]
        sig.save(str(chan_savepath))


def main():

    parser = argparse.ArgumentParser(
        prog="Noise2Void prediction script",
        description="Denoises a dataset with a trained Noise2Void model. Uses config from the training run unless " +
            "optionally overridden",
    )
    parser.add_argument("run_path", type=Path, help="Path to the run folder")
    parser.add_argument("save_path", type=Path, help="Path to the output folder")
    parser.add_argument("--dataset_name", type=str, help="Name of alternative dataset to denoise")

    args = parser.parse_args()
    config_file = args.run_path / "noise2void" / "config.yaml"
    weights_file = args.run_path / "model_state.pt"
    torchscript_file = args.run_path / "traced_model.pt"
    validate_args(args, weights_file, config_file)
    save_root = args.save_path / args.run_path.name
    save_root.mkdir(parents=True, exist_ok=False)
    print("Args validated, loading dataset and model")

    # Initialise model and dataset from config
    config = omegaconf.OmegaConf.load(config_file)
    dataset: MultiChannelDataset = dataset_generators[config.dataset.type](config, predict=True)
    print("Dataset and model loaded")

    # Load model weights
    model = torch.jit.load(torchscript_file)

    # try:  # The predicter config wasn't present in early versions...
    #     max_batch_size = config.predicter.batch_size
    #     video_fps = config.predicter.video_fps
    #     device = torch.device(config.predicter.device)
    # except omegaconf.errors.ConfigAttributeError as err:
    print("Predicter configuration not found, using defaults.")
    max_batch_size = MAX_PREDICT_SIZE
    video_fps = VIDEO_FPS
    device = torch.device("cpu")
    print(f"Using a batch size of {max_batch_size} on {device}")

    model = model.to(device)
    print(f"Model loaded onto device {device}")

    for meta in tqdm.tqdm(dataset.sample_filegroups, desc="Denoising and plotting"):

        datum = dataset.load_interpolate(meta)
        datum = dataset.normalise(datum)

        with torch.no_grad():
            if datum.shape[0] > max_batch_size:  # Process in batches
                process_buf = torch.split(datum, max_batch_size, dim=0)
                pred = torch.cat([  # Predict on the device
                    model(batch.to(device)).to(CPU)
                    for batch in process_buf
                ], dim=0)
            else:
                pred = model(datum.to(device)).to(CPU)
            pred = pred.detach()
            pred = dataset.uninterpolate(meta, pred).numpy()  # Return to original scale

        savepath = save_root / dataset.get_savename(meta)
        savepath.parent.mkdir(parents=True, exist_ok=True)  # Create any sample subfolder if necessary
        if meta.frames is None:  # A single image
            save_plot_image(meta, pred, datum.numpy(), savepath)
        else:
            save_plot_video(meta, pred, datum.numpy(), video_fps, savepath)
        save_hspy(meta, pred, dataset.channels, savepath)

    print("\nComplete.")


if __name__ == "__main__":
    main()