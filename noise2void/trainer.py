"""Trainer

Trains the UNets with Noise2Void training regime.

* INITIAL_LEARNING_RATE - The initial learning rate for the Adam optimiser.
* DEVICE - The device on which to conduct training.

* LossHistory - Contains the loss history during training, and can plot it.
* Trainer - Does the trianing
* MSELoss - A mean-square error loss (i.e. an L2 loss)
* L1Loss - A l1-loss

* void_image_batch - Adds void pixels to an image according to the Noise2Void technique. CPU optimised
* void_image_batch_tensor - As above but on the GPU. Slower but easier to read so I haven't deleted it yet.
* numba_sum_reduce - Optimised sum-reduction
"""

import os
import pathlib
from dataclasses import dataclass

import numpy as np
import numba as nb
import torch.nn
import torch.distributed
import torch.nn.parallel
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from noise2void.models.unet import UNet

CPU = torch.device('cpu')
# This is used to transfer weights from distributed context back to the main process after distributed training has
# finished.
TEMP_FILENAME = "final_model_state.pt"


@dataclass
class TrainConfig:
    """The runtime configuration for the model training"""

    learning_rate: float        # Adam optimiser learning rate
    batch_size: int             # The number of images in each batch, if distributed this is per-device
    epochs: int                 # Total number of epochs to train for
    spacing: int                # Spacing between null/dead pixels in the Noise2Void technique
    image_shape: int            # The size of each image in the batch
    log_dir: pathlib.Path       # Where to save the training loss log
    px_scale: float | None      # The size of pixels in nanometres, for plotting
    test_fraction: float = 0.2  # Fraction of the dataset to be used for testing


@dataclass
class _TrainingProcessConfig:
    """The parameters passed to each distributed training process."""

    model: torch.nn.Module
    train_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    image_size: int  # The size of each image in the batch in H, W
    validation_image: torch.Tensor
    void_spacing: int
    void_size: int
    learning_rate: float
    batch_size: int
    epochs: int
    world_size: int | None
    log_dir: pathlib.Path  # Where to save the training loss log
    plot_dir: pathlib.Path
    px_scale: float | None  # The size of pixels in nanometres, for plotting


class Trainer:
    """Trains a model according to the Noise2Void technique."""

    def __init__(
            self, dataset: torch.utils.data.Dataset, model: torch.nn.Module, distributed: bool, test_image: torch.Tensor,
            config: TrainConfig
    ) -> None:
        """Performs training on a model with Noise2Void training.

        Generates a Noise2Void grid and applies the grid, deletes information in the input at those gridpoints, and
        focuses training attention to those missing regions, according to the Noise2Void technique.

        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            The dataset of training data. Should be images of all the same size (preferably 512x512) with two colour
            channels corresponding to ADF and BF channels in that order.
        model: nn.Module
            The model being trained.
        distributed: bool
            Whether to run on a single GPU or distributed across several
        test_image: torch.Tensor
            A pre-normalised (1, channels, image_shape, image_shape) tensor used to visualise learning
        """

        self.dataset = dataset
        self.model = model
        self.spacing = config.spacing
        self.epochs = config.epochs
        self.learning_rate = config.learning_rate
        self.px_scale = config.px_scale
        self.test_image = test_image
        self.distributed = distributed
        self.image_size = config.image_shape
        self.log_dir = config.log_dir
        self.plot_savedir = pathlib.Path(config.log_dir) / "epoch_plots"
        self.plot_savedir.mkdir(parents=True, exist_ok=False)

        self.num_test_images = max(int(config.test_fraction * len(self.dataset)), 1)
        self.num_train_images = len(self.dataset) - self.num_test_images
        self.batch_size = min(config.batch_size, self.num_test_images)
        if not self.distributed:
            torch.cuda.set_device(0)

        self.train_dataset, self.test_dataset = self.split_datasets()

        self.cpu_workers = max(1, min(self.batch_size, 40))
        self.void_spacing = config.spacing
        self.void_size = config.spacing - 8  # Jitter up to 8px

    def train_distributed_model(self):
        """Trains the model across several GPUs"""

        num_devices = torch.cuda.device_count()
        train_params = _TrainingProcessConfig(
            model=self.model, train_dataset=self.train_dataset, test_dataset=self.test_dataset,
            image_size=self.image_size, validation_image=self.test_image, learning_rate=self.learning_rate,
            batch_size=self.batch_size, epochs=self.epochs, world_size=num_devices, void_size=self.void_size,
            void_spacing=self.void_spacing, log_dir=self.log_dir, plot_dir=self.plot_savedir, px_scale=self.px_scale
        )
        mp.spawn(
            _train_distributed_model,
            args=(train_params,), nprocs=num_devices, join=True
        )  # Waits until all processes rejoin/end
        self.model.to(CPU)
        # Load the final weights that were saved by rank 0 (i.e. GPU 0), transfer them to CPU
        self.model.load_state_dict(torch.load(self.log_dir / TEMP_FILENAME, map_location={"cuda:0": "cpu"}))
        (self.log_dir / TEMP_FILENAME).unlink()  # Delete the temporary file

    def train_model(self):
        """Train the model!

        Returns model to CPU memory when done.
        """

        train_params = _TrainingProcessConfig(
            model=self.model, train_dataset=self.train_dataset, test_dataset=self.test_dataset,
            image_size=self.image_size, validation_image=self.test_image, batch_size=self.batch_size,
            learning_rate=self.learning_rate, epochs=self.epochs, world_size=None, void_size=self.void_size,
            void_spacing=self.void_spacing, log_dir=self.log_dir, plot_dir=self.plot_savedir, px_scale=self.px_scale
        )
        _train_model(train_params)
        self.model.to(CPU)

    def plot_test_image(self, test_image: torch.Tensor, test_image_output: torch.Tensor, tag: str, epoch_index: int):
        """Plots a test image with tensorboard"""

        assert len(test_image.shape) == 4 and test_image.shape[0] == 1
        chans = test_image.shape[1]
        if chans == 1:
            im = ((test_image_output[0, 0] + 1.0) * 128.).to(torch.uint8)
            self.log_writer.add_image(tag, im, dataformats="HW", global_step=epoch_index)
        else:
            im = ((test_image_output[0] + 1.0) * 128.).to(torch.uint8)
            self.log_writer.add_images(tag, im, dataformats="CHW", global_step=epoch_index)

        gridspec_kw = dict(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        fig, axes = plt.subplots(2, chans, gridspec_kw=gridspec_kw, figsize=(chans * 8, 16))
        for row_index, im in enumerate((test_image, test_image_output)):
            for chan_index in range(chans):
                axes[row_index, chan_index].imshow(
                    im[0, chan_index], cmap="inferno", interpolation=None, vmin=-1, vmax=1
                )
                axes[row_index, chan_index].axis("off")
        sbar = ScaleBar(self.px_scale, "nm", location="lower right", color='w', box_color='k', box_alpha=0.7)
        axes[-1, -1].add_artist(sbar)
        fig.savefig(self.plot_savedir / f"epoch_{epoch_index:03d}.png")
        plt.close(fig)

    def apply_mean(self, image_batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replaces pixels in the image_batch with their local mean. Pixels are chosen by the mask.

        Calculates a receptive field around each masked pixel, takes the mean of the receptive field excluding the
        central, masked, pixel, and replaces the masked pixel with that mean value.
        """

        # This was slowing training by >10x. Previously performing on the GPU. Much faster with numba on CPU!
        assert image_batch.device == CPU and mask.device == CPU  # This is an absolute pig if not caught early!
        return torch.from_numpy(void_image_batch(image_batch.numpy(), mask.numpy(), self.spacing))

    def split_datasets(self) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Randomly splits the dataset into test and train. Uses a distributed sampler if distributed. If so, rank and
        world_size must be specified"""

        train_set, test_set = torch.utils.data.random_split(
            self.dataset, (self.num_train_images, self.num_test_images)
        )

        return train_set, test_set


def generate_grid(grid_shape: tuple[int, int], distance: int, jitter: int) -> np.ndarray:
    """Generates an image masked in a grid-shape where mask-points are no closer than distance from themselves or
    the edges."""

    max_jitter = jitter  # Jitter the regular grid of hot pixels by up to
    distance += max_jitter  # Preserve minimum distance even after jittering

    y_grid, x_grid = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]))
    border = np.logical_or(
        np.logical_or(y_grid < distance, y_grid > grid_shape[0] - distance),
        np.logical_or(x_grid < distance, x_grid > grid_shape[1] - distance)
    )
    v_stripes = y_grid % distance == 0
    h_stripes = x_grid % distance == 0
    grid = np.logical_and(  # This is now a regular grid of pixels spaced out by receptive field.
        np.logical_and(v_stripes, h_stripes),
        np.logical_not(border)
    )

    # Add random displacements for each gridpoint to prevent grid artifacts in final result.
    grid_coords = np.argwhere(grid)
    jitter_grid_coords = grid_coords + \
        np.random.default_rng().integers(low=-max_jitter, high=max_jitter, size=grid_coords.shape)
    jitter_grid = np.zeros_like(grid)
    jitter_grid[jitter_grid_coords[:, 0], jitter_grid_coords[:, 1]] = 1
    grid = jitter_grid
    return grid


@nb.njit
def void_image_batch(image_batch: np.ndarray, mask: np.ndarray, void_size: int):
    """Adds information voids to the image_batch at locations specified by match. Numba optimised, operates in-place.
    The image_batch MUST be of floating point type."""

    mask_coords = np.argwhere(mask)
    for mask_coord_index in nb.prange(len(mask_coords)):  # RACE CONDITION IF RECEPTIVE FIELD IS WRONG!!!
        mask_coord = mask_coords[mask_coord_index]
        masked_pixels = image_batch[..., mask_coord[0], mask_coord[1]]
        locality = image_batch[  # Includes the receptive field's centre
            ...,
            mask_coord[0] - void_size // 2: mask_coord[0] + void_size // 2,
            mask_coord[1] - void_size // 2: mask_coord[1] + void_size // 2
        ]
        local_sum = numba_sum_reduce(locality) - masked_pixels  # Receptive field excluding central pixel
        # num_locality is the number of pixels in a single image & channel of the batch, minus the central pixel
        num_locality = locality.size // masked_pixels.size - masked_pixels.size
        local_mean = local_sum / num_locality
        image_batch[..., mask_coord[0], mask_coord[1]] = local_mean
    return image_batch


@nb.njit
def numba_sum_reduce(arr: np.ndarray) -> np.ndarray:
    """Sum reduce along all except the first two axes."""
    res = np.empty(arr.shape[:2], dtype=arr.dtype)
    for f in range(arr.shape[0]):
        for ch in range(arr.shape[1]):
            res[f, ch] = np.sum(arr[f, ch])
    return res


def _train_distributed_model(rank: int, params: _TrainingProcessConfig):
    """Trains a model using Distributed Data Parallel.

    The model is duplicated across each rank/process in `world_size`. The batches are different in each rank. When the
    model is updated after each training batch, the gradients and optimiser updates are synchronised across ranks.

    This function should be run by each process/rank in the distributed process. They then synchronise with eachother.
    """

    assert params.world_size is not None

    if rank == 0:
        log_writer = SummaryWriter(log_dir=str(params.log_dir))
        batch_train_counter, batch_test_counter = 0, 0

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=params.world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    params.model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(params.model, device_ids=[rank])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        params.train_dataset, rank=rank, num_replicas=params.world_size, shuffle=True
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        params.test_dataset, rank=rank, num_replicas=params.world_size, shuffle=True
    )
    train_dataloader = torch.utils.data.DataLoader(
        params.train_dataset, params.batch_size, sampler=train_sampler,
    )
    test_dataloader = torch.utils.data.DataLoader(
        params.test_dataset, params.batch_size, sampler=test_sampler,
    )

    criterion = MSELoss()
    # criterion = L1Loss()
    optimiser = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=5E-6)
    validation_image = params.validation_image.to(device)

    for epoch in range(params.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        test_sampler.set_epoch(epoch)
        epoch_train_loss, epoch_test_loss = 0.0, 0.0

        mask_grid = generate_grid((params.image_size, params.image_size), params.void_spacing, 8)
        mask_grid_device = torch.from_numpy(mask_grid).to(device)

        # Process a training batch
        for image_batch in train_dataloader:

            # Selectively mask a copy of the image, on the CPU
            x = void_image_batch(image_batch.clone().numpy(), mask_grid, params.void_size)
            x = torch.from_numpy(x).to(device)  # Transfer to GPU
            y = image_batch[:, 0][:, None, ...].to(device)  # Transfer the original to the GPU too (as the target)

            prediction = model(x)
            loss = criterion(prediction, y, mask_grid_device) / len(image_batch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # Only perform logging and validation on rank 0
            if rank == 0:

                # Log the train loss
                loss_value = loss.cpu().detach().numpy()
                epoch_train_loss += loss_value
                log_writer.add_scalar("BatchLoss/train", loss, global_step=batch_train_counter)

                # Monitor convergence metrics
                log_writer.add_scalar("BatchStats/mean", torch.mean(prediction), global_step=batch_train_counter)
                log_writer.add_scalar("BatchStats/std", torch.std(prediction), global_step=batch_train_counter)
                gradients = [
                    param.grad.detach().flatten()
                    for param in params.model.parameters()
                    if param.grad is not None
                ]
                gradient_norm = torch.cat(gradients).norm()
                log_writer.add_scalar("BatchStats/gradnorm", gradient_norm, global_step=batch_train_counter)

                batch_train_counter += params.world_size  # Batches are run on each rank in the world-size

            torch.distributed.barrier()  # All ranks to wait for logging to complete before proceeding

        # Test mode
        model.eval()
        for image_batch in test_dataloader:  # Process the test dataset
            x = void_image_batch(image_batch.clone().numpy(), mask_grid, params.void_size)
            x = torch.from_numpy(x).to(device)  # Transfer to GPU
            y = image_batch.to(device)

            prediction = model(x)
            loss = criterion(prediction, y, mask_grid_device) / len(image_batch)

            # Loss is computed over each rank/process/gpu
            # Average across all and send it to rank 0
            torch.distributed.reduce(loss, dst=0, op=torch.distributed.ReduceOp.AVG)

            if rank == 0:  # Only write to the logger from one of the ranks
                loss = loss.cpu().detach().numpy()
                epoch_test_loss += loss
                log_writer.add_scalar("BatchLoss/test", loss, global_step=batch_test_counter)
                batch_test_counter += params.world_size
            torch.distributed.barrier()  # Have every rank wait for this logging to get done before continuing

        if rank == 0:  # At the end of the epoch, log the epoch losses
            epoch_train_loss /= len(train_dataloader)
            epoch_test_loss /= len(test_dataloader)
            log_writer.add_scalar("EpochLoss/train", epoch_train_loss, global_step=epoch)
            log_writer.add_scalar("EpochLoss/test", epoch_test_loss, global_step=epoch)
            _plot_log_test_image(
                log_writer, validation_image.cpu(), params.model(validation_image).detach().cpu(),
                "EpochImage", epoch, params.px_scale, params.plot_dir
            )
        torch.distributed.barrier()  # Make all workers/processes wait for the test step to complete

    if rank == 0:  # Save the model's final weights temporarily, as a way of sending it to the main process
        torch.save(model.module.state_dict(), params.log_dir / TEMP_FILENAME)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def _train_model(params: _TrainingProcessConfig):
    """Trains a model on a single GPU"""

    assert params.world_size is None

    log_writer = SummaryWriter(log_dir=str(params.log_dir))

    device = torch.device("cuda")

    # Constant device arrays
    validation_image = params.validation_image.to(device)

    criterion = MSELoss()
    # criterion = L1Loss()
    optimiser = torch.optim.Adam(params.model.parameters(), lr=params.learning_rate)

    params.model.to(device)

    train_dataloader = torch.utils.data.DataLoader(params.train_dataset, params.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(params.test_dataset, params.batch_size, shuffle=True)

    batch_train_counter, batch_test_counter = 0, 0
    params.model.train()
    for epoch in range(params.epochs):
        epoch_train_loss, epoch_test_loss = 0.0, 0.0

        mask_grid = generate_grid((params.image_size, params.image_size), params.void_spacing, 8)
        mask_grid_device = torch.from_numpy(mask_grid).to(device)

        # For each train batch
        for image_batch in train_dataloader:

            # Selectively mask a copy of the image, on the CPU
            x = void_image_batch(image_batch.clone().numpy(), mask_grid, params.void_size)
            x = torch.from_numpy(x).to(device)  # Transfer to GPU
            y = image_batch[:, 0][:, None, ...].to(device)  # Transfer the original to the GPU too (as the target)

            prediction = params.model(x)
            loss = criterion(prediction, y, mask_grid_device) / len(image_batch)  # Norm over non-uniform batch length

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            loss = loss.cpu().detach().numpy()
            epoch_train_loss += loss
            log_writer.add_scalar("BatchLoss/train", loss, global_step=batch_train_counter)

            # Monitor convergence metrics
            log_writer.add_scalar("BatchStats/mean", torch.mean(prediction), global_step=batch_train_counter)
            log_writer.add_scalar("BatchStats/std", torch.std(prediction), global_step=batch_train_counter)
            gradients = [
                param.grad.detach().flatten()
                for param in params.model.parameters()
                if param.grad is not None
            ]
            gradient_norm = torch.cat(gradients).norm()
            log_writer.add_scalar("BatchStats/gradnorm", gradient_norm, global_step=batch_train_counter)


            batch_train_counter += 1

        # Run a test batch
        params.model.eval()
        for image_batch in test_dataloader:

            x = void_image_batch(image_batch.clone().numpy(), mask_grid, params.void_size)
            x = torch.from_numpy(x).to(device)
            y = image_batch.to(device)

            prediction = params.model(x)
            loss = criterion(prediction, y, mask_grid_device) / len(image_batch)

            loss = loss.cpu().detach().numpy()
            epoch_test_loss += loss
            log_writer.add_scalar("BatchLoss/test", loss, global_step=batch_test_counter)
            batch_test_counter += 1
        params.model.train()

        # Record training progress
        epoch_train_loss /= len(train_dataloader)  # Norm over batches as test & train different length
        epoch_test_loss /= len(test_dataloader)
        log_writer.add_scalar("EpochLoss/train", epoch_train_loss, global_step=epoch)
        log_writer.add_scalar("EpochLoss/test", epoch_test_loss, global_step=epoch)
        _plot_log_test_image(
            log_writer, params.validation_image, params.model(validation_image).detach().cpu(),
            "EpochImage", epoch, params.px_scale, params.plot_dir
        )

    return


def _plot_log_test_image(
    log_writer: SummaryWriter, test_image: torch.Tensor, test_image_output: torch.Tensor, tag: str,
    epoch_index: int, px_scale: float | None, plot_savedir: pathlib.Path
):
    """Plots a test image with tensorboard"""

    assert len(test_image.shape) == 4 and test_image.shape[0] == 1
    chans = test_image.shape[1]  # Stack of three latest frames for each channel
    if chans == 1:
        im = ((test_image_output[0, 0] + 1.0) * 128.).to(torch.uint8)
        log_writer.add_image(tag, im, dataformats="HW", global_step=epoch_index)
    else:
        im = ((test_image_output[0] + 1.0) * 128.).to(torch.uint8)
        log_writer.add_images(tag, im, dataformats="CHW", global_step=epoch_index)

    gridspec_kw = dict(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    if chans == 1:  # If only one channel, plot side-by-side
        fig, axes = plt.subplots(chans, 2, gridspec_kw=gridspec_kw, figsize=(16, 8), squeeze=False)
        axes = axes.T  # Swap axes to confirm with the multi-channel layout in code
    else:  # Otherwise, channels are side-by-side and noisy/denoised is above/below
        fig, axes = plt.subplots(2, chans, gridspec_kw=gridspec_kw, figsize=(chans * 8, 16))
    for row_index, im in enumerate((test_image, test_image_output)):
        for chan_index in range(chans):
            axes[row_index, chan_index].imshow(
                im[0, chan_index], cmap="inferno", interpolation=None, vmin=-1, vmax=1
            )
            axes[row_index, chan_index].axis("off")
    if px_scale is not None:
        sbar = ScaleBar(px_scale, "nm", location="lower right", color='w', box_color='k', box_alpha=0.7)
        axes[-1, -1].add_artist(sbar)
    fig.savefig(plot_savedir / f"epoch_{epoch_index:03d}.png")
    plt.close(fig)


class MSELoss(torch.nn.Module):
    """A MSE loss for Noise2Void architectures, requiring a prediction, target and mask-grid."""

    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        mask_broadcast = torch.broadcast_to(mask[None, None, ...], target.shape)
        sq_error = (prediction - target) ** 2
        masked_sq_error = sq_error * mask_broadcast
        masked_mse_error = masked_sq_error.sum() / mask.sum()
        return masked_mse_error


class L1Loss(torch.nn.Module):
    """An L1 Loss for Noise2Vois"""

    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        mask_broadcast = torch.broadcast_to(mask[None, None, ...], target.shape)
        abs_error = torch.abs(prediction - target)
        masked_abs_error = abs_error * mask_broadcast
        masked_l1_loss = masked_abs_error.sum() / mask.sum()
        return masked_l1_loss


if __name__ == "__main__":

    import skimage.data
    import matplotlib.pyplot as plt
    from noise2void.datasets.iridium_glc_dataset import IridiumVideoDataset

    # Test the voiding functionality
    mask_grid = Trainer.generate_grid((256, 256), 24, 8)
    test_image = skimage.data.brick()[None, None, :256, :256].astype(np.float32)
    void_image = void_image_batch(test_image.copy(), mask_grid, 32 - 8)
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(24, 8))
    axes[0].imshow(mask_grid, cmap="Greys_r")
    axes[1].imshow(test_image[0, 0], cmap="Greys_r")
    axes[2].imshow(void_image[0, 0], cmap="Greys_r")
    axes[0].set_title("Mask grid")
    axes[1].set_title("Test image")
    axes[2].set_title("Void image")

    print("Now attempting real example")
    dset = IridiumVideoDataset(256, 0, ["HAADF Image_movie_0231a_817.dm4"])
    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(24, 8))
    axes[0, 0].imshow(mask_grid, cmap="Greys_r")
    axes[1, 0].imshow(mask_grid, cmap="Greys_r")
    for ax_index in (0, 1):
        im = dset[ax_index][None, ...]
        void_image = void_image_batch(im.clone().numpy(), mask_grid, 24 - 8)
        axes[ax_index, 1].imshow(im[0, 0], cmap="Greys_r")
        axes[ax_index, 2].imshow(void_image[0, 0], cmap="Greys_r")
        axes[ax_index, 3].imshow(im[0, 0] - void_image[0, 0], cmap="Greys_r")
    plt.show(block=True)