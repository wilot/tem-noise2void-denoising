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

import pathlib
from dataclasses import dataclass

import numpy as np
import numba as nb
import torch.nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

from noise2void.models.unet import UNet

CPU = torch.device('cpu')


@dataclass
class TrainConfig:
    """The runtime configuration for the model training"""

    learning_rate: float        # Adam optimiser learning rate
    batch_size: int             # The number of images in each batch
    epochs: int                 # Total number of epochs to train for
    spacing: int                # Spacing between null/dead pixels in the Noise2Void technique
    image_shape: int            # The size of each image in the batch
    log_dir: pathlib.Path       # Where to save the training loss log
    px_scale: float             # The size of pixels in nanometres, for plotting
    test_fraction: float = 0.2  # Fraction of the dataset to be used for testing


class Trainer:
    """Trains a model according to the Noise2Void technique."""

    def __init__(self, dataset: Dataset, model: UNet, test_image: torch.Tensor, config: TrainConfig) -> None:
        """Performs training on a UNet model with Noise2Void training.

        Generates a Noise2Void grid and applies the grid, deletes information in the input at those gridpoints, and
        focuses training attention to those missing regions, according to the Noise2Void technique.

        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            The dataset of training data. Should be images of all the same size (preferably 512x512) with two colour
            channels corresponding to ADF and BF channels in that order.
        model: nn.Module
            The model being trained.
        test_image: torch.Tensor
            A pre-normalised (channels, image_shape, image_shape) tensor used to visualise learning
        """

        self.dataset = dataset
        self.model = model
        self.spacing = config.spacing
        self.epochs = config.epochs
        self.learning_rate = config.learning_rate
        self.px_scale = config.px_scale
        self.test_image = test_image
        self.log_writer = SummaryWriter(log_dir=str(config.log_dir))

        self.num_test_images = max(int(config.test_fraction * len(self.dataset)), 1)
        self.num_train_images = len(self.dataset) - self.num_test_images
        self.batch_size = min(config.batch_size, self.num_test_images)
        self.device = torch.device("cuda:1")

        self.plot_savedir = pathlib.Path(config.log_dir) / "epoch_plots"
        self.plot_savedir.mkdir(parents=True, exist_ok=False)

        self.cpu_workers = max(1, min(self.batch_size, 40))
        self.mask_grid = self.generate_grid((config.image_shape, config.image_shape), config.spacing, 2)

    def train_model(self):
        """Train the model!

        Returns model to CPU memory when done.
        """

        # Randomly split the dataset into test and train
        train_dataloader, test_dataloader, train_frames, test_frames = self.generate_dataloaders()

        # Constant device arrays
        mask_grid_dev = self.mask_grid.to(self.device)
        test_image = self.test_image[None, ...].to(self.device)

        criterion = MSELoss()
        # criterion = L1Loss()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        train_steps = int(train_frames / self.batch_size)
        test_steps = int(test_frames / self.batch_size)

        self.model.to(self.device)
        batch_train_counter, batch_test_counter = 0, 0
        for epoch_index in range(self.epochs):
            print(f"Epoch {epoch_index:03d} of {self.epochs:03d}")
            epoch_train_loss, epoch_test_loss = 0.0, 0.0
            self.model.train()

            # For each train batch
            for image_batch in train_dataloader:

                x = self.apply_mean(
                    image_batch.detach().clone(), self.mask_grid
                ).to(self.device)  # Selectively mask a copy
                y = image_batch.to(self.device)

                prediction = self.model(x)
                loss = criterion(prediction, y, mask_grid_dev) / len(image_batch)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                loss = loss.cpu().detach().numpy()
                epoch_train_loss += loss
                self.log_writer.add_scalar("BatchLoss/train", loss, global_step=batch_train_counter)

                # Monitor convergence metrics
                self.log_writer.add_scalar("BatchStats/mean", torch.mean(prediction), global_step=batch_train_counter)
                self.log_writer.add_scalar("BatchStats/std", torch.std(prediction), global_step=batch_train_counter)
                gradients = [
                    param.grad.detach().flatten()
                    for param in self.model.parameters()
                    if param.grad is not None
                ]
                gradient_norm = torch.cat(gradients).norm()
                self.log_writer.add_scalar("BatchStats/gradnorm", gradient_norm, global_step=batch_train_counter)


                batch_train_counter += 1

            with torch.no_grad():
                self.model.eval()
                for image_batch in test_dataloader:

                    x = self.apply_mean(
                        image_batch.detach().clone(), self.mask_grid
                    ).to(self.device)
                    y = image_batch.to(self.device)

                    prediction = self.model(x)
                    loss = criterion(prediction, y, mask_grid_dev) / len(image_batch)

                    loss = loss.cpu().detach().numpy()
                    epoch_test_loss += loss
                    self.log_writer.add_scalar("BatchLoss/test", loss, global_step=batch_test_counter)
                    batch_test_counter += 1

            # Record training progress
            epoch_train_loss /= train_steps  # Average over batches because there's a different number of batches
            epoch_test_loss /= test_steps    # in the test and train parts
            self.log_writer.add_scalar("EpochLoss/train", epoch_train_loss, global_step=epoch_index)
            self.log_writer.add_scalar("EpochLoss/test", epoch_test_loss, global_step=epoch_index)
            self.plot_test_image(
                test_image.cpu(), self.model(test_image).detach().cpu(), "EpochImage", epoch_index
            )

        self.model.to(torch.device('cpu'))
        return

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

    @staticmethod
    def generate_grid(grid_shape: tuple[int, int], distance: int, jitter: int) -> torch.Tensor:
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

        grid = torch.from_numpy(grid)
        return grid

    def generate_dataloaders(self) -> tuple[DataLoader, DataLoader, int, int]:
        """Randomly splits the dataset into test and train.

        If there are more data in the dataset than required test and train images, the dataset is trimmed here. However
        if there are fewer data in the dataset than number of test and train images required, throws assertion error.

        Returns
        -------
        Tuple[DataLoader, DataLoader, int, int]
            The Train and test dataloaders respectively, along with the number of frames in the train and test sets.
        """

        # assert self.num_test_images + self.num_train_images <= len(self.dataset)
        # if self.num_train_images + self.num_test_images < len(self.dataset):  # Trim the dataset to size if needed
        #     self.dataset = Subset(self.dataset, list(range(self.num_train_images + self.num_test_images)))

        train_set, test_set = random_split(
            self.dataset, (self.num_train_images, self.num_test_images)
        )
        train_dataloader, test_dataloader = [
            DataLoader(dset, self.batch_size, shuffle=True, num_workers=self.cpu_workers, pin_memory=True)
            for dset in (train_set, test_set)
        ]

        return train_dataloader, test_dataloader, len(train_set), len(test_set)


@nb.njit
def void_image_batch(image_batch: np.ndarray, mask: np.ndarray, spacing: int) -> np.ndarray:
    """Adds information voids to the image_batch at locations specified by match. Numba optimised."""

    mask_coords = np.argwhere(mask)
    for mask_coord_index in nb.prange(len(mask_coords)):  # RACE CONDITION IF RECEPTIVE FIELD IS WRONG!!!
        mask_coord = mask_coords[mask_coord_index]
        masked_pixels = image_batch[..., mask_coord[0], mask_coord[1]]
        locality = image_batch[  # Includes the receptive field's centre
            ...,
            mask_coord[0] - spacing // 2: mask_coord[0] + spacing // 2,
            mask_coord[1] - spacing // 2: mask_coord[1] + spacing // 2
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


class MSELoss(torch.nn.Module):
    """A MSE loss for Noise2Void architetures, requiring a prediction, target and mask-grid."""

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
