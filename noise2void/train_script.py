"""Train Script

Trains a U-Net with the Noise2Void training scheme. Must be run as a script.
"""

import time
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig

from dataset import ExperimentalRecentDataset
from trainer import Trainer, TrainConfig
from models import UNet

assert __name__ == '__main__'  # Don't import this!


def setup_rundir(run_path: str):
    """Creates the run-path if necessary, validates and cloned the code/config there"""

    run_path = Path(run_path)
    if run_path.parts[1] == "{SAMPLE_NAME}":
        raise ValueError("Config value for run_directory not set!")

    if not run_path.parent.exists():
        run_path.parent.mkdir()

    if run_path.parts[-1][:-3] == "NNN":  # Just make the next available number
        existing_runs = run_path.parent.glob("run???")
        existing_runs = filter(lambda path: path.is_dir, existing_runs)
        last_run = sorted(existing_runs, key=lambda path: path.name[-3:])[-1]
        last_run_index = int(last_run.name[-3:])
        run_path = run_path.with_name(f"run{last_run_index + 1:03d}")

    run_path.mkdir(exist_ok=False)
    code_dir = Path("noise2void")
    shutil.copytree(code_dir, run_path / "noise2void")


def configure_model(model_config: DictConfig):

    if model_config.model == "UNet":
        model = UNet(
            input_channels=model_config.channels, output_channels=model_config.channels, num_layers=model_config.depth,
            first_layer_channels=model_config.first_layer_channels,
            first_layer_kernel_size=model_config.first_layer_kernel_size, activation=None
        )
    else:
        raise NotImplementedError()

    return model


def configure_dataset(dataset_config: DictConfig):

    # TODO
    ...


@hydra.main(config_path="noise2void", config_name="train_config.yaml")
def main(config: DictConfig):

    setup_rundir(config.run_directory)
    model = configure_model(config.model)
    dataset, test_image = configure_dataset(config.dataset)

    train_config = TrainConfig(**config.trainer)
    trainer = Trainer(dataset, model, test_image, train_config)
    trainer.train_model()


if __name__ == "__main__":
    main()