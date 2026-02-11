"""Train Script

Trains a U-Net with the Noise2Void training scheme. The dataset and model are configured with the `config.yaml`
file. After training, the model weights are saved. Then a traced TorchScript module is saved of the same model.
"""

import sys
import shutil
from pathlib import Path

import torch
import hydra
from omegaconf import DictConfig

from noise2void.datasets.generators import dataset_generators
from noise2void.models.generators import model_generators
from noise2void.datasets.channels import MultiChannelDataset
from noise2void.trainer import Trainer, TrainConfig
from noise2void.models.unet import UNet


PROJECT_DIR = Path(__file__).parents[1]


def setup_rundir(run_path: str) -> Path:
    """Creates the run-path if necessary, validates and cloned the code/config there.

    The run-path can be created as the next increment over the previous existing one i.e. run_001. The code is copied
    to this newly created run directory, and the directory path is returned.
    """

    run_path = PROJECT_DIR /  Path(run_path)
    if run_path.parts[-2] == "{SAMPLE_NAME}":
        raise ValueError("Config value for run_directory not set!")

    if not run_path.parent.exists():
        print(f"{run_path.parent=} does not exist, creating")
        run_path.parent.mkdir(parents=True)

    if "{index}" in run_path.parts[-1]:  # Just make the next available number
        existing_runs = run_path.parent.glob("run_???")
        existing_runs = list(filter(lambda path: path.is_dir, existing_runs))
        if len(existing_runs) == 0:
            run_path = run_path.with_name(f"run_{0:03d}")
        else:
            last_run = sorted(existing_runs, key=lambda path: path.name[-3:])[-1]
            last_run_index = int(last_run.name[-3:])
            run_path = run_path.with_name(f"run_{last_run_index + 1:03d}")

    run_path.mkdir(exist_ok=False)
    code_dir = PROJECT_DIR / Path("noise2void")
    shutil.copytree(code_dir, run_path / "noise2void")
    return run_path


def save_trace_model(model: UNet, example_datum: torch.Tensor, savedir: Path, model_config: DictConfig):
    """Saves the state-dict and then JIT traces/compiles the model with the example tensor."""

    torch.save(model.state_dict(), savedir / model_config.state_savename)
    model.eval()
    model = model.to(torch.device("cpu"))
    traced_model: torch.jit.ScriptModule = torch.jit.trace(model, example_datum)
    traced_model.save(savedir / model_config.traced_savename)


@hydra.main(config_path="", config_name="config.yaml", version_base=None)
def main(config: DictConfig):

    # Validate and apply configs
    run_path = setup_rundir(config.run_directory)

    if config.model.name not in model_generators:
        print(f"Model not recognized: {config.model.name}")
        sys.exit(2)
    model = model_generators[config.model.name](config)
    print("Defined model")
    # model = torch.compile(model)

    if config.dataset.type not in dataset_generators:
        print(f"Dataset not recognized: {config.dataset.type}")
        sys.exit(2)
    dataset: MultiChannelDataset = dataset_generators[config.dataset.type](config, predict=False)
    print(f"Initialised dataset with length {len(dataset)}, commencing training")

    train_config = TrainConfig(
        **config.trainer, log_dir=run_path / "training_log", image_shape=config.image_size,
        px_scale=config.dataset.px_scale
    )
    trainer = Trainer(dataset, model, True, dataset.reserved_example, train_config)
    trainer.train_distributed_model()
    print("Training complete, saving")

    save_trace_model(model, dataset.reserved_example, run_path, config.model)

    print("Complete")


if __name__ == "__main__":
    main()