import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import hydra
from typing import Dict
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def get_train_params(cfg: DictConfig):
    params = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(params, Dict)

    if cfg.log:
        # logging using tensorboard
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.tensorboard.html
        logger = TensorBoardLogger(save_dir="tb_logs")

        # pytorch lightning callbacks
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html#:~:text=A%20callback%20is%20a%20self,your%20lightning%20module%20to%20run.
        callbacks = []

        # save checkpoints of model 
        # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html
        checkpoint_callback = ModelCheckpoint(dirpath=f"{logger.log_dir}/checkpoints")
        callbacks.append(checkpoint_callback)

        # add callbacks
        if cfg.callbacks:
            for _, callback in cfg.callbacks.items():
                callbacks.append(hydra.utils.instantiate(callbacks))

        params["logger"] = logger
        params["callbacks"] = callbacks

    else:
        params["logger"] = False
        params["enable_checkpointing"] = False

    return params


def train(cfg: DictConfig):
    # set random seed
    seed_everything(cfg.seed)

    # build model to be trained
    task = hydra.utils.instantiate(cfg.tasks, cfg)

    # build data for model to be trained on
    data_module = hydra.utils.instantiate(
        cfg.data_module,
        dataset=cfg.dataset,
        transform=cfg.transform,
        _recursive_=False,
    )

    # create the Trainer object with all the wanted configurations
    params = get_train_params(cfg)
    trainer = Trainer(**params)

    # train the model
    trainer.fit(model=task, datamodule=data_module)

    # test the model
    trainer.test(model=task, datamodule=data_module)


@hydra.main(version_base=None, config_path="../config", config_name="defaults")
def run(cfg: DictConfig):
    os.environ["HYDRA_FULL_ERROR"] = os.environ.get("HYDRA_FULL_ERROR", "1")
    train(cfg)

if __name__ == "__main__":
    run()