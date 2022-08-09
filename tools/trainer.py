import os
import hydra
from typing import Dict
from omegaconf import DictConfig, OmegaConf
import tensorboard
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, ModelCheckpoint


def get_train_params(cfg: DictConfig):
    params = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(params, Dict)
    tb_logger = None

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

        if cfg.callbacks:
            for _, callback in cfg.callbacks.items():
                callbacks.append(hydra.utils.instantiate(callbacks))

        params["logger"] = logger
        params["calbacks"] = callbacks

    else:
        params["logger"] = False
        params["enable_checkpointing"] = False

    return params


def train(cfg: DictConfig):
    # set random seed
    seed_everything(cfg.seed)

    # build model to be trained
    task = hydra.utils.instantiate(cfg.task)

    # build data for model to be trained on
    data_module = hydra.utils.instantiate(
        cfg.data_module,
        dataset=cfg.dataset,
        transforms=cfg.transform,
        _recurisive=False,
    )

    # create the Trainer object with all the wanted configurations
    params = get_train_params(cfg)
    trainer = Trainer(**params)

    # train the model
    trainer.fit(task, data_module)

    # test the model
    trainer.test(model=task, data_module=data_module)


@hydra.main(config_path="../config", config_name="defualts")
def run (cfg: DictConfig):
    os.environ["HYDRA_FULL_ERROR"] = os.environ.get("HYDRA_FULL_ERROR")
    train(cfg)

if __name__ == "__main__":
    run()