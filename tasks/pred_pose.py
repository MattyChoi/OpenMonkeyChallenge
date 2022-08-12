import hydra
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchmetrics
from omegaconf import DictConfig


class PoseEstimationModule(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = hydra.utils.instantiate(hparams.model)
        self.loss = hydra.utils.instantiate(hparams.loss)

    def forward(self, images: torch.Tensor, *args, **kwargs):
        return self.model(images)

    def predict(self, batch: torch.Tensor):
        pass

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        images, hmaps = batch