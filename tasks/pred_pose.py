import hydra
import pytorch_lightning as pl
import torch
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

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        # get the training images and output heatmaps
        images, hmaps = batch

        # predict the heatmaps from the images
        pred_hmaps = self.forward(images)

        return self.loss(pred_hmaps, hmaps)

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        images, hmaps = batch

        # predict the heatmaps from the images
        pred_hmaps = self.forward(images)

        return pred_hmaps

    def test_step(
        self, batch: torch.Tensor, batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        images, hmaps = batch

        # predict the heatmaps from the images
        pred_hmaps = self.forward(images)

        return pred_hmaps

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, list(self.model.parameters())
        )
        lr_scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer)
        return [optimizer], [lr_scheduler]
