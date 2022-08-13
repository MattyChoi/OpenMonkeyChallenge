import hydra
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig


class OMCModule(pl.LightningDataModule):
    def __init__(
        self, 
        dataset: DictConfig,        # dictconfig that contains train, val, and test dataloaders
        transform: DictConfig,     # transforms to apply
        train: DictConfig,          # dataloader arguments from the data_module config file
        val: DictConfig = None, 
        test: DictConfig = None
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.train_dataset = hydra.utils.instantiate(dataset.train, transform=transform.train)
        self.val_dataset = None
        self.test_dataset = None

        if val:
            self.val_dataset = hydra.utils.instantiate(dataset.val, transform=transform.val)
        if test:
            self.test_dataset = hydra.utils.instantiate(dataset.test, transform=transform.test)

    def train_dataloader(self) -> DataLoader:
        return self._build_dataloader(
            self.train_dataset,
            self.hparams.train,
        )

    def val_dataloader(self) -> DataLoader:
        return self._build_dataloader(
            self.val_dataset,
            self.hparams.val,
        )

    def test_dataloader(self) -> DataLoader:
        return self._build_dataloader(
            self.test_dataset,
            self.hparams.test,
        )

    def _build_dataloader(self, _dataset: Dataset, cfg: DictConfig):
        return DataLoader(
            _dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory,
            collate_fn=_dataset.collate_fn,
            shuffle=cfg.shuffle,
        )