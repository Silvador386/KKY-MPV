from typing import List, Tuple, Optional, Union, Any, Callable, Dict
from torch.utils.data import DataLoader, Dataset
from src.data.catalog import Catalog, BaseCatalog
from src.data.components.splits import DatasetSplits, SplitTypes
from src.data.datasets import CombinedDataset

from lightning.pytorch import LightningDataModule


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        catalog: BaseCatalog,
        call_features: List[str] = None,
        splits: Optional[Union[DatasetSplits, dict]] = None,
        sampler_partial: Callable | None = None,
        batch_size: int = 32,
        evaluation_mode: str = "basic",
        train_shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        if evaluation_mode.lower() not in ["basic", "retrieval"]:
            raise ValueError(f"Evaluation mode {evaluation_mode.lower()} is not supported.")
        self.evaluation_mode = evaluation_mode.lower()

        self.catalog = catalog
        self.call_features = call_features
        if isinstance(splits, dict):
            splits = DatasetSplits(**splits)
        self.splits = splits
        self.sampler_partial = sampler_partial

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices "
                    f"({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                    self.hparams.batch_size // self.trainer.world_size
            )


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.train_dataset is None:
            raise ValueError("Train dataset not initialized")

        sampler = None
        if self.sampler_partial is not None:
            sampler = self.sampler_partial(dataset=self.train_dataset)
        return DataLoader(
            dataset=self.train_dataset,
            sampler=sampler,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=self.hparams.train_shuffle,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )

    def val_dataloader(self) -> Union[DataLoader, Dict[str, DataLoader]]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.val_dataset is None:
            raise ValueError("Validation dataset not initialized")

        query_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )

        if self.evaluation_mode == "basic":
            return query_loader

        elif self.evaluation_mode == 'retrieval':
            database_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                prefetch_factor=self.hparams.prefetch_factor,
                persistent_workers=self.hparams.persistent_workers,
            )
            return {
                "database": database_loader,
                "query": query_loader
            }

        else:
            raise RuntimeError("Should not happen")

    def test_dataloader(self) -> Union[DataLoader, Dict[str, DataLoader]]:
        """Create and return the test dataloader which copies val data.

        :return: The test dataloader.
        """
        if self.test_dataset is None:
            raise ValueError("Test dataset not initialized")

        query_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            persistent_workers=self.hparams.persistent_workers,
        )

        if self.evaluation_mode == "basic":
            return query_loader
        elif self.evaluation_mode == 'retrieval':
            database_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                prefetch_factor=self.hparams.prefetch_factor,
                persistent_workers=self.hparams.persistent_workers,
            )
            return {
                "database": database_loader,
                "query": query_loader
            }
        else:
            raise RuntimeError("Should not happen")

    def full_dataloader(self, collate_fn=None):
        dataset = CombinedDataset({
            SplitTypes.TRAIN: self.train_dataset,
            SplitTypes.VAL: self.val_dataset,
            SplitTypes.TEST: self.test_dataset,
        })
        loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        return loader

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    pass