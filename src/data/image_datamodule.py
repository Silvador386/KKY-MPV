from typing import Any, Dict, Optional, Tuple, Union, List, Callable

import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from src.data.base_datamodule import BaseDataModule
from src.data.catalog import BaseCatalog, FungiTasticCatalog
from src.data.catalog.base import Catalog
from src.data.components.splits import DatasetSplits
from src.data.datasets import ImageDataset, CombinedDataset


def split_image_dataset(
    df: pd.DataFrame,
    transformations: Tuple[Compose, Compose, Compose],
    splits: Optional[DatasetSplits] = None,
    call_features: List[str] = None,
) -> Tuple[Union[Dataset, None], Union[Dataset, None], Union[Dataset, None]]:
    train_dataset, val_dataset, test_dataset = None, None, None

    if splits is not None:
        train_df = (
            df.iloc[splits.train_indexes, :]
            if splits.train_indexes is not None
            else None
        )
        val_df = (
            df.iloc[splits.val_indexes, :] if splits.val_indexes is not None else None
        )
        test_df = (
            df.iloc[splits.test_indexes, :] if splits.test_indexes is not None else None
        )

        if train_df is not None:
            train_dataset = ImageDataset(train_df, transform=transformations[0], call_features=call_features)
        if val_df is not None:
            val_dataset = ImageDataset(val_df, transform=transformations[1], call_features=call_features)
        if test_df is not None:
            test_dataset = ImageDataset(test_df, transform=transformations[2], call_features=call_features)

    else:
        train_dataset = ImageDataset(df, transform=transformations[0], call_features=call_features)

    return train_dataset, val_dataset, test_dataset


class ImageDataModule(BaseDataModule):
    def __init__(
        self,
        catalog: BaseCatalog,
        call_features: List[str] = None,
        splits: Optional[Union[DatasetSplits, dict]] = None,
        sampler_partial: Callable | None = None,
        transformations: Tuple[Compose, Compose] = None,
        batch_size: int = 32,
        evaluation_mode: str = "basic",
        train_shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__(
            catalog=catalog,
            call_features=call_features,
            splits=splits,
            sampler_partial=sampler_partial,
            batch_size=batch_size,
            evaluation_mode=evaluation_mode,
            train_shuffle=train_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        self.transformations = transformations

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        super().setup(stage=stage)

        # Avoid resplitting
        if self.train_dataset or self.val_dataset or self.test_dataset:
            return

        df = self.catalog.get_metadata()
        splits = self.splits
        if splits is None:
            splits = self.catalog.get_default_splits()

        datasets = split_image_dataset(
            df=df,
            transformations=(self.transformations[0], self.transformations[1], self.transformations[1]),
            splits=splits,
            call_features=self.call_features,
        )
        self.train_dataset, self.val_dataset, self.test_dataset = datasets


if __name__ == "__main__":
    import albumentations as A
    import numpy as np
    from albumentations.pytorch import ToTensorV2

    dataset_dir_path = "~/datasets"
    transformations = A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    metadata = FungiTasticCatalog(
        dataset_root=dataset_dir_path,
        dataset_variant="fewshot",
        dataset_size="720p",
        download=True,
    )
    df = metadata.get_metadata()
    splits = {
        "train_indexes": np.arange(0, len(df) // 2).tolist(),
        "val_indexes": np.arange(len(df) // 2, len(df)).tolist(),
        "test_indexes": None,
    }

    data_module = ImageDataModule(
        catalog=metadata,
        splits=splits,
        transformations=(transformations, transformations),
        batch_size=32,
    )

    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
