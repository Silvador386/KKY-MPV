import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict
from src.data.components.splits import DatasetSplits
from src.config import FILENAME_KEY_NAME, EMBEDDING_KEY_NAME, TRANSFORMATION_KEY_NAME
from pathlib import Path


class Catalog(ABC):

    @abstractmethod
    def __init__(self, dataset_root: str):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        pass

    @abstractmethod
    def get_metadata(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_default_splits(self) -> DatasetSplits:
        pass

    def get_metadata_splits(self) -> Dict[str, Union[pd.DataFrame, None]]:
        pass


class BaseCatalog(Catalog):
    def __init__(self, dataset_root: str):
        self.root = Path(dataset_root).expanduser()
        self.df = None
        self.default_splits = None

    def __getitem__(self, idx: int) -> dict:
        return self.df.iloc[idx].to_dict()

    def get_metadata(self) -> pd.DataFrame:
        return self.df

    def get_default_splits(self) -> DatasetSplits:
        return self.default_splits

    def get_metadata_splits(self) -> Dict[str, Union[pd.DataFrame, None]]:
        return {
            "train": self.df.iloc[self.default_splits.train_indexes] if self.default_splits.train_indexes else None,
            "val": self.df.iloc[self.default_splits.val_indexes] if self.default_splits.train_indexes else None,
            "test": self.df.iloc[self.default_splits.test_indexes] if self.default_splits.test_indexes else None,
        }

    def add_embeddings(self, embeddings: pd.DataFrame, validate: str = "one_to_one"):
        """
        Updates the catalog instance with new embeddings.

        Args:
            embeddings (pd.DataFrame): A DataFrame containing IMAGE_PATH_KEY_NAME, and 'embedding' columns.
            validate
        """
        assert isinstance(embeddings, pd.DataFrame), "Embeddings must be a pandas DataFrame."
        assert EMBEDDING_KEY_NAME in embeddings.columns, f"Embeddings DataFrame must have an '{EMBEDDING_KEY_NAME}' column."
        assert FILENAME_KEY_NAME in embeddings.columns, f"Embeddings DataFrame must have an '{FILENAME_KEY_NAME}' column."

        if TRANSFORMATION_KEY_NAME in embeddings.columns:
            embeddings = embeddings[[FILENAME_KEY_NAME, TRANSFORMATION_KEY_NAME, EMBEDDING_KEY_NAME]]
        else:
            embeddings = embeddings[[FILENAME_KEY_NAME, EMBEDDING_KEY_NAME]]

        self.df = pd.merge(
            self.df,
            embeddings,
            on=FILENAME_KEY_NAME,
            how="left",
            validate=validate,
        )
        if self.df[EMBEDDING_KEY_NAME].isna().any():
            print(f"Missing embeddings for some images: {self.df[EMBEDDING_KEY_NAME].isna().sum()}")
