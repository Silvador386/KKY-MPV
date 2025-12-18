import pandas as pd
import torch

from typing import Union, Dict, List
from PIL import ImageFile
from torch.utils.data import Dataset

from src.data.components.splits import DatasetSplits
from src.config import TARGET_KEY_NAME

ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        call_features: List[str] = None,
    ):
        self.df = df
        self.call_features = call_features
        self._check_metadata_integrity()

    def __len__(self):
        if self.df is None:
            return 0
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        raise NotImplementedError

    def _check_metadata_integrity(self):
        """Checks if the dataset complies with the framework requirements."""
        assert (
            TARGET_KEY_NAME in self.df.columns.values
        ), f"Dataset must contain `{TARGET_KEY_NAME}` column!"

    def get_dataset(self) -> pd.DataFrame:
        """Returns metadata of the dataset."""
        return self.df

    def get_default_splits(self) -> Union[DatasetSplits, None]:
        """Returns default splits of the dataset. Should be overridden by subclasses."""
        raise NotImplementedError

    def get_target_id(self, idx: int) -> int:
        """Get class id of i-th element in the dataset."""
        return self.df[TARGET_KEY_NAME].iloc[idx]

    def get_metadata(self, idx: int) -> dict:
        """Return row data as dict"""
        if self.call_features is None:
            return self.df.iloc[idx].to_dict()
        return self.df.iloc[idx][self.call_features].to_dict()


class CombinedDataset(Dataset):
    def __init__(self, datasets: dict[str, Dataset]):
        """
        Args:
            datasets (dict[str, Dataset]): Mapping from source name to dataset.

        Example: CombinedDataset({"query": ds_test, "test": ds_train})
        """
        self.datasets = datasets
        self.sources = list(datasets.keys())

        # Build index map: list of (source, local_idx)
        self.index_map = []
        for src, ds in self.datasets.items():
            for i in range(len(ds)):
                self.index_map.append((src, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        src, local_idx = self.index_map[idx]
        data = self.datasets[src][local_idx]
        data['source'] = src
        return data


if __name__ == "__main__":
    pass
