import numpy as np
import pandas as pd
import torch
from typing import Tuple, Union, Dict, List

from src.data.datasets.base import BaseDataset
from src.config import TARGET_KEY_NAME, EMBEDDING_KEY_NAME


def resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Convert user-passed dtype into a torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype

    dtype = str(dtype).lower()

    if dtype in ["float32", "fp32"]:
        return torch.float32
    if dtype in ["float16", "fp16", "half"]:
        return torch.float16
    if dtype in ["int8", "uint8"]:
        return torch.int8

    raise ValueError(f"Unsupported dtype option: {dtype}")


class EmbeddingDataset(BaseDataset):

    def __init__(
        self,
        df: pd.DataFrame,
        call_features: List[str] = None,
        dtype: Union[str, torch.dtype] = "float32",
    ):
        super().__init__(
            df=df,
            call_features=call_features,
        )
        self.dtype = resolve_dtype(dtype)
        self._check_metadata_integrity()

    def __getitem__(self, idx: int) -> Dict:
        embedding = self.get_embedding(idx)
        target = self.get_target_id(idx)
        item = {
            EMBEDDING_KEY_NAME: embedding,
            TARGET_KEY_NAME: target,
        }
        if self.call_features is not None:
            item.update(self.get_metadata(idx))
        return item

    def get_embedding(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.df[EMBEDDING_KEY_NAME].iloc[idx], dtype=self.dtype)

    def _check_metadata_integrity(self):
        """Checks if the dataset complies with the framework requirements."""
        assert (
            EMBEDDING_KEY_NAME in self.df.columns.values
        ), f"Dataset must contain `{EMBEDDING_KEY_NAME}` column!"


if __name__ == "__main__":
    data = {
        EMBEDDING_KEY_NAME: [
            np.random.randn(5),
            np.random.randn(5),
        ],
        TARGET_KEY_NAME: [0, 1],
        "user_id": ["A123", "B456"],
    }

    df = pd.DataFrame(data)

    dataset = EmbeddingDataset(
        df=df,
        call_features=["user_id"],
    )

    sample = dataset[0]

    print("Sample output:")
    print(sample)

    print("\nEmbedding:", sample[EMBEDDING_KEY_NAME])
    print("Target:", sample[TARGET_KEY_NAME])
