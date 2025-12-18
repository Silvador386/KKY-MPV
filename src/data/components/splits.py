from enum import Enum
from typing import Optional, Sequence

from pydantic import BaseModel


class SplitTypes(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetSplits(BaseModel):
    train_indexes: Optional[Sequence[int]] = None
    val_indexes: Optional[Sequence[int]] = None
    test_indexes: Optional[Sequence[int]] = None

    model_config = {"from_attributes": True}
