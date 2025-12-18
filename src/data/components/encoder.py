import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional


def create_category_map(keys: np.ndarray, values: np.ndarray, single_value: bool = False) -> dict: 
    """
    Create a mapping dictionary from keys to values.

    Parameters
    ----------
    keys : np.ndarray
        Array of keys (integers or categorical values).
    values : np.ndarray
        Array of values (integers or categorical values).
    single_value : bool, optional
        If True, ensures one key have one value (raises ValueError if duplicates).
        If False, maps each key to a set of values.

    Returns
    -------
    dict
        Mapping of keys to values (single value or set of values).
    """
    if len(keys) != len(values):
        raise ValueError("keys and values must have the same length")

    mapping = {}
    for k, v in zip(keys, values):
        if single_value:
            if k in mapping and mapping[k] != v:
                raise ValueError(
                    f"Duplicate key '{k}' with conflicting values: {mapping[k]} vs {v}"
                )
            mapping[k] = v
        else:
            mapping.setdefault(k, set()).add(v)
    return mapping


class OrdinalEncoder:
    """
    Encodes and decodes categories to integers for consistent handling across training and evaluation.

    - Can be constructed from mappings (cat_to_int, int_to_cat)
    - from_categories builds both mappings (ordinal encoding).
    - encode is available if cat_to_int is present; decode if int_to_cat is present.
    """

    def __init__(self, cat_to_int: Dict, int_to_cat: Optional[Dict] = None):
        """
        Args:
            cat_to_int: Mapping from category to int (for encode)
            int_to_cat: Mapping from int to category (for decode)
        """
        self.cat_to_int = cat_to_int
        self.int_to_cat = int_to_cat


    @classmethod
    def from_categories(cls, categories: Union[List, np.ndarray, pd.Series]):
        """Build both mappings from a list/array of categories (ordinal encoding)."""
        categories_int, categories_map = pd.factorize(categories)
        cat_to_int = {cat: i for i, cat in enumerate(categories_map)}
        int_to_cat = {i: cat for cat, i in cat_to_int.items()}
        return cls(cat_to_int=cat_to_int, int_to_cat=int_to_cat)


    @property
    def categories(self) -> List[str]:
        result = [None] * len(self.cat_to_int)
        for key, index in self.cat_to_int.items():
            result[index] = key
        return result


    def __getitem__(self, key):
        return self.cat_to_int[key]


    def encode(self, cats) -> Union[int, np.ndarray]:
        """Encode categories to integers."""
        if isinstance(cats, (list, np.ndarray, pd.Series)):
            return np.array([self.cat_to_int[c] if c in self.cat_to_int else -1 for c in cats])
        else:
            return self.cat_to_int[cats]


    def decode(self, ints: Union[int, np.ndarray]) -> Union[str, np.ndarray]:
        """Decode integers back to original categories. """
        if self.int_to_cat is None:
            raise NotImplementedError("decode is not available: int_to_cat mapping not provided.")

        if isinstance(ints, (list, np.ndarray, pd.Series)):
            return np.array([self.int_to_cat[i] for i in ints])
        else:
            return self.int_to_cat[ints]


    @property
    def num_categories(self) -> int:
        """Number of unique categories."""
        return len(set(self.cat_to_int.keys()))


    @property
    def num_values(self) -> int:
        """Number of unique values."""
        return len(set(self.cat_to_int.values()))


    def __len__(self) -> int:
        return self.num_categories
