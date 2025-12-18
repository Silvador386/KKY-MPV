import pandas as pd
from src.config import TARGET_KEY_NAME


def add_metadata_info_to_config(
    config: dict, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict:
    """Include information from metadata to the training configuration."""
    assert TARGET_KEY_NAME in train_df and TARGET_KEY_NAME in test_df
    config["number_of_classes"] = len(train_df[TARGET_KEY_NAME].unique())
    config["training_samples"] = len(train_df)
    config["test_samples"] = len(test_df)
    return config


def _load_metadata(metadata: str) -> pd.DataFrame:
    """Load metadata `csv` or `parquet` file."""
    if metadata.lower().endswith(".csv"):
        df = pd.read_csv(metadata)
    elif metadata.lower().endswith(".parquet"):
        df = pd.read_parquet(metadata)
    else:
        raise ValueError(
            f"Unknown metadata file extension: {metadata}. Use either '.csv' or '.parquet'."
        )
    return df


def load_metadata(
    metadata: str
) -> pd.DataFrame:
    """Load metadata.

    Parameters
    ----------
    metadata : str
        File path to the metadata `csv` or `parquet` file.

    Returns
    -------
    df : pd.DataFrame
        Loaded metadata.
    """
    df = _load_metadata(metadata)
    return df
