import os.path as osp
import pandas as pd
import torch
from src.config import TARGET_KEY_NAME, IMAGE_PATH_KEY_NAME, EMBEDDING_KEY_NAME, FILENAME_KEY_NAME


def save_embeddings_metadata(embeddings, split, inference_metadata, save_dir):
    dataset_meta = {
        "split": split,
        "dataset_root": inference_metadata["dataset_root"],
        **inference_metadata,
    }

    save_data = {
        "embeddings_data": embeddings,
        "dataset_metadata": dataset_meta,
    }

    arch = inference_metadata["architecture"].replace("/", "-")
    size = inference_metadata["size"][0]

    save_path = osp.join(save_dir, f"{split}_{arch}_{size}.pth")
    print(f"[+] Saving embeddings → {save_path}")
    torch.save(save_data, save_path, pickle_protocol=5)


def load_df_with_embeddings(embedding_path: str) -> pd.DataFrame:
    """Load embeddings and metadata to recreate a FungiTastic dataset with precomputed features."""
    print(f"[+] Loading embeddings from {embedding_path}")
    saved_data = torch.load(embedding_path, map_location="cpu",  weights_only=False)
    emb_data = saved_data["embeddings_data"]

    emb_df = pd.DataFrame({
        EMBEDDING_KEY_NAME: list(emb_data[EMBEDDING_KEY_NAME]),
        TARGET_KEY_NAME: emb_data[TARGET_KEY_NAME],
        IMAGE_PATH_KEY_NAME: emb_data[IMAGE_PATH_KEY_NAME],
        FILENAME_KEY_NAME: emb_data[FILENAME_KEY_NAME],
    })
    return emb_df
