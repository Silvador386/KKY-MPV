import os
import os.path as osp
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from collections import defaultdict
from typing import Tuple, List, Dict, Any, Optional

from src.config import TARGET_KEY_NAME, IMAGE_PATH_KEY_NAME, IMAGE_KEY_NAME, EMBEDDING_KEY_NAME


def convert_multilabel_targs(targs: List[Dict[str, Tensor]]) -> Dict[str, np.ndarray]:
    _targs = defaultdict(list)
    for item in targs:
        for k, v in item.items():
            _targs[k].append(v.detach().cpu().numpy())
    targs = {k: np.concatenate(v, axis=0) for k, v in _targs.items()}
    return targs


def run_inference_multilabel(
    model: nn.Module,
    dataloader: DataLoader,
    method_mode: Optional[str] = None,
    is_multilabel_dataset: bool = False,
    normalize: bool = False,  # l2
) -> Dict[str, np.ndarray]:
    features, targs, paths = [], [], []
    with torch.no_grad():
        for i, (batch) in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y, path = batch[IMAGE_KEY_NAME], batch[TARGET_KEY_NAME], batch[IMAGE_PATH_KEY_NAME]
            x = x.cuda()
            if not is_multilabel_dataset:
                y = y.numpy()

            if method_mode is not None:
                if method_mode.lower() == "metric":
                    feat_batch = model.forward_features(x)[:, 0, :]
                elif method_mode.lower() == "hfhub":
                    feat_batch = model(x).pooler_output
                else:
                    raise NotImplementedError
            else:
                feat_batch = model.forward(x)

            if normalize:
                feat_batch = F.normalize(feat_batch, p=2, dim=1)

            feat_batch = feat_batch.cpu().numpy()

            features.append(feat_batch)
            targs.append(y)
            paths.append(path)

    features = np.concatenate(features, axis=0)
    paths = np.concatenate(paths, axis=0)
    if not is_multilabel_dataset:
        targs = np.concatenate(targs, axis=0)
    else:
        targs = convert_multilabel_targs(targs)

    return {
        EMBEDDING_KEY_NAME: features,
        TARGET_KEY_NAME: targs,
        IMAGE_PATH_KEY_NAME: paths,
    }


def save_pickle(data: Any, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)
