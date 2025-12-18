import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 32) -> None:
    """Download file with progress bar."""
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
