from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import data_dir  # noqa: E402

DATASETS = {
    "garbage-classification-3": {
        "train": "https://huggingface.co/datasets/keremberke/garbage-object-detection/resolve/main/data/train.zip",
        "valid": "https://huggingface.co/datasets/keremberke/garbage-object-detection/resolve/main/data/valid.zip",
        "test": "https://huggingface.co/datasets/keremberke/garbage-object-detection/resolve/main/data/test.zip",
    }
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already present: {destination}")
        return
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if chunk:
                    handle.write(chunk)


def extract_zip(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zipped:
        zipped.extractall(target)


def main() -> None:
    raw = data_dir() / "raw" / "garbage-classification-3"
    archives = data_dir() / "raw" / "archives"
    urls = DATASETS["garbage-classification-3"]
    for split, url in urls.items():
        archive = archives / f"{split}.zip"
        download_file(url, archive)
        extract_zip(archive, raw / split)
        print(f"Extracted {split} -> {raw / split}")
    print("Download complete.")


if __name__ == "__main__":
    main()
