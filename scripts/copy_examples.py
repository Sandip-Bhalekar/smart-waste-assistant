from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.categories import DATASET_CLASSES  # noqa: E402
from src.paths import data_dir, project_root  # noqa: E402


def classes_in_label(path: Path) -> set[str]:
    names = set()
    if not path.exists():
        return names
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id = int(line.split()[0])
        if 0 <= class_id < len(DATASET_CLASSES):
            names.add(DATASET_CLASSES[class_id])
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a few test images into assets/examples.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=data_dir() / "processed" / "garbage-classification-3",
    )
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    images_dir = args.processed_root / "images" / "test"
    labels_dir = args.processed_root / "labels" / "test"
    dest = project_root() / "assets" / "examples"
    dest.mkdir(parents=True, exist_ok=True)
    mixed = []
    by_class = defaultdict(list)
    for image in images_dir.glob("*"):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        names = classes_in_label(labels_dir / f"{image.stem}.txt")
        if len(names) >= 2:
            mixed.append(image)
        for name in names:
            by_class[name].append(image)
    chosen = list(mixed[: max(3, args.limit // 2)])
    for name in DATASET_CLASSES:
        if by_class[name]:
            chosen.append(by_class[name][0])
    unique = []
    seen = set()
    for item in chosen:
        if item.name not in seen:
            unique.append(item)
            seen.add(item.name)
    random.Random(42).shuffle(unique)
    for image in unique[: args.limit]:
        shutil.copy2(image, dest / image.name)
        print(f"copied {image.name}")


if __name__ == "__main__":
    main()
