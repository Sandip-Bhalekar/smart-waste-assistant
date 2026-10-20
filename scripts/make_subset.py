from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.categories import DATASET_CLASSES  # noqa: E402
from src.paths import data_dir, project_root  # noqa: E402


def label_classes(path: Path) -> set[int]:
    classes = set()
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return classes
    for line in path.read_text(encoding="utf-8").splitlines():
        classes.add(int(line.split()[0]))
    return classes


def sample_split(images_dir: Path, labels_dir: Path, per_class: int, seed: int) -> list[Path]:
    buckets = defaultdict(list)
    mixed = []
    for image in images_dir.glob("*"):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        classes = label_classes(labels_dir / f"{image.stem}.txt")
        if len(classes) >= 2:
            mixed.append(image)
        for cls in classes:
            buckets[cls].append(image)
    rng = random.Random(seed)
    chosen = []
    seen = set()
    for image in mixed:
        if image.name not in seen:
            chosen.append(image)
            seen.add(image.name)
    for cls in range(len(DATASET_CLASSES)):
        rng.shuffle(buckets[cls])
        count = 0
        for image in buckets[cls]:
            if image.name in seen:
                continue
            chosen.append(image)
            seen.add(image.name)
            count += 1
            if count >= per_class:
                break
    return chosen


def copy_pairs(images: list[Path], labels_dir: Path, dest_images: Path, dest_labels: Path) -> None:
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(image, dest_images / image.name)
        label = labels_dir / f"{image.stem}.txt"
        if label.exists():
            shutil.copy2(label, dest_labels / label.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stratified CPU-friendly subset.")
    parser.add_argument(
        "--source",
        type=Path,
        default=data_dir() / "processed" / "garbage-classification-3",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=data_dir() / "processed" / "garbage-classification-3-subset",
    )
    parser.add_argument("--train-per-class", type=int, default=80)
    parser.add_argument("--valid-per-class", type=int, default=25)
    parser.add_argument("--test-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for split, per_class in (
        ("train", args.train_per_class),
        ("valid", args.valid_per_class),
        ("test", args.test_per_class),
    ):
        images = sample_split(
            args.source / "images" / split,
            args.source / "labels" / split,
            per_class,
            args.seed,
        )
        copy_pairs(
            images,
            args.source / "labels" / split,
            args.dest / "images" / split,
            args.dest / "labels" / split,
        )
        print(split, len(images))
    payload = {
        "path": str(args.dest.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(DATASET_CLASSES)},
        "nc": len(DATASET_CLASSES),
    }
    yaml_path = args.dest / "waste.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (project_root() / "waste.subset.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    print("wrote", yaml_path)


if __name__ == "__main__":
    main()
