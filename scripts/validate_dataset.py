from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.categories import DATASET_CLASSES  # noqa: E402
from src.paths import data_dir  # noqa: E402


def file_md5(path: Path, chunk_size: int = 1024 * 128) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
        values = [float(item) for item in parts[1:]]
    except ValueError:
        return None
    return class_id, *values


def validate_split(images_dir: Path, labels_dir: Path) -> dict:
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    missing_labels = []
    unreadable = []
    invalid_class = 0
    invalid_box = 0
    empty_labels = 0
    hashes = {}
    class_counts = Counter()
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            missing_labels.append(image_path.name)
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError):
            unreadable.append(image_path.name)
            continue
        hashes[image_path.name] = file_md5(image_path)
        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            empty_labels += 1
            continue
        for line in content.splitlines():
            parsed = parse_label_line(line)
            if parsed is None:
                invalid_box += 1
                continue
            class_id, x, y, w, h = parsed
            if class_id < 0 or class_id >= len(DATASET_CLASSES):
                invalid_class += 1
                continue
            if min(x, y, w, h) < 0 or max(x, y) > 1 or max(w, h) > 1.05:
                invalid_box += 1
            else:
                class_counts[DATASET_CLASSES[class_id]] += 1
    return {
        "images": len(images),
        "missing_labels": missing_labels[:20],
        "missing_label_count": len(missing_labels),
        "unreadable": unreadable[:20],
        "unreadable_count": len(unreadable),
        "invalid_class_ids": invalid_class,
        "invalid_boxes": invalid_box,
        "empty_labels": empty_labels,
        "class_counts": dict(class_counts),
        "hashes": hashes,
    }


def find_cross_split_duplicates(split_hashes: dict[str, dict[str, str]]) -> list[dict]:
    inverse: dict[str, list[str]] = defaultdict(list)
    for split, mapping in split_hashes.items():
        for name, digest in mapping.items():
            inverse[digest].append(f"{split}/{name}")
    return [{"md5": digest, "files": names} for digest, names in inverse.items() if len(names) > 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO image/label pairs.")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=data_dir() / "processed" / "garbage-classification-3",
    )
    args = parser.parse_args()
    split_hashes = {}
    report = {}
    for split in ("train", "valid", "test"):
        images_dir = args.processed_root / "images" / split
        labels_dir = args.processed_root / "labels" / split
        if not images_dir.exists():
            continue
        result = validate_split(images_dir, labels_dir)
        split_hashes[split] = result.pop("hashes")
        report[split] = result
    report["cross_split_exact_duplicates"] = find_cross_split_duplicates(split_hashes)
    print(json.dumps(report, indent=2))
    if report["cross_split_exact_duplicates"]:
        print("WARNING: exact duplicate images found across splits.")


if __name__ == "__main__":
    main()
