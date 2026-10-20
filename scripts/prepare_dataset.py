from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.convert_annotations import coco_class_lookup, convert_coco_file  # noqa: E402
from src.categories import DATASET_CLASSES  # noqa: E402
from src.paths import data_dir, project_root  # noqa: E402


def copy_images(source_dir: Path, dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in source_dir.iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy2(path, dest_dir / path.name)
            copied += 1
    return copied


def write_data_yaml(processed_root: Path) -> Path:
    payload = {
        "path": str(processed_root.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(DATASET_CLASSES)},
        "nc": len(DATASET_CLASSES),
    }
    yaml_path = processed_root / "waste.yaml"
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    project_yaml = project_root() / "waste.yaml"
    project_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def prepare_split(raw_split: Path, processed_root: Path, split_name: str) -> dict:
    coco_path = raw_split / "_annotations.coco.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"Missing {coco_path}")
    images_dir = processed_root / "images" / split_name
    labels_dir = processed_root / "labels" / split_name
    copied = copy_images(raw_split, images_dir)
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    lookup = coco_class_lookup(coco, remap_taco=False)
    stats = convert_coco_file(coco_path, images_dir, labels_dir, lookup)
    return {"copied_images": copied, "class_lookup": lookup, **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO-format waste dataset.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=data_dir() / "raw" / "garbage-classification-3",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=data_dir() / "processed" / "garbage-classification-3",
    )
    args = parser.parse_args()
    args.processed_root.mkdir(parents=True, exist_ok=True)
    report = {}
    mapping = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
    for folder in args.raw_root.iterdir():
        if not folder.is_dir():
            continue
        split_name = mapping.get(folder.name)
        if not split_name:
            continue
        report[split_name] = prepare_split(folder, args.processed_root, split_name)
    yaml_path = write_data_yaml(args.processed_root)
    report["yaml"] = str(yaml_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
