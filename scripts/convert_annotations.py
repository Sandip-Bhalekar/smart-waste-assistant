from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.categories import DATASET_CLASSES, TACO_TO_APP, dataset_class_id  # noqa: E402


def coco_box_to_yolo(bbox, width: int, height: int) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or width <= 0 or height <= 0:
        return None
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    nw = w / width
    nh = h / height
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < nw <= 1.5 and 0 < nh <= 1.5):
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        nw = min(max(nw, 1e-6), 1)
        nh = min(max(nh, 1e-6), 1)
    return x_center, y_center, nw, nh


def convert_coco_file(coco_path: Path, images_dir: Path, labels_dir: Path, class_lookup: dict[int, str]) -> dict:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in coco["images"]}
    skipped = 0
    written = 0
    labels_dir.mkdir(parents=True, exist_ok=True)
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    for image_id, image_info in images.items():
        file_name = Path(image_info["file_name"]).name
        image_path = images_dir / file_name
        if not image_path.exists():
            skipped += 1
            continue
        width = int(image_info.get("width") or 0)
        height = int(image_info.get("height") or 0)
        lines = []
        for ann in anns_by_image.get(image_id, []):
            class_name = class_lookup.get(ann.get("category_id"))
            if class_name not in DATASET_CLASSES:
                skipped += 1
                continue
            converted = coco_box_to_yolo(ann.get("bbox") or [0, 0, 0, 0], width, height)
            if converted is None:
                skipped += 1
                continue
            class_id = dataset_class_id(class_name)
            x, y, w, h = converted
            lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        label_path = labels_dir / f"{image_path.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    return {"written": written, "skipped": skipped}


def coco_class_lookup(coco: dict, remap_taco: bool = False) -> dict[int, str]:
    lookup = {}
    for item in coco["categories"]:
        name = item["name"]
        if remap_taco:
            mapped = TACO_TO_APP.get(name)
            if mapped:
                lookup[item["id"]] = mapped
        else:
            lowered = name.strip().lower()
            if lowered in DATASET_CLASSES:
                lookup[item["id"]] = lowered
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO detections to YOLO txt labels.")
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--taco", action="store_true", help="Remap TACO category names to application classes.")
    args = parser.parse_args()
    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    lookup = coco_class_lookup(coco, remap_taco=args.taco)
    stats = convert_coco_file(args.coco, args.images, args.labels, lookup)
    print(json.dumps({"class_lookup": lookup, **stats}, indent=2))


if __name__ == "__main__":
    main()
