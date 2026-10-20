from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.categories import DATASET_CLASSES, dataset_class_id  # noqa: E402
from src.paths import data_dir  # noqa: E402


SPLITS = ("train", "valid", "test")
RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _phash_variants(path: Path) -> int:
    with Image.open(path) as source:
        gray = np.asarray(source.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    transforms = [gray, np.rot90(gray), np.rot90(gray, 2), np.rot90(gray, 3)]
    hashes = []
    for image in transforms:
        coeff = cv2.dct(np.float32(image))[:8, :8].flatten()
        median = float(np.median(coeff[1:]))
        value = 0
        for bit in coeff:
            value = (value << 1) | int(bit > median)
        hashes.append(value)
    # Rotation-invariant representative; the Roboflow export includes 90-degree turns.
    return min(hashes)


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def group_near_duplicates(records: list[dict], max_distance: int = 4) -> tuple[list[list[int]], list[dict]]:
    """Group perceptual duplicates across all supplied splits before partitioning.

    Rotation-invariant hashes are indexed in a BK-tree. Candidate pairs are
    confirmed with a full 64-bit Hamming comparison; the threshold is deliberately
    conservative to reduce accidental grouping of different items.
    """
    hashes: list[int] = []
    for index, record in enumerate(records, start=1):
        path = record["path"]
        try:
            hashes.append(_phash_variants(path))
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Cannot read source image for perceptual hash: {path}") from exc
        if index % 1000 == 0:
            print(f"Perceptual hashes: {index}/{len(records)}", flush=True)

    uf = UnionFind(len(records))
    pairs = []
    tree: dict[int, dict] = {}

    def query(value: int) -> list[tuple[int, int]]:
        if not tree:
            return []
        matches = []
        stack = [tree]
        while stack:
            node = stack.pop()
            distance = _hamming(value, node["value"])
            if distance <= max_distance:
                matches.append((node["owner"], distance))
            low, high = distance - max_distance, distance + max_distance
            stack.extend(child for edge, child in node["children"].items() if low <= edge <= high)
        return matches

    def insert(value: int, owner: int) -> None:
        if not tree:
            tree.update({"value": value, "owner": owner, "children": {}})
            return
        node = tree
        while True:
            distance = _hamming(value, node["value"])
            if distance == 0:
                uf.union(node["owner"], owner)
                return
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = {"value": value, "owner": owner, "children": {}}
                return
            node = child

    for image_index, value in enumerate(hashes):
        found: dict[int, int] = {}
        for owner, distance in query(value):
            if owner != image_index:
                found[owner] = min(distance, found.get(owner, distance))
        insert(value, image_index)
        for owner, distance in found.items():
            uf.union(owner, image_index)
            pairs.append({"left": owner, "right": image_index, "hamming_distance": distance})
        if (image_index + 1) % 1000 == 0:
            print(f"Near-duplicate index: {image_index + 1}/{len(records)}", flush=True)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[uf.find(index)].append(index)
    return list(groups.values()), pairs


def _load_records(raw_root: Path) -> list[dict]:
    records = []
    for source_split in SPLITS:
        split_dir = raw_root / source_split
        coco_path = split_dir / "_annotations.coco.json"
        if not coco_path.exists():
            raise FileNotFoundError(f"Missing COCO annotations: {coco_path}")
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        categories = {item["id"]: item["name"].strip().lower() for item in coco["categories"]}
        images = {item["id"]: item for item in coco["images"]}
        labels: dict[int, set[str]] = defaultdict(set)
        boxes: dict[int, list[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            name = categories.get(ann.get("category_id"))
            if name not in DATASET_CLASSES:
                continue
            bbox = ann.get("bbox") or []
            if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            labels[ann["image_id"]].add(name)
            boxes[ann["image_id"]].append({"class": name, "bbox": bbox})
        for image_id, info in images.items():
            path = split_dir / Path(info["file_name"]).name
            if not path.is_file():
                raise FileNotFoundError(f"Image referenced by annotation is missing: {path}")
            records.append({
                "source_split": source_split,
                "file_name": path.name,
                "path": path,
                "width": int(info["width"]),
                "height": int(info["height"]),
                "labels": labels.get(image_id, set()),
                "boxes": boxes.get(image_id, []),
            })
    return records


def make_representative_sheet(records: list[dict], destination: Path) -> None:
    selected = {}
    for record in records:
        for label in record["labels"]:
            selected.setdefault(label, []).append(record)
    cell_w, image_h, caption_h = 240, 170, 30
    canvas = Image.new("RGB", (cell_w * 3, (image_h + caption_h) * 2), "#f4f7f5")
    draw = ImageDraw.Draw(canvas)
    for position, label in enumerate(DATASET_CLASSES):
        candidates = selected.get(label)
        if not candidates:
            continue
        candidates.sort(key=lambda item: sum(box["class"] == label for box in item["boxes"]))
        record = candidates[len(candidates) // 2]
        image = Image.open(record["path"]).convert("RGB")
        image.thumbnail((cell_w - 12, image_h - 8))
        x, y = (position % 3) * cell_w, (position // 3) * (image_h + caption_h)
        offset = (x + 6, y + caption_h)
        canvas.paste(image, offset)
        scale_x, scale_y = image.width / record["width"], image.height / record["height"]
        palette = {"plastic": "#f04d42", "paper": "#3777c8", "cardboard": "#8f5cc2", "metal": "#e29b16", "glass": "#159b75", "biodegradable": "#d23888"}
        for item in record["boxes"]:
            bx, by, bw, bh = item["bbox"]
            color = palette[item["class"]]
            rect = (offset[0] + bx * scale_x, offset[1] + by * scale_y,
                    offset[0] + (bx + bw) * scale_x, offset[1] + (by + bh) * scale_y)
            draw.rectangle(rect, outline=color, width=1)
        class_counts = Counter(item["class"] for item in record["boxes"])
        caption = ", ".join(f"{name}:{count}" for name, count in sorted(class_counts.items()))
        draw.text((x + 7, y + 7), f"median example - {caption}", fill="#173b2d")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=88)


def assign_grouped_splits(
    records: list[dict], groups: list[list[int]], seed: int = 42
) -> dict[str, list[int]]:
    """Assign duplicate groups with a deterministic multilabel greedy stratifier."""
    totals = Counter(label for record in records for label in record["labels"])
    group_data = []
    rng = random.Random(seed)
    for group in groups:
        counts = Counter(label for index in group for label in records[index]["labels"])
        group_data.append((group, counts, rng.random()))
    group_data.sort(key=lambda item: (min((totals[label] for label in item[1]), default=10**9), -len(item[0]), item[2]))

    target = {split: {label: totals[label] * ratio for label in DATASET_CLASSES} for split, ratio in RATIOS.items()}
    assigned_counts = {split: Counter() for split in RATIOS}
    assigned_sizes = Counter()
    assigned = {split: [] for split in RATIOS}
    target_sizes = {split: len(records) * ratio for split, ratio in RATIOS.items()}
    for group, counts, _ in group_data:
        options = []
        for split, ratio in RATIOS.items():
            class_gain = sum(
                (min(counts[label], max(0.0, target[split][label] - assigned_counts[split][label])) / max(totals[label], 1))
                for label in counts
            )
            size_penalty = max(0.0, assigned_sizes[split] + len(group) - target_sizes[split]) / max(target_sizes[split], 1)
            options.append((class_gain - 0.25 * size_penalty, -assigned_sizes[split] / max(target_sizes[split], 1), split))
        split = max(options)[2]
        assigned[split].extend(group)
        assigned_counts[split].update(counts)
        assigned_sizes[split] += len(group)

    # Improve close-to-target ratios by moving the smallest groups greedily is avoided:
    # every component remains indivisible, preserving the no-near-duplicate guarantee.
    return assigned


def _write_dataset(records: list[dict], assignment: dict[str, list[int]], output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output already exists and is not empty: {output}")
    stats = {}
    for split, indices in assignment.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        counts = Counter()
        for index in indices:
            record = records[index]
            destination = image_dir / record["file_name"]
            shutil.copy2(record["path"], destination)
            lines = []
            for item in record["boxes"]:
                x, y, width, height = item["bbox"]
                iw, ih = record["width"], record["height"]
                if iw <= 0 or ih <= 0:
                    continue
                # Clip at image borders, then drop collapsed boxes.
                left, top = max(0.0, x), max(0.0, y)
                right, bottom = min(float(iw), x + width), min(float(ih), y + height)
                if right <= left or bottom <= top:
                    continue
                center_x, center_y = (left + right) / (2 * iw), (top + bottom) / (2 * ih)
                box_w, box_h = (right - left) / iw, (bottom - top) / ih
                class_id = dataset_class_id(item["class"])
                lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}")
                counts[item["class"]] += 1
            (label_dir / f"{destination.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats[split] = {"images": len(indices), "objects": dict(counts)}

    config = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(DATASET_CLASSES)},
        "nc": len(DATASET_CLASSES),
    }
    (output / "waste.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        config["path"] = output.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        pass
    (ROOT / "waste.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-checked, multilabel-stratified YOLO dataset.")
    parser.add_argument("--raw-root", type=Path, default=data_dir() / "raw" / "garbage-classification-3")
    parser.add_argument("--output", type=Path, default=data_dir() / "processed" / "garbage-classification-3-stratified-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phash-distance", type=int, default=4)
    args = parser.parse_args()

    existing_report_path = args.output / "split_report.json"
    if existing_report_path.exists():
        existing = json.loads(existing_report_path.read_text(encoding="utf-8"))
        actual_counts = {
            split: sum(1 for path in (args.output / "images" / split).glob("*") if path.is_file())
            for split in RATIOS
        }
        if (
            existing.get("seed") == args.seed
            and existing.get("phash_hamming_threshold") == args.phash_distance
            and actual_counts == existing.get("split_images")
        ):
            print(f"Reusing validated split at {args.output} ({actual_counts} images).", flush=True)
            return
        raise FileExistsError(
            f"Output exists with different or incomplete split settings: {args.output}. "
            "Choose a new --output directory to preserve the existing data."
        )

    records = _load_records(args.raw_root)
    print(f"Loaded {len(records)} images from the three source splits.", flush=True)
    make_representative_sheet(records, ROOT / "docs" / "dataset_examples.jpg")
    groups, pairs = group_near_duplicates(records, max_distance=args.phash_distance)
    assignment = assign_grouped_splits(records, groups, seed=args.seed)
    split_of = {index: split for split, indices in assignment.items() for index in indices}
    leakage = [pair for pair in pairs if split_of[pair["left"]] != split_of[pair["right"]]]
    if leakage:
        raise RuntimeError(f"Near-duplicate leakage across splits: {len(leakage)} matched pairs")
    stats = _write_dataset(records, assignment, args.output)
    report = {
        "source": "GARBAGE CLASSIFICATION 3; original source splits merged before a new split",
        "seed": args.seed,
        "ratios_requested": RATIOS,
        "phash_method": "minimum of 64-bit DCT perceptual hashes over four rotations (rotation-invariant for the provided export)",
        "phash_hamming_threshold": args.phash_distance,
        "near_duplicate_groups": sum(1 for group in groups if len(group) > 1),
        "near_duplicate_pairs": pairs,
        "cross_split_near_duplicate_pairs": len(leakage),
        "group_count": len(groups),
        "split_stats": stats,
        "split_images": {split: len(indices) for split, indices in assignment.items()},
        "class_image_presence": {
            split: dict(Counter(label for index in indices for label in records[index]["labels"]))
            for split, indices in assignment.items()
        },
        "source_split_distribution": {
            split: dict(Counter(records[index]["source_split"] for index in indices))
            for split, indices in assignment.items()
        },
    }
    (args.output / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "near_duplicate_pairs"}, indent=2))


if __name__ == "__main__":
    main()
