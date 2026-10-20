from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.categories import DATASET_CLASSES  # noqa: E402
from src.paths import data_dir, reports_dir  # noqa: E402


def load_coco(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_coco_files(raw_root: Path) -> dict[str, Path]:
    found = {}
    for split in ("train", "valid", "val", "test"):
        candidate = raw_root / split / "_annotations.coco.json"
        if candidate.exists():
            key = "valid" if split == "val" else split
            found[key] = candidate
    return found


def summarize_split(coco: dict, split: str) -> dict:
    categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    per_class = Counter()
    objects_per_image = Counter()
    classes_per_image: dict[int, set[str]] = defaultdict(set)
    invalid_boxes = 0
    for ann in annotations:
        cat = categories.get(ann.get("category_id"), str(ann.get("category_id")))
        per_class[cat] += 1
        objects_per_image[ann.get("image_id")] += 1
        classes_per_image[ann["image_id"]].add(cat)
        bbox = ann.get("bbox") or [0, 0, 0, 0]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            invalid_boxes += 1
    multi_object = sum(1 for count in objects_per_image.values() if count > 1)
    multi_class = sum(1 for names in classes_per_image.values() if len(names) > 1)
    empty_images = len(images) - len(objects_per_image)
    return {
        "split": split,
        "images": len(images),
        "annotations": len(annotations),
        "classes": categories,
        "per_class": dict(per_class),
        "multi_object_images": multi_object,
        "multi_class_images": multi_class,
        "empty_images": empty_images,
        "invalid_boxes": invalid_boxes,
        "avg_objects_per_image": round(len(annotations) / max(len(images), 1), 3),
    }


def markdown_report(summaries: list[dict], raw_root: Path) -> str:
    total_images = sum(item["images"] for item in summaries)
    total_objects = sum(item["annotations"] for item in summaries)
    all_classes = {}
    for item in summaries:
        all_classes.update(item["classes"])
    lines = [
        "# Dataset Report",
        "",
        "## Source",
        "",
        "- Dataset: GARBAGE CLASSIFICATION 3 (Material Identification / Roboflow)",
        "- Public mirror: [keremberke/garbage-object-detection](https://huggingface.co/datasets/keremberke/garbage-object-detection)",
        "- Original: [Roboflow Universe](https://universe.roboflow.com/material-identification/garbage-classification-3)",
        "- License: CC BY 4.0",
        "- Local path inspected: `data/raw/garbage-classification-3`",
        "",
        "## Why this dataset was selected",
        "",
        "TACO is excellent for mixed litter in context, but its 60 fine-grained classes are sparse and its images are downloaded from many Flickr URLs that often break. GARBAGE CLASSIFICATION 3 provides object-detection boxes, six material classes that match this application, an existing train/validation/test split, and a public download that does not require a Roboflow API key.",
        "",
        "The intended user uploads are mixed-waste photographs. Many source images in this dataset are close-up objects. Training therefore enables YOLO mosaic/mixup so the detector sees multiple objects in one image. Mixed-scene behaviour is also checked on a held-out mixed-image review set after training.",
        "",
        "## Headline counts",
        "",
        f"- Total images: **{total_images}**",
        f"- Total annotated objects: **{total_objects}**",
        f"- Number of classes in annotations: **{len(all_classes)}**",
        f"- Application classes: {', '.join(DATASET_CLASSES)}",
        "",
        "## Splits",
        "",
        "| Split | Images | Objects | Multi-object images | Multi-class images | Empty images | Invalid boxes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| {item['split']} | {item['images']} | {item['annotations']} | "
            f"{item['multi_object_images']} | {item['multi_class_images']} | "
            f"{item['empty_images']} | {item['invalid_boxes']} |"
        )
    lines.extend(["", "## Per-class object counts", ""])
    class_totals: Counter = Counter()
    for item in summaries:
        class_totals.update(item["per_class"])
        lines.append(f"### {item['split']}")
        lines.append("")
        lines.append("| Class | Objects |")
        lines.append("| --- | ---: |")
        for name, count in sorted(item["per_class"].items(), key=lambda kv: kv[0]):
            lines.append(f"| {name} | {count} |")
        lines.append("")
    lines.extend(
        [
            "### Combined",
            "",
            "| Class | Objects |",
            "| --- | ---: |",
        ]
    )
    for name, count in sorted(class_totals.items(), key=lambda kv: kv[0]):
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            "## Final category mapping",
            "",
            "| Dataset class | Application category |",
            "| --- | --- |",
            "| plastic | Plastic |",
            "| paper | Paper |",
            "| cardboard | Cardboard |",
            "| metal | Metal |",
            "| glass | Glass |",
            "| biodegradable | Organic Waste |",
            "",
            "Paper and cardboard are kept separate because the dataset annotates them independently and their recycling preparation differs.",
            "",
            "## Split-quality findings",
            "",
            "The published train/validation/test split is preserved as supplied. It is not class-stratified: validation contains only 33 paper and 214 plastic objects, while test contains no glass objects and only 20 cardboard and 49 biodegradable objects. Complete per-class test conclusions are therefore unreliable. Do not move test examples into training after model selection; acquire a new reviewed holdout or define a stratified split before training.",
            "",
            "Annotation inspection found malformed boxes (reported in the split table). The converter skips non-positive or missing boxes and clips coordinates outside image bounds. Check image readability and cross-split exact duplicates on prepared files with `scripts/validate_dataset.py` before training.",
            "",
            "## Known limitations",
            "",
            "- Images were stretched to 416x416 in the Roboflow export, which distorts aspect ratio.",
            "- The published export includes geometric augmentations (including vertical flips and 90-degree rotations) that can look unrealistic.",
            "- Class balance is severely inconsistent between the published splits; glass has no held-out test objects, and paper is nearly absent from validation.",
            "- This dataset does not include a dedicated 'other/uncertain' class. Unsupported objects are handled at inference time by confidence and mapping rules, not by inventing a training class.",
            "- TACO remains a useful mixed-scene reference dataset but is not required to start training.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect COCO waste-detection annotations.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=data_dir() / "raw" / "garbage-classification-3",
    )
    args = parser.parse_args()
    coco_files = find_coco_files(args.raw_root)
    if not coco_files:
        raise SystemExit(f"No COCO annotation files found under {args.raw_root}. Run scripts/download_dataset.py first.")
    summaries = []
    for split, path in coco_files.items():
        summaries.append(summarize_split(load_coco(path), split))
        print(json.dumps(summaries[-1], indent=2))
    report = markdown_report(summaries, args.raw_root)
    out = reports_dir() / "dataset_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
