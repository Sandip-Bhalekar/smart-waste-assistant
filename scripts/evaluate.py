from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import aggregate_detections  # noqa: E402
from src.categories import DATASET_CLASSES, display_name  # noqa: E402
from src.inference import run_inference  # noqa: E402
from src.model_loader import default_model_path, load_detection_model  # noqa: E402
from src.paths import project_root, reports_dir  # noqa: E402


def coco_style_metrics(model, data_yaml: Path, split: str, imgsz: int, device: str) -> dict:
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        plots=True,
        verbose=False,
    )
    payload = {
        "precision": _metric(results, "mp"),
        "recall": _metric(results, "mr"),
        "mAP50": _metric(results, "map50"),
        "mAP50-95": _metric(results, "map"),
        "fit_time": getattr(results, "speed", {}),
        "save_dir": str(getattr(results, "save_dir", "")),
    }
    names = getattr(results, "names", {i: n for i, n in enumerate(DATASET_CLASSES)})
    maps = getattr(results, "maps", None)
    if maps is not None:
        payload["per_class_ap50_95"] = {
            names.get(i, DATASET_CLASSES[i] if i < len(DATASET_CLASSES) else str(i)): float(value)
            for i, value in enumerate(maps)
        }
    box = getattr(results, "box", None)
    if box is not None:
        payload["box"] = {
            "p": _tolist(getattr(box, "p", None)),
            "r": _tolist(getattr(box, "r", None)),
            "ap50": _tolist(getattr(box, "ap50", None)),
            "ap": _tolist(getattr(box, "ap", None)),
        }
    return payload


def _metric(results, attr: str):
    box = getattr(results, "box", None)
    if box is not None and hasattr(box, attr):
        value = getattr(box, attr)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    value = getattr(results, attr, None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tolist(value):
    if value is None:
        return None
    try:
        return [float(v) for v in value]
    except TypeError:
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)


def image_level_category_scores(model, images_dir: Path, labels_dir: Path, conf: float, imgsz: int) -> dict:
    if not images_dir.exists():
        return {"available": False}
    tp = fp = fn = 0
    per_class = {name: {"tp": 0, "fp": 0, "fn": 0} for name in DATASET_CLASSES}
    image_count = 0
    latencies = []
    samples = []
    error_samples = []
    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        expected = set()
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id = int(line.split()[0])
                if 0 <= class_id < len(DATASET_CLASSES):
                    expected.add(DATASET_CLASSES[class_id])
        started = time.perf_counter()
        raw = run_inference(image_path, model=model, conf=0.15, imgsz=imgsz)
        latencies.append((time.perf_counter() - started) * 1000)
        analysis = aggregate_detections(raw["detections"], confidence_threshold=conf)
        predicted = {item["class_key"] for item in analysis["categories"]}
        image_count += 1
        tp += len(expected & predicted)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        for name in DATASET_CLASSES:
            if name in predicted and name in expected:
                per_class[name]["tp"] += 1
            elif name in predicted:
                per_class[name]["fp"] += 1
            elif name in expected:
                per_class[name]["fn"] += 1
        if len(samples) < 12:
            samples.append(
                {
                    "image": image_path.name,
                    "expected": sorted(expected),
                    "predicted": sorted(predicted),
                    "status": "correct" if expected == predicted else "mismatch",
                }
            )
        if expected != predicted and len(error_samples) < 30:
            error_samples.append(
                {
                    "image": image_path.name,
                    "expected": sorted(expected),
                    "predicted": sorted(predicted),
                    "missed_categories": sorted(expected - predicted),
                    "false_positive_categories": sorted(predicted - expected),
                }
            )
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "available": True,
        "images": image_count,
        "image_level_precision": round(precision, 4),
        "image_level_recall": round(recall, 4),
        "per_class": {
            display_name(name): {
                **counts,
                "precision": round(counts["tp"] / max(counts["tp"] + counts["fp"], 1), 4),
                "recall": round(counts["tp"] / max(counts["tp"] + counts["fn"], 1), 4),
            }
            for name, counts in per_class.items()
        },
        "mean_latency_ms": round(sum(latencies) / max(len(latencies), 1), 2),
        "samples": samples,
        "error_samples": error_samples,
    }


def write_markdown(report: dict) -> Path:
    lines = [
        "# Evaluation Report",
        "",
        "These metrics come from an actual evaluation run. If a field is null, that measurement was not produced by this environment.",
        "",
        f"- Weights: `{report.get('weights')}`",
        f"- Dataset config: `{report.get('data')}`",
        f"- Split evaluated: `{report.get('split')}`",
        f"- Confidence threshold used for image-level category presence: `{report.get('confidence_threshold')}`",
        "",
        "## Object-detection metrics",
        "",
        "These are detection metrics, not ordinary classification accuracy.",
        "",
        f"- Precision: {report['detection'].get('precision')}",
        f"- Recall: {report['detection'].get('recall')}",
        f"- mAP@0.50: {report['detection'].get('mAP50')}",
        f"- mAP@0.50:0.95: {report['detection'].get('mAP50-95')}",
        "",
        "## Image-level category presence",
        "",
        "A category is counted as present if at least one accepted detection of that class appears in the image.",
        "",
        f"- Images scored: {report['image_level'].get('images')}",
        f"- Precision: {report['image_level'].get('image_level_precision')}",
        f"- Recall: {report['image_level'].get('image_level_recall')}",
        f"- Mean latency (ms): {report['image_level'].get('mean_latency_ms')}",
        "",
        "## Image-level error review",
        "",
    ]
    error_samples = report.get("image_level", {}).get("error_samples", [])
    if error_samples:
        lines.extend(
            [
                "Review these mismatches first. They are category-presence errors, not box-level false-positive/false-negative counts.",
                "",
                "| Image | Missed categories | False-positive categories |",
                "| --- | --- | --- |",
            ]
        )
        for sample in error_samples:
            lines.append(
                f"| `{sample['image']}` | {', '.join(sample['missed_categories']) or '—'} "
                f"| {', '.join(sample['false_positive_categories']) or '—'} |"
            )
    else:
        lines.append("No mismatch samples were recorded. Rerun `scripts/evaluate.py` to generate a review queue.")
    lines.extend([
        "",
        "## Notes",
        "",
        "- mAP measures bounding-box detection quality.",
        "- Precision / recall at image level measure whether a material category is present.",
        "- Classification accuracy is not reported because this is a multi-object detector.",
        "- Do not treat a missing category as proof that the material is absent.",
        "",
    ])
    out = reports_dir() / "evaluation_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    json_path = reports_dir() / "evaluation_metrics.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained waste detector on a held-out split.")
    parser.add_argument("--weights", type=Path, default=default_model_path())
    parser.add_argument("--data", type=Path, default=project_root() / "waste.yaml")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()
    if not args.weights.exists():
        raise SystemExit(f"Missing weights at {args.weights}")
    if not args.data.exists():
        raise SystemExit(f"Missing data yaml at {args.data}")

    from scripts.train import detect_device

    device = args.device or detect_device()
    model = load_detection_model(args.weights)
    detection = coco_style_metrics(model, args.data, args.split, args.imgsz, device)
    import yaml

    data_cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(data_cfg["path"])
    split_dir = "valid" if args.split in {"val", "valid"} else args.split
    image_level = image_level_category_scores(
        model,
        root / "images" / split_dir,
        root / "labels" / split_dir,
        conf=args.conf,
        imgsz=args.imgsz,
    )
    size_mb = round(args.weights.stat().st_size / (1024 * 1024), 3)
    report = {
        "weights": str(args.weights),
        "model_size_mb": size_mb,
        "data": str(args.data),
        "split": args.split,
        "device": device,
        "confidence_threshold": args.conf,
        "detection": detection,
        "image_level": image_level,
    }
    markdown = write_markdown(report)
    print(json.dumps(report, indent=2, default=str))
    print(f"Wrote {markdown}")


if __name__ == "__main__":
    main()
