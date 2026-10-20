from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate import image_level_category_scores  # noqa: E402
from src.model_loader import default_model_path, load_detection_model  # noqa: E402
from src.paths import project_root, reports_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune confidence threshold on the validation split.")
    parser.add_argument("--weights", type=Path, default=default_model_path())
    parser.add_argument("--data", type=Path, default=project_root() / "waste.yaml")
    parser.add_argument("--imgsz", type=int, default=416)
    args = parser.parse_args()
    if not args.weights.exists():
        raise SystemExit(f"Missing weights at {args.weights}")
    model = load_detection_model(args.weights)
    data_cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(data_cfg["path"])
    rows = []
    for conf in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        scores = image_level_category_scores(
            model,
            root / "images" / "valid",
            root / "labels" / "valid",
            conf=conf,
            imgsz=args.imgsz,
        )
        precision = scores.get("image_level_precision") or 0
        recall = scores.get("image_level_recall") or 0
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append({"conf": conf, "precision": precision, "recall": recall, "f1": round(f1, 4)})
        print(rows[-1])
    best = max(rows, key=lambda item: item["f1"])
    payload = {
        "rows": rows,
        "selected": best,
        "rationale": (
            "The default reporting threshold is selected on validation image-level F1. "
            "Higher thresholds reduce false categories at the cost of missed materials."
        ),
    }
    out = reports_dir() / "threshold_tuning.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Selected threshold {best['conf']} with F1={best['f1']}")


if __name__ == "__main__":
    main()
