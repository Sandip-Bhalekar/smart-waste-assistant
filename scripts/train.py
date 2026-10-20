from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import data_dir, models_dir, project_root, reports_dir  # noqa: E402


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


def choose_model_name() -> str:
    # Prefer a lightweight YOLO checkpoint compatible with the installed Ultralytics package.
    candidates = ["yolo11n.pt", "yolov8n.pt"]
    return candidates[0]


def hardware_info(device: str) -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": device,
        "gpu_name": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if device != "cpu" and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    return info


def copy_best_weights(run_dir: Path) -> Path | None:
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        return None
    destination = models_dir() / "best.pt"
    shutil.copy2(best, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a lightweight YOLO waste detector.")
    parser.add_argument("--data", type=Path, default=project_root() / "waste.yaml")
    parser.add_argument("--model", type=str, default=None, help="Ultralytics checkpoint or YAML, e.g. yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--name", type=str, default="waste-yolo")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "detect")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Dataset config not found: {args.data}. Run scripts/prepare_dataset.py first.")

    set_seeds(args.seed)
    device = args.device or detect_device()
    if args.batch is None:
        args.batch = 16 if device != "cpu" else 4
    model_name = args.model or choose_model_name()

    from ultralytics import YOLO

    started = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    print(json.dumps({"event": "train_start", "model": model_name, **hardware_info(device)}, indent=2))

    if args.pretrained:
        model = YOLO(model_name)
    else:
        cfg = model_name.replace(".pt", ".yaml")
        print(
            "Training from random initialization. Expect lower accuracy and a much longer "
            "training schedule than fine-tuning."
        )
        model = YOLO(cfg)

    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        patience=args.patience,
        workers=args.workers,
        pretrained=args.pretrained,
        optimizer="AdamW",
        lr0=0.001 if device != "cpu" else 0.0005,
        lrf=0.01,
        mosaic=0.5,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=5.0,
        translate=0.08,
        scale=0.3,
        fliplr=0.5,
        flipud=0.0,
        close_mosaic=10,
        plots=True,
        val=True,
    )

    run_dir = Path(results.save_dir)
    best = copy_best_weights(run_dir)
    duration_sec = round(time.time() - started, 2)
    metrics = {}
    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    if args_yaml.exists():
        recorded_args = yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
    else:
        recorded_args = vars(args)
        recorded_args = {k: str(v) for k, v in recorded_args.items()}

    summary = {
        "started_at": started_at,
        "duration_sec": duration_sec,
        "hardware": hardware_info(device),
        "model_name": model_name,
        "pretrained": args.pretrained,
        "data": str(args.data),
        "run_dir": str(run_dir),
        "best_weights": str(best) if best else None,
        "recorded_args": recorded_args,
        "results_csv": str(results_csv) if results_csv.exists() else None,
        "note": (
            "Validation metrics were used for checkpoint selection. "
            "The held-out test set must be evaluated separately with scripts/evaluate.py."
        ),
    }
    out = reports_dir() / "training_run.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if best:
        print(f"Copied best weights to {best}")
    else:
        print("WARNING: best.pt was not produced.")


if __name__ == "__main__":
    main()
