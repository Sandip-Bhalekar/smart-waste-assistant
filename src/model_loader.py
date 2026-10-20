from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from src.paths import models_dir, project_root


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained waste-detection weights are missing."""


def default_model_path() -> Path:
    configured = os.getenv("MODEL_PATH", "models/best.pt")
    path = Path(configured)
    if not path.is_absolute():
        path = project_root() / path
    return path


def describe_missing_weights(path: Optional[Path] = None) -> str:
    target = path or default_model_path()
    return (
        f"Trained waste-detection weights were not found at '{target}'. "
        "This application does not silently substitute a general-purpose model. "
        "Train the waste model with `python scripts/train.py` after preparing the dataset, "
        "or run the Google Colab notebook in `notebooks/train_waste_model.ipynb` and copy "
        "`best.pt` into the `models/` folder."
    )


def load_detection_model(weights_path: Optional[Path] = None) -> Any:
    path = Path(weights_path) if weights_path else default_model_path()
    if not path.exists():
        raise ModelNotFoundError(describe_missing_weights(path))
    from ultralytics import YOLO

    return YOLO(str(path))


def list_available_checkpoints() -> list[Path]:
    return sorted(models_dir().glob("*.pt"))
