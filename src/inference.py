from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from src.categories import DATASET_CLASSES
from src.model_loader import load_detection_model
from src.validation import load_image


def detections_from_result(result: Any) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    names = getattr(result, "names", None) or {}
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    xyxy = boxes.xyxy.cpu().tolist() if getattr(boxes, "xyxy", None) is not None else []
    confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else []
    clss = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else []
    for index, (box, conf, cls_id) in enumerate(zip(xyxy, confs, clss)):
        class_id = int(cls_id)
        class_name = str(names.get(class_id, DATASET_CLASSES[class_id] if class_id < len(DATASET_CLASSES) else class_id))
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name.lower(),
                "confidence": float(conf),
                "box": [float(v) for v in box],
                "index": index,
            }
        )
    return detections


def run_inference(
    image: Union[str, Path, Image.Image],
    *,
    model: Any = None,
    weights_path: Optional[Path] = None,
    conf: float = 0.15,
    iou: float = 0.45,
    imgsz: int = 416,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run object detection. Bounding boxes are returned for aggregation only."""
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = load_image(image)

    detector = model or load_detection_model(weights_path)
    started = time.perf_counter()
    predict_kwargs: Dict[str, Any] = {
        "source": pil_image,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device
    results = detector.predict(**predict_kwargs)
    latency_ms = (time.perf_counter() - started) * 1000
    result = results[0] if results else None
    detections = detections_from_result(result) if result is not None else []
    return {
        "detections": detections,
        "latency_ms": round(latency_ms, 2),
        "image_size": pil_image.size,
    }
