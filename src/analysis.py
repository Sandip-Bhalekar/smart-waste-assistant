from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

from src.categories import DATASET_CLASSES, display_name
from src.recommendations import get_guidance, recommendation_lines


def _safe_conf(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def aggregate_detections(
    detections: Iterable[Dict[str, Any]],
    *,
    confidence_threshold: float = 0.35,
    uncertain_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Turn raw YOLO detections into a category-wise analysis.

    Bounding-box coordinates are ignored for the returned payload so the UI
    can show a clean summary without drawing boxes.
    """
    if uncertain_threshold is None:
        uncertain_threshold = max(0.15, confidence_threshold - 0.15)

    accepted: List[Dict[str, Any]] = []
    uncertain: List[Dict[str, Any]] = []
    ignored = 0

    for det in detections:
        cls_name = str(det.get("class_name") or "").strip().lower()
        conf = _safe_conf(det.get("confidence"))
        if cls_name not in DATASET_CLASSES:
            uncertain.append(
                {
                    "class_name": cls_name or "unknown",
                    "display_name": display_name(cls_name) if cls_name else "Unsupported",
                    "confidence": round(conf, 4),
                    "reason": "Unsupported or unmapped class.",
                }
            )
            continue
        if conf >= confidence_threshold:
            accepted.append({"class_name": cls_name, "confidence": conf})
        elif conf >= uncertain_threshold:
            uncertain.append(
                {
                    "class_name": cls_name,
                    "display_name": display_name(cls_name),
                    "confidence": round(conf, 4),
                    "reason": "Confidence below the validated reporting threshold.",
                }
            )
        else:
            ignored += 1

    grouped: Dict[str, List[float]] = defaultdict(list)
    for item in accepted:
        grouped[item["class_name"]].append(item["confidence"])

    categories: List[Dict[str, Any]] = []
    for class_name in DATASET_CLASSES:
        scores = grouped.get(class_name)
        if not scores:
            continue
        guide = get_guidance(class_name)
        categories.append(
            {
                "name": display_name(class_name),
                "class_key": class_name,
                "count": len(scores),
                "mean_confidence": round(mean(scores), 4),
                "max_confidence": round(max(scores), 4),
                "min_confidence": round(min(scores), 4),
                "short_description": guide["short_description"],
                "recommendations": recommendation_lines(class_name),
                "guidance": guide,
            }
        )

    if accepted:
        status = "success"
        summary = (
            f"Detected {len(accepted)} object(s) across {len(categories)} supported categor"
            f"{'y' if len(categories) == 1 else 'ies'}."
        )
    elif uncertain:
        status = "uncertain"
        summary = (
            "No category met the reporting threshold. Review uncertain predictions "
            "or upload a clearer photo."
        )
    else:
        status = "empty"
        summary = (
            "No supported waste objects were detected. This does not prove the image "
            "contains no waste; the model may have insufficient evidence."
        )

    checklist: List[Dict[str, Any]] = []
    for cat in categories:
        for step in cat["guidance"]["checklist"]:
            checklist.append({"category": cat["name"], "task": step})

    return {
        "total_detected_objects": len(accepted),
        "total_categories": len(categories),
        "categories": categories,
        "uncertain_predictions": uncertain,
        "ignored_low_confidence": ignored,
        "status": status,
        "summary": summary,
        "checklist": checklist,
        "confidence_threshold": confidence_threshold,
        "uncertain_threshold": uncertain_threshold,
        "absent_categories_note": (
            "A missing category means the model did not find enough evidence for it. "
            "It is not a guarantee that the material is absent."
        ),
    }
