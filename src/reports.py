from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict


def analysis_export_payload(analysis: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "timestamp": analysis.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "model_version": analysis.get("model_version"),
        "status": analysis.get("status"),
        "total_detected_objects": analysis.get("total_detected_objects"),
        "total_categories": analysis.get("total_categories"),
        "confidence_threshold": analysis.get("confidence_threshold"),
        "categories": [
            {
                "name": cat["name"],
                "count": cat["count"],
                "mean_confidence": cat["mean_confidence"],
                "recommendations": cat["recommendations"],
            }
            for cat in analysis.get("categories", [])
        ],
        "uncertain_predictions": analysis.get("uncertain_predictions", []),
        "summary": analysis.get("summary"),
    }


def to_json_bytes(analysis: Dict[str, Any]) -> bytes:
    return json.dumps(analysis_export_payload(analysis), indent=2).encode("utf-8")


def to_csv_bytes(analysis: Dict[str, Any]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["timestamp", "model_version", "category", "count", "mean_confidence", "recommendations"])
    payload = analysis_export_payload(analysis)
    if payload["categories"]:
        for cat in payload["categories"]:
            writer.writerow(
                [
                    payload["timestamp"],
                    payload["model_version"],
                    cat["name"],
                    cat["count"],
                    cat["mean_confidence"],
                    " | ".join(cat["recommendations"]),
                ]
            )
    else:
        writer.writerow(
            [
                payload["timestamp"],
                payload["model_version"],
                "",
                0,
                "",
                payload.get("summary", ""),
            ]
        )
    return buffer.getvalue().encode("utf-8")
