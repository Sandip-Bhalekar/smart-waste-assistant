from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CloudResult:
    enabled: bool
    uploaded: bool
    url: Optional[str] = None
    public_id: Optional[str] = None
    error: Optional[str] = None


def cloudinary_configured() -> bool:
    return all(
        [
            os.getenv("CLOUDINARY_CLOUD_NAME"),
            os.getenv("CLOUDINARY_API_KEY"),
            os.getenv("CLOUDINARY_API_SECRET"),
        ]
    )


def store_images_enabled() -> bool:
    return os.getenv("STORE_IMAGES", "false").lower() in {"1", "true", "yes"}


def upload_image(file_bytes: bytes, filename: str) -> CloudResult:
    if not store_images_enabled():
        return CloudResult(enabled=False, uploaded=False, error="Image storage is disabled.")
    if not cloudinary_configured():
        return CloudResult(
            enabled=False,
            uploaded=False,
            error="Cloud storage is not configured. Set Cloudinary credentials in .env or Streamlit secrets.",
        )
    try:
        import cloudinary
        import cloudinary.uploader
    except ImportError as exc:
        return CloudResult(enabled=False, uploaded=False, error=f"Cloudinary package is not installed: {exc}")

    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True,
    )
    folder = os.getenv("CLOUDINARY_FOLDER", "smart-waste-assistant")
    try:
        response: Dict[str, Any] = cloudinary.uploader.upload(
            file_bytes,
            folder=folder,
            resource_type="image",
            filename=filename,
            use_filename=True,
            unique_filename=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface provider errors to the UI
        return CloudResult(enabled=True, uploaded=False, error=f"Cloud upload failed: {exc}")
    return CloudResult(
        enabled=True,
        uploaded=True,
        url=response.get("secure_url"),
        public_id=response.get("public_id"),
    )
