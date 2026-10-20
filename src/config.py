from __future__ import annotations

import os
from typing import Any


APP_SECRET_KEYS = (
    "CLOUDINARY_CLOUD_NAME",
    "CLOUDINARY_API_KEY",
    "CLOUDINARY_API_SECRET",
    "CLOUDINARY_FOLDER",
    "STORE_IMAGES",
    "DATABASE_URL",
    "MODEL_PATH",
    "CONFIDENCE_THRESHOLD",
    "IOU_THRESHOLD",
    "MAX_UPLOAD_MB",
    "INFERENCE_IMAGE_SIZE",
)


def load_streamlit_secrets(secrets: Any) -> None:
    """Expose configured Streamlit secrets to provider modules using environment variables.

    Existing process environment values take precedence, so local .env settings remain
    straightforward to override in development.
    """
    try:
        for key in APP_SECRET_KEYS:
            value = secrets.get(key)
            if value is not None and str(value).strip():
                os.environ.setdefault(key, str(value))
    except Exception:
        # Streamlit raises when no secrets file/service is configured; that is valid
        # for the fully local, cloud-disabled setup.
        return
