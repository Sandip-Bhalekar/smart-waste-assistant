from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image, UnidentifiedImageError

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_MAX_UPLOAD_MB = 8


class UploadError(ValueError):
    """Raised when an uploaded file cannot be used for inference."""


def validate_filename(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UploadError("Unsupported file format. Please upload a JPG, JPEG, or PNG image.")
    return suffix


def validate_file_size(num_bytes: int, max_mb: int = DEFAULT_MAX_UPLOAD_MB) -> None:
    if num_bytes <= 0:
        raise UploadError("The uploaded file is empty.")
    max_bytes = max_mb * 1024 * 1024
    if num_bytes > max_bytes:
        raise UploadError(f"The image is too large. Maximum size is {max_mb} MB.")


def load_image(source: Union[str, Path, bytes]) -> Image.Image:
    try:
        if isinstance(source, (str, Path)):
            image = Image.open(source)
        else:
            from io import BytesIO

            image = Image.open(BytesIO(source))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UploadError("The file could not be read as an image. It may be corrupted.") from exc
    if image.mode not in {"RGB", "RGBA", "L"}:
        image = image.convert("RGB")
    elif image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode == "L":
        image = image.convert("RGB")
    return image
