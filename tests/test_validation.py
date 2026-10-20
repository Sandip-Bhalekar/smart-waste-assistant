from io import BytesIO

import pytest
from PIL import Image

from src.validation import UploadError, load_image, validate_file_size, validate_filename


def test_accepts_jpeg_and_png_names():
    assert validate_filename("bottle.JPG") == ".jpg"
    assert validate_filename("can.png") == ".png"


def test_rejects_unsupported_format():
    with pytest.raises(UploadError):
        validate_filename("notes.pdf")


def test_file_size_limits():
    validate_file_size(1024, max_mb=1)
    with pytest.raises(UploadError):
        validate_file_size(2 * 1024 * 1024, max_mb=1)
    with pytest.raises(UploadError):
        validate_file_size(0)


def test_load_valid_and_corrupt_image(tmp_path):
    path = tmp_path / "ok.jpg"
    Image.new("RGB", (32, 32), color=(20, 80, 40)).save(path)
    image = load_image(path)
    assert image.mode == "RGB"
    assert image.size == (32, 32)

    with pytest.raises(UploadError):
        load_image(b"this is not an image")

    buffer = BytesIO()
    Image.new("RGBA", (16, 16), color=(10, 20, 30, 128)).save(buffer, format="PNG")
    converted = load_image(buffer.getvalue())
    assert converted.mode == "RGB"
