import pytest

from src.cloud_storage import cloudinary_configured, upload_image
from src.inference import detections_from_result
from src.model_loader import ModelNotFoundError, describe_missing_weights, load_detection_model


class FakeBoxes:
    def __init__(self):
        class _T:
            def __init__(self, data):
                self._data = data

            def cpu(self):
                return self

            def tolist(self):
                return self._data

        self.xyxy = _T([[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 8.0, 16.0]])
        self.conf = _T([0.9, 0.4])
        self.cls = _T([5.0, 1.0])


class FakeResult:
    names = {0: "biodegradable", 1: "cardboard", 5: "plastic"}
    boxes = FakeBoxes()


def test_detections_from_result_maps_class_names():
    detections = detections_from_result(FakeResult())
    assert len(detections) == 2
    assert detections[0]["class_name"] == "plastic"
    assert detections[1]["class_name"] == "cardboard"
    assert detections[0]["confidence"] == pytest.approx(0.9)


def test_missing_weights_message(tmp_path, monkeypatch):
    missing = tmp_path / "best.pt"
    monkeypatch.setenv("MODEL_PATH", str(missing))
    with pytest.raises(ModelNotFoundError):
        load_detection_model(missing)
    text = describe_missing_weights(missing)
    assert "train" in text.lower()
    assert "best.pt" in text


def test_cloud_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("CLOUDINARY_CLOUD_NAME", raising=False)
    monkeypatch.delenv("CLOUDINARY_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDINARY_API_SECRET", raising=False)
    monkeypatch.setenv("STORE_IMAGES", "true")
    assert cloudinary_configured() is False
    result = upload_image(b"fake", "a.jpg")
    assert result.uploaded is False
    assert "not configured" in result.error.lower()


def test_image_storage_opt_in(monkeypatch):
    monkeypatch.setenv("STORE_IMAGES", "false")
    result = upload_image(b"fake", "a.jpg")
    assert result.uploaded is False
    assert result.enabled is False
