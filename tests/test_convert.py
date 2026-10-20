from scripts.convert_annotations import coco_box_to_yolo, coco_class_lookup


def test_coco_box_to_yolo_center():
    box = coco_box_to_yolo([50, 40, 20, 10], width=100, height=100)
    assert box is not None
    x, y, w, h = box
    assert abs(x - 0.6) < 1e-6
    assert abs(y - 0.45) < 1e-6
    assert abs(w - 0.2) < 1e-6
    assert abs(h - 0.1) < 1e-6


def test_invalid_box_rejected():
    assert coco_box_to_yolo([0, 0, 0, 10], 100, 100) is None


def test_class_lookup_filters_unknown():
    coco = {
        "categories": [
            {"id": 1, "name": "plastic"},
            {"id": 2, "name": "unknown-thing"},
        ]
    }
    lookup = coco_class_lookup(coco)
    assert lookup == {1: "plastic"}
