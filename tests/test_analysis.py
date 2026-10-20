from src.analysis import aggregate_detections


def test_aggregates_multiple_categories():
    detections = [
        {"class_name": "plastic", "confidence": 0.91},
        {"class_name": "plastic", "confidence": 0.88},
        {"class_name": "plastic", "confidence": 0.80},
        {"class_name": "cardboard", "confidence": 0.77},
        {"class_name": "cardboard", "confidence": 0.70},
        {"class_name": "metal", "confidence": 0.84},
    ]
    result = aggregate_detections(detections, confidence_threshold=0.35)
    assert result["status"] == "success"
    assert result["total_detected_objects"] == 6
    names = {item["class_key"]: item["count"] for item in result["categories"]}
    assert names == {"plastic": 3, "cardboard": 2, "metal": 1}
    assert "box" not in result
    assert result["categories"][0]["recommendations"]


def test_confidence_filter_and_uncertain_band():
    detections = [
        {"class_name": "glass", "confidence": 0.62},
        {"class_name": "paper", "confidence": 0.28},
        {"class_name": "metal", "confidence": 0.05},
    ]
    result = aggregate_detections(detections, confidence_threshold=0.35, uncertain_threshold=0.20)
    assert result["total_detected_objects"] == 1
    assert result["categories"][0]["class_key"] == "glass"
    assert result["uncertain_predictions"][0]["class_name"] == "paper"
    assert result["ignored_low_confidence"] == 1


def test_empty_detections():
    result = aggregate_detections([], confidence_threshold=0.35)
    assert result["status"] == "empty"
    assert result["total_detected_objects"] == 0
    assert result["categories"] == []


def test_unmapped_class_is_uncertain():
    result = aggregate_detections(
        [{"class_name": "shoe", "confidence": 0.99}],
        confidence_threshold=0.35,
    )
    assert result["status"] == "uncertain"
    assert result["total_detected_objects"] == 0
    assert result["uncertain_predictions"][0]["reason"].startswith("Unsupported")
