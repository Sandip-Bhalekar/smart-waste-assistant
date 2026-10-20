from src.recommendations import get_guidance, recommendation_lines


def test_known_category_contains_local_note():
    guide = get_guidance("plastic")
    assert guide["class_key"] == "plastic"
    assert guide["name"] == "Plastic"
    assert "local" in guide["local_variation_note"].lower()
    assert guide["checklist"]


def test_display_name_alias():
    guide = get_guidance("Organic Waste")
    assert guide["class_key"] == "biodegradable"
    assert "food" in guide["short_description"].lower()


def test_unknown_category_is_cautious():
    guide = get_guidance("battery-pack")
    assert guide["class_key"] == "uncertain"
    assert "residual" in " ".join(guide["disposal"]).lower()
    lines = recommendation_lines("not-a-class")
    assert lines
