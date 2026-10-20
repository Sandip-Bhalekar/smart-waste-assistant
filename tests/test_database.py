import json

from src.database import analytics_summary, init_db, save_analysis


def test_local_sqlite_history(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("src.database.data_dir", lambda: tmp_path)
    init_db()
    payload = {
        "status": "success",
        "total_detected_objects": 2,
        "total_categories": 1,
        "model_version": "test",
        "categories": [{"name": "Plastic", "count": 2}],
    }
    row_id = save_analysis(payload)
    assert row_id >= 1
    summary = analytics_summary()
    assert summary["backend"] == "sqlite"
    assert summary["total_analyses"] == 1
    assert summary["most_frequent_category"] == "Plastic"
    saved = json.loads(summary["recent"][0]["payload_json"])
    assert saved["total_detected_objects"] == 2
