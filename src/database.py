from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.paths import data_dir


def _sqlite_path() -> Path:
    return data_dir() / "history.db"


def uses_postgres() -> bool:
    url = os.getenv("DATABASE_URL", "").strip()
    return url.startswith("postgres")


def _connect_sqlite() -> sqlite3.Connection:
    conn = sqlite3.connect(_sqlite_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    if uses_postgres():
        _init_postgres()
        return
    with _connect_sqlite() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                total_objects INTEGER NOT NULL,
                total_categories INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                image_url TEXT
            )
            """
        )
        conn.commit()


def _init_postgres() -> None:
    import psycopg2

    url = os.getenv("DATABASE_URL", "")
    parsed = urlparse(url)
    with psycopg2.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id SERIAL PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_objects INTEGER NOT NULL,
                    total_categories INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    image_url TEXT
                )
                """
            )
        conn.commit()
    del parsed


def save_analysis(payload: Dict[str, Any], image_url: Optional[str] = None) -> int:
    init_db()
    created_at = datetime.now(timezone.utc).isoformat()
    status = payload.get("status", "unknown")
    total_objects = int(payload.get("total_detected_objects", 0))
    total_categories = int(payload.get("total_categories", 0))
    model_version = str(payload.get("model_version", "unknown"))
    body = json.dumps(payload, ensure_ascii=True)
    if uses_postgres():
        import psycopg2

        with psycopg2.connect(os.environ["DATABASE_URL"]) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO analyses
                    (created_at, status, total_objects, total_categories, model_version, payload_json, image_url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (created_at, status, total_objects, total_categories, model_version, body, image_url),
                )
                row_id = int(cur.fetchone()[0])
            conn.commit()
        return row_id
    with _connect_sqlite() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyses
            (created_at, status, total_objects, total_categories, model_version, payload_json, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (created_at, status, total_objects, total_categories, model_version, body, image_url),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_analyses(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    query = "SELECT * FROM analyses ORDER BY id DESC LIMIT ?"
    if uses_postgres():
        import psycopg2
        import psycopg2.extras

        with psycopg2.connect(os.environ["DATABASE_URL"]) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query.replace("?", "%s"), (limit,))
                rows = cur.fetchall()
        return [dict(row) for row in rows]
    with _connect_sqlite() as conn:
        rows = conn.execute(query, (limit,)).fetchall()
        return [dict(row) for row in rows]


def analytics_summary() -> Dict[str, Any]:
    records = list_analyses(limit=1000)
    category_counts: Dict[str, int] = {}
    for row in records:
        payload = json.loads(row["payload_json"])
        for cat in payload.get("categories", []):
            name = cat.get("name", "Unknown")
            category_counts[name] = category_counts.get(name, 0) + int(cat.get("count", 0))
    most_frequent = None
    if category_counts:
        most_frequent = max(category_counts.items(), key=lambda item: item[1])[0]
    return {
        "total_analyses": len(records),
        "most_frequent_category": most_frequent,
        "category_distribution": category_counts,
        "recent": records[:20],
        "backend": "postgres" if uses_postgres() else "sqlite",
    }
