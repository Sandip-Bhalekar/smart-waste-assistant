from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.database import analytics_summary  # noqa: E402
from src.ui_theme import APP_CSS  # noqa: E402

st.set_page_config(page_title="History · Smart Waste", page_icon="📊", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown("### Prediction history")
st.caption("Statistics come only from analyses saved on this machine or the configured database.")

try:
    summary = analytics_summary()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not read history: {exc}")
    st.stop()

cols = st.columns(3)
cols[0].metric("Saved analyses", summary["total_analyses"])
cols[1].metric("Most frequent category", summary["most_frequent_category"] or "—")
cols[2].metric("Backend", summary["backend"])

distribution = summary["category_distribution"]
if distribution:
    chart_data = pd.DataFrame(
        {"category": list(distribution.keys()), "objects": list(distribution.values())}
    ).set_index("category")
    st.bar_chart(chart_data)
else:
    st.info("No saved analyses yet. Run an analysis on the home page to populate this view.")

rows = []
for item in summary["recent"]:
    payload = json.loads(item["payload_json"])
    names = ", ".join(cat["name"] for cat in payload.get("categories", [])) or "—"
    rows.append(
        {
            "id": item["id"],
            "created_at": item["created_at"],
            "status": item["status"],
            "objects": item["total_objects"],
            "categories": names,
            "model": item["model_version"],
        }
    )
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
