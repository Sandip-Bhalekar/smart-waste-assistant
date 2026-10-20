from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ui_theme import APP_CSS

st.set_page_config(page_title="About · Smart Waste", page_icon="ℹ️", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.markdown("### About this project")
st.write(
    "Smart Waste is a college mini-project that uses a YOLO object detector to find multiple "
    "waste materials in one photograph, then maps each supported category to recycling guidance."
)
st.markdown("**Technology stack**")
st.write("- Ultralytics YOLO for object detection")
st.write("- Streamlit for the user interface")
st.write("- SQLite for local prediction history")
st.write("- Optional Cloudinary image storage")
st.markdown("**Dataset**")
st.write(
    "Training uses GARBAGE CLASSIFICATION 3 (CC BY 4.0), a public object-detection dataset with "
    "plastic, paper, cardboard, metal, glass, and biodegradable labels. TACO was considered for "
    "mixed-scene diversity; see `docs/dataset_report.md`."
)
st.markdown("**Limitations**")
st.write("- The model can miss small, overlapping, or unusual objects.")
st.write("- Material recognition is not a legal determination of recyclability.")
st.write("- Local recycling rules vary and must be followed over this guidance.")
st.write("- Evaluation metrics are reported only after a real test-set run.")
readme = Path(__file__).resolve().parents[1] / "docs" / "limitations.md"
if readme.exists():
    st.markdown(readme.read_text(encoding="utf-8"))
