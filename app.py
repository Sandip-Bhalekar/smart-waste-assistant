from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from src.config import load_streamlit_secrets  # noqa: E402

load_streamlit_secrets(st.secrets)

from src import MODEL_VERSION  # noqa: E402
from src.analysis import aggregate_detections  # noqa: E402
from src.cloud_storage import upload_image  # noqa: E402
from src.database import save_analysis  # noqa: E402
from src.inference import run_inference  # noqa: E402
from src.model_loader import ModelNotFoundError, default_model_path, load_detection_model  # noqa: E402
from src.reports import to_csv_bytes, to_json_bytes  # noqa: E402
from src.ui_theme import APP_CSS  # noqa: E402
from src.validation import UploadError, load_image, validate_file_size, validate_filename  # noqa: E402

st.set_page_config(
    page_title="Smart Waste",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "8"))
DEFAULT_CONF = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
IOU = float(os.getenv("IOU_THRESHOLD", "0.45"))
INFERENCE_IMAGE_SIZE = int(os.getenv("INFERENCE_IMAGE_SIZE", "640"))
EXAMPLES_DIR = ROOT / "assets" / "examples"


@st.cache_resource(show_spinner=False)
def get_model():
    return load_detection_model()


def reset_session() -> None:
    for key in ["uploaded_name", "image_bytes", "analysis", "raw_detections", "busy"]:
        st.session_state.pop(key, None)


def example_images() -> list[Path]:
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(
        [path for path in EXAMPLES_DIR.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )


with st.sidebar:
    st.markdown("### Smart Waste")
    st.caption("AI-assisted waste segregation")
    st.page_link("app.py", label="Analyze", icon="♻️")
    st.page_link("pages/1_History.py", label="History", icon="📊")
    st.page_link("pages/2_About.py", label="About", icon="ℹ️")
    st.divider()
    conf = st.slider(
        "Reporting confidence",
        min_value=0.15,
        max_value=0.75,
        value=float(st.session_state.get("conf", DEFAULT_CONF)),
        step=0.05,
        help="Validated default is 0.35 unless threshold tuning has selected another value.",
    )
    st.session_state["conf"] = conf
    st.caption("Lower values show more categories and more false positives. Higher values are stricter.")
    st.caption(
        "Low-confidence candidates appear under Uncertain results. A missing category does not mean "
        "that material is absent; the model cannot reliably recognize unsupported objects."
    )
    model_path = default_model_path()
    st.markdown(f"**Model file:** `{model_path.name}`")
    st.markdown("Available" if model_path.exists() else "Missing — train or copy `models/best.pt`")

st.markdown(
    """
    <div class="sw-hero">
      <div class="sw-chip">AI-assisted waste segregation • YOLO object detection</div>
      <h1>Smart Waste</h1>
      <p>Upload a photograph of mixed household or street waste. The assistant identifies supported materials and turns them into a segregation checklist. It does not draw boxes on your photo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="sw-kicker">Analyze Your Waste</div>', unsafe_allow_html=True)
st.subheader("Upload a photograph with one or more waste items")
st.write(
    "Clear photos of bottles, cans, paper, cardboard, glass, and food waste work best. "
    "The detector looks for multiple objects in the same image rather than assigning one label to the whole photo."
)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    uploaded = st.file_uploader(
        "Drag and drop a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        key="uploader",
    )
    examples = example_images()
    example_choice = None
    if examples:
        example_choice = st.selectbox(
            "Or try an example image",
            options=["—"] + [path.name for path in examples],
        )
    button_row = st.columns([1, 1, 2])
    analyze = button_row[0].button(
        "Analyze Waste",
        type="primary",
        disabled=st.session_state.get("busy", False),
        use_container_width=True,
    )
    if button_row[1].button("Clear Image", use_container_width=True):
        reset_session()
        st.rerun()

image_bytes = None
filename = None
if uploaded is not None:
    image_bytes = uploaded.getvalue()
    filename = uploaded.name
elif example_choice and example_choice != "—":
    path = EXAMPLES_DIR / example_choice
    image_bytes = path.read_bytes()
    filename = path.name

if image_bytes and filename:
    st.session_state["image_bytes"] = image_bytes
    st.session_state["uploaded_name"] = filename

preview_bytes = st.session_state.get("image_bytes")
preview_name = st.session_state.get("uploaded_name")

with right:
    st.markdown('<div class="sw-card">', unsafe_allow_html=True)
    st.markdown("**Image preview**")
    if preview_bytes:
        st.image(preview_bytes, use_container_width=True, caption=preview_name)
    else:
        st.markdown(
            '<div class="sw-empty">No image yet. Upload a mixed-waste photo or choose an example.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

if analyze:
    if not preview_bytes or not preview_name:
        st.warning("Upload an image before running analysis.")
    else:
        st.session_state["busy"] = True
        try:
            validate_filename(preview_name)
            validate_file_size(len(preview_bytes), MAX_UPLOAD_MB)
            image = load_image(preview_bytes)
            with st.status("Working…", expanded=True) as status:
                st.write("Preparing image.")
                model = get_model()
                st.write("Analyzing waste.")
                raw = run_inference(
                    image,
                    model=model,
                    conf=0.15,
                    iou=IOU,
                    imgsz=INFERENCE_IMAGE_SIZE,
                )
                st.write("Generating segregation guidance.")
                analysis = aggregate_detections(raw["detections"], confidence_threshold=conf)
                analysis["model_version"] = MODEL_VERSION
                analysis["timestamp"] = datetime.now(timezone.utc).isoformat()
                analysis["latency_ms"] = raw["latency_ms"]
                analysis["source_filename"] = preview_name
                cloud = upload_image(preview_bytes, preview_name)
                analysis["cloud"] = {
                    "uploaded": cloud.uploaded,
                    "url": cloud.url,
                    "error": cloud.error,
                    "enabled": cloud.enabled,
                }
                try:
                    save_analysis(analysis, image_url=cloud.url)
                    analysis["history_saved"] = True
                except Exception as exc:  # noqa: BLE001
                    analysis["history_saved"] = False
                    analysis["history_error"] = str(exc)
                st.session_state["analysis"] = analysis
                st.session_state["raw_detections"] = raw["detections"]
                status.update(label="Analysis complete", state="complete")
        except UploadError as exc:
            st.error(str(exc))
        except ModelNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Inference failed: {exc}")
        finally:
            st.session_state["busy"] = False

analysis = st.session_state.get("analysis")
if analysis:
    st.markdown("### Results")
    metrics = st.columns(4)
    metrics[0].markdown(
        f'<div class="sw-metric"><b>{analysis["total_detected_objects"]}</b><span>Detected objects</span></div>',
        unsafe_allow_html=True,
    )
    metrics[1].markdown(
        f'<div class="sw-metric"><b>{analysis["total_categories"]}</b><span>Supported categories</span></div>',
        unsafe_allow_html=True,
    )
    metrics[2].markdown(
        f'<div class="sw-metric"><b>{analysis["confidence_threshold"]:.2f}</b><span>Reporting threshold</span></div>',
        unsafe_allow_html=True,
    )
    metrics[3].markdown(
        f'<div class="sw-metric"><b>{analysis.get("latency_ms", "—")}</b><span>Inference ms</span></div>',
        unsafe_allow_html=True,
    )
    st.write(analysis["summary"])
    st.caption(analysis["absent_categories_note"])

    result_tabs = st.tabs(["Categories", "Guidance", "Checklist", "Uncertain results", "Export"])
    with result_tabs[0]:
        if not analysis["categories"]:
            st.markdown(
                '<div class="sw-empty">No supported category crossed the reporting threshold. Try a closer photo, better lighting, or a slightly lower threshold during development.</div>',
                unsafe_allow_html=True,
            )
        else:
            cards = st.columns(min(3, len(analysis["categories"])))
            for index, cat in enumerate(analysis["categories"]):
                with cards[index % 3]:
                    st.markdown('<div class="sw-card">', unsafe_allow_html=True)
                    st.markdown(f"### {cat['name']}")
                    st.write(cat["short_description"])
                    st.write(f"**{cat['count']}** object(s)")
                    st.write(
                        f"Confidence: mean {cat['mean_confidence']:.2f}, "
                        f"range {cat['min_confidence']:.2f}–{cat['max_confidence']:.2f}"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
    with result_tabs[1]:
        for cat in analysis["categories"]:
            guide = cat["guidance"]
            with st.expander(f"{cat['name']} — recycling and disposal", expanded=True):
                st.write(guide["short_description"])
                st.markdown("**Preparation**")
                for item in guide["preparation"]:
                    st.write(f"- {item}")
                st.markdown("**Recycling**")
                for item in guide["recycling"]:
                    st.write(f"- {item}")
                st.markdown("**Disposal**")
                for item in guide["disposal"]:
                    st.write(f"- {item}")
                st.markdown("**Contamination warnings**")
                for item in guide["contamination_warnings"]:
                    st.write(f"- {item}")
                st.info(guide["local_variation_note"])
        if not analysis["categories"]:
            st.info("Guidance appears after a supported category is detected.")
    with result_tabs[2]:
        if analysis["checklist"]:
            for item in analysis["checklist"]:
                st.checkbox(f"{item['category']}: {item['task']}", key=f"chk-{item['category']}-{item['task']}")
        else:
            st.write("No checklist yet.")
    with result_tabs[3]:
        if analysis["uncertain_predictions"]:
            st.markdown(
                '<div class="sw-warn">These detections were not treated as confirmed categories.</div>',
                unsafe_allow_html=True,
            )
            for item in analysis["uncertain_predictions"]:
                st.write(
                    f"- {item['display_name']} · confidence {item['confidence']:.2f} · {item['reason']}"
                )
        else:
            st.write("No uncertain predictions were retained above the review band.")
        st.caption(
            "Low-confidence results can happen with blur, crowding, unusual objects, or materials outside the six trained classes."
        )
    with result_tabs[4]:
        json_bytes = to_json_bytes(analysis)
        csv_bytes = to_csv_bytes(analysis)
        st.download_button("Download JSON report", data=json_bytes, file_name="waste-analysis.json", mime="application/json")
        st.download_button("Download CSV report", data=csv_bytes, file_name="waste-analysis.csv", mime="text/csv")
        if analysis.get("cloud", {}).get("uploaded"):
            st.success("Image stored in configured cloud storage.")
        elif analysis.get("cloud", {}).get("error"):
            st.caption(analysis["cloud"]["error"])

st.markdown(
    """
    <div class="sw-footer">
      <b>Smart Waste Segregation and Recycling Assistant</b><br/>
      College mini-project using YOLO object detection, Streamlit, and optional Cloudinary/SQLite history.
      Predictions are AI-generated and can be wrong. This tool does not replace municipal recycling rules
      and is not endorsed by any government agency. Dataset: GARBAGE CLASSIFICATION 3 (CC BY 4.0).
    </div>
    """,
    unsafe_allow_html=True,
)
