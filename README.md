# Smart Waste Segregation and Recycling Assistant

AI-assisted waste segregation for mixed photographs. A lightweight YOLO detector finds multiple supported materials in one image. Streamlit shows a category summary, counts, confidence, and recycling guidance. Bounding boxes are used internally and are not drawn in the UI.

Predictions are AI-generated and can be wrong. This project is not municipal guidance and is not government-endorsed.

## Project objective

Users upload ordinary photos that may contain several waste objects: bottles with paper, cans with packaging, outdoor litter, or a mixed pile on a table. The application returns every supported category it can confirm, not a single image-level class.

Supported application categories:

- Plastic
- Paper
- Cardboard
- Metal
- Glass
- Organic Waste (`biodegradable` in the training labels)

## System architecture

See [docs/architecture.md](docs/architecture.md).

- Streamlit: interface
- Ultralytics YOLO: local object detection
- `src/analysis.py`: category aggregation
- `src/recommendations.py`: recycling guidance independent of YOLO
- SQLite: local history
- Cloudinary: optional image storage

## Dataset selection

Inspected candidates:

| Dataset | Mixed scenes | Classes | Access | License |
| --- | --- | --- | --- | --- |
| TACO | Strong (litter in context) | 60 fine-grained | Flickr URLs, often incomplete | Annotations CC BY 4.0 |
| GARBAGE CLASSIFICATION 3 | Mixed capability depends on source photos; mosaic used in training | 6 material classes | Public Hugging Face mirror, no API key | CC BY 4.0 |

**Selected training dataset:** GARBAGE CLASSIFICATION 3 via [keremberke/garbage-object-detection](https://huggingface.co/datasets/keremberke/garbage-object-detection).

Reasons:

- Object-detection boxes, not single-label classification
- Classes already match the application
- Existing train / validation / test split
- Downloadable without a Roboflow API key
- CC BY 4.0

TACO is a better wild-scene dataset, but 60 sparse classes and broken image URLs make it a weaker primary training source for a CPU/Colab mini-project. Mapping tables for TACO are still in `src/categories.py` if you add it later.

Run `python scripts/inspect_dataset.py` after download. The measured counts overwrite [docs/dataset_report.md](docs/dataset_report.md).

The supplied archive is present in this workspace and has been inspected. The original splits have major class-coverage gaps (glass is absent from test; paper has only 33 validation objects). Preprocessing and file/label validation pass, but do not use these split-level metrics to make a full six-class performance claim. For a final model evaluation, first prepare a reviewed, stratified holdout with adequate examples for every supported category, while keeping it separate from training and threshold selection.

### Class mapping

| Dataset class | UI category |
| --- | --- |
| plastic | Plastic |
| paper | Paper |
| cardboard | Cardboard |
| metal | Metal |
| glass | Glass |
| biodegradable | Organic Waste |

Paper and cardboard stay separate. There is no trained "other" class; unsupported objects are handled as uncertain at inference time.

## Training procedure

Fine-tune a nano YOLO checkpoint (`yolo11n.pt` or `yolov8n.pt` depending on Ultralytics). Do not claim scratch training unless you pass `--no-pretrained`.

Windows (dataset prep and optional short CPU runs):

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/inspect_dataset.py
python scripts/prepare_stratified_dataset.py --seed 42 --phash-distance 4
python scripts/validate_dataset.py --processed-root data/processed/garbage-classification-3-stratified-v2
python scripts/train.py --epochs 50 --imgsz 640
python scripts/tune_threshold.py
python scripts/evaluate.py --split test
```

CPU training is slow. Prefer the Colab notebook for the accuracy-focused run.

### Google Colab GPU

1. Upload this repository or copy `notebooks/train_waste_model.ipynb`.
2. Enable GPU: Runtime → Change runtime type → T4 (or better).
3. Run every cell in order.
4. Download `best.pt` from the notebook’s `models/` folder into this project’s `models/best.pt`.

The notebook rebuilds and validates the stratified split, records hardware and duration, and evaluates the held-out test set only after training. Dataset samples and measured counts are in [docs/dataset_report.md](docs/dataset_report.md).

## Evaluation methodology

- Checkpoint selection uses **validation** metrics and early stopping.
- Final numbers use the **held-out test** split.
- Reported detection metrics: precision, recall, mAP@0.50, mAP@0.50:0.95, per-class AP when Ultralytics provides them.
- Image-level category presence treats a class as present if at least one accepted detection of that class appears.
- Confusion matrices and curves are written under `runs/` by Ultralytics when evaluation runs.
- Confidence default is 0.35 until `tune_threshold.py` selects another value on validation F1.

Do not call mAP “accuracy”. Current numbers live in [docs/evaluation_report.md](docs/evaluation_report.md) only after a real eval run. The default training and inference size is 640; Ultralytics letterboxes images to preserve the input aspect ratio. The supplied dataset images are already 416×416, so their original proportions cannot be recovered from that export. Keep newly collected photos at their native dimensions. See [docs/real_world_data.md](docs/real_world_data.md) for preparing mixed-scene data and a separate real-world holdout.

## Run the application locally (Windows)

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 or 3.12 is recommended. The default `python` on this machine may be 3.14, which is a poor match for current PyTorch wheels.

If `models/best.pt` is missing, the app explains how to train. It will not quietly load a COCO general-purpose model.

## Cloud setup

1. Copy `.env.example` to `.env`.
2. For image storage, create a free Cloudinary account and set `CLOUDINARY_*` plus `STORE_IMAGES=true`.
3. Leave `DATABASE_URL` empty to use local SQLite at `data/history.db`.
4. Optional Streamlit Community Cloud: put the same keys in `.streamlit/secrets.toml` (never commit it).

Without credentials the analyzer still runs. Streamlit secrets are read for hosted deployments; local `.env` values take precedence. Set `DATABASE_URL` to a PostgreSQL URL and install the listed PostgreSQL driver for remote prediction history. Failed cloud uploads are reported as failures, not as success.

## Tests

```bat
.venv\Scripts\activate
pytest -q
```

## Repository layout

Matches the mini-project structure: `app.py`, `src/`, `scripts/`, `notebooks/`, `tests/`, `docs/`, and `models/`.

The dataset itself is not committed.

## Known limitations

See [docs/limitations.md](docs/limitations.md).

## Future improvements

- Add TACO mixed-scene images after a reliable image dump is available
- Stronger handling of film plastics vs rigid plastics
- A small manually reviewed mixed-photo test set from the intended campus environment
- Optional municipal-rule lookup if a local dataset is supplied
