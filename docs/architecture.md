# Architecture

## Runtime flow

1. The Streamlit UI accepts a JPG/PNG photograph and validates size, type, and image integrity.
2. A local Ultralytics YOLO detector runs on the image. Raw output includes boxes, class IDs, and confidences.
3. `src/analysis.py` drops boxes from the user-facing payload, filters by a validated confidence threshold, and counts objects per category.
4. `src/recommendations.py` attaches recycling guidance that does not depend on the detector.
5. Optional Cloudinary storage uploads the image only when `STORE_IMAGES=true` and credentials exist.
6. SQLite (default) or PostgreSQL stores the analysis JSON for the History page.

```
Upload image
    -> validation.py
    -> YOLO weights in models/best.pt
    -> analysis.py (category summary)
    -> recommendations.py
    -> Streamlit cards / export
    -> optional Cloudinary + history database
```

## Training flow

```
download_dataset.py
    -> inspect_dataset.py
    -> prepare_dataset.py (COCO to YOLO)
    -> validate_dataset.py
    -> train.py (fine-tune yolo11n / yolov8n)
    -> tune_threshold.py (validation only)
    -> evaluate.py (held-out test split)
    -> copy best.pt to models/
```

The test split is not used for checkpoint selection. `train.py` uses the validation split for early stopping.

## Design choices

- Object detection is required because mixed-waste photos can contain several materials.
- Mosaic augmentation is enabled so the model sees multiple objects in one training image even when many source photos are close-ups.
- Bounding boxes stay internal. The UI shows category cards, counts, and checklists.
- Cloud features are optional. Missing credentials must not block local inference.
