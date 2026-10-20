# Evaluation Report

No test-set metrics have been written yet. This file is updated by `python scripts/evaluate.py` after training produces `models/best.pt`.

Until that command succeeds, do not quote an accuracy, mAP, precision, or recall number for this project.

No trained waste-specific checkpoint is currently present in `models/`, so evaluation and learned-model inference have not been completed. The original source splits had class-coverage gaps. A rebuilt, near-duplicate-screened 70/20/10 split is available at `data/processed/garbage-classification-3-stratified-v2`; every class appears in its validation and test sets. Use a manually reviewed mixed-waste collection for claims about real user-upload photos.

Object-detection mAP is not classification accuracy. Image-level category presence is a separate measurement defined in `scripts/evaluate.py`.
