# Evaluation Report

No test-set metrics have been written yet. This file is updated by `python scripts/evaluate.py` after training produces `models/best.pt`.

Until that command succeeds, do not quote an accuracy, mAP, precision, or recall number for this project.

No trained waste-specific checkpoint is currently present in `models/`, so evaluation and inference against the learned detector have not been completed. The local dataset's supplied test split contains no glass annotations and very few cardboard/biodegradable annotations; the validation split contains only 33 paper annotations. Use a reviewed stratified holdout before claiming comprehensive six-class results.

Object-detection mAP is not classification accuracy. Image-level category presence is a separate measurement defined in `scripts/evaluate.py`.
