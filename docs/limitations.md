# Limitations

- Object detection is not the same as classification accuracy. This project reports precision, recall, and mAP from a held-out detection evaluation. Those numbers are written only after `scripts/evaluate.py` has been run.
- A missing category in the results means the model did not find enough evidence. It does not prove the material is absent.
- Material recognition is not a determination that an item is legally recyclable. Local rules, contamination, and composite packaging matter more than the predicted class.
- The published GARBAGE CLASSIFICATION 3 export is resized to 416x416 with geometric augmentations, including vertical flips. That can distort objects and leak near-duplicates into training.
- Mixed, occluded, tiny, or poorly lit objects are harder than isolated close-ups.
- Batteries, electronics, chemicals, and other hazardous items are not trained classes. Do not recycle them from a guess.
- CPU-only laptops can run inference with YOLO-nano weights, but full-quality training should use a Colab GPU.
- Optional cloud history stores whatever the user saves. Do not upload private photographs if that is not allowed.
# Current project status and limitations

- The dataset archive was inspected and converted to YOLO format. Image readability, image/label pairing, class IDs, and exact duplicates across supplied splits passed validation.
- The source split is not suitable for complete six-class evaluation: glass has zero test annotations, and paper has only 33 validation annotations. Cardboard and biodegradable have 20 and 49 test annotations, respectively. Obtain or curate a stratified, independently reviewed holdout before making claims across all classes.
- Seventeen malformed source boxes were skipped during conversion (10 train, 6 validation, 1 test). Prepared labels contain no invalid class IDs or boxes.
- No trained `models/best.pt` checkpoint is included. The app displays setup instructions and does not substitute a generic detector. Run the Colab workflow after improving the evaluation split, then evaluate once on the untouched test set.
- The dataset has no explicit unknown class. Unknown objects may be missed or assigned to a supported class; the app's uncertainty messaging does not turn this into open-set recognition.
- Predictions identify visual material categories, not guaranteed recyclability. Local collection rules and contamination requirements vary.
