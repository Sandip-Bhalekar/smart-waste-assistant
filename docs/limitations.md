# Limitations

- Object detection is not the same as classification accuracy. This project reports precision, recall, and mAP from a held-out detection evaluation. Those numbers are written only after `scripts/evaluate.py` has been run.
- A missing category in the results means the model did not find enough evidence. It does not prove the material is absent.
- Material recognition is not a determination that an item is legally recyclable. Local rules, contamination, and composite packaging matter more than the predicted class.
- The published GARBAGE CLASSIFICATION 3 export is resized to 416x416 with geometric augmentations, including vertical flips. That can distort objects and leak near-duplicates into training.
- Mixed, occluded, tiny, or poorly lit objects are harder than isolated close-ups.
- Batteries, electronics, chemicals, and other hazardous items are not trained classes. Do not recycle them from a guess.
- CPU-only laptops can run inference with YOLO-nano weights, but full-quality training should use a Colab GPU.
- Optional cloud history stores whatever the user saves. Do not upload private photographs if that is not allowed.
