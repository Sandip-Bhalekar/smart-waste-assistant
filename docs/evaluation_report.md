# Evaluation Report

These metrics come from an actual evaluation run. If a field is null, that measurement was not produced by this environment.

- Weights: `models\best.pt`
- Dataset config: `data\processed\garbage-classification-3-stratified-v2\waste.yaml`
- Split evaluated: `test`
- Confidence threshold used for image-level category presence: `0.35`

## Object-detection metrics

These are detection metrics, not ordinary classification accuracy.

- Precision: 0.6954970158111339
- Recall: 0.5034492552680252
- mAP@0.50: 0.5843602982250031
- mAP@0.50:0.95: 0.4094208127740083

## Image-level category presence

A category is counted as present if at least one accepted detection of that class appears in the image.

- Images scored: 1045
- Precision: 0.8159
- Recall: 0.8391
- Mean latency (ms): 36.76

## Notes

- mAP measures bounding-box detection quality.
- Precision / recall at image level measure whether a material category is present.
- Classification accuracy is not reported because this is a multi-object detector.
- Do not treat a missing category as proof that the material is absent.
