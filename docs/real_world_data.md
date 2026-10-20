# Adding real mixed-scene photos

The existing six-class model was trained from a dataset export whose images are already 416×416. Resizing cannot restore the original aspect ratio or details that were lost during that export. New photos should stay at their native resolution and aspect ratio; YOLO letterboxes them during training and inference.

## Folder layout

Add permission-cleared photos and YOLO labels under:

```text
data/real-world/
  images/
    train/
    valid/
    test/
  labels/
    train/
    valid/
    test/
```

Each image must have a matching `.txt` label file with the same filename stem. A row is `class_id x_center y_center width height`, with normalized coordinates from 0 to 1. Class IDs are:

| ID | Class |
| --- | --- |
| 0 | biodegradable |
| 1 | cardboard |
| 2 | glass |
| 3 | metal |
| 4 | paper |
| 5 | plastic |

Annotate every visible supported waste item consistently. Keep paper and cardboard separate. An empty label file means the image contains no supported waste object; do not leave supported objects unlabelled. Unsupported objects have no class in this model and should not be assigned a made-up class.

Include mixed piles, overlapping items, small objects, different distances, indoor and outdoor backgrounds, and varied lighting. Put whole photo sessions and near-identical frames into only one split. Reserve `test` for the final evaluation; do not use it to tune the confidence threshold or choose a checkpoint. A few dozen photos can expose obvious failure patterns, but they are not enough to establish general accuracy.

## Validate and train

The root-level `waste-real-world.yaml` combines the existing training images with the new real-world training images. Validation and test point only to the real-world splits, so their scores measure the target use case.

```bat
python scripts/validate_dataset.py --processed-root data/real-world
python scripts/train.py --data waste-real-world.yaml --epochs 60 --imgsz 640 --name waste-mixed-scenes
python scripts/evaluate.py --weights models/best.pt --data waste-real-world.yaml --split test --imgsz 640
```

The evaluation report includes per-class metrics and up to 30 image-level mismatch examples, listing missed and false-positive categories. Ultralytics also saves box-level confusion matrices and curves in the run directory. Review the original photos and labels for each mismatch before adding corrected examples to training.

## Current project gaps

No manually reviewed real-world image set is included yet. This file provides the dataset layout and training configuration; real-world accuracy cannot improve until representative images are added, annotated, and the model is retrained on a GPU. The model has no `other` category, so low confidence is a review cue, not reliable open-set recognition.
