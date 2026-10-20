# Dataset Report

## Source

- Dataset: GARBAGE CLASSIFICATION 3 (Material Identification / Roboflow)
- Public mirror: [keremberke/garbage-object-detection](https://huggingface.co/datasets/keremberke/garbage-object-detection)
- Original: [Roboflow Universe](https://universe.roboflow.com/material-identification/garbage-classification-3)
- License: CC BY 4.0
- Local path inspected: `data/raw/garbage-classification-3`

## Why this dataset was selected

TACO is excellent for mixed litter in context, but its 60 fine-grained classes are sparse and its images are downloaded from many Flickr URLs that often break. GARBAGE CLASSIFICATION 3 provides object-detection boxes, six material classes that match this application, an existing train/validation/test split, and a public download that does not require a Roboflow API key.

The intended user uploads are mixed-waste photographs. Many source images in this dataset are close-up objects. Training therefore enables YOLO mosaic/mixup so the detector sees multiple objects in one image. Mixed-scene behaviour is also checked on a held-out mixed-image review set after training.

## Headline counts

- Total images: **10464**
- Total annotated objects: **74090**
- Number of classes in annotations: **6**
- Application classes: biodegradable, cardboard, glass, metal, paper, plastic

## Splits

| Split | Images | Objects | Multi-object images | Multi-class images | Empty images | Invalid boxes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 7324 | 51611 | 3741 | 741 | 0 | 10 |
| valid | 2098 | 18916 | 1200 | 307 | 0 | 6 |
| test | 1042 | 3563 | 442 | 18 | 0 | 1 |

## Per-class object counts

### train

| Class | Objects |
| --- | ---: |
| biodegradable | 31721 |
| cardboard | 3386 |
| glass | 5429 |
| metal | 3948 |
| paper | 2981 |
| plastic | 4146 |

### valid

| Class | Objects |
| --- | ---: |
| biodegradable | 13637 |
| cardboard | 1292 |
| glass | 2380 |
| metal | 1360 |
| paper | 33 |
| plastic | 214 |

### test

| Class | Objects |
| --- | ---: |
| biodegradable | 49 |
| cardboard | 20 |
| metal | 533 |
| paper | 1376 |
| plastic | 1585 |

### Combined

| Class | Objects |
| --- | ---: |
| biodegradable | 45407 |
| cardboard | 4698 |
| glass | 7809 |
| metal | 5841 |
| paper | 4390 |
| plastic | 5945 |

## Final category mapping

| Dataset class | Application category |
| --- | --- |
| plastic | Plastic |
| paper | Paper |
| cardboard | Cardboard |
| metal | Metal |
| glass | Glass |
| biodegradable | Organic Waste |

Paper and cardboard are kept separate because the dataset annotates them independently and their recycling preparation differs.

## Split-quality findings

The published train/validation/test split is preserved as supplied. It is not class-stratified: validation contains only 33 paper and 214 plastic objects, while test contains no glass objects and only 20 cardboard and 49 biodegradable objects. Complete per-class test conclusions are therefore unreliable. Do not move test examples into training after model selection; acquire a new reviewed holdout or define a stratified split before training.

Annotation inspection found malformed boxes (reported in the split table). The converter skips non-positive or missing boxes and clips coordinates outside image bounds. Check image readability and cross-split exact duplicates on prepared files with `scripts/validate_dataset.py` before training.

## Known limitations

- Images were stretched to 416x416 in the Roboflow export, which distorts aspect ratio.
- The published export includes geometric augmentations (including vertical flips and 90-degree rotations) that can look unrealistic.
- Class balance is severely inconsistent between the published splits; glass has no held-out test objects, and paper is nearly absent from validation.
- This dataset does not include a dedicated 'other/uncertain' class. Unsupported objects are handled at inference time by confidence and mapping rules, not by inventing a training class.
- TACO remains a useful mixed-scene reference dataset but is not required to start training.
