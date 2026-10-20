from __future__ import annotations

from typing import Dict, List


# Dataset class names as annotated in GARBAGE CLASSIFICATION 3 / Hugging Face mirror.
DATASET_CLASSES: List[str] = [
    "biodegradable",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]

DISPLAY_NAMES: Dict[str, str] = {
    "plastic": "Plastic",
    "paper": "Paper",
    "cardboard": "Cardboard",
    "metal": "Metal",
    "glass": "Glass",
    "biodegradable": "Organic Waste",
}

# TACO supercategory / category names mapped into the six application classes.
# Categories that cannot be mapped honestly are omitted from training labels.
TACO_TO_APP: Dict[str, str] = {
    "Plastic bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Other plastic bottle": "plastic",
    "Clear plastic bottle": "plastic",
    "Plastic bag": "plastic",
    "Garbage bag": "plastic",
    "Single-use carrier bag": "plastic",
    "Polypropylene bag": "plastic",
    "Plastic film": "plastic",
    "Other plastic wrapper": "plastic",
    "Crisp packet": "plastic",
    "Spread tub": "plastic",
    "Tupperware": "plastic",
    "Disposable plastic cup": "plastic",
    "Other plastic": "plastic",
    "Plastic glooves": "plastic",
    "Plastic utensils": "plastic",
    "Plastic straw": "plastic",
    "Plastic lid": "plastic",
    "Six pack rings": "plastic",
    "Styrofoam piece": "plastic",
    "Foam cup": "plastic",
    "Foam food container": "plastic",
    "Other plastic container": "plastic",
    "Plastic container": "plastic",
    "Squeezable tube": "plastic",
    "Disposable food container": "plastic",
    "Paper": "paper",
    "Paper bag": "paper",
    "Magazine paper": "paper",
    "Newspaper": "paper",
    "Wrapping paper": "paper",
    "Normal paper": "paper",
    "Paper cup": "paper",
    "Toilet tube": "cardboard",
    "Other carton": "cardboard",
    "Egg carton": "cardboard",
    "Drink carton": "cardboard",
    "Corrugated carton": "cardboard",
    "Meal carton": "cardboard",
    "Pizza box": "cardboard",
    "Cardboard": "cardboard",
    "Aluminium foil": "metal",
    "Can": "metal",
    "Aluminium blister pack": "metal",
    "Pop tab": "metal",
    "Scrap metal": "metal",
    "Aerosol": "metal",
    "Drink can": "metal",
    "Food Can": "metal",
    "Metal bottle cap": "metal",
    "Metal lid": "metal",
    "Glass bottle": "glass",
    "Broken glass": "glass",
    "Glass cup": "glass",
    "Glass jar": "glass",
    "Food waste": "biodegradable",
}

HAZARDOUS_TACO_CLASSES = {
    "Battery",
    "Battery (other)",
}


def dataset_class_id(name: str) -> int:
    try:
        return DATASET_CLASSES.index(name)
    except ValueError as exc:
        raise KeyError(f"Unknown dataset class: {name}") from exc


def display_name(class_name: str) -> str:
    return DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())


def canonical_name(label: str) -> str:
    lowered = label.strip().lower()
    if lowered in DATASET_CLASSES:
        return lowered
    reverse = {value.lower(): key for key, value in DISPLAY_NAMES.items()}
    if lowered in reverse:
        return reverse[lowered]
    aliases = {
        "organic": "biodegradable",
        "organic waste": "biodegradable",
        "bio": "biodegradable",
        "paper/cardboard": "paper",
    }
    if lowered in aliases:
        return aliases[lowered]
    raise KeyError(f"Cannot map label '{label}' to a supported class.")
