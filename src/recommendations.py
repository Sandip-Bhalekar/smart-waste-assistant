from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from src.categories import canonical_name, display_name


GUIDANCE: Dict[str, Dict[str, Any]] = {
    "plastic": {
        "short_description": "Plastic bottles, cups, wrappers, and rigid plastic packaging.",
        "preparation": [
            "Empty leftover liquids or food from containers.",
            "Rinse when residue would contaminate a recycling stream.",
            "Flatten bottles only if it helps storage and is accepted locally.",
        ],
        "recycling": [
            "Check the recycling symbol and your municipality's accepted plastic types.",
            "Separate caps and labels only when local rules ask you to.",
            "Keep different plastics together only if your collection program allows mixed plastics.",
        ],
        "disposal": [
            "Place accepted clean plastics in the recycling bin.",
            "Put non-accepted plastic film, multi-layer wrappers, or heavily soiled plastics in residual waste unless a store take-back exists.",
        ],
        "contamination_warnings": [
            "Food-soiled plastic can contaminate a whole recycling batch.",
            "Do not assume every plastic package is recyclable.",
        ],
        "local_variation_note": "Accepted resin types (PET, HDPE, PP, film) vary by city and collection contractor.",
        "checklist": [
            "Empty the container",
            "Rinse if needed",
            "Confirm local plastic rules",
            "Keep film and rigid plastics separate unless told otherwise",
        ],
    },
    "paper": {
        "short_description": "Clean paper, newspaper, magazines, and paper bags.",
        "preparation": [
            "Keep paper dry.",
            "Remove plastic windows, tape, or non-paper packing when practical.",
        ],
        "recycling": [
            "Place clean, dry paper in paper recycling.",
            "Shredded paper is accepted in some programs and rejected in others.",
        ],
        "disposal": [
            "Recycle clean paper.",
            "Compost uncoated food-soiled paper only if your organics program accepts it; otherwise use residual waste.",
        ],
        "contamination_warnings": [
            "Greasy or wet paper often cannot be recycled as fiber.",
            "Thermal receipts and laminated paper may need residual waste.",
        ],
        "local_variation_note": "Some cities collect paper with cardboard; others require separate streams.",
        "checklist": [
            "Keep paper clean and dry",
            "Remove non-paper attachments",
            "Divert greasy paper if required",
        ],
    },
    "cardboard": {
        "short_description": "Corrugated boxes, cartons, and cardboard packaging.",
        "preparation": [
            "Flatten boxes.",
            "Remove plastic inserts, foam, and excessive tape when practical.",
            "Keep cardboard dry.",
        ],
        "recycling": [
            "Recycle clean, dry cardboard with paper or cardboard collection.",
            "Beverage cartons are accepted in some programs and not in others.",
        ],
        "disposal": [
            "Recycling is preferred for clean cardboard.",
            "Pizza boxes with heavy grease stains may belong in organics or residual waste depending on local rules.",
        ],
        "contamination_warnings": [
            "Wet or oily cardboard has lower fiber value and may be rejected.",
        ],
        "local_variation_note": "Cartons (tetra-style) are treated as cardboard in some cities and as residual waste in others.",
        "checklist": [
            "Flatten boxes",
            "Remove plastic inserts",
            "Keep cardboard dry",
        ],
    },
    "metal": {
        "short_description": "Drink cans, food tins, aerosol cans, and other metal packaging.",
        "preparation": [
            "Empty the container.",
            "Rinse food cans when residue would contaminate collection.",
            "Do not crush sharp tins in a way that creates a hazard.",
        ],
        "recycling": [
            "Place empty metal cans in metal or mixed-recycling collection.",
            "Keep scrap metal separate if your city has a scrap stream.",
        ],
        "disposal": [
            "Most empty household cans are recyclable.",
            "Aerosols, paint cans, and pressurized containers may need household-hazardous-waste drop-off.",
        ],
        "contamination_warnings": [
            "Do not recycle cans that still contain chemicals, paint, or fuel.",
            "Foil with heavy food residue may be rejected.",
        ],
        "local_variation_note": "Some programs accept foil and trays; others accept only cans.",
        "checklist": [
            "Empty the can",
            "Rinse if needed",
            "Check rules for aerosols and foil",
        ],
    },
    "glass": {
        "short_description": "Bottles and jars made of glass.",
        "preparation": [
            "Empty and rinse bottles and jars.",
            "Remove lids and sort them by material when the local program requires it.",
            "Handle broken glass with thick gloves and a rigid container.",
        ],
        "recycling": [
            "Place bottles and jars in glass collection if available.",
            "Keep window glass, mirrors, Pyrex, and ceramics out of bottle recycling unless explicitly accepted.",
        ],
        "disposal": [
            "Use glass recycling for bottles and jars.",
            "Wrap broken glass and use residual waste or a designated glass drop-off.",
        ],
        "contamination_warnings": [
            "Ceramics, light bulbs, and heat-resistant glass can ruin a bottle-glass melt.",
            "Do not assume every glass item belongs in bottle recycling.",
        ],
        "local_variation_note": "Color sorting (clear, green, brown) is required in some municipalities.",
        "checklist": [
            "Empty and rinse",
            "Remove lids if required",
            "Keep non-bottle glass separate",
        ],
    },
    "biodegradable": {
        "short_description": "Food scraps and other organic material that may be composted.",
        "preparation": [
            "Separate food from recyclable packaging.",
            "Remove stickers, elastic bands, and plastic liners.",
        ],
        "recycling": [
            "Organic waste is not recycled as packaging. Use composting or municipal organics collection.",
        ],
        "disposal": [
            "Use a compost bin or organics cart where available.",
            "If no organics service exists, use residual waste rather than contaminating recycling.",
        ],
        "contamination_warnings": [
            "Plastic bags and produce stickers contaminate compost.",
            "Meat, dairy, and oils are accepted in some municipal programs and banned in backyard compost.",
        ],
        "local_variation_note": "Organics programs differ widely; follow your city's accepted list.",
        "checklist": [
            "Separate food from packaging",
            "Use organics collection or compost if available",
            "Keep plastic out of organics",
        ],
    },
}

UNCERTAIN_GUIDANCE = {
    "short_description": "The model could not confidently identify this material.",
    "preparation": [
        "Do not force the item into a recycling bin based on a low-confidence prediction.",
    ],
    "recycling": [
        "Check municipal guidance or packaging labels before recycling.",
    ],
    "disposal": [
        "If the material is unknown, residual waste is safer than contaminating recycling.",
    ],
    "contamination_warnings": [
        "Guessing can contaminate recycling streams.",
    ],
    "local_variation_note": "When unsure, follow your municipality's 'when in doubt' rule.",
    "checklist": [
        "Look for a recycling label",
        "Check local rules",
        "Use residual waste if still unsure",
    ],
}


def get_guidance(class_name: str) -> Dict[str, Any]:
    try:
        key = canonical_name(class_name)
    except KeyError:
        payload = deepcopy(UNCERTAIN_GUIDANCE)
        payload["name"] = display_name(class_name) if class_name else "Other / Uncertain Waste"
        payload["class_key"] = "uncertain"
        return payload
    payload = deepcopy(GUIDANCE[key])
    payload["name"] = display_name(key)
    payload["class_key"] = key
    return payload


def recommendation_lines(class_name: str) -> List[str]:
    guide = get_guidance(class_name)
    return [
        *guide["preparation"][:2],
        *guide["recycling"][:2],
    ]
