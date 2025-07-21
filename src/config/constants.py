# src/config/constants.py

MATERIAL_TO_BIN = {
    "Cambridge": {
        "plastic": "blue",
        "glass": "green",
        "metal": "blue",
        "paper": "blue",
        "cardboard": "blue",
        "food": "green",
        "trash": "black",
        "other": "black",  # default catch-all
    },
    "Leeds": {
        "plastic": "green",
        "glass": "brown",
        "metal": "grey",
        "paper": "blue",
        "cardboard": "blue",
        "food": "brown",
        "trash": "black",
        "other": "black",
        "hazardous": "hazardous",
    }
}


LABEL_TO_MATERIAL = {
    "battery": "hazardous",         # hazardous waste, not recycled in normal bins
    "biological": "food",       # organic waste
    "brown-glass": "glass",
    "green-glass": "glass",
    "white-glass": "glass",
    "cardboard": "cardboard",
    "clothes": "other",         # could go to textile recycling, but not standard bin
    "shoes": "other",           # same as above
    "metal": "metal",
    "paper": "paper",
    "plastic": "plastic",
    "trash": "trash"
}

