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
    }
}


LABEL_TO_MATERIAL = {
    "bottle": "plastic",    # assuming PET plastic
    "can": "metal",
    "cup": "plastic",       # or "trash" for disposable
    "book": "paper",
    "vase": "glass",
    "fork": "plastic",
    "spoon": "plastic",
    "banana": "trash",      # if it's the fruit
    "apple": "trash",
    "box": "cardboard",
    "bag": "plastic",       # assuming shopping bag
    "tv": "trash",          # general waste
    # Add more as needed
}

