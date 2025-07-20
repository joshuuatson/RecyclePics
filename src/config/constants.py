# src/config/constants.py

COUNCIL_BIN_MAPPINGS = {
    "Cambridge": {
        "bottle": "blue",        # plastic or glass bottles → blue bin (simplified)
        "banana": "green",       # assume peel
        "apple": "green",        # assume core
        "can": "blue",           # tin can
        "cup": "black",          # assume disposable
        "fork": "black",         # plastic fork
        "spoon": "black",        # plastic spoon
        "book": "blue",          # paper/card
        "vase": "green",         # glass
    },
    "Leeds": {
        "bottle": "green",
        "banana": "brown",
        "apple": "brown",
        "can": "grey",
        "cup": "black",
        "fork": "black",
        "spoon": "black",
        "book": "blue",
        "vase": "brown",
    }
}
