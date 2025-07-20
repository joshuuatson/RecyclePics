from config.constants import COUNCIL_BIN_MAPPINGS

def classify_bin(item_class: str, council: str = "Cambridge") -> str:
    """
    Classify the item into a bin based on its class.

    Args:
        item_class (str): The class of the item to classify.

    Returns:
        str: The bin where the item should be placed.
    """
    bin_map = COUNCIL_BIN_MAPPINGS.get(council, {})
    
    return bin_map.get(item_class, "unknown")

