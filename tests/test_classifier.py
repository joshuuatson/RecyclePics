from utils.classifier import classify_bin

def test_known_items():
    assert classify_bin("plastic_bottle") == "blue"
    assert classify_bin("banana_peel") == "green"
    assert classify_bin("tin_can") == "red" 

def test_unknown_item():
    assert classify_bin("styrofoam_cup") == "unknown"