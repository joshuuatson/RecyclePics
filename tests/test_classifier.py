from utils.classifier import classify_bin

def test_council_specific_bins():
    assert classify_bin("banana") == "green"  # Cambridge default
    assert classify_bin("banana", council="Leeds") == "brown"

def test_fallback_behavior():
    assert classify_bin("nappy", council="UnknownTown") == "unknown"
