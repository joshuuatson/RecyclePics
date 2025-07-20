from model.load_model import load_yolo_model

def test_load_yolov8n_model():
    model = load_yolo_model()
    assert model is not None
    assert hasattr(model, "predict") or hasattr(model, "__call__") # YOLO models are callable

    