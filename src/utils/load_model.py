#potential helper to load the saved weights
from ultralytics import YOLO

def load_yolo_model(model_name = "models/yolov8n.pt"):
    """
    Load a YOLO model from the specified path.
    
    Args:
        model_name (str): The name of the YOLO model file to load.
    
    Returns:
        YOLO: The loaded YOLO model.
    """
    model = YOLO(model_name)
    return model

#note can also use yolov8s for small or yolov8m for medium, yolov8l for large, and yolov8x for extra large models