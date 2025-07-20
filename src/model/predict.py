#this runs the interface: loads the model and returns the detected objects
from model.load_model import load_yolo_model

def predict_image(image_path, model=None, conf = 0.25):
    """
    Predict objects in an image using the YOLO model.
    
    Args:
        image_path (str): Path to the input image.
        model (YOLO, optional): Preloaded YOLO model. If None, loads the default model.
        conf (float): Confidence threshold for predictions.
    
    Returns:
        list: List of tuples containing detected object labels and bounding boxes.
    """
    if isinstance(model, str):
        model = load_yolo_model(model)  #recall this is the function that loads the model given name string
    elif model is None:
        model = load_yolo_model()

    results = model(image_path)[0]

    predictions = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
        w = x2 - x1
        h = y2 - y1
        predictions.append((label, (x1, y1, w, h)))

    return predictions