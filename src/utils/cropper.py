from PIL import Image

def crop_image(image_path, box_coords):
    """
    Crop an image to the specified bounding box coordinates.
    
    Args:
        image_path (str): Path to the input image.
        box_coords (tuple): Bounding box coordinates in the format (x1, y1, x2, y2).
    
    Returns:
        PIL.Image: Cropped image.
    """
    image = Image.open(image_path)
    x, y, w, h = box_coords
    x1 = max(int(x - w / 2), 0)
    y1 = max(int(y - h / 2), 0)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    return image.crop((x1, y1, x2, y2))