from config.constants import MATERIAL_TO_BIN
import torch
from torchvision import transforms

def classify_bin(item_class: str, council: str = "Cambridge") -> str:
    """
    Classify the item into a bin based on its class.

    Args:
        item_class (str): The class of the item to classify.

    Returns:
        str: The bin where the item should be placed.
    """
    bin_map = MATERIAL_TO_BIN.get(council, {})
    
    return bin_map.get(item_class, "unknown")


def load_material_classifier(model_path, device):
    from torchvision import models
    import torch.nn as nn

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_material(model, image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred_idx = output.argmax(1).item()
    return ["glass", "plastic"][pred_idx]  # keep consistent with training