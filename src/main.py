# from utils.predict import predict_image
# from utils.classifier import classify_bin, load_material_classifier, predict_material
# from utils.visualisation import draw_detections
# from utils.cropper import crop_image
# from config.constants import LABEL_TO_MATERIAL
# import torch

# def main():
#     image_path = "assets/real_bottle.jpg"
#     yolo_model_path = "models/online_model_1.pt"
#     glass_plastic_model_path = "models/glass_plastic_model.pt"
#     council = "Cambridge"
#     output_path = "assets/annotated_output.jpg"
    
#     # First load YOLO and material classifiers
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     detections = predict_image(image_path, model=yolo_model_path, conf=0.25, council=council)
#     #so detections is a list of tuples (label, (x, y, w, h), conf_score)
#     print(f"✅ Initial Detections: {(detections)}")

#     final_detections = []

#     for label, (x, y, w, h), conf in detections:
#         material = LABEL_TO_MATERIAL.get(label, "trash")
#         #we now have materials for each label
        
#         if label == "bottle":
#             print('found a bottle')
#             material_model = load_material_classifier(glass_plastic_model_path, device)
#             cropped = crop_image(image_path, (x, y, w, h))
#             material = predict_material(material_model, cropped, device)
#             #so now we have the material of the bottle


#         bin_type = classify_bin(material, council)
#         #this covers the material to bin classification but seemingly only for bottles
#         print(f"✅ Material classified as: {material}, Bin type: {bin_type}")
#         display_label = f"{label} ({material})" if label == "bottle" else label
#         final_detections.append((display_label, (x, y, w, h), bin_type, conf))

#     # Draw detections on the image
#     draw_detections(image_path, final_detections, output_path)
#     print(f"✅ Detections saved to {output_path}")

# if __name__ == "__main__":
#     main()

from utils.predict import predict_image
from utils.classifier import classify_bin, load_material_classifier, predict_material
from utils.visualisation import draw_detections
from utils.cropper import crop_image
from utils.load_model import load_yolo_model
from config.constants import LABEL_TO_MATERIAL

import torch


def main():
    image_path = "assets/real_bottle_3.jpeg"
    general_model_path = "models/yolov8x.pt"
    waste_model_path = "models/online_model_1.pt"
    bottle_model_path = "models/glass_plastic_model.pt"
    council = "Cambridge"
    output_path = "assets/annotated_output.jpg"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load YOLO general model and waste classifier
    general_model = load_yolo_model(general_model_path)
    waste_model = load_yolo_model(waste_model_path)
    bottle_material_model = load_material_classifier(bottle_model_path, device)

    # Use general YOLO model to get initial detections
    general_detections = predict_image(image_path, model=general_model, conf=0.25, council=council)
    print(f"✅ General Detections: {general_detections}")

    final_detections = []

    for label, (x, y, w, h), conf in general_detections:
        # Crop the detected object
        cropped = crop_image(image_path, (x, y, w, h))

        # Use waste model on cropped image
        waste_result = waste_model.predict(cropped, conf=0.25)[0]
        if waste_result.boxes:
            best_box = waste_result.boxes[0]
            cls_id = int(best_box.cls.item())
            waste_label = waste_model.names[cls_id]
            waste_conf = float(best_box.conf.item())
        else:
            waste_label = "trash"
            waste_conf = 0.0

        # Choose label with higher confidence
        chosen_label, chosen_conf = (waste_label, waste_conf) if waste_conf > conf else (label, conf)

        # Special case: bottle
        if chosen_label == "bottle":
            print("🔍 Detected bottle – classifying material...")
            material = predict_material(bottle_material_model, cropped, device)
            display_label = f"bottle ({material})"
        else:
            material = LABEL_TO_MATERIAL.get(chosen_label, "trash")
            display_label = chosen_label

        bin_type = classify_bin(material, council)
        print(f"✅ Final Label: {display_label}, Material: {material}, Bin: {bin_type}, Conf: {chosen_conf:.2f}")

        final_detections.append((display_label, (x, y, w, h), bin_type, chosen_conf))

    # Draw results
    draw_detections(image_path, final_detections, output_path)
    print(f"✅ Annotated output saved to: {output_path}")


if __name__ == "__main__":
    main()
