from utils.predict import predict_image
from utils.classifier import classify_bin, load_material_classifier, predict_material
from utils.visualisation import draw_detections
from utils.cropper import crop_image
from config.constants import LABEL_TO_MATERIAL, MATERIAL_TO_BIN

import torch

def main():
    image_path = "assets/jar.jpg"
    yolo_model_path = "models/yolov8x.pt"
    glass_plastic_model_path = "models/glass_plastic_model.pt"
    council = "Cambridge"
    output_path = "assets/annotated_output.jpg"
    
    # First load YOLO and material classifiers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detections = predict_image(image_path, model=yolo_model_path, conf=0.25, council=council)
    #so detections is a list of tuples (label, (x, y, w, h), conf_score)
    print(f"✅ Initial Detections: {(detections)}")

    final_detections = []

    for label, (x, y, w, h), conf in detections:
        material = LABEL_TO_MATERIAL.get(label, "trash")
        #we now have materials for each label
        
        if label == "bottle":
            print('found a bottle')
            material_model = load_material_classifier(glass_plastic_model_path, device)
            cropped = crop_image(image_path, (x, y, w, h))
            material = predict_material(material_model, cropped, device)
            #so now we have the material of the bottle


        bin_type = classify_bin(material, council)
        #this covers the material to bin classification but seemingly only for bottles
        print(f"✅ Material classified as: {material}, Bin type: {bin_type}")
        display_label = f"{label} ({material})" if label == "bottle" else label
        final_detections.append((display_label, (x, y, w, h), bin_type, conf))

    # Draw detections on the image
    draw_detections(image_path, final_detections, output_path)
    print(f"✅ Detections saved to {output_path}")

if __name__ == "__main__":
    main()