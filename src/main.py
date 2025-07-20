from model.predict import predict_image
from utils.classifier import classify_bin
from utils.visualisation import draw_detections

def main():
    image_path = "assets/test_image_2.png"
    detections = predict_image(image_path, model = "yolov8l.pt") # Example image path
    
    # for label, box in detections:
    #     bin_type = classify_bin(label)
    #     print(f"{label:15s} -> {bin_type:6s} bin @ {box}") 

    visual_output = []
    for label, box in detections:
        bin_type = classify_bin(label)
        visual_output.append((label, box, bin_type))
        print(f"{label:15s} -> {bin_type:6s} bin @ {box}")

    draw_detections(image_path, visual_output, output_path = "assets/annotated_output.jpg")


if __name__ == "__main__":
    main()