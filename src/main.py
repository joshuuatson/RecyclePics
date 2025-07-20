from model.predict import predict_image
from utils.classifier import classify_bin
from utils.visualisation import draw_detections

def main():
    image_path = "assets/test_image_2.png"
    detections = predict_image(image_path, model = "yolov8l.pt") # Example image path

    draw_detections(image_path, detections, output_path = "assets/annotated_output.jpg")


if __name__ == "__main__":
    main()