from model.predict import predict
from utils.classifier import classify_bin

def main():
    image_path = "example.jpg"
    items = predict(image_path)  #Returns: ["plastic_bottle", "banana_peel", "tin_can"]

    for label, box in items:
        bin_type = classify_bin(label)
        print(f"{label} -> {bin_type} bin") 


if __name__ == "__main__":
    main()