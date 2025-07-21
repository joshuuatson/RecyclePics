from ultralytics import YOLO

model = YOLO("models/taco_model.pt")

print(model.names)