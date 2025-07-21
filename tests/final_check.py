import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# --------------------- Configuration ---------------------
DATASET_DIR = 'datasets/dataset-resized'  # Path to dataset
MODEL_PATH = 'models/best_eco_sort_waste_pro_model.keras'  # Trained model file

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
SEED = 100  # Must match validation split seed during training
VALIDATION_SPLIT = 0.2

# --------------------- Load Validation Dataset ---------------------
validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = validation_ds.class_names
print("Class Names:", class_names)


import PIL.Image

# Get one image and label from the validation dataset
for batch_images, batch_labels in validation_ds.take(1):
    sample_img = batch_images[3].numpy().astype("uint8")  # shape: (224, 224, 3)
    sample_label = class_names[batch_labels[3].numpy()]
    break

# Save image to disk for reuse in single-image tests
output_path = "sample_validation_image.jpg"
PIL.Image.fromarray(sample_img).save(output_path)
print(f"Saved a sample validation image as: {output_path} (true label: {sample_label})")



# --------------------- Load Saved Model ---------------------
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

# --------------------- Run Evaluation ---------------------
# Collect all images and labels from dataset
X_val, y_true = [], []
for images, labels in validation_ds:
    X_val.extend(images.numpy())
    y_true.extend(labels.numpy())

X_val = np.array(X_val)
y_true = np.array(y_true)

# Predict
y_probs = model.predict(X_val)
y_pred = np.argmax(y_probs, axis=1)

# --------------------- Metrics ---------------------
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%\n")

print("📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# --------------------- Visual Sample ---------------------
plt.figure(figsize=(12, 6))
for i in range(12):
    idx = np.random.randint(len(X_val))
    img = X_val[idx]
    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred[idx]]
    plt.subplot(3, 4, i+1)
    plt.imshow(img.astype("uint8"))
    plt.title(f"{true_label} → {pred_label}", fontsize=9)
    plt.axis("off")
plt.tight_layout()
plt.savefig("sample_predictions_reloaded.png", dpi=300)
plt.show()
