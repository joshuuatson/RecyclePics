import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------- Configuration ----------------------
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMAGE_PATH = 'assets/sample_validation_image.jpg'
MODEL_PATH = 'models/best_eco_sort_waste_pro_model.keras'

# ---------------------- Load model ----------------------
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------- Load and preprocess image (matching training pipeline) ----------------------
def preprocess_image(path):
    """
    Preprocess image to match the training pipeline preprocessing.
    This function replicates what tf.keras.preprocessing.image_dataset_from_directory does.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    # Keep as uint8 and let the model handle normalization internally
    img = tf.cast(img, tf.uint8)
    # Convert to float32 but do NOT divide by 255.0
    img = tf.cast(img, tf.float32)
    return tf.expand_dims(img, axis=0)      # Add batch dimension

img = preprocess_image(IMAGE_PATH)

# ---------------------- Predict ----------------------
pred = model.predict(img)
pred_class = CLASS_NAMES[np.argmax(pred)]
confidence = 100 * np.max(pred)

# ---------------------- Output ----------------------
print("Probabilities:", np.round(pred, 3))
print(f"Predicted: {pred_class} ({confidence:.2f}%)")

# ---------------------- Additional debugging information ----------------------
print(f"Input image shape: {img.shape}")
print(f"Input image dtype: {img.dtype}")
print(f"Input image value range: [{tf.reduce_min(img):.2f}, {tf.reduce_max(img):.2f}]")