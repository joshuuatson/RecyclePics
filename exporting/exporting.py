import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('models/best_eco_sort_waste_pro_model.keras')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('tflow_model_1.tflite', 'wb') as f:
    f.write(tflite_model)