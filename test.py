import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing import image

# Load trained model
model_path = "model/trained_model.h5"  
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices_path = "model/class_indices.json"
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# Reverse the dictionary to get class labels
class_labels = {v: k for k, v in class_indices.items()}
print("Class Label Mapping:", class_labels)


def preprocess_image(img_path):
    """Load and preprocess an image for prediction"""
    if not os.path.exists(img_path):
        print(f"❌ Error: Image '{img_path}' not found.")
        return None
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_image(img_path):
    """Predict the class of an input image"""
    img_array = preprocess_image(img_path)
    if img_array is None:
        return
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(f"🔍 Image: {os.path.basename(img_path)} → Predicted Class: {class_labels[predicted_class]} ({confidence:.2f}%)")

# **Test multiple images from different nitrogen levels**
test_images = [
    r"C:\Users\vedha\niral-thiruvizha-app\backend\dataset\NitrogenDeficiencyImage\Test\High_Nitrogen\SWAPT4_002.jpg",
    r"C:\Users\vedha\niral-thiruvizha-app\backend\dataset\NitrogenDeficiencyImage\Test\Medium_High_Nitrogen\SWAPT2_001.jpg",
    r"C:\Users\vedha\niral-thiruvizha-app\backend\dataset\NitrogenDeficiencyImage\Test\Medium_Nitrogen\SWAPT3_001.jpg",
    r"C:\Users\vedha\niral-thiruvizha-app\backend\dataset\NitrogenDeficiencyImage\Test\Low_Nitrogen\SWAPT1_001.jpg"
]

# Run predictions on multiple test images
for img_path in test_images:
    predict_image(img_path)
