import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained AI model (Train your own later)
model = tf.keras.models.load_model('leaf_model.h5')

def analyze_leaf(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    nitrogen_level = np.argmax(prediction)  # Example output
    return {"Nitrogen Level": nitrogen_level}

if __name__ == "__main__":
    print(analyze_leaf("test_leaf.jpg"))
