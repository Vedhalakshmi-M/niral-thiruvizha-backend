import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.cnn_model import create_cnn_model  # Import CNN model
import os
import json

# **1️⃣ Set Dataset Paths**
base_dir = r"C:\Users\vedha\niral-thiruvizha-app\backend\dataset\NitrogenDeficiencyImage"
train_dir = os.path.join(base_dir, "Train")

# **2️⃣ Create ImageDataGenerator (Rescale Images)**


datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,  # Increase rotation for more variety
    width_shift_range=0.3,  # Allow more movement in width
    height_shift_range=0.3,  # Allow more movement in height
    shear_range=0.3,  
    zoom_range=0.3,  
    horizontal_flip=True,  # Flip images to avoid memorization
    fill_mode='nearest',
    validation_split=0.3  # ✅ Increase validation data to test generalization
)


# **3️⃣ Load Training Data**
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to fit model input
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    subset='training'
)

# **4️⃣ Load Validation Data**
val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# **5️⃣ Print Class Order (Check Class Indices)**
print("🔍 Class Indices:", train_data.class_indices)  # Verify correct class labels

# **6️⃣ Save Class Indices to JSON for Later Use**
class_indices_path = r"C:\Users\vedha\niral-thiruvizha-app\backend\model\class_indices.json"
with open(class_indices_path, "w") as f:
    json.dump(train_data.class_indices, f)
print(f"✅ Class indices saved to {class_indices_path}")

# **7️⃣ Create CNN Model**
model = create_cnn_model()

# **8️⃣ Train Model**
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10  # Increase epochs for better accuracy
)

# **9️⃣ Save Trained Model**
model_path = r"C:\Users\vedha\niral-thiruvizha-app\backend\model\trained_model.h5"
model.save(model_path)
print(f"✅ Model training completed and saved as {model_path}")
