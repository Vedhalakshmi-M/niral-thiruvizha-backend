import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Update dataset path (modify this path as per your folder structure)
dataset_path = r"C:\Users\vedha\niral-thiruvizha-app\backend\dataset\NitrogenDeficiencyImage"

# Image size and batch size
img_size = (128, 128)
batch_size = 32

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalizing pixel values

# Load training data
train_data = datagen.flow_from_directory(
    dataset_path + r"\Train",  
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Load validation data
val_data = datagen.flow_from_directory(
    dataset_path + r"\Train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load test data (for final model evaluation)
test_data = datagen.flow_from_directory(
    dataset_path + r"\Test",
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

print("\nData successfully loaded! 🎉")
