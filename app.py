from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import os
import requests
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from pymongo import MongoClient
import bcrypt
from datetime import datetime


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection


# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["farmingDB"]  # Database name
users_collection = db["users"]  # Collection for storing user data

# 🔹 Register User (Save Name & Password)
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    password = data.get("password")

    if users_collection.find_one({"name": name}):
        return jsonify({"success": False, "message": "User already exists!"})

    # Hash Password for Security
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    # Insert into MongoDB
    users_collection.insert_one({"name": name, "password": hashed_password})
    return jsonify({"success": True, "message": "User registered successfully!"})

# 🔹 Login User (Verify Credentials)
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    name = data.get("name")
    password = data.get("password")

    user = users_collection.find_one({"name": name})

    if user and bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return jsonify({"success": True, "message": "Login successful!"})
    else:
        return jsonify({"success": False, "message": "Invalid name or password!"})


# Add a new collection for past searches
past_searches_collection = db["past_searches"]

@app.route('/get_data', methods=['POST'])
def get_data():
    data = request.get_json()
    city = data.get("city")
    lang = data.get("lang", "en")

    weather = get_weather(city, lang)
    soil = get_soil_data(city, lang)

    # Save past searches in MongoDB
    past_search = {
        "city": city,
        "timestamp": datetime.utcnow(),
        "weather": weather,
        "soil": soil
    }
    past_searches_collection.insert_one(past_search)

    return jsonify({"weather": weather, "soil": soil})

# API to get past searches
@app.route('/get_past_searches', methods=['GET'])
def get_past_searches():
    past_searches = list(past_searches_collection.find({}, {"_id": 0}))  # Exclude MongoDB ID
    return jsonify({"past_searches": past_searches})


# 🔹 Load trained model
MODEL_PATH = "model/trained_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# 🔹 Load class indices
CLASS_INDICES_PATH = "model/class_indices.json"
with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# 🔹 Reverse class labels
class_labels = {v: k for k, v in class_indices.items()}

# 🔹 Image processing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# 🔹 Prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    # Preprocess and predict
    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    # Remove the file after prediction
    os.remove(file_path)

    # Return prediction response
    return jsonify({
        'predicted_class': class_labels[predicted_class],
        'confidence': confidence
    })

# 🔹 OpenWeather API Key (Replace with your actual key)
OPENWEATHER_API_KEY = "480244c8f817a647e0398edeb0dd03cb"

# 🔹 Function to get weather data
def get_weather(city, lang="en"):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    weather_titles = {
        "en": {"temperature": "Temperature", "humidity": "Humidity", "condition": "Condition"},
        "ta": {"temperature": "வெப்பநிலை", "humidity": "ஈரப்பதம்", "condition": "நிலை"},
        "hi": {"temperature": "तापमान", "humidity": "नमी", "condition": "स्थिति"},
    }

    if response.status_code == 200:
        weather_data = {
            "titles": weather_titles.get(lang, weather_titles["en"]),  # ✅ Get correct language titles
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["description"]
        }
        print("🌤 Weather Data Sent:", weather_data)  # ✅ Debugging
        return weather_data
    else:
        return {"error": "Weather data not found"}



# 🔹 Function to get soil data from JSON file
# 🔹 Function to get soil data from JSON file
def get_soil_data(location, lang="en"):
    soil_data_path = os.path.join(os.path.dirname(__file__), "soil_data.json")

    if not os.path.exists(soil_data_path):
        print(f"❌ ERROR: soil_data.json file not found at {soil_data_path}")
        return {"error": "Soil data file not found!"}

    try:
        with open(soil_data_path, "r", encoding="utf-8") as file:
            soil_data = json.load(file)
    except json.JSONDecodeError:
        print("❌ ERROR: Failed to parse soil_data.json. Check file formatting.")
        return {"error": "Invalid soil data file format!"}

    data = soil_data.get(location)
    if not data:
        print(f"❌ ERROR: No soil data available for {location}")
        return {"error": "No soil data available"}

    print(f"🌱 Fetching soil data for {location} in {lang} language")

    # ✅ Extract values in the selected language
    soil_type = data["soil_type"].get(lang, "Unknown")
    nitrogen = data["nutrients"]["nitrogen"].get(lang, "Unknown")
    phosphorus = data["nutrients"]["phosphorus"].get(lang, "Unknown")
    potassium = data["nutrients"]["potassium"].get(lang, "Unknown")

    # ✅ Titles for UI
    titles = {
        "en": {
            "soil_type": "Soil Type",
            "pH": "pH Level",
            "nutrients": "Nutrients",
            "nitrogen": "Nitrogen",
            "phosphorus": "Phosphorus",
            "potassium": "Potassium",
            "fertilizer_recommendations": "Fertilizer Recommendations"
        },
        "ta": {
            "soil_type": "மண் வகை",
            "pH": "pH நிலை",
            "nutrients": "உட்பொருள்",
            "nitrogen": "நைட்ரஜன்",
            "phosphorus": "பாஸ்பரஸ்",
            "potassium": "பொட்டாசியம்",
            "fertilizer_recommendations": "உர பரிந்துரைகள்"
        },
        "hi": {
            "soil_type": "मिट्टी का प्रकार",
            "pH": "pH स्तर",
            "nutrients": "पोषक तत्व",
            "nitrogen": "नाइट्रोजन",
            "phosphorus": "फास्फोरस",
            "potassium": "पोटैशियम",
            "fertilizer_recommendations": "उर्वरक सिफारिशें"
        }
    }

    # ✅ Fertilizer Recommendations in 3 Languages
    fertilizer_recommendations = {
        "en": [],
        "ta": [],
        "hi": []
    }

    if nitrogen == "Low":
        fertilizer_recommendations["en"].append("Urea, Ammonium Sulfate")
        fertilizer_recommendations["ta"].append("யூரியா, அமோனியம் சல்பேட்")
        fertilizer_recommendations["hi"].append("यूरिया, अमोनियम सल्फेट")
    elif nitrogen == "Medium":
        fertilizer_recommendations["en"].append("Calcium Ammonium Nitrate (CAN)")
        fertilizer_recommendations["ta"].append("கால்சியம் அமோனியம் நைட்ரேட் (CAN)")
        fertilizer_recommendations["hi"].append("कैल्शियम अमोनियम नाइट्रेट (CAN)")
    else:
        fertilizer_recommendations["en"].append("Reduce nitrogen fertilizers")
        fertilizer_recommendations["ta"].append("நைட்ரஜன் உரங்களை குறைக்கவும்")
        fertilizer_recommendations["hi"].append("नाइट्रोजन उर्वरकों को कम करें")

    if phosphorus == "Low":
        fertilizer_recommendations["en"].append("Single Super Phosphate (SSP)")
        fertilizer_recommendations["ta"].append("ஒற்றை சூப்பர் பாஸ்பேட் (SSP)")
        fertilizer_recommendations["hi"].append("सिंगल सुपर फॉस्फेट (SSP)")
    elif phosphorus == "Medium":
        fertilizer_recommendations["en"].append("Di-Ammonium Phosphate (DAP)")
        fertilizer_recommendations["ta"].append("டை-அமோனியம் பாஸ்பேட் (DAP)")
        fertilizer_recommendations["hi"].append("डाई-अमोनियम फॉस्फेट (DAP)")
    else:
        fertilizer_recommendations["en"].append("Reduce phosphorus fertilizers")
        fertilizer_recommendations["ta"].append("பாஸ்பரஸ் உரங்களை குறைக்கவும்")
        fertilizer_recommendations["hi"].append("फॉस्फोरस उर्वरकों को कम करें")

    if potassium == "Low":
        fertilizer_recommendations["en"].append("Muriate of Potash (MOP)")
        fertilizer_recommendations["ta"].append("மியூரியேட் ஆஃப் பொட்டாஷ் (MOP)")
        fertilizer_recommendations["hi"].append("म्यूरिएट ऑफ पोटाश (MOP)")
    elif potassium == "Medium":
        fertilizer_recommendations["en"].append("Potassium Sulfate")
        fertilizer_recommendations["ta"].append("பொட்டாசியம் சல்பேட்")
        fertilizer_recommendations["hi"].append("पोटेशियम सल्फेट")
    else:
        fertilizer_recommendations["en"].append("Reduce potassium fertilizers")
        fertilizer_recommendations["ta"].append("பொட்டாசியம் உரங்களை குறைக்கவும்")
        fertilizer_recommendations["hi"].append("पोटेशियम उर्वरकों को कम करें")

    print(f"✅ Final Fertilizer Recommendations: {fertilizer_recommendations[lang]}")

    return {
        "titles": titles[lang],
        "soil_data": {
            "Soil Type": soil_type,
            "pH": data.get("pH", "Unknown"),
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium
        },
        "fertilizer_recommendations": fertilizer_recommendations[lang]  # ✅ Now returns recommendations in the selected language
    }

   




# 🔹 API Route to Fetch Soil & Weather Data
"""@app.route('/get_data', methods=['POST'])
def get_data():
    data = request.get_json()
    city = data.get("city")

    weather = get_weather(city)
    soil = get_soil_data(city,lang=data.get("lang", "en"))

    return jsonify({"weather": weather, "soil": soil})"""  # Return both data

# Run the Flask app
if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")  # Create uploads folder if not exists
    app.run(debug=True)
