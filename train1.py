import json

try:
    with open("C:/Users/vedha/niral-thiruvizha-app/backend/soil_data.json", "r") as file:
        data = json.load(file)
        print("✅ JSON Loaded Successfully:", data)
except FileNotFoundError:
    print("❌ File not found! Check path and filename.")
except json.JSONDecodeError:
    print("❌ Invalid JSON format. Fix it in soil_data.json.")
