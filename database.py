from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client.farmers_db  # Database Name
farmers_collection = db.farmers_data  # Collection Name

# Function to insert farmer data
def insert_farmer_data(data):
    farmers_collection.insert_one(data)

# Function to get all farmer data
def get_farmer_data():
    return list(farmers_collection.find({}, {"_id": 0}))

if __name__ == "__main__":
    print("Database Connected Successfully!")
