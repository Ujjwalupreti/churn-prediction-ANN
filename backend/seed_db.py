import pandas as pd
import os
from pymongo import MongoClient

# --- CONFIGURATION ---
CSV_PATH = r'backend\WA_Fn-UseC_-Telco-Customer-Churn.csv'
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/") 
DB_NAME = "telco_churn"
COLLECTION_NAME = "training_dataset" 

def seed_database():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found.")
        return

    print("Reading CSV...")
    df = pd.read_csv(CSV_PATH)

    # Basic cleaning before upload (Optional, but good for consistency)
    # Convert TotalCharges to numeric, coerce errors to NaN, fill with 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Standardize Churn Column (Yes/No -> 1/0)
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Convert DataFrame to Dictionary (JSON format for MongoDB)
    data_dict = df.to_dict("records")

    print(f"Connecting to MongoDB ({MONGO_URI})...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Clear existing data to avoid duplicates during testing
    # collection.delete_many({}) 
    
    print(f"Inserting {len(data_dict)} records...")
    collection.insert_many(data_dict)
    
    print("✅ Database Seeded Successfully!")
    client.close()

if __name__ == "__main__":
    seed_database()