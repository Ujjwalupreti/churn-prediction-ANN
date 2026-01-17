import os
from pymongo import MongoClient
from datetime import datetime

MONGO_URI = os.getenv("MONGO_URI", "mongodb://host.docker.internal:27017/")
DB_NAME = "telco_churn"
COLLECTION_NAME = "prediction_logs"

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None

    def connect(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            print("MongoDB Connected Successfully")
        except Exception as e:
            print(f"MongoDB Connection Failed: {e}")

    def close(self):
        if self.client:
            self.client.close()
            print("MongoDB Connection Closed")

    def log_prediction(self, input_data: dict, result: dict):
        if self.collection is not None:
            try:
                record = {
                    "input": input_data,
                    "output": result,
                    "timestamp": datetime.utcnow()
                }
                self.collection.insert_one(record)
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Database not connected. Log skipped.")
db = Database()