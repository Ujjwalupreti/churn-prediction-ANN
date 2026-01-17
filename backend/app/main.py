import os
import joblib
import pandas as pd
import tensorflow as tf
import subprocess
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .schema import CustomerData,CustomerFeedback
from .database import db

app = FastAPI(title="Telco Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
preprocessor = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(BASE_DIR) 
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "churn_model.keras")
PIPELINE_PATH = os.path.join(BASE_DIR, "artifacts", "preprocessor.pkl")
TRAIN_SCRIPT = os.path.join(BACKEND_ROOT, "train_model.py")

@app.on_event("startup")
def startup_event():
    global model, preprocessor
    
    db.connect()

    if not os.path.exists(MODEL_PATH) or not os.path.exists(PIPELINE_PATH):
        try:
            subprocess.run(["python", TRAIN_SCRIPT], check=True)
            print("Automatic training complete.")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not train model: {e}")
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            preprocessor = joblib.load(PIPELINE_PATH)
            print("ML Artifacts loaded successfully.")
        else:
            print("ERROR: Model still missing after training attempt.")
    except Exception as e:
        print(f"Error loading ML artifacts: {e}")

@app.on_event("shutdown")
def shutdown_event():
    db.close()

@app.get("/")
def home():
    return {
        "status": "online", 
        "database": "connected" if db.client else "disconnected",
        "model": "loaded" if model else "not loaded"
    }

@app.post("/predict")
def predict(data: CustomerData, background_tasks: BackgroundTasks):
    if not model or not preprocessor:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded. System may be retraining.")

    try:
        input_dict = data.dict()
        input_df = pd.DataFrame([input_dict])
        processed_data = preprocessor.transform(input_df)

        prediction_raw = model.predict(processed_data)
        churn_prob = float(prediction_raw[0][0])
        is_churn = churn_prob > 0.5

        response_payload = {
            "prediction": "Yes" if is_churn else "No",
            "probability": round(churn_prob, 4),
            "risk_level": "High" if churn_prob > 0.75 else "Medium" if churn_prob > 0.3 else "Low"
        }

        background_tasks.add_task(db.log_prediction, input_dict, response_payload)

        return response_payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {str(e)}")

@app.post("/submit_feedback")
def submit_feedback(data: CustomerFeedback):
    if db.db is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        record = data.dict()
        training_collection = db.db["training_dataset"]
        training_collection.insert_one(record)
        
        return {
            "status": "success", 
            "message": "Data added to training set. Model will learn from this in the next cycle."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Error: {str(e)}")