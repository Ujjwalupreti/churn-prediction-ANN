import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "telco_churn"
COLLECTION_NAME = "training_dataset"

ARTIFACTS_DIR = 'app/artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'churn_model.keras')
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')

def fetch_data_from_db():
    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    data = list(collection.find({}, {'_id': 0}))
    client.close()
    
    if not data:
        raise ValueError("Database is empty! Run seed_db.py first.")
        
    print(f"Fetched {len(data)} records from database.")
    return pd.DataFrame(data)

def remove_outliers(df, cols):
    df_clean = df.copy()
    for col in cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    print(f"Outlier Removal: {len(df)} -> {len(df_clean)} rows.")
    return df_clean

def train():
    # 1. Load Data from DB
    df = fetch_data_from_db()

    # 2. Data Cleaning 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Ensure Churn is integer (1/0)
    if df['Churn'].dtype == 'object':
         df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Drop IDs if they exist
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # 3. Outlier Detection
    numeric_cols_to_check = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df = remove_outliers(df, numeric_cols_to_check)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 4. Pipeline Definitions
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numerical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # 5. Split & Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 6. Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_processed, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

    # 7. Save Artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(preprocessor, PIPELINE_PATH)
    print("Training Complete. Model updated using database records.")

if __name__ == "__main__":
    train()