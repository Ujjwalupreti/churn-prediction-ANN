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

    class_weights = class_weights.compute_class_weight(class_weights="balanced",classes=np.unique(y_train),y=y_train)
    
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Computed Class Weights:{class_weights_dict}")
    
    model = tf.keras.models.Sequential([
        # Input Layer
        tf.keras.layers.Input(shape=(X_train_processed.shape[1],)),
        
        # Layer 1: Wider + BatchNormalization for stability
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),

        # Layer 2: Deep feature extraction
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),

        # Layer 3: Bottleneck
        tf.keras.layers.Dense(32, activation='relu'),
        
        # Output Layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss = "binary_crossentropy",
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),       
            tf.keras.metrics.Recall(name='recall'), 
        ]
    )
    
    #8. Callbacks (The Auto-Pilot)
    
    callbacks = [
        tf.keras.callback.EarlyStopping(monitor = 'val_loss',patiencs=10,restore_best_weights=True,verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=0.00001,verbose=1)
    ]
    
    # 9. Train with Class Weights
    print("Starting Training...")
    history = model.fit(
        X_train_processed, 
        y_train, 
        epochs=100,            
        batch_size=32, 
        validation_split=0.1, 
        class_weight=class_weights_dict, 
        callbacks=callbacks,
        verbose=1
    )

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(preprocessor, PIPELINE_PATH)
    
    print("\n--- Training Complete ---")
    print(f"Final Model saved to: {MODEL_PATH}")
    
    # Quick Evaluation
    loss, acc, auc, recall = model.evaluate(X_test_processed, y_test)
    print(f"Test Set Performance -> Accuracy: {acc:.2f}, AUC: {auc:.2f}, Recall: {recall:.2f}")

if __name__ == "__main__":
    train()