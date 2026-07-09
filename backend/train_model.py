import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import yaml
import logging
import mlflow
import mlflow.keras
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_class_weight


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    logger.error(f"Configuration file {CONFIG_PATH} not found!")
    raise FileNotFoundError(f"{CONFIG_PATH} missing. Please create it.")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = config['data']['db_name']
COLLECTION_NAME = config['data']['collection']

LEARNING_RATE = config['model']['learning_rate']
BATCH_SIZE = config['model']['batch_size']
EPOCHS = config['model']['epochs']
PATIENCE = config['model']['early_stopping_patience']

ARTIFACTS_DIR = 'app/artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'churn_model.keras')
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')

def fetch_data_from_db():
    logger.info("Connecting to MongoDB to fetch raw data...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    data = list(collection.find({}, {'_id': 0}))
    client.close()
    
    if not data:
        logger.error("Database is empty! Run seed_db.py first.")
        raise ValueError("Database is empty!")
        
    logger.info(f"Successfully fetched {len(data)} records from database.")
    return pd.DataFrame(data)

def train():
    mlflow.set_tracking_uri(config['experiment']['tracking_uri'])
    mlflow.set_experiment(config['experiment']['name'])
    
    with mlflow.start_run():
        logger.info("Started MLflow tracking run.")
        
        # Log hyperparameters to MLflow
        mlflow.log_params({
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE
        })

        # Load Data
        df = fetch_data_from_db()

        # Data Cleaning
        logger.info("Processing data and handling missing values...")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        X = df.drop(['Churn', 'customerID'], axis=1, errors='ignore')
        y = df['Churn']
        
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        # Preprocessing Pipeline (Pillar 3 preparation)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ])

        X_processed = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        # Compute Class Weights for Imbalanced Data
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights_dict = dict(zip(classes, weights))

        # Build Neural Network
        logger.info("Compiling Keras model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid') # Output Layer
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall')]
        )
        
        keras_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        ]
        
        # Train Model
        logger.info(f"Starting Training for {EPOCHS} epochs...")
        history = model.fit(
            X_train, 
            y_train, 
            epochs=EPOCHS,            
            batch_size=BATCH_SIZE, 
            validation_split=0.1, 
            class_weight=class_weights_dict, 
            callbacks=keras_callbacks,
            verbose=1
        )

        # Evaluate Model
        logger.info("Evaluating model on test data...")
        loss, accuracy, auc, recall = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Evaluation Results -> Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Recall: {recall:.4f}")
        
        # Log Metrics to MLflow
        mlflow.log_metrics({
            "test_loss": loss,
            "test_accuracy": accuracy,
            "test_auc": auc,
            "test_recall": recall
        })

        # Save Artifacts
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        model.save(MODEL_PATH)
        joblib.dump(preprocessor, PIPELINE_PATH)
        
        mlflow.keras.log_model(model, "keras_model")
        mlflow.log_artifact(PIPELINE_PATH, "preprocessor")

        logger.info("Training complete. Artifacts saved locally and tracked in MLflow.")

if __name__ == "__main__":
    train()