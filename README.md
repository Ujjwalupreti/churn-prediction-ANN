# Telco Customer Churn Prediction System

This project is a full-stack machine learning application designed to predict telecom customer churn. It features a React-based frontend dashboard and a FastAPI backend powered by a custom-trained TensorFlow/Keras neural network.

## Project Structure

The repository is organized into two main directories:

- **`backend/`**: Contains the FastAPI server, the machine learning training pipeline, and database interaction logic.
- **`frontend/`**: Contains the React application providing the user interface, utilizing Material-UI, Tailwind CSS, Redux, and Chart.js.

---

## Backend

The backend is built with **FastAPI** and uses **MongoDB** as its database. It handles model training, prediction serving, and user feedback collection for continuous learning.

### Features
- **Automated Model Training:** The application automatically trains the churn prediction model (a Deep Neural Network) on startup if the model artifacts (`churn_model.keras` and `preprocessor.pkl`) are missing.
- **Real-time Prediction:** Exposes a `/predict` endpoint to evaluate customer data and return a churn probability and risk level (High, Medium, Low).
- **Feedback Loop:** Exposes a `/submit_feedback` endpoint allowing new customer data to be fed back into the MongoDB training dataset for future model retraining.
- **Background Tasks:** Prediction queries are logged asynchronously to the database.

### Tech Stack
- FastAPI & Uvicorn (Web Framework)
- TensorFlow / Keras (Deep Learning)
- Scikit-learn & Pandas (Data Preprocessing)
- MongoDB (Database via pymongo)

### Setup & Installation
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure MongoDB is running locally on `mongodb://localhost:27017/` (or set the `MONGO_URI` environment variable).
5. (Optional) Run the training script manually if you have seed data in the DB:
   ```bash
   python train_model.py
   ```
6. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

---

## Frontend

The frontend is a modern single-page application built with **React**, offering a comprehensive dashboard to interact with the backend API and visualize data.

### Tech Stack
- React 19 & React Router
- Redux Toolkit (State Management)
- Material-UI & Tailwind CSS (Styling/Theming)
- Chart.js & Recharts (Data Visualization)
- Framer Motion (Animations)
- Axios (API requests)

### Setup & Installation
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```
4. The application will be running on `http://localhost:3000`.

---

## Machine Learning Pipeline

The project uses a structured pipeline for predicting churn:
1. **Data Ingestion:** Reads customer records from the `telco_churn.training_dataset` MongoDB collection.
2. **Preprocessing:** 
   - Uses `StandardScaler` for numerical features (`tenure, MonthlyCharges, TotalCharges`).
   - Uses `OneHotEncoder` for all categorical features.
3. **Model Architecture:**
   - 3-Layer Deep Neural Network with Batch Normalization and Dropout for regularization.
   - Output layer uses a sigmoid activation function for binary classification.
4. **Training Strategy:** Implements class weights to handle imbalanced datasets and uses Callbacks (`EarlyStopping`, `ReduceLROnPlateau`) to optimize learning and prevent overfitting.
