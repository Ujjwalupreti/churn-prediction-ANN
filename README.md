# Telco Customer Churn Prediction System

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

> **An End-to-End Machine Learning Platform designed for proactive customer retention, predictive risk scoring, and automated data pipelines.**

## 📌 Project Overview
**Telco Customer Churn Prediction System** is an intelligent web application designed to act as a proactive retention strategist. By leveraging a custom-trained **Deep Neural Network (DNN)**, it analyzes customer telemetry, billing history, and demographic data to predict the likelihood of a customer canceling their service.

The platform bridges the gap between raw database metrics and actionable business intelligence, centralizing real-time inference, model orchestration, and feedback loops into a single environment.

### 🎯 Key Features
* **🤖 Automated Model Orchestration:** Automatically trains and persists the neural network on server startup if serialized artifacts are missing.
* **⚡ Real-Time Inference Engine:** Exposes a high-throughput endpoint to evaluate customer data, instantly returning churn probabilities and categorical risk tiers (High, Medium, Low).
* **🔄 Continuous Learning Loop:** Ingests new customer outcomes back into the MongoDB training dataset via a dedicated feedback endpoint for future model iterations.
* **📋 Asynchronous Logging:** Prediction queries and model metrics are logged asynchronously to the database to ensure zero-blocking inference.

---

## ⚙️ Technology Stack

| Component | Tech Stack | Description |
| :--- | :--- | :--- |
| **Frontend** | React.js, Tailwind CSS | Responsive SPA with Redux Toolkit state management and Chart.js |
| **Backend** | FastAPI (Python) | High-performance asynchronous REST API |
| **Database** | MongoDB | NoSQL storage for Customer Telemetry, Feedback, and Logs |
| **AI / ML** | TensorFlow, Keras, Scikit-Learn | Deep Neural Network for binary classification and data scaling |

---

## 🔄 System Architecture & Workflow

The application follows a decoupled, microservices-inspired architecture to handle heavy mathematical processing without compromising the frontend user experience.

1.  **Data Ingestion:** System reads historical customer records from the MongoDB database.
2.  **Model Bootstrapping:** Backend checks for serialized artifacts (`.keras`, `.pkl`) and triggers automated training if absent.
3.  **Real-Time Evaluation:** React frontend sends customer payloads to FastAPI; the pipeline scales the data and pushes it through the neural network.
4.  **Actionable Output & Feedback:** System returns risk metrics to the dashboard and logs the result. Users can submit ground-truth feedback to improve the next training cycle.

---

## 🧠 Machine Learning & Data Pipeline Implementation
> **Theme:** Automated Deep Learning Pipeline with Continuous Feedback

Since predicting customer churn relies on highly imbalanced datasets and complex non-linear relationships, standard linear models are insufficient. The machine learning architecture is a non-functional requirement of the highest priority.

This project implements rigorous **Data Engineering & ML Ops** principles to ensure the model remains accurate, resilient, and adaptive to shifting market trends.

### 🛡️ Implemented ML Controls

#### 1️⃣ Deep Neural Network (DNN) Architecture
We implemented a **3-Layer** deep learning strategy optimized for tabular data.
* **Mechanism:** Dense layers utilizing Batch Normalization and Dropout.
* **Security/Stability:** Dropout layers aggressively prevent the model from overfitting on training data, ensuring strong generalization.
* **Flow:** Raw Data $\rightarrow$ Scaler/Encoder $\rightarrow$ Hidden Layers (ReLU) $\rightarrow$ Output Layer (Sigmoid).

#### 2️⃣ Automated Model Orchestration
* **Circuit Breaker:** Callbacks (`EarlyStopping`) halt training automatically if the validation loss stops improving.
* **Optimization:** `ReduceLROnPlateau` dynamically lowers the learning rate to navigate complex loss landscapes without manual intervention.

#### 3️⃣ Robust Data Preprocessing
* **Numerical Scaling:** Uses `StandardScaler` for continuous features (`MonthlyCharges`, `tenure`) to ensure gradient descent converges rapidly.
* **Categorical Encoding:** Uses `OneHotEncoder` to handle text features without introducing unintended ordinal bias.

#### 4️⃣ Continuous Learning Loop
* **Data Persistence:** The `/submit_feedback` route writes directly to the core training collection.
* **Lifecycle:** Ground truth data is captured seamlessly, ensuring the model's baseline dataset organically grows with the business.

### 📊 Model Performance & Handling Analysis
| Data Challenge | Mitigation Strategy | Result |
| :--- | :--- | :--- |
| **Imbalanced Classes (More stay than churn)** | Dynamic Class Weights | Prevents the model from biasing toward the majority class |
| **Overfitting on Training Data** | Dropout Layers & Early Stopping | Generalization integrity ensured on unseen test data |
| **Feature Dominance (e.g., High Total Charges)** | Z-Score Standardization | Features contribute proportionately to the network |
| **Stale Model Degradation** | Asynchronous Feedback API | Model adapts to new trends upon retraining |

---

## 🚀 Future Roadmap

### 🧠 Machine Learning Enhancements
- [ ] **Hyperparameter Tuning:** Implement Optuna or KerasTuner for automated network optimization.
- [ ] **Explainable AI (XAI):** Integrate SHAP values to explain *why* a customer was flagged as high risk.
- [ ] **Automated Retraining:** CRON jobs via Apache Airflow to retrain the model weekly using fresh MongoDB data.

### 💻 Application Features
- [ ] **Batch Processing:** Allow CSV uploads on the frontend to score thousands of customers simultaneously.
- [ ] **Authentication:** JWT-based secure login for different admin tiers.

---

## 👥 Contributors
* **Ujjwal Upreti**
