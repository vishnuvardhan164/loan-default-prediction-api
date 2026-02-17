# 🏦 Loan Default Risk Prediction System

Production-Style Machine Learning Pipeline with FastAPI, Streamlit, and Docker

---

## 📌 Overview

This project is an end-to-end machine learning application that predicts the probability of loan default based on applicant financial and demographic information.

It demonstrates a complete production ML workflow including:

* Data preprocessing and feature engineering
* Model training and evaluation
* Model persistence
* REST API deployment
* Interactive web interface
* Containerized deployment using Docker

The system supports real-time loan risk prediction through both an API and a user-friendly web interface.

---

## 🎯 Problem Statement

Financial institutions must assess whether a loan applicant is likely to default.
This project builds a predictive model that analyzes applicant attributes such as income, credit history, loan amount, and employment status to estimate default risk.

---

## 🧠 Machine Learning Pipeline

1. Load historical loan dataset
2. Handle missing values
3. Encode categorical variables
4. Train classification model (Logistic Regression)
5. Evaluate model performance
6. Save trained model and feature schema
7. Serve predictions via API

---

## 🏗 System Architecture

Historical Dataset
↓
Model Training Pipeline (Scikit-learn)
↓
Saved Model Artifact
↓
FastAPI Prediction Service
↓
Streamlit Web Interface
↓
User Prediction Request

---

## ⚙️ Technology Stack

| Component            | Technology             |
| -------------------- | ---------------------- |
| Programming Language | Python                 |
| Data Processing      | Pandas, NumPy          |
| Machine Learning     | Scikit-learn           |
| Model Serialization  | Joblib                 |
| API Backend          | FastAPI                |
| Web UI               | Streamlit              |
| Containerization     | Docker, Docker Compose |

---

## 📊 Model Performance

Example evaluation results:

* Accuracy ≈ 82%
* ROC-AUC ≈ 0.86

Performance may vary depending on dataset split.

---

## 🚀 Features

✔ End-to-end ML training pipeline
✔ Real-time prediction API
✔ Interactive web interface
✔ Containerized deployment
✔ Reproducible environment
✔ Production-ready architecture

---

## 📂 Project Structure

loan-default-prediction-api/

api/ → FastAPI prediction service
src/ → Training pipeline and utilities
ui/ → Streamlit web application
models/ → Trained model artifacts
Dockerfile.api
Dockerfile.ui
docker-compose.yml
requirements.txt
README.md

---

## ▶️ Running the Project Locally

### 1️⃣ Create virtual environment

python -m venv .venv
.venv\Scripts\activate   (Windows)

---

### 2️⃣ Install dependencies

pip install -r requirements.txt

---

### 3️⃣ Train model

python -m src.train

---

### 4️⃣ Start API

uvicorn api.main:app --reload

---

### 5️⃣ Start Streamlit UI

streamlit run ui/app.py

---

API documentation
http://127.0.0.1:8000/docs

Web UI
http://localhost:8501

---

## 🐳 Running with Docker (Recommended)

Run entire system with one command:

docker compose up --build

---

Then open:

FastAPI Docs
http://localhost:8000/docs

Streamlit UI
http://localhost:8501

---

## 🔮 Prediction API Example

POST /predict

{
"data": {
"Gender": "Male",
"Married": "Yes",
"Dependents": "0",
"Education": "Graduate",
"Self_Employed": "No",
"ApplicantIncome": 5000,
"CoapplicantIncome": 1500,
"LoanAmount": 120,
"Loan_Amount_Term": 360,
"Credit_History": 1,
"Property_Area": "Urban"
}
}

Response:

default_prediction
default_probability

---

## 🌍 Deployment

The application can be deployed using Docker on cloud platforms such as:

* Render
* Railway
* AWS
* Azure
* Google Cloud

---

## 👤 Author

Sai Vishnu Vardhan Katroju
Machine Learning | Data Science | AI Systems

---

## ⭐ If you found this project useful

Please consider giving it a star ⭐
