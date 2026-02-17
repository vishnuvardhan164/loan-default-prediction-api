import streamlit as st
import requests

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Loan Default Risk Predictor")
st.write("Enter loan applicant details and get a default risk prediction.")

import os
API_URL = st.text_input(
    "FastAPI URL",
    os.getenv("API_URL", "http://127.0.0.1:8000/predict")
)


st.subheader("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000, step=100)
    coapp_income = st.number_input("Coapplicant Income", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=120, step=10)
    loan_term = st.number_input("Loan Amount Term", min_value=0, value=360, step=12)
    credit_history = st.selectbox("Credit History", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

payload = {
    "data": {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapp_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }
}

if st.button("Predict Default Risk"):
    try:
        resp = requests.post(API_URL, json=payload, timeout=15)
        if resp.status_code != 200:
            st.error(f"API Error {resp.status_code}: {resp.text}")
        else:
            result = resp.json()
            pred = result.get("default_prediction")
            prob = result.get("default_probability")

            st.success("✅ Prediction completed")
            st.write(f"**Default Prediction:** `{pred}`  (1 = risky, 0 = safe)")
            st.write(f"**Default Probability:** `{prob:.4f}`")

    except Exception as e:
        st.error(f"Request failed: {e}")

st.caption("Tip: Keep FastAPI running (uvicorn) while using this UI.")
