from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

from api.schemas import LoanApplication
from src.config import MODEL_PATH, FEATURES_PATH
from src.utils import load_json

app = FastAPI(title="Loan Default Prediction API")

# Load trained model
model = joblib.load(MODEL_PATH)

# Load expected feature columns
feature_cols = load_json(FEATURES_PATH)["feature_columns"]


@app.get("/")
def root():
    return {"message": "Loan Default Prediction API is running"}


@app.get("/schema")
def schema():
    return {"expected_feature_columns": feature_cols}


@app.post("/predict")
def predict(payload: LoanApplication):
    incoming = payload.data

    # Build dataframe with expected columns
    row = {col: incoming.get(col, None) for col in feature_cols}
    df = pd.DataFrame([row])

    try:
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "default_prediction": pred,
        "default_probability": prob
    }
