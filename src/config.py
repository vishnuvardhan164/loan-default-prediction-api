from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "loan_default.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "loan_default_model.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

# Possible target column names
TARGET_CANDIDATES = [
    "Loan_Status",
    "loan_status",
    "Default",
    "default",
    "TARGET",
    "target",
    "Status",
    "status"
]

# Columns to drop if present
ID_COLUMNS = [
    "Loan_ID",
    "loan_id",
    "id",
    "ID",
    "customer_id",
    "applicant_id"
]
