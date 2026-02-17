import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

from src.config import RAW_DATA_PATH, MODEL_DIR, MODEL_PATH, FEATURES_PATH, TARGET_CANDIDATES, ID_COLUMNS
from src.utils import find_target_column, normalize_target, save_json

def train():
    df = pd.read_csv(RAW_DATA_PATH)

    # Drop obvious ID columns if present
    for col in ID_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    target_col = find_target_column(df, TARGET_CANDIDATES)
    if not target_col:
        raise ValueError(
            f"Target column not found. Expected one of: {TARGET_CANDIDATES}. "
            f"Found columns: {list(df.columns)}"
        )

    # Split X/y
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # Normalize target to numeric 0/1
    y = normalize_target(y_raw)

    # Remove rows where target became NaN after normalization
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].astype(int)

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Simple, explainable baseline model
    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 4))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    # Save raw feature columns (expected input schema)
    save_json(FEATURES_PATH, {"feature_columns": X.columns.tolist()})

    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved expected input columns to: {FEATURES_PATH}")
    print(f"Target column used: {target_col}")

if __name__ == "__main__":
    train()
