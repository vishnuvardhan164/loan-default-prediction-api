import json
import pandas as pd
from typing import Optional, List


def find_target_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first configured target column found in the dataframe."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_target(y: pd.Series) -> pd.Series:
    """
    Convert the source Loan_Status label to default risk.

    In the source dataset:
    - Y means the loan was approved, so default-risk label = 0.
    - N means the loan was not approved, so default-risk label = 1.
    """
    return y.map({"Y": 0, "N": 1})


def save_json(path, obj) -> None:
    """Save a dictionary as JSON."""
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2)


def load_json(path):
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
