import json
import pandas as pd
from typing import Optional, List


def find_target_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Finds which column in dataframe is the target label.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_target(y: pd.Series) -> pd.Series:
    """
    Converts Loan_Status column (Y/N) to numeric 1/0.
    """
    return y.map({"Y": 1, "N": 0})

def save_json(path, obj) -> None:
    """
    Saves dictionary to JSON file.
    """
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    """
    Loads JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
