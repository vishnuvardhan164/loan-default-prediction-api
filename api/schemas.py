from pydantic import BaseModel, ConfigDict
from typing import Any, Dict


class LoanApplication(BaseModel):
    """
    Flexible input schema for loan prediction.
    Accepts any loan feature columns as dictionary.
    """

    model_config = ConfigDict(extra="allow")
    data: Dict[str, Any]
