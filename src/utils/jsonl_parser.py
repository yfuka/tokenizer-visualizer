import pandas as pd
from typing import Dict, Any

FORMAT_MESSAGES = "messages"
FORMAT_PROMPT_RESPONSE = "prompt_response"
FORMAT_TEXT = "text"
FORMAT_UNKNOWN = "unknown"

def detect_format(row_data: Dict[str, Any]) -> str:
    """
    Detects the format of a JSONL row.
    """
    # 1. Messages Format
    if "messages" in row_data and isinstance(row_data["messages"], list):
        return FORMAT_MESSAGES
    
    # 2. Prompt/Response Format
    # Check for presence AND non-null values
    if "prompt" in row_data and "response" in row_data:
        # Check if values are not NaN (if using pandas, row_data might have NaNs)
        if pd.notna(row_data["prompt"]) and pd.notna(row_data["response"]):
            return FORMAT_PROMPT_RESPONSE
            
    # 3. Text Format
    if "text" in row_data:
        if pd.notna(row_data["text"]):
            return FORMAT_TEXT
            
    return FORMAT_UNKNOWN
