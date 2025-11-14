from fastapi import Header, HTTPException
from config import API_KEY


def verify_api_key(x_api_key: str = Header(...)) -> None:
    """
    Validates that the correct API key was provided in the X API Key header.
    """
    if API_KEY is None:
        raise HTTPException(
            status_code=500,
            detail="API key not configured on server (INBOUND_API_KEY)."
        )
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key."
        )