from fastapi import APIRouter, Depends
from models import CallLogPayload

from dependencies import verify_api_key  

import csv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent

router = APIRouter(
  prefix="/call-log",
  tags=["call-log"]
)

@router.post(
  "",
    summary="Receive final call log from HappyRobot"
)
def log_call(
    payload: CallLogPayload,
    _: None = Depends(verify_api_key)
) -> dict:
    """
    Final webhook for storing the full call information.

    HappyRobot will call this endpoint once per call, after:
    - AI Classification (outcome, sentiment)
    - AI Extract (structured fields)
    """

    # Save to CSV (persistent call log)
    csv_path = BASE_DIR / "data" / "call_logs.csv"
    csv_path.parent.mkdir(exist_ok=True)

    # Debug log to confirm the function is being called and path used
    print(f"[call-log] Saving call log to: {csv_path}")

    try:
        # Pydantic v2 uses model_dump; if you are on v1, replace with dict()
        row = payload.model_dump()
    except AttributeError:
        row = payload.dict()

    row["logged_at"] = datetime.utcnow().isoformat()

    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return {
        "status": "ok",
        "message": "Call log stored successfully"
    }