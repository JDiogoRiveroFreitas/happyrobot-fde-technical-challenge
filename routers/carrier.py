import requests
from fastapi import APIRouter, Depends, HTTPException

from models import CarrierVerificationResponse
from dependencies import verify_api_key
from config import FMCSA_API_KEY

router = APIRouter(
    prefix="/carrier",
    tags=["carrier"]
)


@router.get(
    "/verify/{mc_number}",
    response_model=CarrierVerificationResponse,
    summary="Verify a carrier using FMCSA"
)
def verify_carrier(
    mc_number: str,
    _: None = Depends(verify_api_key)
) -> CarrierVerificationResponse:
    """
    Thin wrapper around the FMCSA API.

    Notes:
    - Requires FMCSA_API_KEY environment variable.
    - HappyRobot only needs eligible=True/False.
    """

    if FMCSA_API_KEY is None:
        return CarrierVerificationResponse(
            mc_number=mc_number,
            eligible=True,
            status="mock",
            reason="FMCSA_API_KEY not configured; returning eligible=True by default.",
            raw_fmcsa_response=None
        )

    url = f"https://mobile.fmcsa.dot.gov/qc/services/carriers/{mc_number}"
    params = {"webKey": FMCSA_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"FMCSA API request failed: {exc}"
        )

    carrier_data = data.get("content", [{}])[0] if isinstance(data.get("content"), list) else {}
    status = carrier_data.get("carrierStatus", "UNKNOWN")
    allowed = carrier_data.get("allowedToOperate", "Y") == "Y"

    return CarrierVerificationResponse(
        mc_number=mc_number,
        eligible=allowed,
        status=status,
        reason=None if allowed else "Carrier is not authorized to operate according to FMCSA.",
        raw_fmcsa_response=data
    )