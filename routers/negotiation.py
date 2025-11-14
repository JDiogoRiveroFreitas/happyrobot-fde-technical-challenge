from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, Literal

from models import OfferEvaluationRequest, OfferEvaluationResponse
from dependencies import verify_api_key
from config import df

router = APIRouter(
    prefix="/negotiation",
    tags=["negotiation"]
)


@router.post(
    "/evaluate_offer",
    response_model=OfferEvaluationResponse,
    summary="Evaluate a carrier price offer"
)
def evaluate_offer(
    payload: OfferEvaluationRequest,
    _: None = Depends(verify_api_key)
) -> OfferEvaluationResponse:
    """
    Helper endpoint used during rate negotiation.

    Simple logic:
    - If proposed_rate ≥ 97% of anchor_rate → accept
    - If 90% ≤ proposed_rate < 97% → counter
    - If proposed_rate < 90% → reject
    """

    load_row = df[df["load_id"].astype(str) == str(payload.load_id)]
    if load_row.empty:
        raise HTTPException(status_code=404, detail="Load not found.")

    row = load_row.iloc[0]
    anchor_rate = payload.anchor_rate or float(row["loadboard_rate"])

    proposed = payload.proposed_rate
    decision: Literal["accept", "counter", "reject"]
    counter_offer: Optional[float] = None

    if proposed >= 0.97 * anchor_rate:
        decision = "accept"
        comment = "We accept your offer."
    elif proposed >= 0.90 * anchor_rate:
        decision = "counter"
        counter_offer = round(anchor_rate * 0.98, 2)
        comment = "Your offer is close. Can we meet in the middle?"
    else:
        decision = "reject"
        counter_offer = round(anchor_rate * 0.95, 2)
        comment = "Your offer is too low. This would be our best price."

    return OfferEvaluationResponse(
        load_id=str(payload.load_id),
        proposed_rate=proposed,
        anchor_rate=anchor_rate,
        decision=decision,
        counter_offer=counter_offer,
        comment=comment
    )