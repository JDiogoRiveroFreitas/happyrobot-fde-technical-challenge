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

    # Carrier is asking at or below our target -> best for us -> accept immediately
    if proposed <= anchor_rate:
        decision = "accept"
        counter_offer = None
        comment = "We can work with that rate."

    # Carrier is slightly above our target (up to +10%) -> reasonable -> counter down to anchor
    elif proposed <= anchor_rate * 1.10:
        decision = "counter"
        # Never offer more than what the carrier asked, and never exceed our anchor
        counter_offer = round(min(proposed, anchor_rate), 2)
        comment = "You're slightly above our target. Could you meet our target rate?"

    # Carrier is far above market -> reject or counter with a firm lower offer
    else:
        decision = "reject"
        # small concession (96% of anchor) to show our best realistic rate
        counter_offer = round(anchor_rate * 0.96, 2)
        comment = "Your offer is too high for this lane. This would be our best rate."

    return OfferEvaluationResponse(
        load_id=str(payload.load_id),
        proposed_rate=proposed,
        anchor_rate=anchor_rate,
        decision=decision,
        counter_offer=counter_offer,
        comment=comment
    )