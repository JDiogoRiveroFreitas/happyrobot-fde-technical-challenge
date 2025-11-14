from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime

class LoadMatch(BaseModel):
    load_id: str
    origin: str
    destination: str
    pickup_datetime: Optional[datetime] = None
    delivery_datetime: Optional[datetime] = None
    equipment_type: Optional[str] = None
    loadboard_rate: Optional[float] = None
    notes: Optional[str] = None
    weight: Optional[float] = None
    commodity_type: Optional[str] = None
    miles: Optional[float] = None


class LoadSearchRequest(BaseModel):
    origin: str
    destination: str
    equipment_type: str
    min_rate: Optional[float] = None
    max_rate: Optional[float] = None
    earliest_pickup: Optional[datetime] = None
    latest_delivery: Optional[datetime] = None
    max_results: int = 5


class LoadSearchResponse(BaseModel):
    status: Literal["no_match", "matches_found"]
    matches: List[LoadMatch] = []


class OfferEvaluationRequest(BaseModel):
    load_id: str
    proposed_rate: float
    anchor_rate: Optional[float] = None   # Defaults to loadboard_rate


class OfferEvaluationResponse(BaseModel):
    load_id: str
    proposed_rate: float
    anchor_rate: float
    decision: Literal["accept", "counter", "reject"]
    counter_offer: Optional[float] = None
    comment: str


class CarrierVerificationResponse(BaseModel):
    mc_number: str
    eligible: bool
    status: Optional[str] = None
    reason: Optional[str] = None
    raw_fmcsa_response: Optional[dict] = None