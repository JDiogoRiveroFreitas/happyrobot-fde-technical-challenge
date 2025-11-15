from fastapi import APIRouter, Depends
from typing import List

from models import LoadMatch, LoadSearchRequest, LoadSearchResponse
from dependencies import verify_api_key
from config import df

router = APIRouter(
    prefix="/loads",
    tags=["loads"]
)


@router.post(
    "/search",
    response_model=LoadSearchResponse,
    summary="Search available loads"
)
def search_loads(
    payload: LoadSearchRequest,
    _: None = Depends(verify_api_key)
) -> LoadSearchResponse:
    """
    Main endpoint used by HappyRobot to search for loads.

    Filters applied:
    - origin, destination, equipment_type (exact match, case-insensitive)
    - min_rate / max_rate (optional)
    - earliest_pickup / latest_delivery (optional)
    """

    # Base matching
    subset = df[
        (df["origin"].str.lower() == payload.origin.lower()) &
        (df["equipment_type"].str.lower() == payload.equipment_type.lower())
    ]

    # Optional filters
    if payload.destination is not None and "destination" in subset.columns:
        subset = subset[subset["destination"] == payload.destination]
    
    if payload.min_rate is not None and "loadboard_rate" in subset.columns:
        subset = subset[subset["loadboard_rate"] >= payload.min_rate]

    if payload.max_rate is not None and "loadboard_rate" in subset.columns:
        subset = subset[subset["loadboard_rate"] <= payload.max_rate]

    if payload.earliest_pickup is not None and "pickup_datetime" in subset.columns:
        subset = subset[subset["pickup_datetime"] >= payload.earliest_pickup]

    if payload.latest_delivery is not None and "delivery_datetime" in subset.columns:
        subset = subset[subset["delivery_datetime"] <= payload.latest_delivery]

    # If no matches found
    if subset.empty:
        return LoadSearchResponse(status="no_match", matches=[])

    # Limit result count
    subset = subset.head(payload.max_results)

    # Build response objects
    matches: List[LoadMatch] = []
    for _, row in subset.iterrows():
        matches.append(
            LoadMatch(
                load_id=str(row.get("load_id")),
                origin=row.get("origin"),
                destination=row.get("destination"),
                pickup_datetime=row.get("pickup_datetime"),
                delivery_datetime=row.get("delivery_datetime"),
                equipment_type=row.get("equipment_type"),
                loadboard_rate=row.get("loadboard_rate"),
                notes=row.get("notes"),
                weight=row.get("weight"),
                commodity_type=row.get("commodity_type"),
                miles=row.get("miles")
            )
        )

    return LoadSearchResponse(status="matches_found", matches=matches)