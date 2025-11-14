from fastapi import APIRouter, Depends

from dependencies import verify_api_key
from config import df

router = APIRouter(
    prefix="/metrics",
    tags=["metrics"]
)


@router.get(
    "/load_metrics",
    summary="Get basic load dataset metrics"
)
def load_metrics(
    _: None = Depends(verify_api_key)
) -> dict:
    """
    Returns basic dataset metrics for dashboards or monitoring.
    """
    total_loads = int(len(df))
    equipment_counts = df["equipment_type"].value_counts(dropna=False).to_dict() if "equipment_type" in df.columns else {}
    origin_counts = df["origin"].value_counts().head(10).to_dict() if "origin" in df.columns else {}
    destination_counts = df["destination"].value_counts().head(10).to_dict() if "destination" in df.columns else {}

    rate_stats = {}
    if "loadboard_rate" in df.columns:
        rate_stats = {
            "min": float(df["loadboard_rate"].min()),
            "max": float(df["loadboard_rate"].max()),
            "mean": float(df["loadboard_rate"].mean())
        }

    return {
        "total_loads": total_loads,
        "equipment_distribution": equipment_counts,
        "top_origins": origin_counts,
        "top_destinations": destination_counts,
        "rate_stats": rate_stats
    }