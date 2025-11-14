from fastapi import FastAPI

from routers.loads import router as loads_router
from routers.negotiation import router as negotiation_router
from routers.carrier import router as carrier_router
from routers.metrics import router as metrics_router
from routers.call_log import router as call_log_router

import uvicorn

app = FastAPI(
    title="Inbound Carrier Search API",
    description="API for load searching, carrier verification, and inbound call flow support."
)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "message": "Inbound Carrier Search API running"}


# Register routers
app.include_router(loads_router)
app.include_router(negotiation_router)
app.include_router(carrier_router)
app.include_router(metrics_router)
app.include_router(call_log_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)