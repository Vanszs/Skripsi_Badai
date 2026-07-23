from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])


@router.get("/live")
def get_live_dashboard():
    """Contract boundary for the public dashboard's live weather snapshot.

    A production adapter must return five ERA5-cell observations, MAIN forecast
    quantiles, and optional risk assessment using the DashboardLiveSnapshot shape.
    The endpoint intentionally fails closed until that adapter is configured.
    """
    raise HTTPException(
        status_code=503,
        detail="Sumber data cuaca real-time belum dikonfigurasi",
    )
