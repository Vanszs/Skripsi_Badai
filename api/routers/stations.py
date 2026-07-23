from fastapi import APIRouter
from api.schemas import StationListResponse, StationInfo

router = APIRouter(prefix="/api/v1/stations", tags=["Stations"])

STATIONS_DATA = [
    StationInfo(
        id="Puncak",
        name="Pos Puncak Pangrango",
        latitude=-6.769797,
        longitude=106.963583,
        elevation_mdpl=2958.0,
        role="Target & Dynamic Node",
        era5_grid="(-6.75, 107.00)"
    ),
    StationInfo(
        id="Lereng_Cibodas",
        name="Pos Lereng Cibodas",
        latitude=-6.751722,
        longitude=106.987160,
        elevation_mdpl=1275.0,
        role="Dynamic Conditioning Node",
        era5_grid="(-6.75, 107.00)"
    ),
    StationInfo(
        id="Hilir_Cianjur",
        name="Pos Hilir Cianjur",
        latitude=-6.816000,
        longitude=107.133000,
        elevation_mdpl=450.0,
        role="Dynamic Conditioning Node",
        era5_grid="(-6.75, 107.25)"
    ),
]

@router.get("", response_model=StationListResponse)
def get_stations():
    """Mendapatkan daftar node stasiun pengamatan Gede-Pangrango."""
    return StationListResponse(total=len(STATIONS_DATA), stations=STATIONS_DATA)
