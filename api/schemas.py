from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class StationInfo(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    elevation_mdpl: float
    role: str
    era5_grid: str

class StationListResponse(BaseModel):
    total: int
    stations: List[StationInfo]

class TargetMetrics(BaseModel):
    rmse: float
    mae: float
    correlation: float

class ProbabilisticMetrics(BaseModel):
    crps: float
    brier: float
    pod: float
    far: float
    csi: float

class WeatherMetricsSummary(BaseModel):
    model_name: str
    analyzed_at: str
    test_period: str
    deterministic: Dict[str, TargetMetrics]
    probabilistic: Dict[str, ProbabilisticMetrics]
    threshold_sensitivity_precip: List[Dict[str, Any]]

class NowcastRequest(BaseModel):
    station_id: str = Field(default="Puncak", description="Station ID (Puncak, Lereng_Cibodas, Hilir_Cianjur)")
    num_samples: int = Field(default=30, ge=1, le=50, description="Ensemble size (1-50)")
    ddim_steps: int = Field(default=20, ge=5, le=50, description="DDIM sampling steps")

class VariableForecast(BaseModel):
    unit: str
    median: float
    p10: float
    p90: float
    samples: List[float]

class NowcastResponse(BaseModel):
    station_id: str
    timestamp: str
    forecast_step_hours: int = 1
    ensemble_size: int
    ddim_steps: int
    targets: Dict[str, VariableForecast]
    risk_level: str  # "LOW", "MODERATE", "HIGH", "EXTREME"
    risk_summary: str
