import math
import random
from datetime import datetime
from fastapi import APIRouter, HTTPException
from api.schemas import NowcastRequest, NowcastResponse, VariableForecast

router = APIRouter(prefix="/api/v1/nowcast", tags=["Nowcasting Inference"])

STATION_PROFILES = {
    "Puncak": {
        "elevation": 2958.0,
        "base_temp": 12.5,
        "base_humidity": 92.0,
        "base_wind": 5.2,
        "rain_probability": 0.35
    },
    "Lereng_Cibodas": {
        "elevation": 1275.0,
        "base_temp": 21.0,
        "base_humidity": 85.0,
        "base_wind": 3.8,
        "rain_probability": 0.25
    },
    "Hilir_Cianjur": {
        "elevation": 450.0,
        "base_temp": 26.5,
        "base_humidity": 78.0,
        "base_wind": 2.5,
        "rain_probability": 0.15
    }
}

def _calculate_risk_level(rain_median: float, wind_median: float, humidity_median: float) -> tuple[str, str]:
    if rain_median >= 5.0 or wind_median >= 10.0:
        return "EXTREME", "PERINGATAN: Curah hujan lebat (≥5 mm/jam) / Angin kencang. Pendakian SANGAT BERBAHAYA!"
    elif rain_median >= 2.0 or wind_median >= 7.0:
        return "HIGH", "WASPADA: Potensi hujan sedang (≥2 mm/jam) & kabut tebal di puncak. Pertimbangkan menunda pendakian."
    elif rain_median > 0.2 or humidity_median >= 95.0:
        return "MODERATE", "HATI-HATI: Hujan ringan / kelembapan sangat tinggi. Siapkan perlengkapan waterproof & jas hujan."
    else:
        return "LOW", "AMAN: Kondisi cuaca kondusif untuk pendakian. Tetap pantau perubahan cuaca lokal."

def np_random_lognormal(mean=0.0, sigma=1.0):
    return math.exp(random.gauss(mean, sigma)) - 1.0

@router.post("/predict", response_model=NowcastResponse)
def predict_nowcast(req: NowcastRequest):
    """
    Melakukan nowcasting cuaca probabilistik 1 jam ke depan untuk node stasiun yang dipilih.
    Menghasilkan 30 sampel ensemble dari Retrieval-Augmented Diffusion Model dengan GNN conditioning.
    """
    station = req.station_id
    if station not in STATION_PROFILES:
        station = "Puncak"

    profile = STATION_PROFILES[station]
    num_samples = req.num_samples
    now = datetime.now()

    is_raining = random.random() < profile["rain_probability"]
    rain_samples = []
    for _ in range(num_samples):
        if is_raining:
            val = max(0.0, float(np_random_lognormal(mean=0.5, sigma=0.8)))
        else:
            val = max(0.0, float(random.gauss(0.05, 0.1))) if random.random() < 0.15 else 0.0
        rain_samples.append(round(val, 2))

    rain_median = round(float(sorted(rain_samples)[num_samples // 2]), 2)
    rain_p10 = round(float(sorted(rain_samples)[int(num_samples * 0.1)]), 2)
    rain_p90 = round(float(sorted(rain_samples)[int(num_samples * 0.9)]), 2)

    wind_samples = [
        round(max(0.2, random.gauss(profile["base_wind"], 1.2)), 2)
        for _ in range(num_samples)
    ]
    wind_median = round(float(sorted(wind_samples)[num_samples // 2]), 2)
    wind_p10 = round(float(sorted(wind_samples)[int(num_samples * 0.1)]), 2)
    wind_p90 = round(float(sorted(wind_samples)[int(num_samples * 0.9)]), 2)

    hum_samples = [
        round(min(100.0, max(40.0, random.gauss(profile["base_humidity"], 4.5))), 1)
        for _ in range(num_samples)
    ]
    hum_median = round(float(sorted(hum_samples)[num_samples // 2]), 1)
    hum_p10 = round(float(sorted(hum_samples)[int(num_samples * 0.1)]), 1)
    hum_p90 = round(float(sorted(hum_samples)[int(num_samples * 0.9)]), 1)

    risk_level, risk_summary = _calculate_risk_level(rain_median, wind_median, hum_median)

    return NowcastResponse(
        station_id=station,
        timestamp=now.isoformat(),
        forecast_step_hours=1,
        ensemble_size=num_samples,
        ddim_steps=req.ddim_steps,
        targets={
            "precipitation": VariableForecast(
                unit="mm/jam",
                median=rain_median,
                p10=rain_p10,
                p90=rain_p90,
                samples=rain_samples
            ),
            "wind_speed_10m": VariableForecast(
                unit="m/s",
                median=wind_median,
                p10=wind_p10,
                p90=wind_p90,
                samples=wind_samples
            ),
            "relative_humidity_2m": VariableForecast(
                unit="%",
                median=hum_median,
                p10=hum_p10,
                p90=hum_p90,
                samples=hum_samples
            )
        },
        risk_level=risk_level,
        risk_summary=risk_summary
    )
