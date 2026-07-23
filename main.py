import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add current root directory and core engine directory to python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from api.routers import dashboard, metrics, nowcast, stations

app = FastAPI(
    title="RA-Diffusion Gede-Pangrango Weather Nowcasting API",
    description="Backend API untuk Nowcasting Probabilistik Cuaca Multi-Variabel di Gunung Gede-Pangrango (Retrieval-Augmented Diffusion Model + Spatio-Temporal GNN)",
    version="1.0.0",
)

# CORS Middleware for Vite Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(stations.router)
app.include_router(metrics.router)
app.include_router(nowcast.router)
app.include_router(dashboard.router)

# Mount plots static files if core/results/result_test/plots exists
plots_dir = os.path.join(CORE_DIR, "results", "result_test", "plots")
if os.path.exists(plots_dir):
    app.mount("/static/plots", StaticFiles(directory=plots_dir), name="plots")

@app.get("/")
def root():
    return {
        "status": "online",
        "system": "RA-Diffusion Gede-Pangrango Weather Nowcasting API",
        "docs": "/docs",
        "endpoints": [
            "/api/v1/stations",
            "/api/v1/metrics",
            "/api/v1/nowcast/predict",
            "/api/v1/dashboard/live"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
