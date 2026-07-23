import os
import json
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/metrics", tags=["Evaluation Metrics"])

SUMMARY_PATH = "core/results/result_test/comparison/comparison_summary.json"

@router.get("")
def get_metrics_summary():
    """Mendapatkan hasil perbandingan evaluasi skenario model."""
    if not os.path.exists(SUMMARY_PATH):
        raise HTTPException(status_code=404, detail="Metrics summary not found")

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data
