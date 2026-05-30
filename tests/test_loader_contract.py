import pandas as pd
import unittest

from src.config import FINAL_FEATURE_COLS
from src.data.temporal_loader import TemporalGraphDataset


def _make_rows(ts, node):
    return {
        "date": ts,
        "node": node,
        "temperature_2m": 20.0,
        "relative_humidity_2m": 80.0,
        "dewpoint_2m": 18.0,
        "surface_pressure": 900.0,
        "wind_speed_10m": 2.0,
        "wind_direction_10m": 180.0,
        "cloud_cover": 50.0,
        "precipitation_lag1": 0.0,
        "elevation": 100.0,
        "precipitation": 0.1,
    }


def _build_df(node_order):
    ts1 = pd.Timestamp("2024-01-01 00:00:00+00:00")
    ts2 = pd.Timestamp("2024-01-01 01:00:00+00:00")
    rows = []
    for n in node_order:
        rows.append(_make_rows(ts1, n))
    for n in node_order:
        rows.append(_make_rows(ts2, n))
    return pd.DataFrame(rows)


class LoaderContractTest(unittest.TestCase):
    def test_loader_accepts_canonical_order(self):
        df = _build_df(["MAIN", "UP", "DOWN", "LEFT", "RIGHT"])
        ds = TemporalGraphDataset(df=df, feature_cols=FINAL_FEATURE_COLS, seq_len=1)
        self.assertEqual(len(ds), 1)

    def test_loader_rejects_wrong_order(self):
        df = _build_df(["UP", "MAIN", "DOWN", "LEFT", "RIGHT"])
        with self.assertRaisesRegex(ValueError, "Wrong node order detected"):
            TemporalGraphDataset(df=df, feature_cols=FINAL_FEATURE_COLS, seq_len=1)

    def test_loader_rejects_missing_node(self):
        df = _build_df(["MAIN", "UP", "DOWN", "LEFT"])
        with self.assertRaisesRegex(ValueError, "Missing required nodes"):
            TemporalGraphDataset(df=df, feature_cols=FINAL_FEATURE_COLS, seq_len=1)
