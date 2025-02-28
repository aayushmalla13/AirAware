#!/usr/bin/env python3
"""
CLI: Backtest residual-based ensemble blending over last N days and tune thresholds.

Sweeps:
- AIRWARE_BLEND_DISTANCE_UG (e.g., 1.5, 2, 3, 4, 5)
- AIRWARE_WEIGHT_FLOOR (e.g., 0.1, 0.2, 0.25, 0.3)

Outputs best settings to data/artifacts/ensemble_tuning.json
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from airaware.api.services import ModelService, ForecastService
from airaware.api.models import ForecastRequest, ModelType, Language


def evaluate_mae(actual: List[float], pred: List[float]) -> float:
    m = min(len(actual), len(pred))
    if m == 0:
        return float("inf")
    a = np.array(actual[:m], dtype=float)
    p = np.array(pred[:m], dtype=float)
    return float(np.mean(np.abs(a - p)))


def backtest(params: Tuple[float, float], data: pd.DataFrame, station_ids: List[str], horizon: int) -> float:
    blend_distance, weight_floor = params
    # Inject env vars for this run
    os.environ["AIRWARE_BLEND_DISTANCE_UG"] = str(blend_distance)
    os.environ["AIRWARE_WEIGHT_FLOOR"] = str(weight_floor)

    # Fresh services to pick up env vars
    ms = ModelService()
    fs = ForecastService(ms)

    maes: List[float] = []
    for sid in station_ids:
        sd = data[data["station_id"] == int(sid)].copy()
        if sd.empty:
            continue
        # Actual next-horizon values
        actual = sd.tail(horizon)["pm25"].tolist()
        # Request forecast
        req = ForecastRequest(
            station_ids=[sid], horizon_hours=horizon, model_type=ModelType.PROPHET, uncertainty_level=0.9, language=Language.EN
        )
        resp = fs.generate_forecast(req)
        fc = resp.station_forecasts.get(sid, [])
        preds = [pt.pm25_mean for pt in fc]
        mae = evaluate_mae(actual, preds)
        if np.isfinite(mae):
            maes.append(mae)

    return float(np.mean(maes)) if maes else float("inf")


def main():
    parser = argparse.ArgumentParser(description="Tune ensemble blending thresholds over last N days")
    parser.add_argument("--data-path", default="data/processed/joined_data.parquet", help="Processed data path")
    parser.add_argument("--days", type=int, default=60, help="Backtest window in days")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (hours)")
    parser.add_argument("--stations", type=str, nargs="*", default=None, help="Optional subset of station_ids")
    parser.add_argument("--out", default="data/artifacts/ensemble_tuning.json", help="Output JSON path")
    parser.add_argument("--fast", action="store_true", help="Sample fewer stations for speed")

    args = parser.parse_args()

    df = pd.read_parquet(args.data_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    # Ensure UTC-aware cutoff
    cutoff = pd.Timestamp.utcnow().tz_convert("UTC") - pd.Timedelta(days=args.days)
    df = df[df["datetime_utc"] >= cutoff].copy()

    station_ids = [str(int(s)) for s in sorted(df["station_id"].unique())]
    if args.stations:
        req = set(args.stations)
        station_ids = [s for s in station_ids if s in req]
    if args.fast:
        station_ids = station_ids[:3]

    # Parameter grids
    blend_grid = [1.5, 2.0, 3.0, 4.0, 5.0]
    floor_grid = [0.10, 0.20, 0.25, 0.30]

    trials = []
    for bd in blend_grid:
        for wf in floor_grid:
            score = backtest((bd, wf), df, station_ids, args.horizon)
            trials.append({"blend_distance_ug": bd, "weight_floor": wf, "mae": score})

    # Pick best
    best = min(trials, key=lambda x: x["mae"]) if trials else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"trials": trials, "best": best}, f, indent=2)

    if best:
        print("BEST:", best)
    else:
        print("No valid trials.")


if __name__ == "__main__":
    main()


