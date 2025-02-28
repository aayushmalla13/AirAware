# Codebase Overview

## Key Directories
- `airaware/api/`
  - `services.py`: `ModelService`, `ForecastService`, `ExplainabilityService`, `WhatIfService`
  - `app.py`: FastAPI entrypoint and routes
- `airaware/ui/`
  - `app.py`: Streamlit app (Forecast, Explainability, What-If, Agents, Cross-Station)
- `airaware/features/`
  - `era5_land_extractor.py`: ERA5-Land fetch + caching + derived features
  - `bias_correction.py`: Hourly and Adaptive bias correctors
- `airaware/deep_models/`
  - `patchtst.py`: PatchTST model, train/eval, save/load
- `scripts/`
  - `deep_models_cli.py`: train/evaluate/compare deep models
  - `ensemble_backtest_cli.py`: tune blend threshold and weight floor
  - other utilities: evaluation, visualization, residual analysis

## How to Extend
- Add new model: create forecaster class, wire into `ModelService.load_model`
- Add new features: extend extractor and pass to Prophet/forecasters
- Add new endpoints: implement service method, expose route in `api/app.py`

## Conventions
- Artifacts in `data/artifacts/*` with a `best/` stable symlink
- Processed data in `data/processed/`
- Environment variables override tuned configs when present
