# AirAware - PM2.5 Nowcasting (API + UI)

AirAware is a production-grade system for short-term PM2.5 forecasting in the Kathmandu Valley. It combines classical (Prophet) and deep learning (PatchTST) models, residual-based dynamic ensembling, hourly bias learning, and ERA5-Land weather features, served via FastAPI with a Streamlit UI.

## Quick start

Prerequisites:
- Python 3.11+ (3.12 recommended)
- Linux/macOS (Windows via WSL recommended)

Create environment and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Core runtime deps
pip install -r requirements.txt  # if present
# Or install minimal set (fallback):
pip install fastapi uvicorn[standard] streamlit pandas numpy plotly requests cmdstanpy prophet torch scikit-learn
```

Run the API:
```bash
# From repo root
python -m airaware.api.app
# API up at http://127.0.0.1:8000
```

Run the UI:
```bash
# From repo root
streamlit run airaware/ui/app.py --server.port 8501 --server.address 0.0.0.0
# UI up at http://127.0.0.1:8501
```

Minimal data expectation:
- Processed dataset: `data/processed/joined_data.parquet`
- Artifacts (created/updated automatically):
  - Prophet grid: `data/artifacts/prophet_grid.json`
  - PatchTST best: `data/artifacts/deep_models/best/patchtst_model.pth`
  - Ensemble tuning: `data/artifacts/ensemble_tuning.json`

## Configuration

Environment variables (optional; sensible defaults apply):
- `AIRWARE_BLEND_DISTANCE_UG` (float): Ensemble blend threshold in µg/m³ (default 3.0; tuned result is auto-read from `ensemble_tuning.json`).
- `AIRWARE_WEIGHT_FLOOR` (float): Minimum per-model weight in the ensemble (default 0.25; tuned result auto-read).

## Training and tuning

PatchTST training (60-day window example):
```bash
python3 scripts/deep_models_cli.py patchtst \
  --data-path data/processed/joined_data.parquet \
  --output-dir data/artifacts/deep_models/sweep_long \
  --target-col pm25 --context-length 192 --prediction-length 24 \
  --epochs 30 --batch-size 64 --learning-rate 6e-5 --device auto --days 60
```
This generates `patchtst_model.pth` and `patchtst_results.json`. The service auto-resolves the best model via `data/artifacts/deep_models/best/*` symlinks or by scanning results.

Ensemble backtest (60 days):
```bash
python3 scripts/ensemble_backtest_cli.py \
  --data-path data/processed/joined_data.parquet \
  --days 60 --horizon 24 \
  --out data/artifacts/ensemble_tuning.json
```
The service auto-loads the best tuned `blend_distance_ug` and `weight_floor` values.

## Project layout (high level)
- `airaware/api/` FastAPI business logic and services
- `airaware/ui/` Streamlit application
- `airaware/features/` ERA5 extractor, bias correction
- `airaware/deep_models/` PatchTST implementation
- `scripts/` CLIs for training, evaluation, tuning
- `data/processed/` Processed datasets
- `data/artifacts/` Trained models, tuning results

## Documentation (MkDocs)
We ship a full documentation site with architecture, algorithms, and code walkthrough.

- Local preview:
```bash
pip install mkdocs mkdocs-material
mkdocs serve  # http://127.0.0.1:8000 by default
```
- Build static docs:
```bash
mkdocs build  # outputs to site/
```

Start here: `docs/index.md`. Key pages:
- Workflow and pipeline: `docs/workflow.md`
- Tech stack: `docs/techstack.md`
- Algorithms: `docs/algorithms.md`
- Architecture: `docs/architecture.md`
- Code overview: `docs/codebase.md`

## Extra materials
- Model performance and artifacts are exposed via the API `/models` endpoint and reflected in the UI badges.
- Feature importance and what-if analysis are available in dedicated UI tabs.

## Troubleshooting
- ERA5 fetch issues: the system falls back to sensible defaults; cached ERA5 data is used when available.
- Prophet backend (cmdstanpy) must be installed and able to compile/optimize; if needed, follow Prophet docs to set up CmdStan.
- If UI shows deprecation for `use_container_width`, we’ve already migrated to `width='stretch'`.

## License
Add your license here.
