# Workflow

## Data pipeline
- Input: station PM2.5 measurements (joined 60-day window) at `data/processed/joined_data.parquet`
- QC: deduplication, clipping outliers, timezone alignment
- Weather: ERA5-Land via Open-Meteo with local caching and fallbacks

## Modeling
- Prophet (with tuned seasonality and ERA5 regressors)
- PatchTST (context-length 192, horizon 24; trained on rolling 60 days)
- Hourly Bias Learning (station × hour smoothing)

## Ensembling
- Residual-based adaptive blending of Prophet and PatchTST
- Thresholded blending with weight floor; inverse-error weighting
- Tuned via 60-day backtest to minimize MAE

## Serving
- FastAPI service loads best artifacts (Prophet params, PatchTST best)
- Ensemble + uncertainty generation in `ForecastService`
- Streamlit UI displays forecasts, uncertainty, badges, and analysis
