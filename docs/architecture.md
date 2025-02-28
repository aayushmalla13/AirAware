# Architecture

## High-Level Components
- Data layer: processed parquet, ERA5 cache
- Modeling layer: Prophet, PatchTST, ensembling, bias correction
- Serving layer: FastAPI services
- Presentation: Streamlit UI

## Component Diagram (textual)
- UI (Streamlit)
  -> API `/forecast`, `/models`, `/explainability`, `/what-if`
- API (FastAPI)
  -> `ForecastService`, `ModelService`, `ExplainabilityService`
- Models
  -> ProphetBaseline, PatchTSTForecaster
- Features
  -> `ERA5LandExtractor`, bias correctors

## Data Flow
1) Request arrives at API
2) Load models (best auto-resolved)
3) Extract/prepare features (ERA5 cache/fallback)
4) Predict per model, apply bias correction
5) Blend predictions, add uncertainty
6) Return to UI

## Deployment Notes
- Stateless API; artifacts on disk under `data/artifacts`
- Caching for models and ERA5 to reduce latency
