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

## Microservices Architecture

### Docker Services
- **API Service**: FastAPI backend with health checks
- **UI Service**: Streamlit frontend with live reloading
- **Processor Service**: Batch job processing
- **Docs Service**: MkDocs documentation server
- **Redis** (optional): Caching layer
- **PostgreSQL** (optional): Database storage

### Service Communication
- UI → API: HTTP requests with configurable base URL
- API → External: ERA5 API, model artifacts
- Inter-service: Docker networking with service names

### Container Configuration
- Multi-stage builds for optimized images
- Environment-based configuration
- Volume mounts for data persistence
- Health checks for service monitoring
- Restart policies for reliability
