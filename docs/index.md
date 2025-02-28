# AirAware Documentation

Welcome to the AirAware docs. This site covers the end-to-end PM2.5 nowcasting system including data flow, algorithms, architecture, and the codebase organization.

## Quick Start

### Docker (Recommended)
```bash
docker compose up --build
# API: http://localhost:8000
# UI: http://localhost:8501
```

### Local Development
```bash
python -m airaware.api.app  # API
streamlit run airaware/ui/app.py  # UI
```

## Documentation Structure

Use the left navigation to explore:
- **Workflow**: data → models → ensembling → API → UI
- **Tech Stack**: languages, frameworks, libraries, containerization
- **Algorithms**: Prophet, PatchTST, residual-based ensembling, bias learning
- **Architecture**: components, interfaces, deployment view, microservices
- **Codebase**: where things live and how to extend
- **Docker Setup**: containerization, deployment, and management
