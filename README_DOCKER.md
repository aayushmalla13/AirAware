# AirAware Docker Setup

This document provides comprehensive instructions for running the AirAware PM2.5 forecasting system using Docker.

## Quick Start

### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 4GB RAM available for containers
- 10GB free disk space for data and models

### Basic Usage

1. **Build and start all services:**
   ```bash
   # Using the helper script
   ./scripts/docker_helper.sh build
   ./scripts/docker_helper.sh up --detach
   
   # Or using docker-compose directly
   docker-compose up --build -d
   ```

2. **Access the services:**
   - **API**: http://localhost:8000
   - **UI**: http://localhost:8501
   - **API Docs**: http://localhost:8000/docs
   - **Health Check**: http://localhost:8000/health

3. **Stop services:**
   ```bash
   ./scripts/docker_helper.sh down
   # or
   docker-compose down
   ```

## Service Architecture

### Core Services

- **`api`**: FastAPI backend service (port 8000)
- **`ui`**: Streamlit frontend service (port 8501)
- **`processor`**: Data processing service (batch jobs)
- **`docs`**: MkDocs documentation service (port 8001)

### Optional Services

- **`redis`**: Caching service (port 6379)
- **`postgres`**: Database service (port 5432)

## Docker Commands

### Using the Helper Script

The `scripts/docker_helper.sh` script provides convenient commands:

```bash
# Build images
./scripts/docker_helper.sh build [--no-cache]

# Start services
./scripts/docker_helper.sh up [service] [--detach]

# Stop services
./scripts/docker_helper.sh down

# View logs
./scripts/docker_helper.sh logs [--follow]
./scripts/docker_helper.sh logs-api
./scripts/docker_helper.sh logs-ui

# Open shell in container
./scripts/docker_helper.sh shell

# Check status and health
./scripts/docker_helper.sh status
./scripts/docker_helper.sh health

# Clean up
./scripts/docker_helper.sh clean
```

### Using Docker Compose Directly

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d api

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build -d

# Scale services
docker-compose up -d --scale api=3
```

## Development Mode

For development with live reload:

```bash
# Use development compose file
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Or build with development Dockerfile
docker-compose -f docker-compose.dev.yml up --build
```

## Service Profiles

### Processing Profile
Run data processing jobs:
```bash
docker-compose --profile processing up processor
```

### Documentation Profile
Start documentation server:
```bash
docker-compose --profile docs up docs
```

### Cache Profile
Include Redis caching:
```bash
docker-compose --profile cache up
```

### Database Profile
Include PostgreSQL database:
```bash
docker-compose --profile database up
```

## Data Persistence

### Volume Mounts

The following directories are mounted as volumes:
- `./data` → `/app/data` (data files)
- `./results` → `/app/results` (model results)
- `./logs` → `/app/logs` (application logs)
- `./configs` → `/app/configs` (configuration files)

### Data Directory Structure

```
data/
├── artifacts/          # Model artifacts and checkpoints
├── cache/             # Cached processed data
├── interim/           # Intermediate processing files
├── processed/         # Final processed datasets
├── raw/              # Raw data files
└── test_data/        # Test datasets
```

## Environment Variables

### API Service
- `ENVIRONMENT`: `production` or `development`
- `PYTHONPATH`: `/app`
- `LOG_LEVEL`: `info`, `debug`, `warning`, `error`

### UI Service
- `ENVIRONMENT`: `production` or `development`
- `PYTHONPATH`: `/app`

## Health Checks

### API Health Check
```bash
curl http://localhost:8000/health
```

### UI Health Check
```bash
curl http://localhost:8501
```

### Container Health Status
```bash
docker-compose ps
```

## Troubleshooting

### Common Issues

1. **Port conflicts:**
   ```bash
   # Check what's using the ports
   lsof -i :8000
   lsof -i :8501
   
   # Change ports in docker-compose.yml
   ports:
     - "8001:8000"  # Use port 8001 instead
   ```

2. **Permission issues:**
   ```bash
   # Fix file permissions
   sudo chown -R $USER:$USER data/ results/ logs/
   ```

3. **Out of memory:**
   ```bash
   # Check container resource usage
   docker stats
   
   # Increase Docker memory limit in Docker Desktop
   ```

4. **Build failures:**
   ```bash
   # Clean build
   docker-compose down
   docker system prune -f
   docker-compose build --no-cache
   ```

### Logs and Debugging

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs ui

# Follow logs in real-time
docker-compose logs -f

# View logs with timestamps
docker-compose logs -t
```

### Container Shell Access

```bash
# Access API container
docker-compose exec api /bin/bash

# Access UI container
docker-compose exec ui /bin/bash

# Run commands in container
docker-compose exec api python scripts/start_api.py --help
```

## Production Deployment

### Security Considerations

1. **Use secrets for sensitive data:**
   ```yaml
   services:
     api:
       secrets:
         - api_key
         - database_password
   
   secrets:
     api_key:
       file: ./secrets/api_key.txt
     database_password:
       file: ./secrets/db_password.txt
   ```

2. **Use production environment:**
   ```yaml
   environment:
     - ENVIRONMENT=production
     - LOG_LEVEL=warning
   ```

3. **Limit resource usage:**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 2G
         cpus: '1.0'
   ```

### Scaling

```bash
# Scale API service
docker-compose up -d --scale api=3

# Use load balancer (nginx)
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

## Monitoring

### Resource Monitoring

```bash
# Container resource usage
docker stats

# Service status
docker-compose ps

# Health checks
./scripts/docker_helper.sh health
```

### Application Monitoring

- **API Metrics**: http://localhost:8000/metrics
- **Health Endpoint**: http://localhost:8000/health
- **Logs**: Available in `./logs/` directory

## Backup and Recovery

### Data Backup

```bash
# Backup data directory
tar -czf airaware_data_backup_$(date +%Y%m%d).tar.gz data/

# Backup results
tar -czf airaware_results_backup_$(date +%Y%m%d).tar.gz results/
```

### Container Backup

```bash
# Save container state
docker-compose down
docker save airaware_api:latest | gzip > airaware_api_backup.tar.gz
```

## Performance Optimization

### Build Optimization

1. **Use multi-stage builds** (already implemented)
2. **Leverage build cache** with proper layer ordering
3. **Use .dockerignore** to exclude unnecessary files

### Runtime Optimization

1. **Use production images** for deployment
2. **Configure resource limits**
3. **Use health checks** for better container management
4. **Implement proper logging** and monitoring

## Support

For issues and questions:
1. Check the logs: `./scripts/docker_helper.sh logs`
2. Verify health: `./scripts/docker_helper.sh health`
3. Review this documentation
4. Check the main project README.md

## Contributing

When adding new services or modifying Docker configuration:

1. Update this README
2. Test with both development and production configurations
3. Ensure proper health checks are implemented
4. Update the helper script if needed

