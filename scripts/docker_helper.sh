#!/bin/bash
# Docker Helper Script for AirAware Project

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "AirAware Docker Helper Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  build           Build Docker images"
    echo "  up              Start all services"
    echo "  down            Stop all services"
    echo "  restart         Restart all services"
    echo "  logs            Show logs for all services"
    echo "  logs-api        Show logs for API service"
    echo "  logs-ui         Show logs for UI service"
    echo "  shell           Open shell in API container"
    echo "  clean           Clean up Docker resources"
    echo "  status          Show status of all services"
    echo "  health          Check health of all services"
    echo ""
    echo "Service-specific commands:"
    echo "  api             Start only API service"
    echo "  ui              Start only UI service"
    echo "  docs            Start documentation service"
    echo "  processing      Start data processing service"
    echo ""
    echo "Options:"
    echo "  --no-cache      Build without cache"
    echo "  --detach        Run in background"
    echo "  --follow        Follow logs"
    echo ""
    echo "Examples:"
    echo "  $0 build --no-cache"
    echo "  $0 up --detach"
    echo "  $0 logs --follow"
    echo "  $0 shell"
}

# Function to build Docker images
build_images() {
    local no_cache=""
    if [[ "$1" == "--no-cache" ]]; then
        no_cache="--no-cache"
        print_warning "Building without cache..."
    fi
    
    print_status "Building Docker images..."
    docker compose build $no_cache
    print_success "Docker images built successfully!"
}

# Function to start services
start_services() {
    local service="$1"
    local detach=""
    
    if [[ "$2" == "--detach" ]]; then
        detach="-d"
    fi
    
    if [[ -n "$service" ]]; then
        print_status "Starting $service service..."
        docker compose up $detach $service
    else
        print_status "Starting all services..."
        docker compose up $detach
    fi
    
    print_success "Services started successfully!"
}

# Function to stop services
stop_services() {
    print_status "Stopping all services..."
    docker compose down
    print_success "Services stopped successfully!"
}

# Function to restart services
restart_services() {
    print_status "Restarting all services..."
    docker compose restart
    print_success "Services restarted successfully!"
}

# Function to show logs
show_logs() {
    local service="$1"
    local follow=""
    
    if [[ "$2" == "--follow" ]]; then
        follow="-f"
    fi
    
    if [[ -n "$service" ]]; then
        print_status "Showing logs for $service service..."
        docker compose logs $follow $service
    else
        print_status "Showing logs for all services..."
        docker compose logs $follow
    fi
}

# Function to open shell in container
open_shell() {
    print_status "Opening shell in API container..."
    docker compose exec api /bin/bash
}

# Function to clean up Docker resources
cleanup() {
    print_warning "This will remove all Docker resources for AirAware..."
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Cleaning up Docker resources..."
        docker compose down -v --remove-orphans
        docker system prune -f
        print_success "Cleanup completed!"
    else
        print_status "Cleanup cancelled."
    fi
}

# Function to show status
show_status() {
    print_status "Service status:"
    docker compose ps
}

# Function to check health
check_health() {
    print_status "Checking service health..."
    
    # Check API health
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        print_success "API service is healthy"
    else
        print_error "API service is not responding"
    fi
    
    # Check UI health
    if curl -f http://localhost:8501 >/dev/null 2>&1; then
        print_success "UI service is healthy"
    else
        print_error "UI service is not responding"
    fi
}

# Main script logic
case "$1" in
    "build")
        build_images "$2"
        ;;
    "up")
        start_services "$2" "$3"
        ;;
    "down")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "logs")
        show_logs "$2" "$3"
        ;;
    "logs-api")
        show_logs "api" "$2"
        ;;
    "logs-ui")
        show_logs "ui" "$2"
        ;;
    "shell")
        open_shell
        ;;
    "clean")
        cleanup
        ;;
    "status")
        show_status
        ;;
    "health")
        check_health
        ;;
    "api")
        start_services "api" "$2"
        ;;
    "ui")
        start_services "ui" "$2"
        ;;
    "docs")
        start_services "docs" "$2"
        ;;
    "processing")
        start_services "processor" "$2"
        ;;
    "help"|"-h"|"--help"|"")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
