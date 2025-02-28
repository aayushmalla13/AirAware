"""Pydantic models for AirAware configuration validation."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class BoundingBox(BaseModel):
    """Geographic bounding box for the Kathmandu Valley."""
    north: float = Field(..., ge=-90, le=90, description="Northern latitude")
    south: float = Field(..., ge=-90, le=90, description="Southern latitude") 
    east: float = Field(..., ge=-180, le=180, description="Eastern longitude")
    west: float = Field(..., ge=-180, le=180, description="Western longitude")
    
    @model_validator(mode='after')
    def validate_coordinates(self):
        if self.north <= self.south:
            raise ValueError('north must be greater than south')
        if self.east <= self.west:
            raise ValueError('east must be greater than west')
        return self


class StationConfig(BaseModel):
    """Configuration for a PM₂.₅ monitoring station."""
    station_id: int = Field(..., description="Unique station identifier")
    name: str = Field(..., min_length=1, description="Human-readable station name")
    latitude: float = Field(..., ge=-90, le=90, description="Station latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Station longitude")
    distance_km: float = Field(..., ge=0, description="Distance from valley center (km)")
    quality_score: float = Field(..., ge=0, le=1, description="CP-1 quality score")
    missingness_pct: float = Field(..., ge=0, le=100, description="Data missingness percentage")
    pm25_sensor_ids: List[int] = Field(default_factory=list, description="Associated PM₂.₅ sensor IDs")
    first_measurement: Optional[datetime] = Field(None, description="First measurement timestamp")
    last_measurement: Optional[datetime] = Field(None, description="Last measurement timestamp")
    data_span_days: int = Field(default=0, ge=0, description="Total data span in days")
    selected_reason: str = Field(default="quality_threshold", description="Reason for selection")


class DataConfig(BaseModel):
    """Data configuration for the AirAware system."""
    # Geographic boundaries
    valley_bbox: BoundingBox = Field(..., description="Kathmandu Valley bounding box")
    center_lat: float = Field(27.7172, description="Valley center latitude")
    center_lon: float = Field(85.3240, description="Valley center longitude")
    search_radius_km: float = Field(25.0, ge=1, le=100, description="Station search radius")
    
    # Station selection
    stations: List[StationConfig] = Field(..., min_length=3, description="Selected PM₂.₅ stations")
    min_quality_score: float = Field(0.3, ge=0, le=1, description="Minimum quality threshold")
    max_missingness_pct: float = Field(30.0, ge=0, le=100, description="Maximum missingness threshold")
    
    # Temporal boundaries
    date_from: datetime = Field(..., description="Data collection start date")
    date_to: datetime = Field(..., description="Data collection end date")
    frequency: str = Field("1H", description="Data frequency (pandas offset)")
    timezone_storage: str = Field("UTC", description="Storage timezone")
    timezone_display: str = Field("Asia/Kathmandu", description="Display timezone")
    
    # Data quality settings
    min_data_span_days: int = Field(60, ge=1, description="Minimum data span required")
    backup_stations: List[StationConfig] = Field(default_factory=list, description="Backup stations")
    
    @model_validator(mode='after')
    def validate_temporal_and_quality(self):
        # Validate date range
        if self.date_to <= self.date_from:
            raise ValueError('date_to must be after date_from')
        
        # Validate station quality
        quality_stations = [
            s for s in self.stations 
            if s.quality_score >= self.min_quality_score and s.missingness_pct <= self.max_missingness_pct
        ]
        
        if len(quality_stations) < 3:
            raise ValueError(
                f"Need at least 3 stations meeting quality criteria "
                f"(quality≥{self.min_quality_score}, missingness≤{self.max_missingness_pct}%), "
                f"but only {len(quality_stations)} found"
            )
        return self


class UIConfig(BaseModel):
    """User interface configuration."""
    language_default: str = Field("auto", description="Default language (en/ne/auto)")
    language_options: List[str] = Field(["en", "ne"], description="Available languages")
    timezone_display: str = Field("Asia/Kathmandu", description="UI timezone")
    forecast_horizons: List[int] = Field([6, 12, 24], description="Forecast horizons (hours)")
    uncertainty_levels: List[float] = Field([0.8, 0.9, 0.95], description="Uncertainty levels")
    refresh_interval_minutes: int = Field(60, ge=1, description="Data refresh interval")
    max_stations_display: int = Field(10, ge=1, description="Maximum stations to show")


class AppConfig(BaseModel):
    """Application configuration for AirAware."""
    # Application metadata
    app_name: str = Field("AirAware", description="Application name")
    version: str = Field("1.0.0", description="Application version")
    
    # UI settings
    ui: UIConfig = Field(default_factory=UIConfig, description="UI configuration")
    
    # API settings
    api_host: str = Field("0.0.0.0", description="API host")
    api_port: int = Field(8000, ge=1, le=65535, description="API port")
    api_workers: int = Field(1, ge=1, description="API worker processes")
    
    # Model settings
    model_cache_ttl_hours: int = Field(24, ge=1, description="Model cache TTL")
    prediction_batch_size: int = Field(32, ge=1, description="Prediction batch size")
    max_forecast_horizon: int = Field(24, ge=1, description="Maximum forecast horizon")
    
    # Logging
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    
    # Security
    enable_cors: bool = Field(True, description="Enable CORS")
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")
    rate_limit_requests: int = Field(100, ge=1, description="Rate limit requests per minute")
