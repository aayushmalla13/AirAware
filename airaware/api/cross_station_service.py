"""Cross-station prediction service for external data integration."""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..models.cross_station_forecaster import CrossStationForecaster, CrossStationConfig
from ..features.replacement_external_data_integration import RealExternalDataIntegrator, RealExternalDataConfig
from ..features.spatial_correlation import SpatialCorrelationGenerator, SpatialCorrelationConfig

logger = logging.getLogger(__name__)


class CrossStationService:
    """Service for cross-station air quality prediction using external data."""
    
    def __init__(self, data_path: str = "data/processed/joined_data.parquet"):
        self.data_path = data_path
        self.forecaster = None
        self.external_integrator = None
        self.spatial_generator = None
        
        # Initialize components
        self._initialize_components()
        
        logger.info("CrossStationService initialized")
    
    def _initialize_components(self):
        """Initialize cross-station components."""
        
        # Initialize REAL external data integrator - NO SYNTHETIC DATA
        external_config = RealExternalDataConfig(
            cache_real_data=True,
            cache_ttl_hours=2,  # Short cache for fresh real data
            enable_india_real_data=True,
            enable_china_real_data=True,
            min_real_data_points=10,  # Lower threshold for real data
            max_data_age_hours=48  # Accept data up to 48h old
        )
        self.external_integrator = RealExternalDataIntegrator(external_config)
        
        # Initialize spatial correlation generator with REAL data
        spatial_config = SpatialCorrelationConfig(
            use_synthetic_data=False,
            cache_ttl_hours=24,  # Longer cache for better performance
            india_stations=[  # Reduced number of stations for faster processing
                {"id": "IN001", "name": "Delhi", "lat": 28.6139, "lon": 77.2090, "weight": 0.8},
                {"id": "IN002", "name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "weight": 0.6},
                {"id": "IN003", "name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "weight": 0.7}
            ],
            china_stations=[  # Reduced number of stations for faster processing
                {"id": "CN001", "name": "Beijing", "lat": 39.9042, "lon": 116.4074, "weight": 0.6},
                {"id": "CN002", "name": "Shanghai", "lat": 31.2304, "lon": 121.4737, "weight": 0.4},
                {"id": "CN003", "name": "Guangzhou", "lat": 23.1291, "lon": 113.2644, "weight": 0.3}
            ],
            us_stations=[  # US stations for global coverage
                {"id": "US001", "name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "weight": 0.1},
                {"id": "US002", "name": "New York", "lat": 40.7128, "lon": -74.0060, "weight": 0.1},
                {"id": "US003", "name": "Chicago", "lat": 41.8781, "lon": -87.6298, "weight": 0.1}
            ],
            wind_transport_hours=[6, 24]  # Reduced time lags for faster processing
        )
        self.spatial_generator = SpatialCorrelationGenerator(spatial_config)
        
        # Initialize cross-station forecaster
        cross_station_config = CrossStationConfig(
            use_spatial_features=True,
            use_external_data=True,
            use_temporal_features=True,
            spatial_config=spatial_config,
            external_config=external_config
        )
        self.forecaster = CrossStationForecaster(cross_station_config)
    
    def generate_cross_station_forecast(
        self, 
        station_id: str,
        horizon_hours: int = 24,
        include_external_stations: List[str] = ["india", "china"]
    ) -> Dict[str, Any]:
        """Generate cross-station forecast using external data."""
        
        logger.info(f"Generating cross-station forecast for station {station_id}")
        
        try:
            # Load target station data
            target_data = self._load_target_data(station_id)
            
            if target_data.empty:
                raise ValueError(f"No data found for station {station_id}")
            
            # Fetch external data
            external_data = self._fetch_external_data(
                target_data['datetime_utc'].min(),
                target_data['datetime_utc'].max(),
                include_external_stations
            )
            
            # Train model if not already fitted (with fast mode for demo)
            if not self.forecaster.is_fitted:
                logger.info("Training cross-station model (fast mode)...")
                # Use only recent data for faster training
                recent_target = target_data.tail(500) if len(target_data) > 500 else target_data
                recent_external = external_data.tail(500) if len(external_data) > 500 else external_data
                
                training_result = self.forecaster.fit(recent_target, recent_external)
                logger.info(f"Model training completed: {training_result}")
            
            # Generate predictions
            predictions = self.forecaster.predict(
                target_data, 
                external_data, 
                return_uncertainty=True
            )
            
            # Format response
            # Latest observed for precision comparison
            latest_row = target_data.dropna(subset=["datetime_utc"]).iloc[-1]
            latest_obs = {
                "timestamp": pd.Timestamp(latest_row["datetime_utc"]).tz_convert(timezone.utc).isoformat()
                if pd.api.types.is_datetime64_any_dtype(type(latest_row["datetime_utc"])) else str(latest_row["datetime_utc"]),
                "pm25": float(latest_row.get("pm25", np.nan))
            }

            response = self._format_forecast_response(
                predictions, 
                station_id, 
                horizon_hours,
                latest_obs,
                include_external_stations
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Cross-station forecast failed: {e}")
            raise
    
    def _load_target_data(self, station_id: str) -> pd.DataFrame:
        """Load target station data."""
        
        try:
            data = pd.read_parquet(self.data_path)
            station_data = data[data['station_id'] == int(station_id)].copy()
            
            if station_data.empty:
                logger.warning(f"No data found for station {station_id}")
                return pd.DataFrame()
            
            # Sort by datetime and use recent data window
            station_data = station_data.sort_values('datetime_utc').reset_index(drop=True)
            
            # Use last 30 days for training as requested
            recent_data = station_data[station_data['datetime_utc'] >= (datetime.now(timezone.utc) - timedelta(days=30))]
            
            if not recent_data.empty:
                station_data = recent_data
                logger.info(f"Using recent data: {len(station_data)} records from last 7 days")
            
            logger.info(f"Loaded {len(station_data)} records for station {station_id}")
            return station_data
            
        except Exception as e:
            logger.error(f"Failed to load target data: {e}")
            return pd.DataFrame()
    
    def _fetch_external_data(
        self, 
        start_date: datetime, 
        end_date: datetime,
        countries: List[str]
    ) -> pd.DataFrame:
        """Fetch external station data."""
        
        try:
            # Use last 30 days of external data for training
            max_start_date = datetime.now(timezone.utc) - timedelta(days=30)
            if start_date < max_start_date:
                start_date = max_start_date
                logger.info(f"Limited external data range to last 3 days for performance")
            
            external_data = self.external_integrator.fetch_real_external_data(
                start_date, 
                end_date, 
                countries
            )
            
            if not external_data.empty:
                # Validate and clean data
                logger.info(f"🌍 Received {len(external_data)} REAL external data points")
                logger.info(f"Fetched {len(external_data)} external data points")
            else:
                logger.warning("No external data available")
            
            return external_data
            
        except Exception as e:
            logger.error(f"Failed to fetch external data: {e}")
            return pd.DataFrame()
    
    def _format_forecast_response(
        self, 
        predictions: Dict[str, Any], 
        station_id: str, 
        horizon_hours: int,
        latest_observed: Optional[Dict[str, Any]] = None,
        external_used: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Format forecast response."""
        
        # Get current time aligned to the next hour (UTC) and generate future timestamps
        now_utc = datetime.now(timezone.utc)
        current_time = now_utc.replace(minute=0, second=0, microsecond=0)
        if now_utc > current_time:
            current_time = current_time + timedelta(hours=1)
        future_timestamps = [
            current_time + timedelta(hours=i)
            for i in range(0, horizon_hours)
        ]
        
        # Format forecast points
        forecast_points = []
        
        for i, timestamp in enumerate(future_timestamps):
            if i < len(predictions["predictions"]):
                point = {
                    "timestamp": timestamp.isoformat(),
                    "pm25_mean": predictions["predictions"][i],
                    "confidence_level": 0.9
                }
                
                # Add uncertainty if available
                if "uncertainty" in predictions:
                    uncertainty = predictions["uncertainty"]
                    if "q10" in uncertainty and i < len(uncertainty["q10"]):
                        point["pm25_lower"] = uncertainty["q10"][i]
                    if "q90" in uncertainty and i < len(uncertainty["q90"]):
                        point["pm25_upper"] = uncertainty["q90"][i]
                
                forecast_points.append(point)
        
        # Format response
        response = {
            "station_id": station_id,
            "forecast_type": "cross_station",
            "horizon_hours": horizon_hours,
            "forecasts": forecast_points,
            "model_info": {
                "model_type": "cross_station",
                "external_stations_used": external_used or ["india", "china"],
                "spatial_features": True,
                "temporal_features": True
            },
            "feature_importance": predictions.get("feature_importance", {}),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

        if latest_observed is not None:
            response["latest_observed"] = latest_observed
        
        return response
    
    def get_external_stations_info(self) -> Dict[str, Any]:
        """Get information about available external stations."""
        
        try:
            stations_info = self.external_integrator.get_real_station_info()
            
            return {
                "total_stations": sum(len(stations) for stations in stations_info.values()),
                "countries": list(stations_info.keys()),
                "stations": stations_info,
                "use_synthetic_data": False,  # NO SYNTHETIC DATA
                "data_type": "REAL_ONLY",
                "note": "All data extracted from real air quality monitoring stations"
            }
            
        except Exception as e:
            logger.error(f"Failed to get external stations info: {e}")
            return {"error": str(e)}
    
    def get_spatial_correlation_features(self) -> Dict[str, Any]:
        """Get spatial correlation feature information."""
        
        try:
            feature_categories = self.spatial_generator.get_feature_categories()
            
            return {
                "feature_categories": feature_categories,
                "total_features": sum(len(features) for features in feature_categories.values()),
                "description": "Spatial correlation features for cross-border pollution transport"
            }
            
        except Exception as e:
            logger.error(f"Failed to get spatial correlation features: {e}")
            return {"error": str(e)}
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get cross-station model performance metrics."""
        
        try:
            if not self.forecaster.is_fitted:
                return {"error": "Model not fitted yet"}
            
            training_history = self.forecaster.get_training_history()
            feature_importance = self.forecaster.get_feature_importance()
            model_info = self.forecaster.get_model_info()
            
            return {
                "model_info": model_info,
                "training_history": training_history,
                "feature_importance": feature_importance,
                "performance_summary": {
                    "latest_test_mae": training_history[-1]["test_mae"] if training_history else None,
                    "latest_test_rmse": training_history[-1]["test_rmse"] if training_history else None,
                    "n_features": model_info["n_features"],
                    "n_training_samples": training_history[-1]["n_samples"] if training_history else None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            return {"error": str(e)}
    
    def retrain_model(
        self, 
        station_id: str,
        include_external_stations: List[str] = ["india", "china", "usa"]
    ) -> Dict[str, Any]:
        """Retrain the cross-station model with fresh data."""
        
        logger.info(f"Retraining cross-station model for station {station_id}")
        
        try:
            # Load target data
            target_data = self._load_target_data(station_id)
            
            if target_data.empty:
                raise ValueError(f"No data found for station {station_id}")
            
            # Fetch external data
            external_data = self._fetch_external_data(
                target_data['datetime_utc'].min(),
                target_data['datetime_utc'].max(),
                include_external_stations
            )
            
            # Retrain model
            training_result = self.forecaster.fit(target_data, external_data)
            
            return {
                "status": "success",
                "training_result": training_result,
                "retrained_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return {"status": "error", "error": str(e)}
