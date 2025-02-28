"""
Service layer for AirAware API

This module provides the business logic for the API endpoints.
"""

import time
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import torch

from .models import (
    ForecastRequest, ForecastResponse, ForecastPoint,
    ExplainabilityRequest, ExplainabilityResponse, FeatureImportance,
    WhatIfRequest, WhatIfResponse, WhatIfScenario,
    ModelInfo, ModelType, Language
)

# Import our existing components
from ..baselines import ProphetBaseline, ProphetConfig
from ..deep_models import PatchTSTForecaster, PatchTSTConfig
from ..calibration import ConformalPredictor, ConformalConfig
from ..explainability import FeatureImportanceAnalyzer, FeatureImportanceConfig
from ..explainability import WhatIfAnalyzer, WhatIfConfig
from ..features.era5_land_extractor import ERA5LandExtractor
from ..features.bias_correction import AdaptiveBiasCorrector

logger = logging.getLogger(__name__)


class ModelService:
    """Service for managing and loading models"""
    
    def __init__(self, data_path: str = "data/processed/joined_data.parquet"):
        self.data_path = data_path
        self.models = {}
        self.model_cache = {}
        self.cache_ttl = 1 * 3600  # 1 hour for now to avoid stale predictions
        
    def load_model(self, model_type: ModelType) -> Any:
        """Load a model by type"""
        if model_type in self.model_cache:
            cached_model, timestamp = self.model_cache[model_type]
            if time.time() - timestamp < self.cache_ttl:
                return cached_model
        
        try:
            if model_type == ModelType.PROPHET:
                # Load tuned Prophet params if available
                tuned_config = ProphetConfig()
                try:
                    grid_path = Path("data/artifacts/prophet_grid.json")
                    if grid_path.exists():
                        import json
                        with open(grid_path) as f:
                            g = json.load(f)
                        params = g.get("best_params", {})
                        if "changepoint_prior_scale" in params:
                            tuned_config.changepoint_prior_scale = float(params["changepoint_prior_scale"])
                        if "seasonality_prior_scale" in params:
                            tuned_config.seasonality_prior_scale = float(params["seasonality_prior_scale"])
                        # Keep daily/weekly seasonality per tuned approach
                        tuned_config.yearly_seasonality = False
                        tuned_config.weekly_seasonality = True
                        tuned_config.daily_seasonality = True
                except Exception as cfg_err:
                    logger.warning(f"Failed to load Prophet tuned params: {cfg_err}")

                model = ProphetBaseline(tuned_config)
                # Load pre-trained model if available
                model_path = Path("data/artifacts/prophet_model.pkl")
                if model_path.exists():
                    model.load_model(str(model_path))
                    
            elif model_type == ModelType.PATCHTST:
                # Prefer a stable symlink to the best model if present
                best_dir = Path("data/artifacts/deep_models/best")
                resolved_model_path: Optional[Path] = None
                if best_dir.exists():
                    candidate = best_dir / "patchtst_model.pth"
                    if candidate.exists():
                        resolved_model_path = candidate
                
                # Fallback: direct default location
                if resolved_model_path is None:
                    default_path = Path("data/artifacts/deep_models/patchtst_model.pth")
                    if default_path.exists():
                        resolved_model_path = default_path
                
                # Fallback: scan sweep directories for the best results.json (lowest MAE)
                if resolved_model_path is None:
                    try:
                        artifacts_root = Path("data/artifacts/deep_models")
                        best_mae = float("inf")
                        best_path = None
                        if artifacts_root.exists():
                            for sub in artifacts_root.glob("**/patchtst_results.json"):
                                try:
                                    import json
                                    with open(sub) as f:
                                        res = json.load(f)
                                    mae = res.get("validation_metrics", {}).get("mae")
                                    if mae is not None and float(mae) < best_mae:
                                        model_candidate = sub.parent / "patchtst_model.pth"
                                        if model_candidate.exists():
                                            best_mae = float(mae)
                                            best_path = model_candidate
                                except Exception:
                                    continue
                        if best_path is not None:
                            # Point best symlink to this best_path for stability next time
                            try:
                                best_dir.mkdir(parents=True, exist_ok=True)
                                # Update/replace symlink or copy if symlink not supported
                                link_target = best_dir / "patchtst_model.pth"
                                if link_target.exists() or link_target.is_symlink():
                                    try:
                                        link_target.unlink()
                                    except Exception:
                                        pass
                                try:
                                    # Attempt to create a relative symlink
                                    link_target.symlink_to(best_path.resolve())
                                except Exception:
                                    # Fallback: copy the file
                                    import shutil
                                    shutil.copy2(str(best_path), str(link_target))
                                resolved_model_path = link_target
                            except Exception:
                                resolved_model_path = best_path
                    except Exception:
                        resolved_model_path = None

                if resolved_model_path is not None and resolved_model_path.exists():
                    # Load the saved config first
                    model_state = torch.load(str(resolved_model_path), map_location='cpu', weights_only=False)
                    saved_config = model_state['config']
                    model = PatchTSTForecaster(saved_config)
                    model.load_model(str(resolved_model_path))
                else:
                    model = PatchTSTForecaster(PatchTSTConfig())
                    
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Cache the model
            self.model_cache[model_type] = (model, time.time())
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {e}")
            raise
    
    def get_model_info(self, model_type: ModelType) -> ModelInfo:
        """Get information about a model"""
        try:
            model = self.load_model(model_type)
            
            # Load performance metrics if available
            metrics = {}
            if model_type == ModelType.PROPHET:
                metrics_path = Path("data/artifacts/baseline_evaluation.json")
            elif model_type in [ModelType.PATCHTST]:
                # Prefer best symlink directory if available
                best_metrics = Path("data/artifacts/deep_models/best/patchtst_results.json")
                if best_metrics.exists():
                    metrics_path = best_metrics
                else:
                    metrics_path = Path(f"data/artifacts/deep_models/{model_type.value}_results.json")
            else:
                metrics_path = None
                
            if metrics_path and metrics_path.exists():
                import json
                with open(metrics_path) as f:
                    data = json.load(f)
                    if "validation_metrics" in data:
                        metrics = data["validation_metrics"]
            
            return ModelInfo(
                model_type=model_type,
                version="1.0.0",
                training_date=datetime.now() - timedelta(days=1),  # Placeholder
                performance_metrics=metrics,
                is_available=True,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_type}: {e}")
            return ModelInfo(
                model_type=model_type,
                version="1.0.0",
                training_date=datetime.now(),
                performance_metrics={},
                is_available=False,
                last_updated=datetime.now()
            )


class ForecastService:
    """Service for generating forecasts"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.data_path = "data/processed/joined_data.parquet"
        # Per-station rolling residual tracker: {'station_id': {'prophet': [errs], 'patchtst': [errs]}}
        self._residual_tracker: Dict[str, Dict[str, list]] = {}
        self._max_residuals = 48  # keep last 48 hours
        # Ensemble tuning knobs
        try:
            import os, json
            # Defaults
            default_blend = 3.0
            default_floor = 0.25
            # Load from tuning artifact if present
            tuning_path = Path("data/artifacts/ensemble_tuning.json")
            tuned_blend = None
            tuned_floor = None
            if tuning_path.exists():
                try:
                    with open(tuning_path) as f:
                        tuning = json.load(f)
                    best = tuning.get("best", {})
                    tuned_blend = float(best.get("blend_distance_ug")) if best.get("blend_distance_ug") is not None else None
                    tuned_floor = float(best.get("weight_floor")) if best.get("weight_floor") is not None else None
                except Exception:
                    pass
            # Env overrides tuned values if set
            env_blend = os.environ.get("AIRWARE_BLEND_DISTANCE_UG")
            env_floor = os.environ.get("AIRWARE_WEIGHT_FLOOR")
            self._blend_distance_ug = float(env_blend) if env_blend is not None else (tuned_blend if tuned_blend is not None else default_blend)
            self._weight_floor = float(env_floor) if env_floor is not None else (tuned_floor if tuned_floor is not None else default_floor)
        except Exception:
            self._blend_distance_ug = 3.0
            self._weight_floor = 0.25
        
        # Initialize ERA5-Land weather data extractor
        self.era5_extractor = ERA5LandExtractor()
        
        # Initialize adaptive bias corrector
        self.bias_corrector = AdaptiveBiasCorrector()
        
    def load_data(self) -> pd.DataFrame:
        """Load the processed data"""
        try:
            data = pd.read_parquet(self.data_path)
            data['datetime_utc'] = pd.to_datetime(data['datetime_utc'])
            return data
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def _update_residuals(self, station_id: str, prophet_pred: Optional[float], patch_pred: Optional[float], latest_obs: Optional[float]) -> None:
        """Update rolling residuals for a station."""
        try:
            if latest_obs is None:
                return
            tracker = self._residual_tracker.setdefault(str(station_id), {"prophet": [], "patchtst": []})
            if prophet_pred is not None and np.isfinite(prophet_pred):
                tracker["prophet"].append(float(abs(prophet_pred - latest_obs)))
            if patch_pred is not None and np.isfinite(patch_pred):
                tracker["patchtst"].append(float(abs(patch_pred - latest_obs)))
            # Trim
            for k in ("prophet", "patchtst"):
                if len(tracker[k]) > self._max_residuals:
                    tracker[k] = tracker[k][-self._max_residuals:]
        except Exception as err:
            logger.warning(f"Residual update failed for station {station_id}: {err}")

    def _compute_adaptive_weights(self, station_id: str, p_err_inst: Optional[float], d_err_inst: Optional[float]) -> Tuple[float, float]:
        """Compute prophet/patchtst weights using rolling mean absolute residuals with instant error fallback."""
        try:
            tracker = self._residual_tracker.get(str(station_id), {"prophet": [], "patchtst": []})
            p_hist = float(np.mean(tracker["prophet"])) if tracker["prophet"] else None
            d_hist = float(np.mean(tracker["patchtst"])) if tracker["patchtst"] else None
            # Prefer historical means; fall back to instantaneous error when needed
            p_use = p_hist if p_hist is not None and np.isfinite(p_hist) else (p_err_inst if p_err_inst is not None else None)
            d_use = d_hist if d_hist is not None and np.isfinite(d_hist) else (d_err_inst if d_err_inst is not None else None)
            if p_use is None or d_use is None or (p_use + d_use) <= 1e-6:
                return 0.6, 0.4  # default
            # Inverse-error weighting (lower error => higher weight)
            w_p = d_use / (p_use + d_use)
            w_d = p_use / (p_use + d_use)
            # Clamp and normalize
            w_p = float(np.clip(w_p, self._weight_floor, 1.0 - self._weight_floor))
            w_d = float(np.clip(w_d, self._weight_floor, 1.0 - self._weight_floor))
            s = w_p + w_d
            return w_p / s, w_d / s
        except Exception as err:
            logger.warning(f"Adaptive weight calc failed for station {station_id}: {err}")
            return 0.6, 0.4
    
    def get_available_stations(self) -> List[Dict[str, Any]]:
        """Get list of available stations"""
        try:
            data = self.load_data()
            stations = data.groupby('station_id').agg({
                'latitude': 'first',
                'longitude': 'first'
            }).reset_index()
            
            # Map station IDs to actual locations in Kathmandu Valley
            station_names = {
                5506835: "Thamel (Central Kathmandu)",
                5509787: "Boudhanath (East Kathmandu)", 
                5633032: "Lalitpur (South Kathmandu)"
            }
            
            return [
                {
                    "station_id": str(int(row['station_id'])),
                    "name": station_names.get(int(row['station_id']), f"Station {int(row['station_id'])}"),
                    "latitude": float(row['latitude']),
                    "longitude": float(row['longitude']),
                    "city": "Kathmandu",
                    "country": "NP"
                }
                for _, row in stations.iterrows()
            ]
        except Exception as e:
            logger.error(f"Failed to get stations: {e}")
            return []
    
    def generate_forecast(
        self, 
        request: ForecastRequest
    ) -> ForecastResponse:
        """Generate forecasts for the given request"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Load models (both for auto-selection)
            prophet_model = self.model_service.load_model(ModelType.PROPHET)
            patchtst_model = self.model_service.load_model(ModelType.PATCHTST)
            
            # Load data
            data = self.load_data()
            
            # Generate forecasts for each station
            station_forecasts = {}
            
            for station_id in request.station_ids:
                # Filter data for this station (convert to int for comparison)
                station_data = data[data['station_id'] == int(station_id)].copy()
                
                if len(station_data) == 0:
                    logger.warning(f"No data found for station {station_id}")
                    continue
                
                # Generate both forecasts for accuracy optimization
                prophet_fc = self._generate_prophet_forecast(
                    prophet_model, station_data, request.horizon_hours
                )
                patch_fc = self._generate_deep_forecast(
                    patchtst_model, station_data, request.horizon_hours, ModelType.PATCHTST
                )

                # Auto-select model closest to latest observation for first horizon
                latest_obs = station_data['pm25'].dropna().iloc[-1] if 'pm25' in station_data.columns and not station_data['pm25'].dropna().empty else None
                chosen = prophet_fc
                chosen_name = 'prophet'
                if latest_obs is not None and len(prophet_fc) > 0 and len(patch_fc) > 0:
                    p_err = abs(prophet_fc[0].pm25_mean - latest_obs)
                    d_err = abs(patch_fc[0].pm25_mean - latest_obs)
                    # Update residual tracker using first-step predictions
                    self._update_residuals(station_id, prophet_fc[0].pm25_mean, patch_fc[0].pm25_mean, latest_obs)
                    # Residual-based dynamic blending: weights inversely proportional to first-step error
                    # If predictions are very far apart (> threshold ug/m3), choose the lower-error model instead of blending
                    distance = abs(prophet_fc[0].pm25_mean - patch_fc[0].pm25_mean)
                    if np.isfinite(p_err) and np.isfinite(d_err) and (p_err + d_err) > 1e-6 and distance <= self._blend_distance_ug:
                        # Compute adaptive weights using rolling residuals with instantaneous fallback
                        w_p, w_d = self._compute_adaptive_weights(station_id, p_err, d_err)
                        blended = []
                        for i in range(min(len(prophet_fc), len(patch_fc))):
                            mean = w_p * prophet_fc[i].pm25_mean + w_d * patch_fc[i].pm25_mean
                            lower = None
                            upper = None
                            if prophet_fc[i].pm25_lower is not None and patch_fc[i].pm25_lower is not None:
                                lower = w_p * prophet_fc[i].pm25_lower + w_d * patch_fc[i].pm25_lower
                            if prophet_fc[i].pm25_upper is not None and patch_fc[i].pm25_upper is not None:
                                upper = w_p * prophet_fc[i].pm25_upper + w_d * patch_fc[i].pm25_upper
                            blended.append(ForecastPoint(
                                timestamp=prophet_fc[i].timestamp,
                                pm25_mean=float(mean),
                                pm25_lower=float(lower) if lower is not None else None,
                                pm25_upper=float(upper) if upper is not None else None,
                                confidence_level=prophet_fc[i].confidence_level or patch_fc[i].confidence_level
                            ))
                        chosen = blended
                        chosen_name = 'ensemble'
                    else:
                        if d_err < p_err:
                            chosen = patch_fc
                            chosen_name = 'patchtst'
                        else:
                            chosen = prophet_fc
                            chosen_name = 'prophet'
                else:
                    # Fallback to requested model behavior
                    chosen = prophet_fc if request.model_type == ModelType.PROPHET else patch_fc
                    chosen_name = request.model_type.value
                
                # Add uncertainty if requested
                if request.uncertainty_level < 1.0:
                    chosen = self._add_uncertainty(
                        chosen, request.uncertainty_level
                    )
                
                station_forecasts[station_id] = chosen
            
            processing_time = (time.time() - start_time) * 1000
            
            return ForecastResponse(
                request_id=request_id,
                timestamp=datetime.now(),
                station_forecasts=station_forecasts,
                model_info={
                    "model_type": chosen_name,
                    "horizon_hours": request.horizon_hours,
                    "uncertainty_level": request.uncertainty_level,
                    "stations_count": len(request.station_ids)
                },
                processing_time_ms=processing_time,
                language=request.language
            )
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            raise
    
    def _generate_prophet_forecast(
        self, 
        model: Any, 
        data: pd.DataFrame, 
        horizon_hours: int
    ) -> List[ForecastPoint]:
        """Generate Prophet forecast with ERA5-Land weather features"""
        try:
            # Get station coordinates for ERA5-Land data
            lat = data['latitude'].iloc[0] if 'latitude' in data.columns else 27.7172
            lon = data['longitude'].iloc[0] if 'longitude' in data.columns else 85.3240
            
            # Create future timestamps starting from current time
            from datetime import datetime, timezone
            current_time = datetime.now(timezone.utc)
            # Round to the next hour
            current_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            future_timestamps = [current_hour + timedelta(hours=i) for i in range(horizon_hours)]
            
            logger.info(f"Current time: {current_time}")
            logger.info(f"Forecast start: {current_hour}")
            logger.info(f"First 3 timestamps: {future_timestamps[:3]}")
            
            # Extract ERA5-Land weather data for forecast period
            forecast_start = current_hour
            forecast_end = current_hour + timedelta(hours=horizon_hours)
            
            era5_data = self.era5_extractor.extract_era5_data(
                forecast_start, forecast_end, lat, lon
            )
            
            # Train on recent window (rolling retrain) for robustness
            recent_cutoff = current_hour - timedelta(days=60)
            recent_data = data[data['datetime_utc'] >= recent_cutoff].copy()
            if not hasattr(model, 'is_fitted') or not model.is_fitted:
                model.fit(recent_data, target_col='pm25', group_col='station_id')

            forecast = model.predict(
                timestamps=future_timestamps,
                station_id=int(data['station_id'].iloc[0]),
                horizon_hours=horizon_hours,
                meteo_data=era5_data
            )
            
            # Extract forecast points from ProphetForecast object
            forecast_points = []
            for i in range(horizon_hours):
                point = ForecastPoint(
                    timestamp=future_timestamps[i],
                    pm25_mean=float(forecast.predictions[i]),
                    pm25_lower=float(forecast.confidence_intervals['q0.1'][i]),
                    pm25_upper=float(forecast.confidence_intervals['q0.9'][i]),
                    confidence_level=0.8  # Default confidence
                )
                forecast_points.append(point)
            
            return forecast_points
            
        except Exception as e:
            logger.error(f"Prophet forecast failed: {e}")
            raise
    
    def _generate_deep_forecast(
        self, 
        model: Any, 
        data: pd.DataFrame, 
        horizon_hours: int,
        model_type: ModelType
    ) -> List[ForecastPoint]:
        """Generate deep learning model forecast"""
        try:
            # Generate predictions
            predictions = model.predict(data)
            # PatchTST returns numpy array
            pred_values = predictions
            
            # Create forecast points starting from current time
            forecast_points = []
            from datetime import datetime, timezone
            current_time = datetime.now(timezone.utc)
            # Round to the next hour
            current_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            
            # Enhanced bias correction using adaptive learning
            latest_obs = float(data['pm25'].dropna().iloc[-1]) if 'pm25' in data.columns and not data['pm25'].dropna().empty else None
            first_pred = None
            if isinstance(pred_values, np.ndarray):
                if pred_values.ndim > 1 and pred_values.size > 0:
                    first_pred = float(pred_values[0, 0])
                elif pred_values.ndim == 1 and pred_values.size > 0:
                    first_pred = float(pred_values[0])

            # Get station ID for bias correction
            station_id = str(data['station_id'].iloc[0]) if 'station_id' in data.columns else "unknown"
            
            # Apply adaptive bias correction
            bias_offset = 0.0
            if latest_obs is not None and first_pred is not None and np.isfinite(latest_obs) and np.isfinite(first_pred):
                # Update bias corrector with latest error
                self.bias_corrector.update_bias(station_id, current_hour, first_pred, latest_obs)
                
                # Get adaptive bias correction
                adaptive_bias = self.bias_corrector.get_bias_correction(station_id, current_hour)
                
                # Combine instantaneous and adaptive bias
                instantaneous_bias = latest_obs - first_pred
                bias_offset = 0.7 * instantaneous_bias + 0.3 * adaptive_bias

            for i in range(horizon_hours):
                timestamp = current_hour + timedelta(hours=i)
                
                if isinstance(pred_values, np.ndarray) and pred_values.ndim > 1:
                    # Multi-step prediction
                    base_pred = pred_values[0, i] if i < pred_values.shape[1] else pred_values[0, -1]
                else:
                    # Single prediction
                    base_pred = pred_values[0] if isinstance(pred_values, np.ndarray) else pred_values
                # Apply adaptive bias correction for each forecast step
                step_bias = self.bias_corrector.get_bias_correction(station_id, timestamp)
                pred_value = float(max(0.0, base_pred + bias_offset + 0.1 * step_bias))
                
                point = ForecastPoint(
                    timestamp=timestamp,
                    pm25_mean=float(pred_value),
                    confidence_level=0.8
                )
                forecast_points.append(point)
            
            return forecast_points
            
        except Exception as e:
            logger.error(f"Deep model forecast failed: {e}")
            raise
    
    def _add_uncertainty(
        self, 
        forecast: List[ForecastPoint], 
        confidence_level: float
    ) -> List[ForecastPoint]:
        """Add uncertainty bounds to forecast"""
        # Simple uncertainty estimation based on historical variance
        uncertainty_factor = 1.96 if confidence_level == 0.95 else 1.645 if confidence_level == 0.9 else 1.28
        
        for point in forecast:
            # Estimate uncertainty as percentage of mean
            uncertainty = point.pm25_mean * 0.2 * uncertainty_factor  # 20% coefficient of variation
            point.pm25_lower = max(0, point.pm25_mean - uncertainty)
            point.pm25_upper = point.pm25_mean + uncertainty
            point.confidence_level = confidence_level
        
        return forecast


class ExplainabilityService:
    """Service for explainability analysis"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.data_path = "data/processed/joined_data.parquet"
    
    def analyze_explainability(
        self, 
        request: ExplainabilityRequest
    ) -> ExplainabilityResponse:
        """Perform explainability analysis"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Load data
            data = pd.read_parquet(self.data_path)
            # Convert station_id to int for comparison
            station_id_int = int(request.station_id)
            station_data = data[data['station_id'] == station_id_int].copy()
            
            logger.info(f"Loaded data shape: {data.shape}")
            logger.info(f"Station IDs in data: {data['station_id'].unique()}")
            logger.info(f"Requested station ID: {request.station_id} (int: {station_id_int})")
            logger.info(f"Filtered station data shape: {station_data.shape}")
            
            if len(station_data) == 0:
                raise ValueError(f"No data found for station {request.station_id}")
            
            # Initialize explainability analyzer
            config = FeatureImportanceConfig(
                use_permutation_importance="permutation" in request.methods,
                use_tree_importance="tree" in request.methods,
                use_correlation_analysis="correlation" in request.methods,
                use_mutual_information="mutual_information" in request.methods
            )
            
            # Prepare features and target - only numeric columns
            feature_cols = [col for col in station_data.columns 
                           if col not in ['pm25', 'datetime_utc', 'station_id'] 
                           and station_data[col].dtype in ['int64', 'float64']]
            X = station_data[feature_cols].values
            y = station_data['pm25'].values
            
            # Create a simple model for analysis
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            analyzer = FeatureImportanceAnalyzer(config)
            
            # Perform analysis
            analyzer.fit(model, X, y, feature_names=feature_cols)
            results = analyzer.importance_results
            
            # Feature name mapping for user-friendly display
            feature_name_mapping = {
                'blh': 'Boundary Layer Height',
                'u10': 'East-West Wind',
                'v10': 'North-South Wind', 
                'wind_speed': 'Wind Speed',
                'wind_direction': 'Wind Direction',
                't2m_celsius': 'Temperature',
                'latitude': 'Latitude',
                'longitude': 'Longitude'
            }
            
            # Convert results to response format
            feature_importance = []
            for method in request.methods:
                if method in results:
                    method_results = results[method]
                    if 'importances' in method_results and 'feature_names' in method_results:
                        for i, (feature, importance) in enumerate(zip(
                            method_results['feature_names'],
                            method_results['importances']
                        )):
                            # Use user-friendly name if available
                            display_name = feature_name_mapping.get(feature, feature)
                            feature_importance.append(FeatureImportance(
                                feature_name=display_name,
                                importance_score=float(importance),
                                method=method
                            ))
            
            processing_time = (time.time() - start_time) * 1000
            
            return ExplainabilityResponse(
                request_id=request_id,
                timestamp=datetime.now(),
                station_id=request.station_id,
                feature_importance=feature_importance,
                processing_time_ms=processing_time,
                language=request.language
            )
            
        except Exception as e:
            logger.error(f"Explainability analysis failed: {e}")
            raise


class WhatIfService:
    """Service for what-if analysis"""
    
    def __init__(self, model_service: ModelService):
        self.model_service = model_service
        self.data_path = "data/processed/joined_data.parquet"
    
    def analyze_what_if(
        self, 
        request: WhatIfRequest
    ) -> WhatIfResponse:
        """Perform what-if analysis"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Load data
            data = pd.read_parquet(self.data_path)
            # Convert station_id to int for comparison
            station_id_int = int(request.station_id)
            station_data = data[data['station_id'] == station_id_int].copy()
            
            if len(station_data) == 0:
                raise ValueError(f"No data found for station {request.station_id}")
            
            # Prepare features and target
            feature_cols = [col for col in station_data.columns 
                           if col not in ['pm25', 'datetime_utc', 'station_id'] 
                           and station_data[col].dtype in ['int64', 'float64']]
            X = station_data[feature_cols].values
            y = station_data['pm25'].values
            
            # Create a simple model for analysis
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Get baseline prediction (latest data point)
            baseline_data = X[-1:].copy()
            baseline_prediction = model.predict(baseline_data)[0]
            
            logger.info(f"Baseline prediction: {baseline_prediction}")
            logger.info(f"Available features: {feature_cols}")
            
            # Convert results to response format
            scenarios = []
            for i, scenario in enumerate(request.scenarios):
                # Apply user's parameter changes to baseline
                scenario_data = baseline_data.copy()
                changes = scenario.get('changes', {})
                
                logger.info(f"Processing scenario {i}: {changes}")
                
                # Apply changes to features
                for feature_name, change_value in changes.items():
                    if feature_name in feature_cols:
                        feature_idx = feature_cols.index(feature_name)
                        original_value = baseline_data[0, feature_idx]
                        
                        if feature_name in ['wind_speed', 'humidity']:
                            # Multiplier changes
                            scenario_data[0, feature_idx] = baseline_data[0, feature_idx] * change_value
                        elif feature_name in ['t2m_celsius', 'pressure']:
                            # Absolute changes (temperature, pressure)
                            scenario_data[0, feature_idx] = baseline_data[0, feature_idx] + change_value
                        else:
                            # Default to absolute changes for other features
                            scenario_data[0, feature_idx] = baseline_data[0, feature_idx] + change_value
                        
                        new_value = scenario_data[0, feature_idx]
                        logger.info(f"Feature {feature_name}: {original_value} -> {new_value}")
                    else:
                        logger.warning(f"Feature {feature_name} not found in available features")
                
                # Get scenario prediction
                scenario_prediction = model.predict(scenario_data)[0]
                logger.info(f"Scenario prediction: {scenario_prediction}")
                
                # Calculate impact
                impact = scenario_prediction - baseline_prediction
                impact_percentage = (impact / baseline_prediction) * 100 if baseline_prediction != 0 else 0
                
                try:
                    impact_analysis = {
                        'baseline_mean': float(baseline_prediction),
                        'scenario_mean': float(scenario_prediction),
                        'impact': float(impact),
                        'impact_percentage': float(impact_percentage),
                        'changes_applied': changes
                    }
                    
                    scenario_result = WhatIfScenario(
                        scenario_name=scenario.get('name', f'Scenario {i+1}'),
                        scenario_description=scenario.get('description', 'Custom scenario'),
                        baseline_forecast=[],  # Placeholder
                        scenario_forecast=[],  # Placeholder
                        impact_analysis=impact_analysis
                    )
                    scenarios.append(scenario_result)
                except Exception as scenario_error:
                    logger.error(f"Failed to create scenario result: {scenario_error}")
                    # Create minimal scenario result
                    scenario_result = WhatIfScenario(
                        scenario_name=scenario.get('name', f'Scenario {i+1}'),
                        scenario_description=scenario.get('description', 'Custom scenario'),
                        baseline_forecast=[],
                        scenario_forecast=[],
                        impact_analysis={}
                    )
                    scenarios.append(scenario_result)
            
            processing_time = (time.time() - start_time) * 1000
            
            return WhatIfResponse(
                request_id=request_id,
                timestamp=datetime.now(),
                station_id=request.station_id,
                scenarios=scenarios,
                processing_time_ms=processing_time,
                language=request.language
            )
            
        except Exception as e:
            logger.error(f"What-if analysis failed: {e}")
            raise
