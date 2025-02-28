"""Hourly bias correction system for PM2.5 forecasting."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class BiasCorrectionConfig:
    """Configuration for bias correction system."""
    
    def __init__(
        self,
        learning_window_days: int = 14,
        min_observations: int = 24,
        update_frequency_hours: int = 1,
        max_bias_correction: float = 10.0,
        smoothing_factor: float = 0.1
    ):
        self.learning_window_days = learning_window_days
        self.min_observations = min_observations
        self.update_frequency_hours = update_frequency_hours
        self.max_bias_correction = max_bias_correction
        self.smoothing_factor = smoothing_factor


class HourlyBiasCorrector:
    """Hourly bias correction system that learns from recent forecast errors."""
    
    def __init__(self, config: Optional[BiasCorrectionConfig] = None, cache_dir: str = "data/cache/bias_correction"):
        self.config = config or BiasCorrectionConfig()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-station bias tracking: {station_id: {hour: deque([errors])}}
        # Track last 14 days by hour for more stability
        self._bias_tracker: Dict[str, Dict[int, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=336)))  # 14 days * 24 hours
        
        # Per-station bias corrections: {station_id: {hour: bias_value}}
        self._bias_corrections: Dict[str, Dict[int, float]] = defaultdict(dict)
        
        # Load existing bias corrections
        self._load_bias_corrections()
    
    def update_bias(
        self, 
        station_id: str, 
        forecast_time: datetime, 
        predicted_value: float, 
        observed_value: float
    ) -> None:
        """Update bias correction based on forecast error."""
        
        if pd.isna(predicted_value) or pd.isna(observed_value):
            return
        
        hour = forecast_time.hour
        error = observed_value - predicted_value
        
        # Store the error for this hour
        self._bias_tracker[station_id][hour].append(error)
        
        # Update bias correction for this hour
        self._update_hourly_bias(station_id, hour)
        
        # Save updated corrections
        self._save_bias_corrections()
        
        logger.debug(f"Updated bias for station {station_id}, hour {hour}: error={error:.2f}")
    
    def _update_hourly_bias(self, station_id: str, hour: int) -> None:
        """Update bias correction for a specific hour."""
        
        errors = list(self._bias_tracker[station_id][hour])
        
        if len(errors) < self.config.min_observations:
            return
        
        # Calculate mean error for this hour
        mean_error = np.mean(errors)
        
        # Apply stricter exponential smoothing and clamp step size
        current_bias = self._bias_corrections[station_id].get(hour, 0.0)
        alpha = max(0.05, min(0.2, self.config.smoothing_factor))
        raw_update = alpha * (mean_error - current_bias)
        # Limit per-update change to avoid oscillations
        max_step = 0.8  # ug/m3 per update
        raw_update = float(np.clip(raw_update, -max_step, max_step))
        new_bias = current_bias + raw_update
        
        # Clamp bias correction
        new_bias = np.clip(new_bias, -self.config.max_bias_correction, self.config.max_bias_correction)
        
        self._bias_corrections[station_id][hour] = new_bias
    
    def get_bias_correction(self, station_id: str, forecast_time: datetime) -> float:
        """Get bias correction for a specific station and time."""
        
        hour = forecast_time.hour
        return self._bias_corrections[station_id].get(hour, 0.0)
    
    def apply_bias_correction(
        self, 
        station_id: str, 
        forecast_time: datetime, 
        predicted_value: float
    ) -> float:
        """Apply bias correction to a predicted value."""
        
        if pd.isna(predicted_value):
            return predicted_value
        
        bias = self.get_bias_correction(station_id, forecast_time)
        corrected_value = predicted_value + bias
        
        # Ensure non-negative PM2.5 values
        corrected_value = max(0.0, corrected_value)
        
        return corrected_value
    
    def get_bias_summary(self, station_id: str) -> Dict[str, any]:
        """Get bias correction summary for a station."""
        
        if station_id not in self._bias_corrections:
            return {"station_id": station_id, "bias_corrections": {}, "total_observations": 0}
        
        bias_corrections = self._bias_corrections[station_id]
        total_observations = sum(len(self._bias_tracker[station_id][hour]) for hour in range(24))
        
        return {
            "station_id": station_id,
            "bias_corrections": dict(bias_corrections),
            "total_observations": total_observations,
            "mean_bias": np.mean(list(bias_corrections.values())) if bias_corrections else 0.0,
            "max_bias": np.max(list(bias_corrections.values())) if bias_corrections else 0.0,
            "min_bias": np.min(list(bias_corrections.values())) if bias_corrections else 0.0
        }
    
    def _get_cache_path(self, station_id: str) -> Path:
        """Get cache file path for bias corrections."""
        return self.cache_dir / f"bias_corrections_{station_id}.json"
    
    def _save_bias_corrections(self) -> None:
        """Save bias corrections to cache."""
        
        for station_id in self._bias_corrections:
            cache_path = self._get_cache_path(station_id)
            
            try:
                data = {
                    "station_id": station_id,
                    "bias_corrections": self._bias_corrections[station_id],
                    "last_updated": datetime.now().isoformat()
                }
                
                with open(cache_path, 'w') as f:
                    json.dump(data, f, indent=2)
                    
            except Exception as e:
                logger.warning(f"Failed to save bias corrections for station {station_id}: {e}")
    
    def _load_bias_corrections(self) -> None:
        """Load bias corrections from cache."""
        
        for cache_file in self.cache_dir.glob("bias_corrections_*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                station_id = data["station_id"]
                self._bias_corrections[station_id] = data["bias_corrections"]
                
                logger.info(f"Loaded bias corrections for station {station_id}")
                
            except Exception as e:
                logger.warning(f"Failed to load bias corrections from {cache_file}: {e}")
    
    def clear_old_data(self, cutoff_date: datetime) -> None:
        """Clear bias data older than cutoff date."""
        
        # This is a simplified implementation
        # In practice, you'd want to track timestamps with the errors
        logger.info(f"Clearing bias data older than {cutoff_date}")
        
        # For now, just clear stations with very old data
        stations_to_clear = []
        for station_id, hour_data in self._bias_tracker.items():
            total_obs = sum(len(errors) for errors in hour_data.values())
            if total_obs > 1000:  # Clear if too many observations
                stations_to_clear.append(station_id)
        
        for station_id in stations_to_clear:
            del self._bias_tracker[station_id]
            if station_id in self._bias_corrections:
                del self._bias_corrections[station_id]
        
        logger.info(f"Cleared bias data for {len(stations_to_clear)} stations")


class AdaptiveBiasCorrector:
    """Advanced bias correction with adaptive learning rates."""
    
    def __init__(self, config: Optional[BiasCorrectionConfig] = None):
        self.config = config or BiasCorrectionConfig()
        self.base_corrector = HourlyBiasCorrector(config)
        
        # Adaptive learning rates per station and hour
        self._learning_rates: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(lambda: 0.1))
        
        # Performance tracking
        self._performance_history: Dict[str, List[float]] = defaultdict(list)
    
    def update_bias(
        self, 
        station_id: str, 
        forecast_time: datetime, 
        predicted_value: float, 
        observed_value: float
    ) -> None:
        """Update bias with adaptive learning rate."""
        
        if pd.isna(predicted_value) or pd.isna(observed_value):
            return
        
        hour = forecast_time.hour
        error = observed_value - predicted_value
        
        # Update base bias correction
        self.base_corrector.update_bias(station_id, forecast_time, predicted_value, observed_value)
        
        # Adapt learning rate based on recent performance
        self._adapt_learning_rate(station_id, hour, abs(error))
        
        # Track performance
        self._performance_history[station_id].append(abs(error))
        if len(self._performance_history[station_id]) > 100:
            self._performance_history[station_id] = self._performance_history[station_id][-100:]
    
    def _adapt_learning_rate(self, station_id: str, hour: int, error_magnitude: float) -> None:
        """Adapt learning rate based on error magnitude and recent performance."""
        
        current_rate = self._learning_rates[station_id][hour]
        
        # If error is large, increase learning rate
        if error_magnitude > 5.0:
            new_rate = min(0.3, current_rate * 1.1)
        # If error is small, decrease learning rate for stability
        elif error_magnitude < 2.0:
            new_rate = max(0.05, current_rate * 0.95)
        else:
            new_rate = current_rate
        
        self._learning_rates[station_id][hour] = new_rate
    
    def get_bias_correction(self, station_id: str, forecast_time: datetime) -> float:
        """Get bias correction with adaptive learning."""
        return self.base_corrector.get_bias_correction(station_id, forecast_time)
    
    def apply_bias_correction(
        self, 
        station_id: str, 
        forecast_time: datetime, 
        predicted_value: float
    ) -> float:
        """Apply bias correction with adaptive learning."""
        return self.base_corrector.apply_bias_correction(station_id, forecast_time, predicted_value)
    
    def get_performance_summary(self, station_id: str) -> Dict[str, any]:
        """Get performance summary for a station."""
        
        base_summary = self.base_corrector.get_bias_summary(station_id)
        
        if station_id in self._performance_history:
            recent_errors = self._performance_history[station_id][-24:]  # Last 24 observations
            base_summary.update({
                "recent_mae": np.mean(recent_errors) if recent_errors else 0.0,
                "learning_rates": dict(self._learning_rates[station_id]),
                "performance_trend": self._calculate_performance_trend(station_id)
            })
        
        return base_summary
    
    def _calculate_performance_trend(self, station_id: str) -> str:
        """Calculate performance trend (improving, stable, degrading)."""
        
        if station_id not in self._performance_history:
            return "unknown"
        
        errors = self._performance_history[station_id]
        if len(errors) < 10:
            return "insufficient_data"
        
        # Compare recent vs older performance
        recent_errors = errors[-10:]
        older_errors = errors[-20:-10] if len(errors) >= 20 else errors[:-10]
        
        recent_mae = np.mean(recent_errors)
        older_mae = np.mean(older_errors)
        
        if recent_mae < older_mae * 0.9:
            return "improving"
        elif recent_mae > older_mae * 1.1:
            return "degrading"
        else:
            return "stable"
