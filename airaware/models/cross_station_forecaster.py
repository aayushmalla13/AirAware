"""Cross-station forecaster for multi-location air quality prediction."""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from pathlib import Path

from ..features.spatial_correlation import SpatialCorrelationGenerator, SpatialCorrelationConfig
from ..features.replacement_external_data_integration import RealExternalDataIntegrator, RealExternalDataConfig

logger = logging.getLogger(__name__)


class CrossStationConfig(BaseModel):
    """Configuration for cross-station forecasting."""
    
    # Model parameters
    model_type: str = Field("random_forest", description="Model type for cross-station prediction")
    n_estimators: int = Field(100, description="Number of estimators for ensemble models")
    max_depth: int = Field(10, description="Maximum depth for tree-based models")
    random_state: int = Field(42, description="Random state for reproducibility")
    
    # Feature engineering
    use_spatial_features: bool = Field(True, description="Use spatial correlation features")
    use_external_data: bool = Field(True, description="Use external station data")
    use_temporal_features: bool = Field(True, description="Use temporal features")
    
    # Spatial correlation settings
    spatial_config: SpatialCorrelationConfig = Field(
        default_factory=SpatialCorrelationConfig,
        description="Spatial correlation configuration"
    )
    
    # External data settings (REAL DATA ONLY)
    external_config: RealExternalDataConfig = Field(
        default_factory=RealExternalDataConfig,
        description="REAL external data configuration - NO SYNTHETIC DATA"
    )
    
    # Training parameters
    train_test_split: float = Field(0.8, description="Train-test split ratio")
    validation_split: float = Field(0.2, description="Validation split ratio")
    min_training_samples: int = Field(50, description="Minimum training samples required")
    
    # Prediction parameters
    prediction_horizon_hours: int = Field(24, description="Prediction horizon in hours")
    uncertainty_quantiles: List[float] = Field(
        default=[0.1, 0.25, 0.5, 0.75, 0.9],
        description="Uncertainty quantiles"
    )


class CrossStationForecaster:
    """Cross-station air quality forecaster using external station data."""
    
    def __init__(self, config: Optional[CrossStationConfig] = None):
        self.config = config or CrossStationConfig()
        
        # Initialize components
        self.spatial_generator = SpatialCorrelationGenerator(self.config.spatial_config)
        self.external_integrator = RealExternalDataIntegrator(self.config.external_config)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        
        # Performance tracking
        self.training_history = []
        self.feature_importance = {}
        
        logger.info("CrossStationForecaster initialized")
    
    def fit(
        self, 
        target_data: pd.DataFrame,
        external_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Fit the cross-station forecasting model."""
        
        logger.info("Starting cross-station model training")
        
        # Prepare training data
        training_data = self._prepare_training_data(target_data, external_data)
        
        if len(training_data) < self.config.min_training_samples:
            logger.warning(f"Limited training data: {len(training_data)} < {self.config.min_training_samples}. Using adaptive training.")
            
            # Adapt model parameters based on dataset size
            if len(training_data) < 200:
                # Use simpler model for small datasets
                self.config.n_estimators = min(50, max(10, len(training_data) // 10))
                self.config.max_depth = min(5, max(3, len(training_data) // 50))
            elif len(training_data) < 500:
                # Use medium complexity model
                self.config.n_estimators = min(100, len(training_data) // 8)
                self.config.max_depth = min(8, len(training_data) // 40)
        
        # Split data
        train_data, test_data = self._split_data(training_data)
        
        # Prepare features and targets
        X_train, y_train = self._prepare_features_targets(train_data)
        X_test, y_test = self._prepare_features_targets(test_data)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Store feature importance
        self.feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        # Update training history
        training_result = {
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "n_features": len(self.feature_columns),
            "n_samples": len(training_data)
        }
        self.training_history.append(training_result)
        
        self.is_fitted = True
        
        logger.info(f"Model training completed. Test MAE: {test_mae:.2f}, Test RMSE: {test_rmse:.2f}")
        
        return training_result
    
    def predict(
        self, 
        target_data: pd.DataFrame,
        external_data: Optional[pd.DataFrame] = None,
        return_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """Make predictions using the cross-station model."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare prediction data
        prediction_data = self._prepare_training_data(target_data, external_data)
        
        if prediction_data.empty:
            raise ValueError("No prediction data available")
        
        # Prepare features
        X_pred = self._prepare_features(prediction_data)
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make predictions
        predictions = self.model.predict(X_pred_scaled)
        
        result = {
            "predictions": predictions.tolist(),
            "timestamps": prediction_data['datetime_utc'].tolist(),
            "feature_importance": self.feature_importance
        }
        
        # Add uncertainty estimates if requested
        if return_uncertainty:
            uncertainty = self._estimate_uncertainty(X_pred_scaled)
            result["uncertainty"] = uncertainty
        
        return result
    
    def _prepare_training_data(
        self, 
        target_data: pd.DataFrame,
        external_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Prepare training data with all features."""
        
        # Start with target data
        training_data = target_data.copy()
        
        # Add spatial correlation features
        if self.config.use_spatial_features:
            training_data = self.spatial_generator.generate_spatial_features(training_data)
        
        # Add external data features
        if self.config.use_external_data and external_data is not None:
            training_data = self._merge_external_data(training_data, external_data)
        
        # Add temporal features
        if self.config.use_temporal_features:
            training_data = self._add_temporal_features(training_data)
        
        return training_data
    
    def _merge_external_data(
        self, 
        target_data: pd.DataFrame, 
        external_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge external station data with target data."""
        
        # Merge external data by datetime
        merged_data = target_data.merge(
            external_data, 
            on='datetime_utc', 
            how='left',
            suffixes=('', '_external')
        )
        
        # Add external station features
        if 'pm25_external' in merged_data.columns:
            # Calculate external station statistics
            merged_data['external_pm25_mean'] = merged_data.groupby('datetime_utc')['pm25_external'].transform('mean')
            merged_data['external_pm25_std'] = merged_data.groupby('datetime_utc')['pm25_external'].transform('std')
            merged_data['external_pm25_max'] = merged_data.groupby('datetime_utc')['pm25_external'].transform('max')
            merged_data['external_pm25_min'] = merged_data.groupby('datetime_utc')['pm25_external'].transform('min')
        
        return merged_data
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to the data."""
        
        # Hour of day
        df['hour'] = df['datetime_utc'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week
        df['dayofweek'] = df['datetime_utc'].dt.dayofweek
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Month
        df['month'] = df['datetime_utc'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Season
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3  # Autumn
        })
        
        return df
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        
        # Sort by datetime
        data = data.sort_values('datetime_utc').reset_index(drop=True)
        
        # Split by time
        split_idx = int(len(data) * self.config.train_test_split)
        
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        return train_data, test_data
    
    def _prepare_features_targets(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training."""
        
        # Select feature columns
        feature_cols = [col for col in data.columns 
                       if col not in ['pm25', 'datetime_utc', 'station_id'] 
                       and data[col].dtype in ['int64', 'float64']]
        
        # Store feature columns for later use
        if not self.feature_columns:
            self.feature_columns = feature_cols
        
        # Prepare features and targets
        X = data[feature_cols].values
        y = data['pm25'].values
        
        return X, y
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction."""
        
        # Use the same feature columns as training
        feature_cols = self.feature_columns
        
        # Prepare features
        X = data[feature_cols].values
        
        return X
    
    def _estimate_uncertainty(self, X: np.ndarray) -> Dict[str, List[float]]:
        """Estimate prediction uncertainty using quantile regression."""
        
        # For now, use a simple approach based on prediction variance
        # In practice, you might want to use quantile regression or ensemble methods
        
        predictions = self.model.predict(X)
        
        # Estimate uncertainty based on feature importance and prediction variance
        uncertainty = {}
        
        for quantile in self.config.uncertainty_quantiles:
            # Simple uncertainty estimation
            if quantile == 0.5:
                uncertainty[f"q{int(quantile*100)}"] = predictions.tolist()
            else:
                # Add some uncertainty based on quantile
                uncertainty_range = np.std(predictions) * (quantile - 0.5) * 2
                if quantile < 0.5:
                    uncertainty[f"q{int(quantile*100)}"] = (predictions - uncertainty_range).tolist()
                else:
                    uncertainty[f"q{int(quantile*100)}"] = (predictions + uncertainty_range).tolist()
        
        return uncertainty
    
    def save_model(self, path: str) -> None:
        """Save the trained model."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            "config": self.config,
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model."""
        
        model_data = joblib.load(path)
        
        self.config = model_data["config"]
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]
        self.feature_importance = model_data["feature_importance"]
        self.training_history = model_data["training_history"]
        
        self.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return self.feature_importance
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history."""
        
        return self.training_history
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        
        return {
            "is_fitted": self.is_fitted,
            "n_features": len(self.feature_columns),
            "model_type": self.config.model_type,
            "training_history": self.training_history,
            "feature_importance": self.feature_importance
        }
