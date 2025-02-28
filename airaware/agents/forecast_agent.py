"""
Forecast Optimization Agent for AirAware

This agent optimizes forecast generation, model selection, and performance monitoring.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from pathlib import Path

from .base_agent import BaseAgent, AgentConfig, AgentStatus


@dataclass
class ForecastAgentConfig(AgentConfig):
    """Configuration for Forecast Optimization Agent"""
    # Model management settings
    model_performance_file: str = "data/model_performance.json"
    model_selection_strategy: str = "adaptive"  # "best", "adaptive", "ensemble"
    
    # Performance optimization
    performance_window_hours: int = 168  # 1 week
    min_performance_samples: int = 10
    performance_threshold: float = 0.8  # 80% of best performance
    
    # Model selection criteria
    primary_metric: str = "mae"  # "mae", "rmse", "coverage"
    secondary_metric: str = "rmse"
    uncertainty_weight: float = 0.3  # Weight for uncertainty quality
    
    # Adaptive settings
    adaptation_sensitivity: float = 0.1  # How quickly to adapt
    model_switching_threshold: float = 0.05  # 5% performance difference
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=lambda: {
        "update_interval": 3600,  # 1 hour
        "performance_cache_duration": 7200,  # 2 hours
        "max_models_to_evaluate": 5,
        "ensemble_weight_decay": 0.95
    })


@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_id: str
    model_type: str
    mae: float
    rmse: float
    coverage: float
    interval_width: float
    execution_time: float
    timestamp: datetime
    station_id: Optional[str] = None
    horizon_hours: int = 24
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "mae": self.mae,
            "rmse": self.rmse,
            "coverage": self.coverage,
            "interval_width": self.interval_width,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "station_id": self.station_id,
            "horizon_hours": self.horizon_hours
        }


@dataclass
class ModelRecommendation:
    """Model recommendation for specific conditions"""
    recommended_model: str
    confidence: float
    reasoning: str
    expected_performance: Dict[str, float]
    alternatives: List[Tuple[str, float]]  # (model_id, score)
    valid_until: datetime


class ForecastOptimizationAgent(BaseAgent):
    """Intelligent forecast optimization agent"""
    
    def __init__(self, config: ForecastAgentConfig):
        super().__init__(config)
        self.model_performance: List[ModelPerformance] = []
        self.model_recommendations: Dict[str, ModelRecommendation] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.last_performance_update: Optional[datetime] = None
        
        # Load historical performance data
        self._load_model_performance()
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute forecast optimization logic"""
        try:
            # Extract context data
            station_id = context.get("station_id")
            horizon_hours = context.get("horizon_hours", 24)
            available_models = context.get("available_models", [])
            current_conditions = context.get("current_conditions", {})
            
            # Update model performance if new data available
            if context.get("new_performance_data"):
                await self._update_model_performance(context["new_performance_data"])
            
            # Generate model recommendations
            recommendations = await self._generate_model_recommendations(
                station_id, horizon_hours, available_models, current_conditions
            )
            
            # Optimize ensemble weights
            ensemble_optimization = await self._optimize_ensemble_weights(
                station_id, horizon_hours
            )
            
            # Generate performance insights
            performance_insights = self._generate_performance_insights(
                station_id, horizon_hours
            )
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "model_recommendations": recommendations,
                "ensemble_optimization": ensemble_optimization,
                "performance_insights": performance_insights,
                "station_id": station_id,
                "horizon_hours": horizon_hours
            }
            
        except Exception as e:
            self.logger.error(f"Forecast optimization execution failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Perform health check for the agent"""
        try:
            # Check if performance data is available
            if not self.model_performance:
                self.logger.warning("No model performance data available")
                return False
            
            # Check if recommendations are up to date
            if not self.model_recommendations:
                self.logger.warning("No model recommendations available")
            
            # Update health check timestamp
            self.last_health_check = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def _generate_model_recommendations(
        self,
        station_id: str,
        horizon_hours: int,
        available_models: List[str],
        current_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate model recommendations for specific conditions"""
        # Get recent performance data for this station and horizon
        recent_performance = self._get_recent_performance(station_id, horizon_hours)
        
        if not recent_performance:
            # Fallback to global performance
            recent_performance = self._get_recent_performance(None, horizon_hours)
        
        if not recent_performance:
            # No performance data available, use default recommendation
            recommendation = self._get_default_recommendation(available_models)
            return recommendation.__dict__
        
        # Calculate model scores based on strategy
        if self.config.model_selection_strategy == "best":
            recommendation = self._get_best_model_recommendation(recent_performance, available_models)
        elif self.config.model_selection_strategy == "adaptive":
            recommendation = self._get_adaptive_model_recommendation(
                recent_performance, available_models, current_conditions
            )
        elif self.config.model_selection_strategy == "ensemble":
            recommendation = self._get_ensemble_recommendation(recent_performance, available_models)
        else:
            recommendation = self._get_default_recommendation(available_models)
        
        # Cache recommendation
        cache_key = f"{station_id}_{horizon_hours}"
        self.model_recommendations[cache_key] = recommendation
        
        return recommendation.__dict__
    
    def _get_recent_performance(
        self, 
        station_id: Optional[str], 
        horizon_hours: int
    ) -> List[ModelPerformance]:
        """Get recent performance data"""
        cutoff_time = datetime.now() - timedelta(hours=self.config.performance_window_hours)
        
        recent_performance = []
        for perf in self.model_performance:
            if (perf.timestamp >= cutoff_time and 
                perf.horizon_hours == horizon_hours and
                (station_id is None or perf.station_id == station_id)):
                recent_performance.append(perf)
        
        return recent_performance
    
    def _get_best_model_recommendation(
        self, 
        performance_data: List[ModelPerformance], 
        available_models: List[str]
    ) -> ModelRecommendation:
        """Get recommendation for best performing model"""
        if not performance_data:
            return self._get_default_recommendation(available_models)
        
        # Calculate average performance for each model
        model_scores = {}
        for perf in performance_data:
            if perf.model_id not in model_scores:
                model_scores[perf.model_id] = []
            model_scores[perf.model_id].append(perf)
        
        # Find best model based on primary metric
        best_model = None
        best_score = float('inf')
        model_rankings = []
        
        for model_id, perfs in model_scores.items():
            if model_id not in available_models:
                continue
                
            # Calculate average performance
            avg_mae = np.mean([p.mae for p in perfs])
            avg_rmse = np.mean([p.rmse for p in perfs])
            avg_coverage = np.mean([p.coverage for p in perfs])
            avg_execution_time = np.mean([p.execution_time for p in perfs])
            
            # Calculate composite score
            if self.config.primary_metric == "mae":
                primary_score = avg_mae
            elif self.config.primary_metric == "rmse":
                primary_score = avg_rmse
            elif self.config.primary_metric == "coverage":
                primary_score = 1.0 - abs(avg_coverage - 0.9)  # Closer to 90% is better
            else:
                primary_score = avg_mae
            
            # Add execution time penalty
            execution_penalty = avg_execution_time / 1000.0  # Convert to seconds
            composite_score = primary_score + execution_penalty * 0.1
            
            model_rankings.append((model_id, composite_score))
            
            if composite_score < best_score:
                best_score = composite_score
                best_model = model_id
        
        # Sort by score
        model_rankings.sort(key=lambda x: x[1])
        
        # Create recommendation
        reasoning = f"Best performing model based on {self.config.primary_metric} over the last {self.config.performance_window_hours} hours"
        
        return ModelRecommendation(
            recommended_model=best_model or available_models[0],
            confidence=0.8 if len(performance_data) >= self.config.min_performance_samples else 0.6,
            reasoning=reasoning,
            expected_performance={
                "mae": avg_mae,
                "rmse": avg_rmse,
                "coverage": avg_coverage,
                "execution_time": avg_execution_time
            },
            alternatives=model_rankings[:3],
            valid_until=datetime.now() + timedelta(hours=1)
        )
    
    def _get_adaptive_model_recommendation(
        self,
        performance_data: List[ModelPerformance],
        available_models: List[str],
        current_conditions: Dict[str, Any]
    ) -> ModelRecommendation:
        """Get adaptive model recommendation based on current conditions"""
        if not performance_data:
            return self._get_default_recommendation(available_models)
        
        # Analyze performance trends
        trend_analysis = self._analyze_performance_trends(performance_data)
        
        # Consider current conditions
        condition_factors = self._analyze_current_conditions(current_conditions)
        
        # Calculate adaptive scores
        adaptive_scores = {}
        for model_id in available_models:
            model_perfs = [p for p in performance_data if p.model_id == model_id]
            if not model_perfs:
                continue
            
            # Base performance score
            base_score = np.mean([p.mae for p in model_perfs])
            
            # Trend adjustment
            trend_factor = trend_analysis.get(model_id, 1.0)
            
            # Condition adjustment
            condition_factor = condition_factors.get(model_id, 1.0)
            
            # Adaptive score
            adaptive_score = base_score * trend_factor * condition_factor
            adaptive_scores[model_id] = adaptive_score
        
        if not adaptive_scores:
            return self._get_default_recommendation(available_models)
        
        # Find best adaptive model
        best_model = min(adaptive_scores.items(), key=lambda x: x[1])[0]
        best_score = adaptive_scores[best_model]
        
        # Calculate confidence based on score consistency
        scores = list(adaptive_scores.values())
        score_std = np.std(scores)
        confidence = max(0.5, 1.0 - score_std / np.mean(scores))
        
        # Create alternatives
        alternatives = sorted(adaptive_scores.items(), key=lambda x: x[1])[:3]
        
        reasoning = f"Adaptive selection considering performance trends and current conditions"
        
        return ModelRecommendation(
            recommended_model=best_model,
            confidence=confidence,
            reasoning=reasoning,
            expected_performance={
                "mae": best_score,
                "rmse": best_score * 1.2,  # Estimate
                "coverage": 0.9,  # Estimate
                "execution_time": 1000  # Estimate
            },
            alternatives=alternatives,
            valid_until=datetime.now() + timedelta(hours=1)
        )
    
    def _get_ensemble_recommendation(
        self,
        performance_data: List[ModelPerformance],
        available_models: List[str]
    ) -> ModelRecommendation:
        """Get ensemble model recommendation"""
        if not performance_data:
            return self._get_default_recommendation(available_models)
        
        # Calculate ensemble weights based on recent performance
        model_weights = {}
        total_weight = 0.0
        
        for model_id in available_models:
            model_perfs = [p for p in performance_data if p.model_id == model_id]
            if not model_perfs:
                continue
            
            # Calculate inverse MAE as weight (lower MAE = higher weight)
            avg_mae = np.mean([p.mae for p in model_perfs])
            weight = 1.0 / (avg_mae + 1e-6)  # Add small epsilon to avoid division by zero
            
            model_weights[model_id] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for model_id in model_weights:
                model_weights[model_id] /= total_weight
        
        # Update ensemble weights with decay
        for model_id, weight in model_weights.items():
            if model_id in self.ensemble_weights:
                # Apply exponential decay
                self.ensemble_weights[model_id] = (
                    self.config.custom_settings["ensemble_weight_decay"] * self.ensemble_weights[model_id] +
                    (1 - self.config.custom_settings["ensemble_weight_decay"]) * weight
                )
            else:
                self.ensemble_weights[model_id] = weight
        
        # Find primary model (highest weight)
        primary_model = max(self.ensemble_weights.items(), key=lambda x: x[1])[0]
        
        reasoning = f"Ensemble recommendation with weights: {dict(self.ensemble_weights)}"
        
        return ModelRecommendation(
            recommended_model=primary_model,
            confidence=0.7,
            reasoning=reasoning,
            expected_performance={
                "mae": np.mean([p.mae for p in performance_data]),
                "rmse": np.mean([p.rmse for p in performance_data]),
                "coverage": np.mean([p.coverage for p in performance_data]),
                "execution_time": np.mean([p.execution_time for p in performance_data])
            },
            alternatives=list(self.ensemble_weights.items()),
            valid_until=datetime.now() + timedelta(hours=1)
        )
    
    def _get_default_recommendation(self, available_models: List[str]) -> ModelRecommendation:
        """Get intelligent default recommendation when no performance data is available"""
        # Define model preferences based on general characteristics
        model_preferences = {
            "prophet": {
                "score": 0.9,
                "reasoning": "Robust for time series with seasonality and trends",
                "expected_mae": 8.5,
                "expected_rmse": 10.2,
                "expected_coverage": 0.88,
                "expected_execution_time": 800
            },
            "patchtst": {
                "score": 0.8,
                "reasoning": "Advanced transformer model for complex patterns",
                "expected_mae": 7.8,
                "expected_rmse": 9.5,
                "expected_coverage": 0.85,
                "expected_execution_time": 1200
            },
            "simple_tft": {
                "score": 0.7,
                "reasoning": "Good balance of performance and interpretability",
                "expected_mae": 8.2,
                "expected_rmse": 9.8,
                "expected_coverage": 0.87,
                "expected_execution_time": 1000
            },
            "ensemble": {
                "score": 0.6,
                "reasoning": "Combines multiple models for better reliability",
                "expected_mae": 7.5,
                "expected_rmse": 9.0,
                "expected_coverage": 0.90,
                "expected_execution_time": 2000
            }
        }
        
        # Find best available model
        best_model = "prophet"  # Default fallback
        best_score = 0.0
        
        for model in available_models:
            if model in model_preferences:
                if model_preferences[model]["score"] > best_score:
                    best_score = model_preferences[model]["score"]
                    best_model = model
        
        # Create alternatives list
        alternatives = []
        for model in available_models:
            if model in model_preferences:
                alternatives.append((model, model_preferences[model]["score"]))
        
        # Sort by score
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        # Get expected performance for best model
        expected_perf = model_preferences[best_model]
        
        return ModelRecommendation(
            recommended_model=best_model,
            confidence=0.7,  # Higher confidence for intelligent default
            reasoning=f"Intelligent default: {expected_perf['reasoning']}",
            expected_performance={
                "mae": expected_perf["expected_mae"],
                "rmse": expected_perf["expected_rmse"],
                "coverage": expected_perf["expected_coverage"],
                "execution_time": expected_perf["expected_execution_time"]
            },
            alternatives=alternatives[:3],
            valid_until=datetime.now() + timedelta(hours=1)
        )
    
    def _analyze_performance_trends(self, performance_data: List[ModelPerformance]) -> Dict[str, float]:
        """Analyze performance trends for each model"""
        trends = {}
        
        # Group by model
        model_perfs = {}
        for perf in performance_data:
            if perf.model_id not in model_perfs:
                model_perfs[perf.model_id] = []
            model_perfs[perf.model_id].append(perf)
        
        # Calculate trends
        for model_id, perfs in model_perfs.items():
            if len(perfs) < 3:
                trends[model_id] = 1.0
                continue
            
            # Sort by timestamp
            perfs.sort(key=lambda x: x.timestamp)
            
            # Calculate trend (improving = < 1.0, degrading = > 1.0)
            recent_mae = np.mean([p.mae for p in perfs[-3:]])
            older_mae = np.mean([p.mae for p in perfs[:3]])
            
            if older_mae > 0:
                trend_factor = recent_mae / older_mae
            else:
                trend_factor = 1.0
            
            trends[model_id] = trend_factor
        
        return trends
    
    def _analyze_current_conditions(self, current_conditions: Dict[str, Any]) -> Dict[str, float]:
        """Analyze current conditions and their impact on model performance"""
        factors = {}
        
        # Default factor for all models
        default_factor = 1.0
        
        # Analyze specific conditions
        if "pm25_level" in current_conditions:
            pm25 = current_conditions["pm25_level"]
            if pm25 > 100:  # High pollution
                # Some models may perform better in extreme conditions
                default_factor = 0.95
            elif pm25 < 20:  # Low pollution
                # Some models may perform better in clean conditions
                default_factor = 1.05
        
        if "weather_conditions" in current_conditions:
            weather = current_conditions["weather_conditions"]
            if weather == "stormy":
                # Weather models may be more important
                default_factor *= 0.9
            elif weather == "calm":
                # Statistical models may be more reliable
                default_factor *= 1.1
        
        # Apply default factor to all models
        for model_id in ["prophet", "patchtst", "simple_tft"]:
            factors[model_id] = default_factor
        
        return factors
    
    async def _optimize_ensemble_weights(
        self, 
        station_id: str, 
        horizon_hours: int
    ) -> Dict[str, Any]:
        """Optimize ensemble weights for better performance"""
        # Get recent performance data
        recent_performance = self._get_recent_performance(station_id, horizon_hours)
        
        if not recent_performance:
            return {"status": "no_data", "weights": self.ensemble_weights}
        
        # Calculate optimal weights using performance-based optimization
        optimal_weights = {}
        total_weight = 0.0
        
        # Group by model
        model_perfs = {}
        for perf in recent_performance:
            if perf.model_id not in model_perfs:
                model_perfs[perf.model_id] = []
            model_perfs[perf.model_id].append(perf)
        
        # Calculate weights based on inverse MAE
        for model_id, perfs in model_perfs.items():
            avg_mae = np.mean([p.mae for p in perfs])
            avg_coverage = np.mean([p.coverage for p in perfs])
            
            # Weight based on MAE and coverage
            mae_weight = 1.0 / (avg_mae + 1e-6)
            coverage_weight = 1.0 - abs(avg_coverage - 0.9)  # Closer to 90% is better
            
            # Combine weights
            combined_weight = mae_weight * coverage_weight
            optimal_weights[model_id] = combined_weight
            total_weight += combined_weight
        
        # Normalize weights
        if total_weight > 0:
            for model_id in optimal_weights:
                optimal_weights[model_id] /= total_weight
        
        # Update ensemble weights with learning rate
        learning_rate = 0.1
        for model_id, weight in optimal_weights.items():
            if model_id in self.ensemble_weights:
                self.ensemble_weights[model_id] = (
                    (1 - learning_rate) * self.ensemble_weights[model_id] +
                    learning_rate * weight
                )
            else:
                self.ensemble_weights[model_id] = weight
        
        return {
            "status": "optimized",
            "weights": self.ensemble_weights,
            "optimal_weights": optimal_weights,
            "performance_samples": len(recent_performance)
        }
    
    def _generate_performance_insights(
        self, 
        station_id: str, 
        horizon_hours: int
    ) -> Dict[str, Any]:
        """Generate performance insights and recommendations"""
        recent_performance = self._get_recent_performance(station_id, horizon_hours)
        
        if not recent_performance:
            return {"status": "no_data", "insights": []}
        
        insights = []
        
        # Model performance comparison
        model_perfs = {}
        for perf in recent_performance:
            if perf.model_id not in model_perfs:
                model_perfs[perf.model_id] = []
            model_perfs[perf.model_id].append(perf)
        
        # Calculate insights
        for model_id, perfs in model_perfs.items():
            avg_mae = np.mean([p.mae for p in perfs])
            avg_rmse = np.mean([p.rmse for p in perfs])
            avg_coverage = np.mean([p.coverage for p in perfs])
            avg_execution_time = np.mean([p.execution_time for p in perfs])
            
            insights.append({
                "model_id": model_id,
                "performance": {
                    "mae": avg_mae,
                    "rmse": avg_rmse,
                    "coverage": avg_coverage,
                    "execution_time": avg_execution_time
                },
                "samples": len(perfs)
            })
        
        # Find best performing model
        best_model = min(insights, key=lambda x: x["performance"]["mae"])
        
        # Generate recommendations
        recommendations = []
        
        if best_model["performance"]["mae"] < 10:
            recommendations.append("Model performance is excellent")
        elif best_model["performance"]["mae"] < 20:
            recommendations.append("Model performance is good")
        else:
            recommendations.append("Consider model retraining or parameter tuning")
        
        if best_model["performance"]["coverage"] < 0.85:
            recommendations.append("Uncertainty calibration needs improvement")
        
        if best_model["performance"]["execution_time"] > 5000:
            recommendations.append("Consider model optimization for faster execution")
        
        return {
            "status": "success",
            "insights": insights,
            "best_model": best_model,
            "recommendations": recommendations,
            "total_samples": len(recent_performance)
        }
    
    async def _update_model_performance(self, new_data: List[Dict[str, Any]]):
        """Update model performance with new data"""
        for data in new_data:
            try:
                perf = ModelPerformance(
                    model_id=data["model_id"],
                    model_type=data["model_type"],
                    mae=data["mae"],
                    rmse=data["rmse"],
                    coverage=data["coverage"],
                    interval_width=data["interval_width"],
                    execution_time=data["execution_time"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    station_id=data.get("station_id"),
                    horizon_hours=data.get("horizon_hours", 24)
                )
                self.model_performance.append(perf)
            except Exception as e:
                self.logger.error(f"Failed to add performance data: {e}")
        
        # Keep only recent performance data
        cutoff_time = datetime.now() - timedelta(hours=self.config.performance_window_hours * 2)
        self.model_performance = [
            p for p in self.model_performance if p.timestamp >= cutoff_time
        ]
        
        # Save updated performance data
        self._save_model_performance()
        self.last_performance_update = datetime.now()
    
    def _load_model_performance(self):
        """Load model performance data from file"""
        try:
            perf_path = Path(self.config.model_performance_file)
            if perf_path.exists():
                with open(perf_path, 'r') as f:
                    data = json.load(f)
                
                self.model_performance = []
                for item in data:
                    perf = ModelPerformance(
                        model_id=item["model_id"],
                        model_type=item["model_type"],
                        mae=item["mae"],
                        rmse=item["rmse"],
                        coverage=item["coverage"],
                        interval_width=item["interval_width"],
                        execution_time=item["execution_time"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        station_id=item.get("station_id"),
                        horizon_hours=item.get("horizon_hours", 24)
                    )
                    self.model_performance.append(perf)
                
                self.logger.info(f"Loaded {len(self.model_performance)} performance records")
            else:
                self.model_performance = []
                self.logger.info("No performance data file found, starting with empty data")
        except Exception as e:
            self.logger.error(f"Failed to load model performance: {e}")
            self.model_performance = []
    
    def _save_model_performance(self):
        """Save model performance data to file"""
        try:
            perf_path = Path(self.config.model_performance_file)
            perf_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = [perf.to_dict() for perf in self.model_performance]
            
            with open(perf_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save model performance: {e}")
    
    def add_performance_data(self, performance: ModelPerformance):
        """Add new performance data"""
        self.model_performance.append(performance)
        self._save_model_performance()
    
    def get_model_recommendation(self, station_id: str, horizon_hours: int) -> Optional[ModelRecommendation]:
        """Get cached model recommendation"""
        cache_key = f"{station_id}_{horizon_hours}"
        recommendation = self.model_recommendations.get(cache_key)
        
        if recommendation and recommendation.valid_until > datetime.now():
            return recommendation
        
        return None
