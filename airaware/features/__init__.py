"""Feature engineering pipeline for AirAware PM₂.₅ nowcasting."""

from .data_joiner import DataJoiner, JoinResult
from .feature_builder import FeatureBuilder, FeatureConfig
from .temporal_features import TemporalFeatureGenerator
from .met_features import MeteorologicalFeatureGenerator
from .feature_validator import FeatureValidator, FeatureQualityMetrics
from .advanced_feature_selector import AdvancedFeatureSelector, FeatureSelectionConfig
from .domain_features import DomainFeatureGenerator, DomainFeatureConfig
from .interaction_detector import FeatureInteractionDetector, InteractionConfig

__all__ = [
    "DataJoiner",
    "JoinResult", 
    "FeatureBuilder",
    "FeatureConfig",
    "TemporalFeatureGenerator",
    "MeteorologicalFeatureGenerator",
    "FeatureValidator",
    "FeatureQualityMetrics",
    "AdvancedFeatureSelector",
    "FeatureSelectionConfig",
    "DomainFeatureGenerator", 
    "DomainFeatureConfig",
    "FeatureInteractionDetector",
    "InteractionConfig",
]
