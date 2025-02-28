"""Environment-specific configuration management for AirAware."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from .loader import load_data_config, save_data_config
from .models import AppConfig, DataConfig

logger = logging.getLogger(__name__)


class EnvironmentConfig(BaseModel):
    """Environment-specific configuration settings."""
    name: str
    description: str
    data_config_overrides: Dict = {}
    app_config_overrides: Dict = {}
    station_filters: Dict = {}  # Filters to apply when selecting stations
    created_at: str
    created_by: Optional[str] = None


class ConfigEnvironmentManager:
    """Manages configuration across different environments (dev, staging, prod)."""
    
    def __init__(self, environments_dir: str = "configs/environments"):
        self.environments_dir = Path(environments_dir)
        self.environments_dir.mkdir(parents=True, exist_ok=True)
    
    def create_environment(self, name: str, description: str, 
                          base_config: Optional[DataConfig] = None,
                          station_count_limit: Optional[int] = None,
                          quality_threshold_override: Optional[float] = None,
                          created_by: Optional[str] = None) -> EnvironmentConfig:
        """Create a new environment configuration."""
        from datetime import datetime
        
        # Build overrides based on parameters
        data_overrides = {}
        station_filters = {}
        
        if quality_threshold_override is not None:
            data_overrides["min_quality_score"] = quality_threshold_override
        
        if station_count_limit is not None:
            station_filters["max_stations"] = station_count_limit
        
        # Environment-specific app config overrides
        app_overrides = {}
        if name == "development":
            app_overrides.update({
                "log_level": "DEBUG",
                "enable_cors": True,
                "cors_origins": ["*"]
            })
        elif name == "staging":
            app_overrides.update({
                "log_level": "INFO",
                "enable_cors": True,
                "cors_origins": ["https://staging.airaware.com"]
            })
        elif name == "production":
            app_overrides.update({
                "log_level": "WARNING",
                "enable_cors": False,
                "rate_limit_requests": 50
            })
        
        env_config = EnvironmentConfig(
            name=name,
            description=description,
            data_config_overrides=data_overrides,
            app_config_overrides=app_overrides,
            station_filters=station_filters,
            created_at=datetime.now().isoformat(),
            created_by=created_by
        )
        
        # Save environment configuration
        env_file = self.environments_dir / f"{name}.json"
        with open(env_file, 'w') as f:
            json.dump(env_config.model_dump(), f, indent=2)
        
        logger.info(f"Created environment configuration: {name}")
        return env_config
    
    def load_environment(self, name: str) -> EnvironmentConfig:
        """Load environment configuration."""
        env_file = self.environments_dir / f"{name}.json"
        
        if not env_file.exists():
            raise FileNotFoundError(f"Environment '{name}' not found")
        
        with open(env_file) as f:
            env_data = json.load(f)
        
        return EnvironmentConfig(**env_data)
    
    def list_environments(self) -> List[str]:
        """List available environment configurations."""
        return [f.stem for f in self.environments_dir.glob("*.json")]
    
    def apply_environment(self, env_name: str, base_data_config: Optional[DataConfig] = None) -> DataConfig:
        """Apply environment configuration to base config."""
        env_config = self.load_environment(env_name)
        
        # Load base config if not provided
        if base_data_config is None:
            base_data_config = load_data_config()
        
        # Create a copy to modify
        modified_config_dict = base_data_config.model_dump()
        
        # Apply data config overrides
        for key, value in env_config.data_config_overrides.items():
            if key in modified_config_dict:
                modified_config_dict[key] = value
                logger.info(f"Applied override: {key} = {value}")
        
        # Apply station filters
        if env_config.station_filters:
            stations = modified_config_dict["stations"]
            
            # Filter by maximum station count
            if "max_stations" in env_config.station_filters:
                max_stations = env_config.station_filters["max_stations"]
                if len(stations) > max_stations:
                    # Keep highest quality stations
                    stations.sort(key=lambda s: s["quality_score"], reverse=True)
                    modified_config_dict["stations"] = stations[:max_stations]
                    logger.info(f"Limited stations to {max_stations} for environment {env_name}")
        
        # Create new config with modifications
        return DataConfig(**modified_config_dict)
    
    def export_for_environment(self, env_name: str, output_dir: str = "exports") -> str:
        """Export configuration for specific environment."""
        export_dir = Path(output_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply environment configuration
        env_config = self.apply_environment(env_name)
        
        # Export data config
        data_export_path = export_dir / f"data_config_{env_name}.yaml"
        save_data_config(env_config, str(data_export_path))
        
        # Create deployment info
        from datetime import datetime
        deployment_info = {
            "environment": env_name,
            "exported_at": datetime.now().isoformat(),
            "station_count": len(env_config.stations),
            "quality_threshold": env_config.min_quality_score,
            "export_files": [str(data_export_path)]
        }
        
        info_path = export_dir / f"deployment_info_{env_name}.json"
        with open(info_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"Exported {env_name} configuration to {export_dir}")
        return str(export_dir)
    
    def validate_environment_compatibility(self, env_name: str) -> Dict:
        """Validate that environment configuration is compatible with base config."""
        try:
            env_config = self.load_environment(env_name)
            applied_config = self.apply_environment(env_name)
            
            validation_result = {
                "compatible": True,
                "warnings": [],
                "errors": [],
                "station_count": len(applied_config.stations),
                "quality_threshold": applied_config.min_quality_score
            }
            
            # Check minimum requirements
            if len(applied_config.stations) < 3:
                validation_result["errors"].append(
                    f"Environment {env_name} results in only {len(applied_config.stations)} stations (minimum 3 required)"
                )
                validation_result["compatible"] = False
            
            # Check quality thresholds
            if applied_config.min_quality_score < 0.1:
                validation_result["warnings"].append(
                    f"Very low quality threshold ({applied_config.min_quality_score}) in environment {env_name}"
                )
            
            return validation_result
            
        except Exception as e:
            return {
                "compatible": False,
                "errors": [f"Failed to validate environment {env_name}: {e}"],
                "warnings": [],
                "station_count": 0,
                "quality_threshold": 0.0
            }


def create_default_environments():
    """Create default environment configurations."""
    manager = ConfigEnvironmentManager()
    
    # Development environment
    manager.create_environment(
        name="development",
        description="Development environment with verbose logging and relaxed CORS",
        quality_threshold_override=0.2,  # Lower threshold for testing
        created_by="system"
    )
    
    # Staging environment  
    manager.create_environment(
        name="staging",
        description="Staging environment for testing with production-like settings",
        station_count_limit=5,  # Limit for cost control
        created_by="system"
    )
    
    # Production environment
    manager.create_environment(
        name="production", 
        description="Production environment with strict security and logging",
        quality_threshold_override=0.4,  # Higher threshold for reliability
        created_by="system"
    )
    
    return ["development", "staging", "production"]
