"""Configuration loading utilities for AirAware."""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from .models import AppConfig, DataConfig


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    # Check if running from project root
    if Path("configs").exists():
        return Path("configs")
    
    # Check relative to this file
    config_dir = Path(__file__).parent.parent.parent / "configs"
    if config_dir.exists():
        return config_dir
    
    raise FileNotFoundError("configs/ directory not found")


def load_yaml_config(config_name: str) -> dict:
    """Load YAML configuration file."""
    config_dir = get_config_dir()
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML config {config_path}: {e}")


def load_data_config(config_path: Optional[str] = None) -> DataConfig:
    """Load and validate data configuration.
    
    Args:
        config_path: Optional path to config file. If None, uses configs/data.yaml
        
    Returns:
        Validated DataConfig object
        
    Raises:
        FileNotFoundError: If config file not found
        ValidationError: If config validation fails
    """
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = load_yaml_config("data")
    
    try:
        return DataConfig(**config_dict)
    except ValidationError as e:
        raise ValueError(f"Data configuration validation failed: {e}")


def load_app_config(config_path: Optional[str] = None) -> AppConfig:
    """Load and validate application configuration.
    
    Args:
        config_path: Optional path to config file. If None, uses configs/app.yaml
        
    Returns:
        Validated AppConfig object
        
    Raises:
        FileNotFoundError: If config file not found  
        ValidationError: If config validation fails
    """
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = load_yaml_config("app")
    
    try:
        return AppConfig(**config_dict)
    except ValidationError as e:
        raise ValueError(f"Application configuration validation failed: {e}")


def save_data_config(config: DataConfig, config_path: Optional[str] = None) -> None:
    """Save data configuration to YAML file.
    
    Args:
        config: DataConfig object to save
        config_path: Optional path to save to. If None, uses configs/data.yaml
    """
    if config_path:
        output_path = Path(config_path)
    else:
        config_dir = get_config_dir()
        output_path = config_dir / "data.yaml"
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and save
    config_dict = config.model_dump()
    
    # Convert datetime objects to ISO strings for YAML serialization
    for key, value in config_dict.items():
        if hasattr(value, 'isoformat'):
            config_dict[key] = value.isoformat()
    
    # Handle nested datetime objects in stations
    if 'stations' in config_dict:
        for station in config_dict['stations']:
            for dt_field in ['first_measurement', 'last_measurement']:
                if station.get(dt_field):
                    station[dt_field] = station[dt_field].isoformat() if hasattr(station[dt_field], 'isoformat') else station[dt_field]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def save_app_config(config: AppConfig, config_path: Optional[str] = None) -> None:
    """Save application configuration to YAML file.
    
    Args:
        config: AppConfig object to save
        config_path: Optional path to save to. If None, uses configs/app.yaml
    """
    if config_path:
        output_path = Path(config_path)
    else:
        config_dir = get_config_dir()
        output_path = config_dir / "app.yaml"
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict and save
    config_dict = config.model_dump()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
