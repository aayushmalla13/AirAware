"""Configuration versioning and change tracking for AirAware."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from .models import DataConfig

logger = logging.getLogger(__name__)


class ConfigChange(BaseModel):
    """Record of a configuration change."""
    timestamp: datetime
    version: str
    change_type: str  # "station_added", "station_removed", "threshold_changed", etc.
    description: str
    user: Optional[str] = None
    config_hash: str
    changes: Dict = {}  # Detailed change information


class ConfigVersion(BaseModel):
    """Configuration version metadata."""
    version: str
    timestamp: datetime
    config_hash: str
    station_count: int
    avg_quality_score: float
    description: str
    is_active: bool = False


class ConfigVersionManager:
    """Manages configuration versions and change tracking."""
    
    def __init__(self, versions_file: str = "data/artifacts/config_versions.json"):
        self.versions_file = Path(versions_file)
        self.versions_file.parent.mkdir(parents=True, exist_ok=True)
        
    def _calculate_config_hash(self, config: DataConfig) -> str:
        """Calculate hash of configuration for change detection."""
        # Create normalized dict for hashing
        config_dict = config.model_dump()
        
        # Remove timestamp fields that change on every save
        config_for_hash = {
            "stations": [
                {
                    "station_id": s["station_id"],
                    "quality_score": s["quality_score"],
                    "missingness_pct": s["missingness_pct"]
                }
                for s in config_dict["stations"]
            ],
            "min_quality_score": config_dict["min_quality_score"],
            "max_missingness_pct": config_dict["max_missingness_pct"],
            "search_radius_km": config_dict["search_radius_km"]
        }
        
        # Calculate SHA256 hash
        config_str = json.dumps(config_for_hash, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _load_versions(self) -> List[ConfigVersion]:
        """Load existing configuration versions."""
        if not self.versions_file.exists():
            return []
        
        try:
            with open(self.versions_file) as f:
                data = json.load(f)
                return [ConfigVersion(**v) for v in data.get("versions", [])]
        except Exception as e:
            logger.warning(f"Failed to load config versions: {e}")
            return []
    
    def _save_versions(self, versions: List[ConfigVersion]) -> None:
        """Save configuration versions to file."""
        try:
            data = {
                "versions": [v.model_dump() for v in versions],
                "last_updated": datetime.now().isoformat()
            }
            
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save config versions: {e}")
    
    def create_version(self, config: DataConfig, description: str, user: Optional[str] = None) -> str:
        """Create a new configuration version."""
        versions = self._load_versions()
        
        # Calculate version details
        config_hash = self._calculate_config_hash(config)
        version_number = f"v{len(versions) + 1:03d}"
        avg_quality = sum(s.quality_score for s in config.stations) / len(config.stations)
        
        # Check if this is actually a new version
        if versions and versions[-1].config_hash == config_hash:
            logger.info("Configuration unchanged, not creating new version")
            return versions[-1].version
        
        # Deactivate previous versions
        for version in versions:
            version.is_active = False
        
        # Create new version
        new_version = ConfigVersion(
            version=version_number,
            timestamp=datetime.now(),
            config_hash=config_hash,
            station_count=len(config.stations),
            avg_quality_score=avg_quality,
            description=description,
            is_active=True
        )
        
        versions.append(new_version)
        self._save_versions(versions)
        
        logger.info(f"Created configuration version {version_number}: {description}")
        return version_number
    
    def get_version_history(self) -> List[ConfigVersion]:
        """Get configuration version history."""
        return sorted(self._load_versions(), key=lambda v: v.timestamp, reverse=True)
    
    def get_active_version(self) -> Optional[ConfigVersion]:
        """Get the currently active configuration version."""
        versions = self._load_versions()
        for version in versions:
            if version.is_active:
                return version
        return None
    
    def compare_versions(self, version1: str, version2: str) -> Dict:
        """Compare two configuration versions."""
        versions = {v.version: v for v in self._load_versions()}
        
        if version1 not in versions or version2 not in versions:
            raise ValueError(f"Version not found")
        
        v1, v2 = versions[version1], versions[version2]
        
        return {
            "version_1": v1.model_dump(),
            "version_2": v2.model_dump(),
            "differences": {
                "station_count_change": v2.station_count - v1.station_count,
                "quality_score_change": v2.avg_quality_score - v1.avg_quality_score,
                "time_difference": (v2.timestamp - v1.timestamp).total_seconds() / 3600  # hours
            }
        }


