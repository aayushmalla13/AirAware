"""Manifest management for ETL pipeline integration with CP-0 data scanner."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ETLArtifact(BaseModel):
    """Represents an ETL-generated data artifact."""
    file_path: str
    file_type: str  # 'parquet', 'netcdf', 'hdf5'
    etl_stage: str  # 'raw', 'interim', 'processed'
    data_source: str  # 'openaq', 'era5', 'imerg'
    station_id: Optional[int] = None
    date_partition: Optional[str] = None  # YYYY-MM format
    created_at: datetime
    file_size_bytes: int
    record_count: Optional[int] = None
    data_quality_score: Optional[float] = None
    schema_version: str = "1.0"


class ManifestManager:
    """Manages data manifest integration for ETL pipeline."""
    
    def __init__(self, manifest_path: str = "data/artifacts/local_data_manifest.json"):
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_manifest(self) -> Dict:
        """Load existing data manifest."""
        if not self.manifest_path.exists():
            logger.info("No existing manifest found, starting fresh")
            return {
                "scan_timestamp": datetime.now().isoformat(),
                "files": [],
                "summary": {
                    "total_files": 0,
                    "total_size_mb": 0.0,
                    "datasets": {}
                },
                "etl_artifacts": []
            }
        
        try:
            with open(self.manifest_path) as f:
                manifest = json.load(f)
                
            # Ensure ETL artifacts section exists
            if "etl_artifacts" not in manifest:
                manifest["etl_artifacts"] = []
                
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return {}
    
    def save_manifest(self, manifest: Dict) -> None:
        """Save updated manifest."""
        try:
            manifest["last_updated"] = datetime.now().isoformat()
            
            with open(self.manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
                
            logger.debug(f"Manifest saved to {self.manifest_path}")
            
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
    
    def add_etl_artifact(self, artifact: ETLArtifact) -> None:
        """Add a new ETL artifact to the manifest."""
        manifest = self.load_manifest()
        
        # Convert to dict for storage
        artifact_dict = artifact.model_dump()
        
        # Check for existing artifact (avoid duplicates)
        existing_idx = None
        for i, existing in enumerate(manifest["etl_artifacts"]):
            if (existing.get("file_path") == artifact.file_path and
                existing.get("etl_stage") == artifact.etl_stage):
                existing_idx = i
                break
        
        if existing_idx is not None:
            # Update existing artifact
            manifest["etl_artifacts"][existing_idx] = artifact_dict
            logger.info(f"Updated existing ETL artifact: {artifact.file_path}")
        else:
            # Add new artifact
            manifest["etl_artifacts"].append(artifact_dict)
            logger.info(f"Added new ETL artifact: {artifact.file_path}")
        
        # Update summary statistics
        self._update_etl_summary(manifest)
        
        self.save_manifest(manifest)
    
    def get_artifacts_by_source(self, data_source: str, etl_stage: Optional[str] = None) -> List[ETLArtifact]:
        """Get ETL artifacts filtered by data source and optionally by stage."""
        manifest = self.load_manifest()
        
        artifacts = []
        for artifact_dict in manifest.get("etl_artifacts", []):
            if artifact_dict.get("data_source") == data_source:
                if etl_stage is None or artifact_dict.get("etl_stage") == etl_stage:
                    try:
                        artifact = ETLArtifact(**artifact_dict)
                        artifacts.append(artifact)
                    except Exception as e:
                        logger.warning(f"Failed to parse artifact: {e}")
        
        return artifacts
    
    def get_station_artifacts(self, station_id: int, data_source: str = "openaq") -> List[ETLArtifact]:
        """Get artifacts for a specific station."""
        manifest = self.load_manifest()
        
        artifacts = []
        for artifact_dict in manifest.get("etl_artifacts", []):
            if (artifact_dict.get("station_id") == station_id and
                artifact_dict.get("data_source") == data_source):
                try:
                    artifact = ETLArtifact(**artifact_dict)
                    artifacts.append(artifact)
                except Exception as e:
                    logger.warning(f"Failed to parse station artifact: {e}")
        
        return artifacts
    
    def get_date_partitions(self, station_id: int, data_source: str = "openaq") -> List[str]:
        """Get available date partitions for a station."""
        artifacts = self.get_station_artifacts(station_id, data_source)
        
        partitions = set()
        for artifact in artifacts:
            if artifact.date_partition:
                partitions.add(artifact.date_partition)
        
        return sorted(list(partitions))
    
    def check_data_exists(self, station_id: int, year_month: str, data_source: str = "openaq") -> bool:
        """Check if data exists for a specific station and time period."""
        artifacts = self.get_station_artifacts(station_id, data_source)
        
        for artifact in artifacts:
            if artifact.date_partition == year_month:
                # Check if file actually exists
                if Path(artifact.file_path).exists():
                    return True
                else:
                    logger.warning(f"Manifest references missing file: {artifact.file_path}")
        
        return False
    
    def get_missing_partitions(self, station_id: int, required_partitions: List[str], 
                              data_source: str = "openaq") -> List[str]:
        """Get list of missing data partitions for a station."""
        existing_partitions = set(self.get_date_partitions(station_id, data_source))
        required_partitions_set = set(required_partitions)
        
        missing = required_partitions_set - existing_partitions
        return sorted(list(missing))
    
    def _update_etl_summary(self, manifest: Dict) -> None:
        """Update ETL summary statistics in manifest."""
        etl_artifacts = manifest.get("etl_artifacts", [])
        
        # Calculate ETL-specific stats
        etl_summary = {
            "total_etl_artifacts": len(etl_artifacts),
            "by_source": {},
            "by_stage": {},
            "total_etl_size_mb": 0.0,
            "total_records": 0
        }
        
        for artifact in etl_artifacts:
            source = artifact.get("data_source", "unknown")
            stage = artifact.get("etl_stage", "unknown")
            
            # Count by source
            etl_summary["by_source"][source] = etl_summary["by_source"].get(source, 0) + 1
            
            # Count by stage
            etl_summary["by_stage"][stage] = etl_summary["by_stage"].get(stage, 0) + 1
            
            # Sum sizes and records
            if artifact.get("file_size_bytes"):
                etl_summary["total_etl_size_mb"] += artifact["file_size_bytes"] / (1024 * 1024)
            
            if artifact.get("record_count"):
                etl_summary["total_records"] += artifact["record_count"]
        
        # Add ETL summary to manifest
        if "summary" not in manifest:
            manifest["summary"] = {}
        
        manifest["summary"]["etl"] = etl_summary
        
        logger.debug(f"Updated ETL summary: {etl_summary['total_etl_artifacts']} artifacts, "
                    f"{etl_summary['total_etl_size_mb']:.1f} MB, {etl_summary['total_records']} records")
    
    def cleanup_stale_artifacts(self, max_age_days: int = 30) -> int:
        """Remove manifest entries for files that no longer exist."""
        manifest = self.load_manifest()
        
        etl_artifacts = manifest.get("etl_artifacts", [])
        cleaned_artifacts = []
        removed_count = 0
        
        for artifact in etl_artifacts:
            file_path = Path(artifact.get("file_path", ""))
            
            if file_path.exists():
                cleaned_artifacts.append(artifact)
            else:
                logger.info(f"Removing stale artifact reference: {file_path}")
                removed_count += 1
        
        if removed_count > 0:
            manifest["etl_artifacts"] = cleaned_artifacts
            self._update_etl_summary(manifest)
            self.save_manifest(manifest)
            
            logger.info(f"Cleaned up {removed_count} stale artifact references")
        
        return removed_count
    
    def get_etl_stats(self) -> Dict:
        """Get comprehensive ETL statistics from manifest."""
        manifest = self.load_manifest()
        return manifest.get("summary", {}).get("etl", {})
    
    def export_etl_catalog(self, output_path: str) -> None:
        """Export ETL artifact catalog for external tools."""
        manifest = self.load_manifest()
        
        catalog = {
            "export_timestamp": datetime.now().isoformat(),
            "etl_artifacts": manifest.get("etl_artifacts", []),
            "summary": manifest.get("summary", {}).get("etl", {})
        }
        
        with open(output_path, 'w') as f:
            json.dump(catalog, f, indent=2, default=str)
        
        logger.info(f"ETL catalog exported to {output_path}")
    
    def validate_manifest_integrity(self) -> bool:
        """Validate manifest integrity and file references."""
        manifest = self.load_manifest()
        
        issues = []
        total_artifacts = len(manifest.get("etl_artifacts", []))
        
        for i, artifact in enumerate(manifest.get("etl_artifacts", [])):
            file_path = artifact.get("file_path")
            
            if not file_path:
                issues.append(f"Artifact {i}: Missing file_path")
                continue
            
            if not Path(file_path).exists():
                issues.append(f"Artifact {i}: File not found: {file_path}")
        
        if issues:
            logger.warning(f"Manifest integrity issues found: {len(issues)}/{total_artifacts}")
            for issue in issues[:10]:  # Show first 10 issues
                logger.warning(f"  - {issue}")
            return False
        
        logger.info(f"Manifest integrity validated: {total_artifacts} artifacts OK")
        return True


