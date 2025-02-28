"""ERA5 ETL Pipeline for meteorological data with reuse-or-fetch logic."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..config import load_data_config
from ..feasibility.met.era5_check import ERA5Checker
from .data_validator import OpenAQDataValidator
from .lineage_tracker import DataLineageTracker
from .manifest_manager import ETLArtifact, ManifestManager
from .quality_monitor import QualityMonitor
from .met_data_validator import MeteorologicalDataValidator
from .performance_optimizer import ETLPerformanceOptimizer, MemoryMonitor
from .error_recovery import ETLErrorRecovery, ResilientETLWrapper
from .data_completeness import DataCompletenessAnalyzer

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ERA5ETL:
    """ERA5 meteorological data ETL pipeline with spatial processing."""
    
    def __init__(self, 
                 raw_data_dir: str = "data/raw/era5",
                 interim_data_dir: str = "data/interim",
                 use_local: bool = True,
                 cache_ttl_hours: int = 24,
                 max_workers: int = 2):
        
        self.raw_data_dir = Path(raw_data_dir)
        self.interim_data_dir = Path(interim_data_dir)
        self.use_local = use_local
        self.cache_ttl_hours = cache_ttl_hours
        self.max_workers = max_workers
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.interim_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        cds_api_key = os.getenv("CDSAPI_KEY")
        self.era5_checker = ERA5Checker(
            cds_api_key=cds_api_key,
            use_local=use_local and not cds_api_key
        )
        self.manifest_manager = ManifestManager()
        self.quality_monitor = QualityMonitor()
        self.lineage_tracker = DataLineageTracker()
        self.console = Console()
        
        # Enhanced components
        self.met_validator = MeteorologicalDataValidator()
        self.performance_optimizer = ETLPerformanceOptimizer()
        self.memory_monitor = MemoryMonitor()
        self.error_recovery = ETLErrorRecovery()
        self.resilient_wrapper = ResilientETLWrapper(self.error_recovery)
        self.completeness_analyzer = DataCompletenessAnalyzer()
        
        # Load configuration
        self.config = load_data_config()
        self.valley_bbox = self.config.valley_bbox
        
        # ERA5 variables to download
        self.era5_variables = ["u10", "v10", "t2m", "blh"]
        
        logger.info(f"ERA5 ETL initialized for bounding box: {self.valley_bbox}")
    
    def generate_date_range(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate daily date range for ERA5 downloads."""
        dates = []
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        
        return dates
    
    def get_netcdf_path(self, date: datetime) -> Path:
        """Get NetCDF file path for a specific date."""
        return self.raw_data_dir / f"era5_{date.strftime('%Y%m%d')}.nc"
    
    def check_file_exists_and_valid(self, file_path: Path) -> bool:
        """Check if NetCDF file exists and is valid."""
        if not file_path.exists():
            return False
        
        try:
            # Quick validation - try to open with xarray
            with xr.open_dataset(file_path) as ds:
                # Check required variables
                missing_vars = [var for var in self.era5_variables if var not in ds.variables]
                if missing_vars:
                    logger.warning(f"File {file_path} missing variables: {missing_vars}")
                    return False
                
                # Check time dimension (should have 24 hours)
                time_dim = 'time' if 'time' in ds.dims else 'valid_time'
                if time_dim in ds.dims and ds.sizes[time_dim] != 24:
                    logger.warning(f"File {file_path} has {ds.sizes[time_dim]} time steps, expected 24")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to validate {file_path}: {e}")
            return False
    
    def download_era5_day(self, date: datetime, force: bool = False) -> bool:
        """Download ERA5 data for a specific day."""
        file_path = self.get_netcdf_path(date)
        
        # Check if file exists and is valid
        if not force and self.check_file_exists_and_valid(file_path):
            logger.info(f"REUSED: {file_path}")
            return True
        
        try:
            logger.info(f"DOWNLOADING: ERA5 data for {date.strftime('%Y-%m-%d')}")
            
            # Use ERA5Checker internal method to download
            bbox = {
                "north": self.valley_bbox.north,
                "south": self.valley_bbox.south,
                "east": self.valley_bbox.east,
                "west": self.valley_bbox.west
            }
            
            try:
                self.era5_checker._download_era5_day(
                    date=date,
                    bbox=bbox,
                    variables=self.era5_variables,
                    output_file=file_path
                )
                success = True
            except Exception as e:
                logger.error(f"ERA5 download failed: {e}")
                success = False
            
            if success and self.check_file_exists_and_valid(file_path):
                logger.info(f"DOWNLOADED: {file_path}")
                
                # Track lineage
                self.lineage_tracker.track_extraction(
                    source="era5_cds_api",
                    station_id=None,
                    record_count=24,  # 24 hourly records
                    metadata={
                        "date": date.strftime('%Y-%m-%d'),
                        "variables": self.era5_variables,
                        "bbox": self.valley_bbox.model_dump()
                    }
                )
                
                return True
            else:
                logger.error(f"FAILED: Download failed for {date.strftime('%Y-%m-%d')}")
                return False
                
        except Exception as e:
            logger.error(f"FAILED: Error downloading {date.strftime('%Y-%m-%d')}: {e}")
            return False
    
    def process_netcdf_to_hourly(self, file_path: Path) -> pd.DataFrame:
        """Process NetCDF file to hourly DataFrame with spatial averaging."""
        try:
            logger.info(f"Processing NetCDF: {file_path}")
            
            with xr.open_dataset(file_path) as ds:
                # Spatial averaging over Kathmandu Valley bounding box
                bbox_data = ds.sel(
                    latitude=slice(self.valley_bbox.north, self.valley_bbox.south),
                    longitude=slice(self.valley_bbox.west, self.valley_bbox.east)
                )
                
                # Calculate spatial means
                bbox_means = bbox_data.mean(dim=['latitude', 'longitude'])
                
                # Convert to DataFrame
                df_data = []
                time_coord = 'time' if 'time' in bbox_means.coords else 'valid_time'
                time_values = bbox_means[time_coord].values
                
                for time_idx, time_val in enumerate(time_values):
                    timestamp = pd.to_datetime(time_val).replace(tzinfo=None)
                    
                    row = {
                        'datetime_utc': timestamp,
                        'u10': float(bbox_means.u10.isel({time_coord: time_idx}).values),
                        'v10': float(bbox_means.v10.isel({time_coord: time_idx}).values), 
                        't2m': float(bbox_means.t2m.isel({time_coord: time_idx}).values),
                        'blh': float(bbox_means.blh.isel({time_coord: time_idx}).values)
                    }
                    
                    # Add derived wind features
                    row['wind_speed'] = np.sqrt(row['u10']**2 + row['v10']**2)
                    row['wind_direction'] = np.degrees(np.arctan2(row['v10'], row['u10'])) % 360
                    
                    # Convert temperature from Kelvin to Celsius
                    row['t2m_celsius'] = row['t2m'] - 273.15
                    
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df = df.sort_values('datetime_utc').reset_index(drop=True)
                
                logger.info(f"Processed {len(df)} hourly records from {file_path}")
                return df
                
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return pd.DataFrame()
    
    def run_etl(self, 
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                force: bool = False) -> Dict:
        """Run ERA5 ETL pipeline."""
        
        # Use config dates if not provided
        if start_date is None:
            start_date = pd.to_datetime(self.config.date_from).replace(tzinfo=None)
        if end_date is None:
            end_date = pd.to_datetime(self.config.date_to).replace(tzinfo=None)
        
        logger.info(f"Starting ERA5 ETL: {start_date.date()} to {end_date.date()}")
        
        # Generate date range
        dates = self.generate_date_range(start_date, end_date)
        
        results = {
            "start_time": datetime.now(),
            "dates_processed": 0,
            "dates_downloaded": 0,
            "dates_reused": 0,
            "dates_failed": 0,
            "total_records": 0
        }
        
        all_data = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task(f"Processing {len(dates)} days...", total=len(dates))
            
            for date in dates:
                progress.update(task, description=f"Processing {date.strftime('%Y-%m-%d')}")
                
                # Download if needed
                download_success = self.download_era5_day(date, force=force)
                results["dates_processed"] += 1
                
                if download_success:
                    file_path = self.get_netcdf_path(date)
                    
                    if file_path.exists():
                        # Simple tracking - assume downloaded if force was used
                        if force:
                            results["dates_downloaded"] += 1
                        else:
                            results["dates_reused"] += 1
                        
                        # Process to hourly data
                        daily_df = self.process_netcdf_to_hourly(file_path)
                        if not daily_df.empty:
                            all_data.append(daily_df)
                            results["total_records"] += len(daily_df)
                    
                else:
                    results["dates_failed"] += 1
                
                progress.advance(task)
        
        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('datetime_utc').reset_index(drop=True)
            
            # Enhanced data quality validation
            quality_metrics = self.met_validator.validate_era5_data(combined_df)
            logger.info(f"ERA5 data quality score: {quality_metrics.overall_quality_score:.3f}")
            
            # Data completeness analysis
            completeness_metrics = self.completeness_analyzer.analyze_era5_completeness(
                combined_df, start_date, end_date
            )
            logger.info(f"ERA5 completeness: {completeness_metrics.completeness_percentage:.1f}% "
                       f"(Grade: {completeness_metrics.data_quality_grade})")
            
            # Save to interim parquet with performance optimization
            output_path = self.interim_data_dir / "era5_hourly.parquet"
            perf_result = self.performance_optimizer.optimize_parquet_writing(
                combined_df, output_path, compression='snappy'
            )
            
            logger.info(f"Saved {len(combined_df)} records to {output_path}")
            
            # Track lineage for processing
            transform_event_id = self.lineage_tracker.track_transformation(
                source="data/raw/era5/",
                target=str(output_path),
                operation="netcdf_to_parquet_spatial_avg",
                record_count=len(combined_df),
                metadata={
                    "variables": list(combined_df.columns),
                    "date_range": f"{start_date.date()} to {end_date.date()}",
                    "spatial_processing": "kathmandu_valley_bbox_mean",
                    "quality_score": quality_metrics.overall_quality_score,
                    "completeness_grade": completeness_metrics.data_quality_grade
                }
            )
            
            # Update manifest with enhanced metrics
            artifact = ETLArtifact(
                file_path=str(output_path),
                file_type="parquet",
                etl_stage="interim",
                data_source="era5",
                created_at=datetime.now(),
                file_size_bytes=output_path.stat().st_size,
                record_count=len(combined_df),
                data_quality_score=quality_metrics.overall_quality_score
            )
            
            self.manifest_manager.add_etl_artifact(artifact)
            
            # Store enhanced results
            results["output_file"] = str(output_path)
            results["success"] = True
            results["quality_metrics"] = quality_metrics.model_dump()
            results["completeness_metrics"] = completeness_metrics.model_dump()
            results["performance_metrics"] = perf_result
            
        else:
            logger.warning("No data processed successfully")
            results["success"] = False
        
        results["end_time"] = datetime.now()
        results["duration_minutes"] = (results["end_time"] - results["start_time"]).total_seconds() / 60
        
        return results
    
    def get_era5_status(self) -> Dict:
        """Get current ERA5 ETL status."""
        output_file = self.interim_data_dir / "era5_hourly.parquet"
        
        status = {
            "era5_file_exists": output_file.exists(),
            "era5_file_path": str(output_file),
            "raw_netcdf_files": len(list(self.raw_data_dir.glob("*.nc"))),
            "latest_data": None,
            "record_count": 0,
            "date_range": None
        }
        
        if output_file.exists():
            try:
                df = pd.read_parquet(output_file)
                status["record_count"] = len(df)
                
                if not df.empty:
                    status["latest_data"] = df['datetime_utc'].max().isoformat()
                    status["date_range"] = f"{df['datetime_utc'].min().date()} to {df['datetime_utc'].max().date()}"
                    
            except Exception as e:
                logger.warning(f"Error reading ERA5 file: {e}")
        
        return status
