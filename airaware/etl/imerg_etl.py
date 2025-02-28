"""IMERG ETL Pipeline for precipitation data with JWT authentication."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..config import load_data_config
from ..feasibility.met.imerg_check import IMERGChecker
from .lineage_tracker import DataLineageTracker
from .manifest_manager import ETLArtifact, ManifestManager
from .quality_monitor import QualityMonitor

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class IMERGETL:
    """IMERG precipitation data ETL pipeline with spatial processing."""
    
    def __init__(self, 
                 raw_data_dir: str = "data/raw/imerg",
                 interim_data_dir: str = "data/interim",
                 use_local: bool = True,
                 cache_ttl_hours: int = 24):
        
        self.raw_data_dir = Path(raw_data_dir)
        self.interim_data_dir = Path(interim_data_dir)
        self.use_local = use_local
        self.cache_ttl_hours = cache_ttl_hours
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.interim_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        earthdata_token = os.getenv("EARTHDATA_TOKEN")
        self.imerg_checker = IMERGChecker(
            earthdata_token=earthdata_token,
            use_local=use_local and not earthdata_token
        )
        self.manifest_manager = ManifestManager()
        self.quality_monitor = QualityMonitor()
        self.lineage_tracker = DataLineageTracker()
        self.console = Console()
        
        # Load configuration
        self.config = load_data_config()
        self.valley_bbox = self.config.valley_bbox
        
        logger.info(f"IMERG ETL initialized for bounding box: {self.valley_bbox}")
    
    def generate_half_hourly_timestamps(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate half-hourly timestamps for IMERG data."""
        timestamps = []
        current = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        while current <= end_date:
            # IMERG has data every 30 minutes
            timestamps.append(current)
            timestamps.append(current + timedelta(minutes=30))
            current += timedelta(hours=1)
        
        return [ts for ts in timestamps if ts <= end_date]
    
    def get_hdf5_filename(self, timestamp: datetime) -> str:
        """Generate IMERG HDF5 filename for a specific timestamp."""
        # IMERG filename format: 3B-HHR.MS.MRG.3IMERG.20240601-S000000-E002959.0000.V07B.HDF5
        date_str = timestamp.strftime('%Y%m%d')
        
        # Start time (S) and end time (E) in format HHMMSS
        start_time = timestamp.strftime('%H%M%S')
        end_timestamp = timestamp + timedelta(minutes=29, seconds=59)
        end_time = end_timestamp.strftime('%H%M%S')
        
        return f"3B-HHR.MS.MRG.3IMERG.{date_str}-S{start_time}-E{end_time}.0000.V07B.HDF5"
    
    def get_hdf5_path(self, timestamp: datetime) -> Path:
        """Get HDF5 file path for a specific timestamp."""
        filename = self.get_hdf5_filename(timestamp)
        return self.raw_data_dir / filename
    
    def _download_imerg_file_direct(self, timestamp: datetime, file_path: Path) -> bool:
        """Download IMERG file directly using NASA Earthdata API."""
        try:
            # Construct IMERG URL for half-hourly data
            year = timestamp.year
            month = timestamp.month
            day = timestamp.day
            
            # IMERG half-hourly filename pattern
            filename = self.get_hdf5_filename(timestamp)
            url = f"https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGHH.07/{year:04d}/{month:02d}/{filename}"
            
            # Use session from IMERG checker
            if not self.imerg_checker.session:
                logger.error("No authentication session available")
                return False
            
            # Download with redirect handling
            response = self.imerg_checker.session.get(url, stream=True)
            
            # Handle 303 redirects (common for NASA Earthdata)
            if response.status_code == 303:
                redirect_url = response.headers.get('Location')
                if redirect_url:
                    logger.debug(f"Following redirect to: {redirect_url}")
                    response = self.imerg_checker.session.get(redirect_url, stream=True)
            
            if response.status_code == 200:
                # Write file
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.debug(f"Downloaded {file_path.stat().st_size / 1024 / 1024:.1f} MB")
                return True
            else:
                logger.warning(f"HTTP {response.status_code}: {response.reason}")
                return False
                
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def check_hdf5_valid(self, file_path: Path) -> bool:
        """Check if HDF5 file exists and is valid."""
        if not file_path.exists():
            return False
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Check for precipitation dataset
                if 'Grid/precipitationCal' not in f:
                    logger.warning(f"File {file_path} missing precipitation dataset")
                    return False
                
                # Check data shape
                precip_data = f['Grid/precipitationCal']
                if len(precip_data.shape) != 2:
                    logger.warning(f"File {file_path} has invalid precipitation data shape")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to validate {file_path}: {e}")
            return False
    
    def download_imerg_file(self, timestamp: datetime, force: bool = False) -> bool:
        """Download IMERG file for a specific timestamp."""
        file_path = self.get_hdf5_path(timestamp)
        
        # Check if file exists and is valid
        if not force and self.check_hdf5_valid(file_path):
            logger.info(f"REUSED: {file_path}")
            return True
        
        try:
            logger.info(f"DOWNLOADING: IMERG data for {timestamp}")
            
            # Download IMERG file using direct requests
            success = self._download_imerg_file_direct(timestamp, file_path)
            
            if success and self.check_hdf5_valid(file_path):
                logger.info(f"DOWNLOADED: {file_path}")
                
                # Track lineage
                self.lineage_tracker.track_extraction(
                    source="imerg_earthdata_api",
                    station_id=None,
                    record_count=1,  # One half-hourly record
                    metadata={
                        "timestamp": timestamp.isoformat(),
                        "bbox": self.valley_bbox.model_dump(),
                        "temporal_resolution": "30min"
                    }
                )
                
                return True
            else:
                logger.error(f"FAILED: Download failed for {timestamp}")
                return False
                
        except Exception as e:
            logger.error(f"FAILED: Error downloading {timestamp}: {e}")
            return False
    
    def process_hdf5_to_precip(self, file_path: Path, timestamp: datetime) -> Optional[Dict]:
        """Process HDF5 file to extract precipitation for Kathmandu Valley."""
        try:
            logger.debug(f"Processing HDF5: {file_path}")
            
            with h5py.File(file_path, 'r') as f:
                # Get precipitation data
                precip_data = f['Grid/precipitationCal'][:]
                
                # Get coordinate information
                lat_bounds = f['Grid/lat_bnds'][:]
                lon_bounds = f['Grid/lon_bnds'][:]
                
                # Find indices for Kathmandu Valley bounding box
                lat_mask = (
                    (lat_bounds[:, 0] <= self.valley_bbox.north) & 
                    (lat_bounds[:, 1] >= self.valley_bbox.south)
                )
                lon_mask = (
                    (lon_bounds[:, 0] <= self.valley_bbox.east) & 
                    (lon_bounds[:, 1] >= self.valley_bbox.west)
                )
                
                # Extract data for bounding box
                bbox_precip = precip_data[np.ix_(lat_mask, lon_mask)]
                
                # Calculate spatial mean (handle missing values)
                valid_precip = bbox_precip[bbox_precip >= 0]  # IMERG uses negative values for missing
                
                if len(valid_precip) > 0:
                    mean_precip = np.mean(valid_precip)
                else:
                    mean_precip = 0.0  # No valid precipitation data
                
                return {
                    'datetime_utc': timestamp,
                    'precipitation_mm_30min': float(mean_precip),
                    'valid_pixels': len(valid_precip),
                    'total_pixels': bbox_precip.size
                }
                
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return None
    
    def resample_to_hourly(self, half_hourly_data: List[Dict]) -> pd.DataFrame:
        """Resample half-hourly precipitation to hourly."""
        if not half_hourly_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(half_hourly_data)
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        df = df.sort_values('datetime_utc')
        
        # Resample to hourly by summing precipitation
        hourly_df = df.set_index('datetime_utc').resample('H').agg({
            'precipitation_mm_30min': 'sum',
            'valid_pixels': 'mean',
            'total_pixels': 'mean'
        }).reset_index()
        
        # Rename column
        hourly_df = hourly_df.rename(columns={
            'precipitation_mm_30min': 'precipitation_mm_hourly'
        })
        
        return hourly_df
    
    def run_etl(self, 
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                force: bool = False) -> Dict:
        """Run IMERG ETL pipeline."""
        
        # Use config dates if not provided
        if start_date is None:
            start_date = pd.to_datetime(self.config.date_from).replace(tzinfo=None)
        if end_date is None:
            end_date = pd.to_datetime(self.config.date_to).replace(tzinfo=None)
        
        logger.info(f"Starting IMERG ETL: {start_date.date()} to {end_date.date()}")
        
        # Generate half-hourly timestamps
        timestamps = self.generate_half_hourly_timestamps(start_date, end_date)
        
        results = {
            "start_time": datetime.now(),
            "timestamps_processed": 0,
            "files_downloaded": 0,
            "files_reused": 0,
            "files_failed": 0,
            "total_records": 0
        }
        
        half_hourly_data = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task(f"Processing {len(timestamps)} files...", total=len(timestamps))
            
            for timestamp in timestamps:
                progress.update(task, description=f"Processing {timestamp}")
                
                # Download if needed
                download_success = self.download_imerg_file(timestamp, force=force)
                results["timestamps_processed"] += 1
                
                if download_success:
                    file_path = self.get_hdf5_path(timestamp)
                    
                    if file_path.exists():
                        # Process precipitation data
                        precip_data = self.process_hdf5_to_precip(file_path, timestamp)
                        
                        if precip_data:
                            half_hourly_data.append(precip_data)
                            results["total_records"] += 1
                        
                        # Track download vs reuse
                        if f"DOWNLOADED: {file_path}" in str(logging.getLogger().handlers):
                            results["files_downloaded"] += 1
                        else:
                            results["files_reused"] += 1
                    
                else:
                    results["files_failed"] += 1
                
                progress.advance(task)
        
        # Resample to hourly and save
        if half_hourly_data:
            hourly_df = self.resample_to_hourly(half_hourly_data)
            
            if not hourly_df.empty:
                # Save to interim parquet
                output_path = self.interim_data_dir / "imerg_hourly.parquet"
                hourly_df.to_parquet(output_path, index=False)
                
                logger.info(f"Saved {len(hourly_df)} hourly records to {output_path}")
                
                # Track lineage for processing
                self.lineage_tracker.track_transformation(
                    source="data/raw/imerg/",
                    target=str(output_path),
                    operation="hdf5_to_parquet_spatial_avg_resample",
                    record_count=len(hourly_df),
                    metadata={
                        "temporal_resolution": "hourly",
                        "date_range": f"{start_date.date()} to {end_date.date()}",
                        "spatial_processing": "kathmandu_valley_bbox_mean"
                    }
                )
                
                # Update manifest
                artifact = ETLArtifact(
                    file_path=str(output_path),
                    file_type="parquet",
                    etl_stage="interim",
                    data_source="imerg",
                    created_at=datetime.now(),
                    file_size_bytes=output_path.stat().st_size,
                    record_count=len(hourly_df),
                    data_quality_score=0.9  # IMERG generally good but can have gaps
                )
                
                self.manifest_manager.add_etl_artifact(artifact)
                
                results["output_file"] = str(output_path)
                results["success"] = True
            else:
                logger.warning("No valid hourly data after resampling")
                results["success"] = False
        else:
            logger.warning("No data processed successfully")
            results["success"] = False
        
        results["end_time"] = datetime.now()
        results["duration_minutes"] = (results["end_time"] - results["start_time"]).total_seconds() / 60
        
        return results
    
    def get_imerg_status(self) -> Dict:
        """Get current IMERG ETL status."""
        output_file = self.interim_data_dir / "imerg_hourly.parquet"
        
        status = {
            "imerg_file_exists": output_file.exists(),
            "imerg_file_path": str(output_file),
            "raw_hdf5_files": len(list(self.raw_data_dir.glob("*.HDF5"))),
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
                logger.warning(f"Error reading IMERG file: {e}")
        
        return status
