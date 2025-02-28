"""Automatic data updater that fetches latest data and extends historical coverage."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..etl.openaq_etl import OpenAQETL
from ..etl.era5_etl import ERA5ETL
from ..etl.imerg_etl import IMERGETL
from ..features import FeatureBuilder, FeatureConfig

logger = logging.getLogger(__name__)


class UpdateConfig(BaseModel):
    """Configuration for automatic data updates."""
    # Data collection parameters
    start_date: str = Field("2024-06-01", description="Start date for historical collection")
    end_date: str = Field("2025-09-30", description="End date for collection")
    auto_extend_to_latest: bool = Field(True, description="Automatically extend to latest available")
    
    # Update frequency
    update_frequency_hours: int = Field(24, description="How often to check for updates (hours)")
    max_gap_days: int = Field(7, description="Maximum gap before forcing update")
    
    # Data sources to update
    update_openaq: bool = Field(True, description="Update OpenAQ data")
    update_era5: bool = Field(True, description="Update ERA5 data")
    update_imerg: bool = Field(False, description="Update IMERG data (optional)")
    
    # Feature engineering
    regenerate_features: bool = Field(True, description="Regenerate features after data update")
    
    # Performance settings
    batch_size_days: int = Field(30, description="Process data in batches of N days")
    parallel_stations: bool = Field(True, description="Process stations in parallel")


class UpdateResult(BaseModel):
    """Result of data update operation."""
    success: bool
    start_date: str
    end_date: str
    records_added: int
    sources_updated: List[str]
    features_regenerated: bool
    update_duration_minutes: float
    next_update_due: str
    data_quality_score: float = Field(ge=0, le=1)


class AutoDataUpdater:
    """Automatically update data from multiple sources with intelligent scheduling."""
    
    def __init__(self, config: Optional[UpdateConfig] = None):
        self.config = config or UpdateConfig()
        self.console = Console()
        
        # Initialize ETL components
        self.openaq_etl = OpenAQETL() if self.config.update_openaq else None
        self.era5_etl = ERA5ETL() if self.config.update_era5 else None
        self.imerg_etl = IMERGETL() if self.config.update_imerg else None
        
        # Initialize feature builder
        self.feature_builder = FeatureBuilder() if self.config.regenerate_features else None
        
        logger.info("AutoDataUpdater initialized")
    
    def check_and_update(self) -> UpdateResult:
        """Check if update is needed and perform it."""
        
        logger.info("Checking if data update is needed")
        
        # Check current data status
        data_status = self._assess_current_data()
        
        # Determine if update is needed
        needs_update = self._needs_update(data_status)
        
        if not needs_update:
            logger.info("Data is up to date, no update needed")
            return UpdateResult(
                success=True,
                start_date=data_status["latest_date"],
                end_date=data_status["latest_date"], 
                records_added=0,
                sources_updated=[],
                features_regenerated=False,
                update_duration_minutes=0.0,
                next_update_due=self._calculate_next_update().isoformat(),
                data_quality_score=data_status.get("quality_score", 1.0)
            )
        
        # Perform update
        return self.perform_full_update()
    
    def perform_full_update(self) -> UpdateResult:
        """Perform comprehensive data update from all sources."""
        
        start_time = datetime.now()
        logger.info("Starting comprehensive data update")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            # Calculate date range for update
            start_date, end_date = self._calculate_update_range()
            
            task = progress.add_task("Updating data sources...", total=5)
            
            sources_updated = []
            total_records = 0
            
            # 1. Update OpenAQ data
            if self.openaq_etl:
                progress.update(task, description="Updating OpenAQ PM2.5 data...")
                openaq_result = self._update_openaq_data(start_date, end_date)
                if openaq_result["success"]:
                    sources_updated.append("openaq")
                    total_records += openaq_result["records_added"]
                
                progress.update(task, advance=1)
            
            # 2. Update ERA5 data
            if self.era5_etl:
                progress.update(task, description="Updating ERA5 meteorological data...")
                era5_result = self._update_era5_data(start_date, end_date)
                if era5_result["success"]:
                    sources_updated.append("era5")
                    total_records += era5_result["records_added"]
                
                progress.update(task, advance=1)
            
            # 3. Update IMERG data (optional)
            if self.imerg_etl:
                progress.update(task, description="Updating IMERG precipitation data...")
                imerg_result = self._update_imerg_data(start_date, end_date)
                if imerg_result["success"]:
                    sources_updated.append("imerg")
                    total_records += imerg_result["records_added"]
                
                progress.update(task, advance=1)
            else:
                progress.update(task, advance=1)
            
            # 4. Regenerate features
            features_regenerated = False
            if self.feature_builder and sources_updated:
                progress.update(task, description="Regenerating features...")
                feature_result = self._regenerate_features()
                features_regenerated = feature_result["success"]
                
                progress.update(task, advance=1)
            else:
                progress.update(task, advance=1)
            
            # 5. Assess final data quality
            progress.update(task, description="Assessing data quality...")
            final_status = self._assess_current_data()
            quality_score = final_status.get("quality_score", 0.0)
            
            progress.update(task, advance=1)
        
        # Calculate duration
        end_time = datetime.now()
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Calculate next update time
        next_update = self._calculate_next_update()
        
        logger.info(f"Data update complete: {total_records:,} records added from {len(sources_updated)} sources")
        
        return UpdateResult(
            success=len(sources_updated) > 0,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            records_added=total_records,
            sources_updated=sources_updated,
            features_regenerated=features_regenerated,
            update_duration_minutes=duration_minutes,
            next_update_due=next_update.isoformat(),
            data_quality_score=quality_score
        )
    
    def extend_historical_data(self, target_end_date: Optional[str] = None) -> UpdateResult:
        """Extend historical data collection to target date."""
        
        if target_end_date:
            self.config.end_date = target_end_date
        
        logger.info(f"Extending historical data to {self.config.end_date}")
        
        return self.perform_full_update()
    
    def _assess_current_data(self) -> Dict:
        """Assess current data status and coverage."""
        
        data_status = {
            "has_data": False,
            "latest_date": None,
            "earliest_date": None,
            "total_records": 0,
            "stations_count": 0,
            "quality_score": 0.0,
            "data_gaps": []
        }
        
        try:
            # Check for existing processed features
            features_path = Path("data/processed/enhanced_features.parquet")
            
            if features_path.exists():
                df = pd.read_parquet(features_path)
                
                if not df.empty:
                    data_status["has_data"] = True
                    data_status["latest_date"] = df['datetime_utc'].max().isoformat()
                    data_status["earliest_date"] = df['datetime_utc'].min().isoformat()
                    data_status["total_records"] = len(df)
                    data_status["stations_count"] = df['station_id'].nunique() if 'station_id' in df.columns else 1
                    
                    # Assess data quality
                    data_status["quality_score"] = self._calculate_data_quality(df)
                    
                    # Detect gaps
                    data_status["data_gaps"] = self._detect_data_gaps(df)
        
        except Exception as e:
            logger.warning(f"Failed to assess current data: {e}")
        
        return data_status
    
    def _needs_update(self, data_status: Dict) -> bool:
        """Determine if data update is needed."""
        
        # No data exists
        if not data_status["has_data"]:
            return True
        
        # Check if data is too old
        if data_status["latest_date"]:
            latest_date = pd.to_datetime(data_status["latest_date"])
            hours_since_update = (datetime.now() - latest_date).total_seconds() / 3600
            
            if hours_since_update > self.config.update_frequency_hours:
                return True
        
        # Check for significant data gaps
        if len(data_status["data_gaps"]) > 0:
            max_gap_hours = max(gap["hours"] for gap in data_status["data_gaps"])
            if max_gap_hours > (self.config.max_gap_days * 24):
                return True
        
        # Check if we need to extend to latest available
        if self.config.auto_extend_to_latest:
            target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            if data_status["latest_date"]:
                latest_date = pd.to_datetime(data_status["latest_date"]).replace(tzinfo=None)
                if latest_date < target_date - timedelta(days=2):  # Allow 2-day lag
                    return True
        
        return False
    
    def _calculate_update_range(self) -> Tuple[datetime, datetime]:
        """Calculate the date range for data update."""
        
        # Start from configured start date
        start_date = pd.to_datetime(self.config.start_date).replace(tzinfo=None)
        
        # End at configured end date or latest available
        if self.config.auto_extend_to_latest:
            end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=0)
            
            # Don't go beyond configured end date
            config_end = pd.to_datetime(self.config.end_date).replace(tzinfo=None)
            end_date = min(end_date, config_end)
        else:
            end_date = pd.to_datetime(self.config.end_date).replace(tzinfo=None)
        
        # Check existing data to avoid re-processing
        current_status = self._assess_current_data()
        if current_status["has_data"] and current_status["latest_date"]:
            existing_latest = pd.to_datetime(current_status["latest_date"]).replace(tzinfo=None)
            start_date = max(start_date, existing_latest + timedelta(hours=1))
        
        return start_date, end_date
    
    def _update_openaq_data(self, start_date: datetime, end_date: datetime) -> Dict:
        """Update OpenAQ data for the specified date range."""
        
        try:
            logger.info(f"Updating OpenAQ data from {start_date} to {end_date}")
            
            # Run OpenAQ ETL for date range
            result = self.openaq_etl.run_etl(
                start_date=start_date,
                end_date=end_date,
                reuse_existing=True
            )
            
            return {
                "success": True,
                "records_added": result.get("total_records", 0),
                "message": "OpenAQ data updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to update OpenAQ data: {e}")
            return {
                "success": False,
                "records_added": 0,
                "message": f"OpenAQ update failed: {e}"
            }
    
    def _update_era5_data(self, start_date: datetime, end_date: datetime) -> Dict:
        """Update ERA5 data for the specified date range."""
        
        try:
            logger.info(f"Updating ERA5 data from {start_date} to {end_date}")
            
            # Process in daily batches to avoid overwhelming CDS API
            current_date = start_date
            total_records = 0
            
            while current_date <= end_date:
                try:
                    daily_result = self.era5_etl.run_etl(
                        start_date=current_date,
                        end_date=current_date,
                        reuse_existing=True
                    )
                    
                    total_records += daily_result.get("total_records", 0)
                    current_date += timedelta(days=1)
                    
                    # Small delay to respect API limits
                    import time
                    time.sleep(1)
                    
                except Exception as daily_error:
                    logger.warning(f"Failed to update ERA5 for {current_date}: {daily_error}")
                    current_date += timedelta(days=1)
                    continue
            
            return {
                "success": True,
                "records_added": total_records,
                "message": "ERA5 data updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to update ERA5 data: {e}")
            return {
                "success": False,
                "records_added": 0,
                "message": f"ERA5 update failed: {e}"
            }
    
    def _update_imerg_data(self, start_date: datetime, end_date: datetime) -> Dict:
        """Update IMERG data for the specified date range."""
        
        try:
            logger.info(f"Updating IMERG data from {start_date} to {end_date}")
            
            result = self.imerg_etl.run_etl(
                start_date=start_date,
                end_date=end_date,
                reuse_existing=True
            )
            
            return {
                "success": True,
                "records_added": result.get("total_records", 0),
                "message": "IMERG data updated successfully"
            }
            
        except Exception as e:
            logger.warning(f"IMERG update failed (optional): {e}")
            return {
                "success": False,
                "records_added": 0,
                "message": f"IMERG update failed: {e}"
            }
    
    def _regenerate_features(self) -> Dict:
        """Regenerate features after data update."""
        
        try:
            logger.info("Regenerating features with updated data")
            
            # Run enhanced feature engineering pipeline
            result = self.feature_builder.build_features()
            
            return {
                "success": result.success,
                "features_count": result.total_features,
                "message": "Features regenerated successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to regenerate features: {e}")
            return {
                "success": False,
                "features_count": 0,
                "message": f"Feature regeneration failed: {e}"
            }
    
    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score."""
        
        # Basic quality metrics
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        # Temporal completeness
        if 'datetime_utc' in df.columns:
            time_range = df['datetime_utc'].max() - df['datetime_utc'].min()
            expected_records = time_range.total_seconds() / 3600  # Hourly data
            completeness = len(df) / max(expected_records, 1)
        else:
            completeness = 1.0
        
        # Target variable quality
        if 'pm25' in df.columns:
            target_missing = df['pm25'].isnull().sum() / len(df)
            target_quality = 1.0 - target_missing
        else:
            target_quality = 1.0
        
        # Combined quality score
        quality_score = (
            (1.0 - missing_rate) * 0.3 +
            min(completeness, 1.0) * 0.4 +
            target_quality * 0.3
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def _detect_data_gaps(self, df: pd.DataFrame) -> List[Dict]:
        """Detect significant gaps in time series data."""
        
        gaps = []
        
        if 'datetime_utc' not in df.columns or len(df) < 2:
            return gaps
        
        # Sort by time
        df_sorted = df.sort_values('datetime_utc')
        
        # Calculate time differences
        time_diffs = df_sorted['datetime_utc'].diff()
        
        # Find gaps longer than 2 hours
        gap_threshold = pd.Timedelta(hours=2)
        gap_mask = time_diffs > gap_threshold
        
        if gap_mask.any():
            gap_indices = df_sorted[gap_mask].index
            
            for idx in gap_indices:
                prev_time = df_sorted.loc[idx-1, 'datetime_utc']
                curr_time = df_sorted.loc[idx, 'datetime_utc']
                gap_hours = (curr_time - prev_time).total_seconds() / 3600
                
                gaps.append({
                    "start": prev_time.isoformat(),
                    "end": curr_time.isoformat(),
                    "hours": gap_hours
                })
        
        return gaps
    
    def _calculate_next_update(self) -> datetime:
        """Calculate when the next update should occur."""
        
        return datetime.now() + timedelta(hours=self.config.update_frequency_hours)
    
    def get_update_status(self) -> Dict:
        """Get current update status and recommendations."""
        
        current_data = self._assess_current_data()
        needs_update = self._needs_update(current_data)
        next_update = self._calculate_next_update()
        
        return {
            "current_data": current_data,
            "needs_update": needs_update,
            "next_scheduled_update": next_update.isoformat(),
            "update_frequency_hours": self.config.update_frequency_hours,
            "auto_extend_enabled": self.config.auto_extend_to_latest
        }


