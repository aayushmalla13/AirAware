"""OpenAQ ETL Pipeline with reuse-or-fetch logic and partitioned storage."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ..config import load_data_config
from ..feasibility.openaq_client import OpenAQClient
from .data_validator import OpenAQDataValidator
from .manifest_manager import ETLArtifact, ManifestManager
from .quality_monitor import QualityMonitor
from .lineage_tracker import DataLineageTracker

logger = logging.getLogger(__name__)


class OpenAQETL:
    """Production-grade OpenAQ ETL pipeline with reuse-or-fetch logic."""
    
    def __init__(self, 
                 raw_data_dir: str = "data/raw/openaq",
                 interim_data_dir: str = "data/interim",
                 use_local: bool = True,
                 cache_ttl_hours: int = 24,
                 max_workers: int = 3):
        
        self.raw_data_dir = Path(raw_data_dir)
        self.interim_data_dir = Path(interim_data_dir)
        self.use_local = use_local
        self.cache_ttl_hours = cache_ttl_hours
        self.max_workers = max_workers
        
        # Create directories
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.interim_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        openaq_api_key = os.getenv("OPENAQ_KEY")
        self.openaq_client = OpenAQClient(
            api_key=openaq_api_key,
            use_local=use_local and not openaq_api_key
        )
        self.data_validator = OpenAQDataValidator()
        self.manifest_manager = ManifestManager()
        self.quality_monitor = QualityMonitor()
        self.lineage_tracker = DataLineageTracker()
        self.console = Console()
        
        # Load station configuration from CP-2
        self.config = load_data_config()
        self.stations = [
            {
                "station_id": s.station_id, 
                "name": s.name, 
                "quality_score": s.quality_score,
                "pm25_sensor_ids": s.pm25_sensor_ids
            }
            for s in self.config.stations
        ]
        
        logger.info(f"OpenAQ ETL initialized for {len(self.stations)} stations")
    
    def generate_date_partitions(self, start_date: datetime, end_date: datetime) -> List[str]:
        """Generate YYYY-MM date partitions between start and end dates."""
        partitions = []
        current = start_date.replace(day=1)  # Start from beginning of month
        
        while current <= end_date:
            partitions.append(current.strftime("%Y-%m"))
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return partitions
    
    def get_partition_file_path(self, station_id: int, year_month: str) -> Path:
        """Get file path for a station/date partition."""
        year, month = year_month.split('-')
        return self.raw_data_dir / f"station_{station_id}" / f"year={year}" / f"month={month}" / f"data.parquet"
    
    def check_partition_exists(self, station_id: int, year_month: str) -> bool:
        """Check if partition exists and is valid."""
        file_path = self.get_partition_file_path(station_id, year_month)
        
        if not file_path.exists():
            return False
        
        # Check if file is valid and not too old
        try:
            file_age_hours = (datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)).total_seconds() / 3600
            
            if file_age_hours > self.cache_ttl_hours:
                logger.info(f"Partition {file_path} is stale ({file_age_hours:.1f}h old)")
                return False
            
            # Basic validation
            if not self.data_validator.validate_parquet_file(str(file_path)):
                logger.warning(f"Partition {file_path} failed validation")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking partition {file_path}: {e}")
            return False
    
    def fetch_station_data(self, station_id: int, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch data for a single station using OpenAQ API."""
        try:
            logger.info(f"Fetching OpenAQ data for station {station_id}: {start_date.date()} to {end_date.date()}")
            
            # Get sensor IDs for this station
            station_info = next((s for s in self.stations if s["station_id"] == station_id), None)
            if not station_info:
                logger.error(f"Station {station_id} not found in configuration")
                return pd.DataFrame()
            
            sensor_ids = station_info.get("pm25_sensor_ids", [])
            if not sensor_ids:
                logger.warning(f"No PM2.5 sensor IDs found for station {station_id}")
                return pd.DataFrame()
            
            # Fetch data for each sensor
            all_measurements = []
            for sensor_id in sensor_ids:
                logger.info(f"Fetching data for sensor {sensor_id} (station {station_id})")
                measurements = self.openaq_client.get_measurements(
                    sensor_id=sensor_id,
                    station_id=station_id,
                    parameter="pm25",
                    date_from=start_date,
                    date_to=end_date,
                    limit=1000  # OpenAQ v3 limit
                )
                all_measurements.extend(measurements)
            
            if not all_measurements:
                logger.warning(f"No data returned for station {station_id}")
                return pd.DataFrame()
            
            # Convert to DataFrame with standardized schema
            data = []
            for measurement in all_measurements:
                data.append({
                    'station_id': station_id,
                    'sensor_id': measurement.get('sensor_id', 0),
                    'parameter': 'pm25',
                    'value': measurement.get('value'),
                    'unit': measurement.get('unit', 'µg/m³'),
                    'datetime_utc': pd.to_datetime(measurement.get('datetime')),
                    'latitude': measurement.get('coordinates', {}).get('latitude') if measurement.get('coordinates') else None,
                    'longitude': measurement.get('coordinates', {}).get('longitude') if measurement.get('coordinates') else None,
                    'quality_flag': measurement.get('quality', 'unknown'),
                    'data_source': 'openaq_v3'
                })
            
            df = pd.DataFrame(data)
            
            # Data cleaning and validation
            if not df.empty:
                # Remove rows with missing critical values
                df = df.dropna(subset=['value', 'datetime_utc'])
                
                # Sort by timestamp
                df = df.sort_values('datetime_utc')
                
                # Remove duplicates
                df = df.drop_duplicates(subset=['station_id', 'datetime_utc', 'parameter'])
                
                # Quality monitoring and alerting
                alerts = self.quality_monitor.assess_data_quality(station_id, df)
                if alerts:
                    self.quality_monitor.save_alerts(alerts)
                    high_severity_alerts = [a for a in alerts if a.severity in ["high", "critical"]]
                    if high_severity_alerts:
                        logger.warning(f"High severity quality alerts for station {station_id}: {len(high_severity_alerts)}")
                
                # Track data lineage
                extract_event_id = self.lineage_tracker.track_extraction(
                    source="openaq_v3_api",
                    station_id=station_id,
                    record_count=len(df),
                    metadata={"date_range": f"{start_date.date()} to {end_date.date()}"}
                )
                
                logger.info(f"Fetched {len(df)} valid records for station {station_id}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for station {station_id}: {e}")
            return pd.DataFrame()
    
    def save_partition(self, df: pd.DataFrame, station_id: int, year_month: str) -> bool:
        """Save data partition to Parquet format."""
        if df.empty:
            logger.warning(f"No data to save for station {station_id} partition {year_month}")
            return False
        
        file_path = self.get_partition_file_path(station_id, year_month)
        
        try:
            # Create directory structure
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as Parquet with compression
            table = pa.Table.from_pandas(df)
            pq.write_table(table, file_path, compression='snappy')
            
            # Validate saved file
            if not self.data_validator.validate_parquet_file(str(file_path)):
                logger.error(f"Validation failed for saved partition: {file_path}")
                return False
            
            # Add to manifest
            file_size = file_path.stat().st_size
            artifact = ETLArtifact(
                file_path=str(file_path),
                file_type="parquet",
                etl_stage="raw",
                data_source="openaq",
                station_id=station_id,
                date_partition=year_month,
                created_at=datetime.now(),
                file_size_bytes=file_size,
                record_count=len(df),
                data_quality_score=None,  # Will be calculated later
                schema_version="1.0"
            )
            
            self.manifest_manager.add_etl_artifact(artifact)
            
            logger.info(f"Saved partition {file_path} ({file_size:,} bytes, {len(df)} records)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save partition {file_path}: {e}")
            return False
    
    def process_station_partitions(self, station_info: Dict, required_partitions: List[str],
                                 force_refresh: bool = False) -> Tuple[int, int, int]:
        """Process all partitions for a single station with reuse-or-fetch logic."""
        station_id = station_info["station_id"]
        station_name = station_info["name"]
        
        reused_count = 0
        downloaded_count = 0
        failed_count = 0
        
        self.console.print(f"\n[bold blue]Processing station {station_id}: {station_name}[/bold blue]")
        
        # Check which partitions already exist
        missing_partitions = []
        
        for partition in required_partitions:
            if force_refresh or not self.check_partition_exists(station_id, partition):
                missing_partitions.append(partition)
            else:
                reused_count += 1
                self.console.print(f"  ✅ [green]REUSED[/green] {partition}")
        
        if not missing_partitions:
            self.console.print(f"  🎯 All partitions exist and are valid")
            return reused_count, downloaded_count, failed_count
        
        self.console.print(f"  📥 Need to download {len(missing_partitions)} partitions")
        
        # Download missing partitions
        for partition in missing_partitions:
            try:
                year, month = partition.split('-')
                start_date = datetime(int(year), int(month), 1)
                
                # End date is last day of month
                if int(month) == 12:
                    end_date = datetime(int(year) + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(int(year), int(month) + 1, 1) - timedelta(days=1)
                
                # Fetch data
                df = self.fetch_station_data(station_id, start_date, end_date)
                
                if not df.empty:
                    if self.save_partition(df, station_id, partition):
                        downloaded_count += 1
                        self.console.print(f"  ✅ [cyan]DOWNLOADED[/cyan] {partition} ({len(df)} records)")
                    else:
                        failed_count += 1
                        self.console.print(f"  ❌ [red]SAVE FAILED[/red] {partition}")
                else:
                    failed_count += 1
                    self.console.print(f"  ⚠️ [yellow]NO DATA[/yellow] {partition}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"Failed to process partition {partition} for station {station_id}: {e}")
                self.console.print(f"  ❌ [red]ERROR[/red] {partition}: {str(e)[:50]}")
        
        return reused_count, downloaded_count, failed_count
    
    def run_etl_pipeline(self, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        force_refresh: bool = False,
                        station_ids: Optional[List[int]] = None) -> Dict:
        """Run the complete OpenAQ ETL pipeline."""
        
        # Set default date range (last 6 months if not specified)
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=180)
        
        # Generate required partitions
        required_partitions = self.generate_date_partitions(start_date, end_date)
        
        # Filter stations if specified
        if station_ids:
            stations_to_process = [s for s in self.stations if s["station_id"] in station_ids]
        else:
            stations_to_process = self.stations
        
        self.console.print(f"\n[bold green]🚀 Starting OpenAQ ETL Pipeline[/bold green]")
        self.console.print(f"Date range: {start_date.date()} to {end_date.date()}")
        self.console.print(f"Partitions: {len(required_partitions)} ({required_partitions[0]} to {required_partitions[-1]})")
        self.console.print(f"Stations: {len(stations_to_process)}")
        self.console.print(f"Force refresh: {force_refresh}")
        self.console.print(f"Use local: {self.use_local}")
        
        # Process stations
        total_reused = 0
        total_downloaded = 0
        total_failed = 0
        
        if self.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_station = {
                    executor.submit(self.process_station_partitions, station, required_partitions, force_refresh): station
                    for station in stations_to_process
                }
                
                for future in as_completed(future_to_station):
                    station = future_to_station[future]
                    try:
                        reused, downloaded, failed = future.result()
                        total_reused += reused
                        total_downloaded += downloaded
                        total_failed += failed
                    except Exception as e:
                        logger.error(f"Station {station['station_id']} processing failed: {e}")
                        total_failed += len(required_partitions)
        else:
            # Sequential processing
            for station in stations_to_process:
                reused, downloaded, failed = self.process_station_partitions(station, required_partitions, force_refresh)
                total_reused += reused
                total_downloaded += downloaded
                total_failed += failed
        
        # Generate combined targets file
        combined_file_path = self.create_combined_targets_file(stations_to_process, required_partitions)
        
        # ETL Summary
        etl_results = {
            "execution_timestamp": datetime.now().isoformat(),
            "date_range": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "stations_processed": len(stations_to_process),
            "partitions_requested": len(required_partitions),
            "results": {
                "reused_partitions": total_reused,
                "downloaded_partitions": total_downloaded,
                "failed_partitions": total_failed,
                "success_rate": (total_reused + total_downloaded) / (total_reused + total_downloaded + total_failed) if (total_reused + total_downloaded + total_failed) > 0 else 0
            },
            "combined_targets_file": str(combined_file_path) if combined_file_path else None,
            "manifest_updated": True
        }
        
        self.console.print(f"\n[bold green]✅ ETL Pipeline Complete![/bold green]")
        self.console.print(f"📊 Results:")
        self.console.print(f"  • Reused: {total_reused} partitions")
        self.console.print(f"  • Downloaded: {total_downloaded} partitions") 
        self.console.print(f"  • Failed: {total_failed} partitions")
        self.console.print(f"  • Success rate: {etl_results['results']['success_rate']:.1%}")
        
        if combined_file_path:
            self.console.print(f"  • Combined targets: {combined_file_path}")
        
        return etl_results
    
    def create_combined_targets_file(self, stations: List[Dict], partitions: List[str]) -> Optional[Path]:
        """Create combined targets.parquet file from all station partitions."""
        try:
            self.console.print(f"\n[blue]📋 Creating combined targets file...[/blue]")
            
            all_data = []
            
            for station in stations:
                station_id = station["station_id"]
                
                for partition in partitions:
                    file_path = self.get_partition_file_path(station_id, partition)
                    
                    if file_path.exists():
                        try:
                            df = pd.read_parquet(file_path)
                            if not df.empty:
                                all_data.append(df)
                        except Exception as e:
                            logger.warning(f"Failed to read partition {file_path}: {e}")
            
            if not all_data:
                logger.warning("No data found for combined targets file")
                return None
            
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Sort by station and timestamp
            combined_df = combined_df.sort_values(['station_id', 'datetime_utc'])
            
            # Remove duplicates
            combined_df = combined_df.drop_duplicates(subset=['station_id', 'datetime_utc', 'parameter'])
            
            # Validate combined data
            quality_metrics = self.data_validator.validate_data_quality(
                combined_df, 
                [s["station_id"] for s in stations]
            )
            
            # Save combined file
            combined_file_path = self.interim_data_dir / "targets.parquet"
            table = pa.Table.from_pandas(combined_df)
            pq.write_table(table, combined_file_path, compression='snappy')
            
            # Add to manifest
            file_size = combined_file_path.stat().st_size
            artifact = ETLArtifact(
                file_path=str(combined_file_path),
                file_type="parquet",
                etl_stage="interim",
                data_source="openaq",
                station_id=None,  # Combined file
                date_partition=None,
                created_at=datetime.now(),
                file_size_bytes=file_size,
                record_count=len(combined_df),
                data_quality_score=quality_metrics.quality_score,
                schema_version="1.0"
            )
            
            self.manifest_manager.add_etl_artifact(artifact)
            
            self.console.print(f"✅ Combined targets file created: {combined_file_path}")
            self.console.print(f"   Records: {len(combined_df):,}")
            self.console.print(f"   Size: {file_size / 1024 / 1024:.1f} MB")
            self.console.print(f"   Quality score: {quality_metrics.quality_score:.2f}")
            
            return combined_file_path
            
        except Exception as e:
            logger.error(f"Failed to create combined targets file: {e}")
            return None
    
    def get_etl_status(self) -> Dict:
        """Get current ETL status and statistics."""
        manifest_stats = self.manifest_manager.get_etl_stats()
        
        status = {
            "stations_configured": len(self.stations),
            "etl_artifacts": manifest_stats.get("total_etl_artifacts", 0),
            "total_size_mb": manifest_stats.get("total_etl_size_mb", 0.0),
            "total_records": manifest_stats.get("total_records", 0),
            "by_source": manifest_stats.get("by_source", {}),
            "by_stage": manifest_stats.get("by_stage", {}),
            "last_updated": datetime.now().isoformat()
        }
        
        return status
