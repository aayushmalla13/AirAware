"""Multi-source data joining with temporal alignment for AirAware PM₂.₅ forecasting."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)


class JoinConfig(BaseModel):
    """Configuration for data joining operations."""
    target_frequency: str = Field("1H", description="Target temporal resolution")
    time_tolerance: str = Field("15min", description="Tolerance for time alignment")
    station_coords: Dict[int, Tuple[float, float]] = Field(default_factory=dict, description="Station coordinates")
    fill_method: str = Field("interpolate", description="Method for filling missing values")
    max_gap_hours: int = Field(6, description="Maximum gap to fill (hours)")


class JoinResult(BaseModel):
    """Result of data joining operation."""
    success: bool
    record_count: int
    date_range: Tuple[str, str]
    stations_joined: List[int]
    features_created: List[str]
    data_quality_score: float = Field(ge=0, le=1)
    join_statistics: Dict = Field(default_factory=dict)
    output_path: Optional[str] = None


class DataJoiner:
    """Multi-source data joiner with temporal alignment and quality control."""
    
    def __init__(self, 
                 interim_data_dir: str = "data/interim",
                 processed_data_dir: str = "data/processed"):
        
        self.interim_data_dir = Path(interim_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.console = Console()
        
        # Create directories
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DataJoiner initialized")
    
    def join_all_sources(self, config: Optional[JoinConfig] = None) -> JoinResult:
        """Join OpenAQ targets with ERA5 meteorological and optional IMERG data."""
        if config is None:
            config = JoinConfig()
        
        logger.info("Starting multi-source data joining")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            # Load data sources
            task = progress.add_task("Loading data sources...", total=4)
            
            # 1. Load OpenAQ targets (required)
            targets_df = self._load_openaq_targets()
            if targets_df.empty:
                logger.error("No OpenAQ target data available")
                return JoinResult(success=False, record_count=0, date_range=("", ""), 
                                stations_joined=[], features_created=[])
            
            progress.update(task, description="✅ OpenAQ targets loaded", advance=1)
            
            # 2. Load ERA5 meteorological data (required)
            era5_df = self._load_era5_data()
            if era5_df.empty:
                logger.error("No ERA5 meteorological data available")
                return JoinResult(success=False, record_count=0, date_range=("", ""), 
                                stations_joined=[], features_created=[])
            
            progress.update(task, description="✅ ERA5 meteorological data loaded", advance=1)
            
            # 3. Load IMERG precipitation data (optional)
            imerg_df = self._load_imerg_data()
            has_imerg = not imerg_df.empty
            
            progress.update(task, description=f"✅ IMERG data {'loaded' if has_imerg else 'not available'}", advance=1)
            
            # 4. Perform temporal alignment and joining
            progress.update(task, description="Performing temporal alignment...", advance=1)
            
            joined_df = self._align_and_join(targets_df, era5_df, imerg_df, config)
            
            if joined_df.empty:
                logger.error("Data joining failed - no aligned records")
                return JoinResult(success=False, record_count=0, date_range=("", ""), 
                                stations_joined=[], features_created=[])
        
        # Calculate join statistics
        join_stats = self._calculate_join_statistics(joined_df, targets_df, era5_df, imerg_df)
        
        # Save joined data
        output_path = self.processed_data_dir / "joined_data.parquet"
        joined_df.to_parquet(output_path, index=False)
        
        # Calculate data quality score
        quality_score = self._calculate_join_quality(joined_df, join_stats)
        
        # Get date range
        date_range = (
            joined_df['datetime_utc'].min().strftime('%Y-%m-%d %H:%M'),
            joined_df['datetime_utc'].max().strftime('%Y-%m-%d %H:%M')
        )
        
        # Get unique stations
        stations = joined_df['station_id'].unique().tolist() if 'station_id' in joined_df.columns else []
        
        # Get feature columns
        feature_cols = [col for col in joined_df.columns 
                       if col not in ['datetime_utc', 'station_id', 'pm25']]
        
        logger.info(f"Data joining complete: {len(joined_df):,} records, "
                   f"{len(stations)} stations, {len(feature_cols)} features")
        
        return JoinResult(
            success=True,
            record_count=len(joined_df),
            date_range=date_range,
            stations_joined=stations,
            features_created=feature_cols,
            data_quality_score=quality_score,
            join_statistics=join_stats,
            output_path=str(output_path)
        )
    
    def _load_openaq_targets(self) -> pd.DataFrame:
        """Load OpenAQ PM₂.₅ target data."""
        targets_path = self.interim_data_dir / "targets.parquet"
        
        if not targets_path.exists():
            logger.warning(f"OpenAQ targets file not found: {targets_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(targets_path)
            
            # Ensure required columns
            required_cols = ['datetime_utc', 'pm25', 'station_id']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing required columns in targets: {missing_cols}")
                return pd.DataFrame()
            
            # Convert datetime and sort
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            df = df.sort_values(['station_id', 'datetime_utc']).reset_index(drop=True)
            
            logger.info(f"Loaded {len(df):,} OpenAQ target records for {df['station_id'].nunique()} stations")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load OpenAQ targets: {e}")
            return pd.DataFrame()
    
    def _load_era5_data(self) -> pd.DataFrame:
        """Load ERA5 meteorological data."""
        era5_path = self.interim_data_dir / "era5_hourly.parquet"
        
        if not era5_path.exists():
            logger.warning(f"ERA5 data file not found: {era5_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(era5_path)
            
            # Ensure datetime column and sort
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            df = df.sort_values('datetime_utc').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df):,} ERA5 meteorological records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load ERA5 data: {e}")
            return pd.DataFrame()
    
    def _load_imerg_data(self) -> pd.DataFrame:
        """Load IMERG precipitation data (optional)."""
        imerg_path = self.interim_data_dir / "imerg_hourly.parquet"
        
        if not imerg_path.exists():
            logger.info("IMERG data file not found (optional)")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(imerg_path)
            
            # Ensure datetime column and sort
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
            df = df.sort_values('datetime_utc').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df):,} IMERG precipitation records")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load IMERG data: {e}")
            return pd.DataFrame()
    
    def _align_and_join(self, targets_df: pd.DataFrame, era5_df: pd.DataFrame, 
                       imerg_df: pd.DataFrame, config: JoinConfig) -> pd.DataFrame:
        """Perform temporal alignment and joining of all data sources."""
        
        if targets_df.empty or era5_df.empty:
            logger.error("Cannot join - missing required data sources")
            return pd.DataFrame()
        
        logger.info("Performing temporal alignment and joining")
        
        # Start with targets as the base
        joined_df = targets_df.copy()
        
        # Round timestamps to nearest hour for alignment
        joined_df = joined_df.copy()
        era5_df = era5_df.copy()
        
        joined_df['datetime_hour'] = joined_df['datetime_utc'].dt.round('h')
        era5_df['datetime_hour'] = era5_df['datetime_utc'].dt.round('h')
        
        if not imerg_df.empty:
            imerg_df = imerg_df.copy()
            imerg_df['datetime_hour'] = imerg_df['datetime_utc'].dt.round('h')
        
        # Join with ERA5 data (meteorological variables)
        era5_join_cols = ['datetime_hour'] + [col for col in era5_df.columns 
                                             if col not in ['datetime_utc', 'datetime_hour']]
        
        joined_df = joined_df.merge(
            era5_df[era5_join_cols],
            on='datetime_hour',
            how='left',
            suffixes=('', '_era5')
        )
        
        # Join with IMERG data (precipitation) if available
        if not imerg_df.empty:
            imerg_join_cols = ['datetime_hour'] + [col for col in imerg_df.columns 
                                                  if col not in ['datetime_utc', 'datetime_hour']]
            
            joined_df = joined_df.merge(
                imerg_df[imerg_join_cols],
                on='datetime_hour',
                how='left',
                suffixes=('', '_imerg')
            )
        
        # Clean up temporary columns
        joined_df = joined_df.drop(columns=['datetime_hour'])
        
        # Handle missing values based on configuration
        joined_df = self._handle_missing_values(joined_df, config)
        
        logger.info(f"Temporal alignment complete: {len(joined_df):,} aligned records")
        return joined_df
    
    def _handle_missing_values(self, df: pd.DataFrame, config: JoinConfig) -> pd.DataFrame:
        """Handle missing values in joined dataset."""
        
        if df.empty:
            return df
        
        logger.info("Handling missing values in joined dataset")
        
        # Get numeric columns (excluding datetime and categorical)
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove target variable from interpolation
        if 'pm25' in numeric_cols:
            numeric_cols.remove('pm25')
        
        # Apply filling method based on configuration
        if config.fill_method == "interpolate":
            # Linear interpolation with limits
            max_gap = pd.Timedelta(hours=config.max_gap_hours)
            
            for col in numeric_cols:
                # Calculate time gaps
                df_sorted = df.sort_values(['station_id', 'datetime_utc']) if 'station_id' in df.columns else df.sort_values('datetime_utc')
                
                # Interpolate within each station group if station_id exists
                if 'station_id' in df.columns:
                    for station in df['station_id'].unique():
                        mask = df['station_id'] == station
                        df.loc[mask, col] = df.loc[mask, col].interpolate(method='linear', limit=config.max_gap_hours)
                else:
                    df[col] = df[col].interpolate(method='linear', limit=config.max_gap_hours)
        
        elif config.fill_method == "forward_fill":
            # Forward fill with limits
            for col in numeric_cols:
                if 'station_id' in df.columns:
                    for station in df['station_id'].unique():
                        mask = df['station_id'] == station
                        df.loc[mask, col] = df.loc[mask, col].fillna(method='ffill', limit=config.max_gap_hours)
                else:
                    df[col] = df[col].fillna(method='ffill', limit=config.max_gap_hours)
        
        # Log missing value statistics
        missing_stats = df[numeric_cols].isnull().sum()
        total_missing = missing_stats.sum()
        
        if total_missing > 0:
            logger.warning(f"Remaining missing values after filling: {total_missing}")
            for col, missing_count in missing_stats.items():
                if missing_count > 0:
                    pct = (missing_count / len(df)) * 100
                    logger.warning(f"  {col}: {missing_count} ({pct:.1f}%)")
        
        return df
    
    def _calculate_join_statistics(self, joined_df: pd.DataFrame, targets_df: pd.DataFrame,
                                  era5_df: pd.DataFrame, imerg_df: pd.DataFrame) -> Dict:
        """Calculate comprehensive join statistics."""
        
        stats = {
            "input_records": {
                "targets": len(targets_df),
                "era5": len(era5_df),
                "imerg": len(imerg_df) if not imerg_df.empty else 0
            },
            "output_records": len(joined_df),
            "join_success_rate": 0.0,
            "temporal_coverage": {},
            "missing_values": {},
            "data_sources_used": []
        }
        
        if not joined_df.empty:
            # Calculate join success rate
            stats["join_success_rate"] = len(joined_df) / len(targets_df) if len(targets_df) > 0 else 0
            
            # Temporal coverage
            if 'datetime_utc' in joined_df.columns:
                stats["temporal_coverage"] = {
                    "start_date": joined_df['datetime_utc'].min().isoformat(),
                    "end_date": joined_df['datetime_utc'].max().isoformat(),
                    "duration_hours": (joined_df['datetime_utc'].max() - joined_df['datetime_utc'].min()).total_seconds() / 3600
                }
            
            # Missing values analysis
            numeric_cols = joined_df.select_dtypes(include=['number']).columns
            missing_analysis = {}
            
            for col in numeric_cols:
                missing_count = joined_df[col].isnull().sum()
                missing_pct = (missing_count / len(joined_df)) * 100
                missing_analysis[col] = {
                    "count": int(missing_count),
                    "percentage": float(missing_pct)
                }
            
            stats["missing_values"] = missing_analysis
            
            # Data sources used
            stats["data_sources_used"] = ["openaq", "era5"]
            if not imerg_df.empty and 'precipitation_mm_hourly' in joined_df.columns:
                stats["data_sources_used"].append("imerg")
        
        return stats
    
    def _calculate_join_quality(self, joined_df: pd.DataFrame, join_stats: Dict) -> float:
        """Calculate overall data join quality score."""
        
        if joined_df.empty:
            return 0.0
        
        # Base score from join success rate
        join_success = join_stats.get("join_success_rate", 0.0)
        
        # Penalty for missing values
        missing_penalty = 0.0
        missing_stats = join_stats.get("missing_values", {})
        
        for col, stats in missing_stats.items():
            if col != 'pm25':  # Don't penalize target variable missing values
                missing_pct = stats.get("percentage", 0.0)
                missing_penalty += missing_pct / 100.0
        
        # Average missing penalty across features
        num_features = len(missing_stats) - (1 if 'pm25' in missing_stats else 0)
        if num_features > 0:
            missing_penalty = missing_penalty / num_features
        
        # Calculate quality score (0-1 scale)
        quality_score = join_success * (1.0 - missing_penalty)
        quality_score = max(0.0, min(1.0, quality_score))
        
        return quality_score
    
    def get_join_summary(self, join_result: JoinResult) -> str:
        """Generate human-readable join summary report."""
        
        if not join_result.success:
            return "❌ Data joining failed - check logs for details"
        
        stats = join_result.join_statistics
        
        report = f"""
📊 Data Join Summary Report
============================

✅ Join Status: SUCCESS
📈 Records: {join_result.record_count:,}
🏭 Stations: {len(join_result.stations_joined)}
🔧 Features: {len(join_result.features_created)}
📅 Date Range: {join_result.date_range[0]} to {join_result.date_range[1]}
⭐ Quality Score: {join_result.data_quality_score:.1%}

Data Sources:
- OpenAQ Targets: {stats['input_records']['targets']:,} records
- ERA5 Meteorological: {stats['input_records']['era5']:,} records
- IMERG Precipitation: {stats['input_records']['imerg']:,} records

Join Statistics:
- Success Rate: {stats['join_success_rate']:.1%}
- Temporal Coverage: {stats['temporal_coverage'].get('duration_hours', 0):.1f} hours
- Sources Used: {', '.join(stats['data_sources_used'])}

Output: {join_result.output_path}
"""
        
        # Add missing value summary if significant
        missing_stats = stats.get("missing_values", {})
        high_missing = {col: data for col, data in missing_stats.items() 
                       if data.get("percentage", 0) > 5.0}
        
        if high_missing:
            report += "\n⚠️ Variables with >5% missing values:\n"
            for col, data in high_missing.items():
                report += f"   • {col}: {data['percentage']:.1f}%\n"
        
        return report
