"""Comprehensive feature engineering pipeline for AirAware PM₂.₅ forecasting."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .data_joiner import DataJoiner, JoinConfig, JoinResult
from .temporal_features import TemporalFeatureGenerator, TemporalConfig
from .met_features import MeteorologicalFeatureGenerator, MeteorologicalConfig

logger = logging.getLogger(__name__)


class FeatureConfig(BaseModel):
    """Comprehensive configuration for feature engineering."""
    # Data joining configuration
    join_config: JoinConfig = Field(default_factory=JoinConfig)
    
    # Temporal features configuration
    temporal_config: TemporalConfig = Field(default_factory=TemporalConfig)
    
    # Meteorological features configuration  
    meteorological_config: MeteorologicalConfig = Field(default_factory=MeteorologicalConfig)
    
    # Feature selection and filtering
    min_feature_variance: float = Field(0.001, description="Minimum variance for feature selection")
    max_correlation_threshold: float = Field(0.95, description="Maximum correlation for feature selection")
    remove_constant_features: bool = Field(True, description="Remove constant features")
    
    # Output configuration
    save_feature_importance: bool = Field(True, description="Save feature importance analysis")
    save_intermediate_steps: bool = Field(False, description="Save intermediate processing steps")


class FeatureBuildResult(BaseModel):
    """Result of comprehensive feature building process."""
    success: bool
    total_features: int
    selected_features: int
    record_count: int
    date_range: tuple[str, str]
    stations_processed: List[int]
    feature_categories: Dict[str, int]
    data_quality_score: float = Field(ge=0, le=1)
    join_result: Optional[JoinResult] = None
    output_files: Dict[str, str] = Field(default_factory=dict)
    processing_time_minutes: float = 0.0


class FeatureBuilder:
    """Comprehensive feature engineering pipeline for PM₂.₅ forecasting."""
    
    def __init__(self, 
                 interim_data_dir: str = "data/interim",
                 processed_data_dir: str = "data/processed",
                 config: Optional[FeatureConfig] = None):
        
        self.interim_data_dir = Path(interim_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.config = config or FeatureConfig()
        self.console = Console()
        
        # Create directories
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_joiner = DataJoiner(interim_data_dir, processed_data_dir)
        self.temporal_generator = TemporalFeatureGenerator(self.config.temporal_config)
        self.met_generator = MeteorologicalFeatureGenerator(self.config.meteorological_config)
        
        logger.info("FeatureBuilder initialized")
    
    def build_features(self) -> FeatureBuildResult:
        """Build comprehensive feature set for PM₂.₅ forecasting."""
        
        start_time = datetime.now()
        logger.info("Starting comprehensive feature building pipeline")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            # Step 1: Join all data sources
            task = progress.add_task("Joining data sources...", total=6)
            
            join_result = self.data_joiner.join_all_sources(self.config.join_config)
            
            if not join_result.success:
                logger.error("Data joining failed")
                return FeatureBuildResult(
                    success=False, total_features=0, selected_features=0,
                    record_count=0, date_range=("", ""), stations_processed=[],
                    feature_categories={}, join_result=join_result
                )
            
            progress.update(task, description="✅ Data sources joined", advance=1)
            
            # Load joined data
            joined_df = pd.read_parquet(join_result.output_path)
            
            if self.config.save_intermediate_steps:
                self._save_intermediate_step(joined_df, "01_joined_data.parquet")
            
            progress.update(task, description="Loading joined data...", advance=1)
            
            # Step 2: Generate temporal features
            progress.update(task, description="Generating temporal features...")
            
            df_with_temporal = self.temporal_generator.generate_features(
                joined_df, target_col='pm25', group_col='station_id'
            )
            
            if self.config.save_intermediate_steps:
                self._save_intermediate_step(df_with_temporal, "02_temporal_features.parquet")
            
            progress.update(task, description="✅ Temporal features generated", advance=1)
            
            # Step 3: Generate meteorological features
            progress.update(task, description="Generating meteorological features...")
            
            df_with_met = self.met_generator.generate_features(
                df_with_temporal, group_col='station_id'
            )
            
            if self.config.save_intermediate_steps:
                self._save_intermediate_step(df_with_met, "03_meteorological_features.parquet")
            
            progress.update(task, description="✅ Meteorological features generated", advance=1)
            
            # Step 4: Feature selection and cleaning
            progress.update(task, description="Performing feature selection...")
            
            df_selected, feature_selection_report = self._perform_feature_selection(df_with_met)
            
            progress.update(task, description="✅ Feature selection complete", advance=1)
            
            # Step 5: Final processing and validation
            progress.update(task, description="Finalizing features...")
            
            df_final = self._finalize_features(df_selected)
            
            # Save final feature set
            output_path = self.processed_data_dir / "features.parquet"
            df_final.to_parquet(output_path, index=False)
            
            progress.update(task, description="✅ Feature building complete", advance=1)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() / 60
        
        # Analyze feature categories
        feature_categories = self._analyze_feature_categories(df_final)
        
        # Calculate final quality score
        quality_score = self._calculate_feature_quality(df_final, join_result)
        
        # Get date range and stations
        date_range = (
            df_final['datetime_utc'].min().strftime('%Y-%m-%d %H:%M'),
            df_final['datetime_utc'].max().strftime('%Y-%m-%d %H:%M')
        )
        
        stations = df_final['station_id'].unique().tolist() if 'station_id' in df_final.columns else []
        
        # Prepare output files
        output_files = {
            "features": str(output_path),
            "feature_selection_report": str(self.processed_data_dir / "feature_selection_report.json")
        }
        
        # Save feature selection report
        self._save_feature_selection_report(feature_selection_report, df_final)
        
        # Generate feature importance analysis if enabled
        if self.config.save_feature_importance:
            importance_file = self._generate_feature_importance_analysis(df_final)
            output_files["feature_importance"] = importance_file
        
        logger.info(f"Feature building complete: {len(df_final):,} records, "
                   f"{len(df_final.columns)-3} features, {len(stations)} stations")
        
        return FeatureBuildResult(
            success=True,
            total_features=len(df_final.columns) - 3,  # Exclude datetime_utc, station_id, pm25
            selected_features=len([col for col in df_final.columns if col not in ['datetime_utc', 'station_id', 'pm25']]),
            record_count=len(df_final),
            date_range=date_range,
            stations_processed=stations,
            feature_categories=feature_categories,
            data_quality_score=quality_score,
            join_result=join_result,
            output_files=output_files,
            processing_time_minutes=processing_time
        )
    
    def _perform_feature_selection(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict]:
        """Perform comprehensive feature selection."""
        
        logger.info("Performing feature selection and cleaning")
        
        selection_report = {
            "initial_features": len(df.columns),
            "removed_features": {
                "constant": [],
                "low_variance": [],
                "high_correlation": [],
                "high_missing": []
            },
            "final_features": 0
        }
        
        # Get feature columns (exclude target and metadata)
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime_utc', 'station_id', 'pm25']]
        
        selection_report["initial_features"] = len(feature_cols)
        
        # 1. Remove constant features
        if self.config.remove_constant_features:
            constant_features = []
            for col in feature_cols:
                if df[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                logger.info(f"Removing {len(constant_features)} constant features")
                feature_cols = [col for col in feature_cols if col not in constant_features]
                selection_report["removed_features"]["constant"] = constant_features
        
        # 2. Remove low variance features
        low_variance_features = []
        for col in feature_cols:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                variance = df[col].var()
                if variance < self.config.min_feature_variance:
                    low_variance_features.append(col)
        
        if low_variance_features:
            logger.info(f"Removing {len(low_variance_features)} low variance features")
            feature_cols = [col for col in feature_cols if col not in low_variance_features]
            selection_report["removed_features"]["low_variance"] = low_variance_features
        
        # 3. Remove highly correlated features
        numeric_features = [col for col in feature_cols 
                           if df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        if len(numeric_features) > 1:
            corr_matrix = df[numeric_features].corr().abs()
            
            # Find highly correlated pairs
            high_corr_features = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.config.max_correlation_threshold:
                        # Remove the feature with lower variance
                        feature1, feature2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        var1, var2 = df[feature1].var(), df[feature2].var()
                        
                        feature_to_remove = feature1 if var1 < var2 else feature2
                        high_corr_features.add(feature_to_remove)
            
            if high_corr_features:
                logger.info(f"Removing {len(high_corr_features)} highly correlated features")
                feature_cols = [col for col in feature_cols if col not in high_corr_features]
                selection_report["removed_features"]["high_correlation"] = list(high_corr_features)
        
        # 4. Remove features with excessive missing values (>50%)
        high_missing_features = []
        for col in feature_cols:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > 0.5:
                high_missing_features.append(col)
        
        if high_missing_features:
            logger.info(f"Removing {len(high_missing_features)} features with >50% missing values")
            feature_cols = [col for col in feature_cols if col not in high_missing_features]
            selection_report["removed_features"]["high_missing"] = high_missing_features
        
        # Create final feature set
        final_cols = ['datetime_utc', 'station_id', 'pm25'] + feature_cols
        df_selected = df[final_cols].copy()
        
        selection_report["final_features"] = len(feature_cols)
        
        logger.info(f"Feature selection complete: {selection_report['initial_features']} → {selection_report['final_features']} features")
        
        return df_selected, selection_report
    
    def _finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final feature processing and validation."""
        
        logger.info("Finalizing feature processing")
        
        # Ensure proper data types
        df = df.copy()
        
        # Convert datetime
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        
        # Sort by station and time
        if 'station_id' in df.columns:
            df = df.sort_values(['station_id', 'datetime_utc']).reset_index(drop=True)
        else:
            df = df.sort_values('datetime_utc').reset_index(drop=True)
        
        # Fill remaining missing values in feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime_utc', 'station_id', 'pm25']]
        
        for col in feature_cols:
            if df[col].dtype in ['float64', 'float32']:
                # Fill with median for numeric features
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            elif df[col].dtype in ['object', 'category']:
                # Fill with mode for categorical features
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                df[col] = df[col].fillna(mode_val)
        
        # Convert categorical features to category dtype for efficiency
        categorical_cols = [col for col in feature_cols if df[col].dtype == 'object']
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        
        return df
    
    def _analyze_feature_categories(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze feature categories for reporting."""
        
        feature_categories = {
            "temporal": 0,
            "meteorological": 0,
            "lag": 0,
            "rolling": 0,
            "calendar": 0,
            "cyclical": 0,
            "other": 0
        }
        
        for col in df.columns:
            if col in ['datetime_utc', 'station_id', 'pm25']:
                continue
                
            if col.startswith('lag_'):
                feature_categories["lag"] += 1
            elif col.startswith('rolling_') or col.startswith('met_rolling_'):
                feature_categories["rolling"] += 1
            elif col.startswith('calendar_'):
                feature_categories["calendar"] += 1
            elif col.startswith('cyclical_') or col.startswith('seasonal_'):
                feature_categories["cyclical"] += 1
            elif any(col.startswith(prefix) for prefix in ['wind_', 'temp_', 'stability_', 'comfort_', 'pollution_', 'bl_', 'weather_']):
                feature_categories["meteorological"] += 1
            elif any(col.startswith(prefix) for prefix in ['time_since_', 'trend_']):
                feature_categories["temporal"] += 1
            else:
                feature_categories["other"] += 1
        
        return feature_categories
    
    def _calculate_feature_quality(self, df: pd.DataFrame, join_result: JoinResult) -> float:
        """Calculate overall feature quality score."""
        
        # Base score from join quality
        base_score = join_result.data_quality_score
        
        # Calculate missing values penalty
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime_utc', 'station_id', 'pm25']]
        
        if feature_cols:
            missing_rates = df[feature_cols].isnull().sum() / len(df)
            avg_missing_rate = missing_rates.mean()
            missing_penalty = avg_missing_rate * 0.5  # 50% penalty for missing values
        else:
            missing_penalty = 0.0
        
        # Calculate feature diversity bonus
        total_features = len(feature_cols)
        diversity_bonus = min(0.1, total_features / 1000)  # Up to 10% bonus for many features
        
        # Final quality score
        quality_score = max(0.0, min(1.0, base_score - missing_penalty + diversity_bonus))
        
        return quality_score
    
    def _save_intermediate_step(self, df: pd.DataFrame, filename: str):
        """Save intermediate processing step."""
        
        output_path = self.processed_data_dir / "intermediate" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        logger.debug(f"Saved intermediate step: {filename}")
    
    def _save_feature_selection_report(self, report: Dict, df: pd.DataFrame):
        """Save feature selection report."""
        
        import json
        
        # Add final feature list to report
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime_utc', 'station_id', 'pm25']]
        
        report["selected_features"] = feature_cols
        report["feature_count_by_category"] = self._analyze_feature_categories(df)
        
        output_path = self.processed_data_dir / "feature_selection_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Feature selection report saved: {output_path}")
    
    def _generate_feature_importance_analysis(self, df: pd.DataFrame) -> str:
        """Generate feature importance analysis."""
        
        logger.info("Generating feature importance analysis")
        
        # Get feature categories
        temporal_cats = self.temporal_generator.get_feature_importance_categories()
        met_cats = self.met_generator.get_feature_categories()
        
        # Combine all categories
        all_categories = {**temporal_cats, **met_cats}
        
        # Create importance analysis
        importance_analysis = {
            "feature_categories": all_categories,
            "category_counts": {},
            "feature_statistics": {}
        }
        
        # Count features in each category
        for category, features in all_categories.items():
            available_features = [f for f in features if f in df.columns]
            importance_analysis["category_counts"][category] = len(available_features)
        
        # Calculate basic statistics for numeric features
        numeric_cols = df.select_dtypes(include=['number']).columns
        feature_cols = [col for col in numeric_cols 
                       if col not in ['station_id', 'pm25']]
        
        for col in feature_cols:
            if col in df.columns:
                importance_analysis["feature_statistics"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "missing_rate": float(df[col].isnull().sum() / len(df))
                }
        
        # Save analysis
        import json
        
        output_path = self.processed_data_dir / "feature_importance_analysis.json"
        
        with open(output_path, 'w') as f:
            json.dump(importance_analysis, f, indent=2, default=str)
        
        logger.info(f"Feature importance analysis saved: {output_path}")
        
        return str(output_path)
    
    def get_build_summary(self, result: FeatureBuildResult) -> str:
        """Generate human-readable feature build summary."""
        
        if not result.success:
            return "❌ Feature building failed - check logs for details"
        
        report = f"""
🔧 Feature Engineering Summary
===============================

✅ Build Status: SUCCESS
📊 Total Features: {result.total_features}
📈 Selected Features: {result.selected_features}
📋 Records: {result.record_count:,}
🏭 Stations: {len(result.stations_processed)}
📅 Date Range: {result.date_range[0]} to {result.date_range[1]}
⭐ Quality Score: {result.data_quality_score:.1%}
⏱️ Processing Time: {result.processing_time_minutes:.1f} minutes

Feature Categories:
"""
        
        for category, count in result.feature_categories.items():
            report += f"- {category.title()}: {count} features\n"
        
        report += f"\nOutput Files:\n"
        for file_type, path in result.output_files.items():
            report += f"- {file_type.title()}: {path}\n"
        
        if result.join_result:
            report += f"\nData Join Summary:\n"
            report += f"- Success Rate: {result.join_result.join_statistics.get('join_success_rate', 0):.1%}\n"
            report += f"- Sources Used: {', '.join(result.join_result.join_statistics.get('data_sources_used', []))}\n"
        
        return report
