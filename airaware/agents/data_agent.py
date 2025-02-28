"""
Data Quality Monitoring Agent for AirAware

This agent monitors data quality, detects anomalies, and ensures data integrity.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .base_agent import BaseAgent, AgentConfig, AgentStatus


@dataclass
class DataAgentConfig(AgentConfig):
    """Configuration for Data Quality Monitoring Agent"""
    # Data quality settings
    data_quality_file: str = "data/data_quality.json"
    anomaly_detection_enabled: bool = True
    missing_data_threshold: float = 0.1  # 10% missing data threshold
    
    # Anomaly detection parameters
    contamination: float = 0.1  # Expected proportion of outliers
    anomaly_threshold: float = 0.5  # Threshold for anomaly score
    window_size: int = 24  # Hours for rolling window analysis
    
    # Data validation rules
    pm25_min: float = 0.0
    pm25_max: float = 1000.0
    temperature_min: float = -50.0
    temperature_max: float = 60.0
    humidity_min: float = 0.0
    humidity_max: float = 100.0
    
    # Quality metrics
    completeness_threshold: float = 0.95  # 95% completeness required
    consistency_threshold: float = 0.9   # 90% consistency required
    accuracy_threshold: float = 0.85     # 85% accuracy required
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=lambda: {
        "update_interval": 1800,  # 30 minutes
        "quality_cache_duration": 3600,  # 1 hour
        "max_anomalies_per_hour": 10,
        "data_retention_days": 30
    })


@dataclass
class DataQualityMetric:
    """Data quality metric structure"""
    metric_name: str
    value: float
    threshold: float
    status: str  # "good", "warning", "critical"
    timestamp: datetime
    station_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "status": self.status,
            "timestamp": self.timestamp.isoformat(),
            "station_id": self.station_id,
            "details": self.details
        }


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    timestamp: datetime
    station_id: str
    variable: str
    value: float
    anomaly_score: float
    is_anomaly: bool
    severity: str  # "low", "medium", "high", "critical"
    description: str
    suggested_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "station_id": self.station_id,
            "variable": self.variable,
            "value": self.value,
            "anomaly_score": self.anomaly_score,
            "is_anomaly": self.is_anomaly,
            "severity": self.severity,
            "description": self.description,
            "suggested_action": self.suggested_action
        }


@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    report_id: str
    timestamp: datetime
    station_id: str
    overall_quality_score: float
    quality_metrics: List[DataQualityMetric]
    anomalies: List[AnomalyDetection]
    recommendations: List[str]
    status: str  # "good", "warning", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp.isoformat(),
            "station_id": self.station_id,
            "overall_quality_score": self.overall_quality_score,
            "quality_metrics": [m.to_dict() for m in self.quality_metrics],
            "anomalies": [a.to_dict() for a in self.anomalies],
            "recommendations": self.recommendations,
            "status": self.status
        }


class DataQualityAgent(BaseAgent):
    """Intelligent data quality monitoring agent"""
    
    def __init__(self, config: DataAgentConfig):
        super().__init__(config)
        self.quality_metrics: List[DataQualityMetric] = []
        self.anomaly_detector = IsolationForest(contamination=config.contamination, random_state=42)
        self.scaler = StandardScaler()
        self.quality_reports: List[DataQualityReport] = []
        self.last_quality_update: Optional[datetime] = None
        
        # Load historical quality data
        self._load_quality_data()
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data quality monitoring logic"""
        try:
            # Extract context data
            data = context.get("data")
            station_id = context.get("station_id")
            data_source = context.get("data_source", "unknown")
            
            if data is None:
                return {"status": "error", "message": "No data provided for quality assessment"}
            
            # Convert data to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # Perform data quality assessment
            quality_report = await self._assess_data_quality(df, station_id, data_source)
            
            # Detect anomalies if enabled
            anomalies = []
            if self.config.anomaly_detection_enabled:
                anomalies = await self._detect_anomalies(df, station_id)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(quality_report, anomalies)
            
            # Update quality report
            quality_report.anomalies = anomalies
            quality_report.recommendations = recommendations
            
            # Store quality report
            self.quality_reports.append(quality_report)
            self._save_quality_data()
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "quality_report": quality_report.to_dict(),
                "station_id": station_id,
                "data_source": data_source
            }
            
        except Exception as e:
            self.logger.error(f"Data quality monitoring execution failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Perform health check for the agent"""
        try:
            # Check if quality data is available
            if not self.quality_metrics:
                self.logger.warning("No quality metrics available")
                return False
            
            # Check if anomaly detector is trained
            if not hasattr(self.anomaly_detector, 'decision_function'):
                self.logger.warning("Anomaly detector not trained")
            
            # Update health check timestamp
            self.last_health_check = datetime.now()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    async def _assess_data_quality(
        self, 
        df: pd.DataFrame, 
        station_id: str, 
        data_source: str
    ) -> DataQualityReport:
        """Assess data quality for the given dataset"""
        quality_metrics = []
        
        # 1. Completeness assessment
        completeness_metric = self._assess_completeness(df, station_id)
        quality_metrics.append(completeness_metric)
        
        # 2. Consistency assessment
        consistency_metric = self._assess_consistency(df, station_id)
        quality_metrics.append(consistency_metric)
        
        # 3. Accuracy assessment
        accuracy_metric = self._assess_accuracy(df, station_id)
        quality_metrics.append(accuracy_metric)
        
        # 4. Validity assessment
        validity_metric = self._assess_validity(df, station_id)
        quality_metrics.append(validity_metric)
        
        # 5. Timeliness assessment
        timeliness_metric = self._assess_timeliness(df, station_id)
        quality_metrics.append(timeliness_metric)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality_score(quality_metrics)
        
        # Determine status
        status = self._determine_quality_status(overall_score, quality_metrics)
        
        # Create quality report
        report = DataQualityReport(
            report_id=f"quality_{station_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            station_id=station_id,
            overall_quality_score=overall_score,
            quality_metrics=quality_metrics,
            anomalies=[],  # Will be filled later
            recommendations=[],  # Will be filled later
            status=status
        )
        
        return report
    
    def _assess_completeness(self, df: pd.DataFrame, station_id: str) -> DataQualityMetric:
        """Assess data completeness"""
        total_records = len(df)
        missing_records = df.isnull().sum().sum()
        completeness_ratio = 1.0 - (missing_records / (total_records * len(df.columns)))
        
        # Determine status
        if completeness_ratio >= self.config.completeness_threshold:
            status = "good"
        elif completeness_ratio >= 0.8:
            status = "warning"
        else:
            status = "critical"
        
        return DataQualityMetric(
            metric_name="completeness",
            value=completeness_ratio,
            threshold=self.config.completeness_threshold,
            status=status,
            timestamp=datetime.now(),
            station_id=station_id,
            details={
                "total_records": total_records,
                "missing_records": missing_records,
                "missing_columns": df.isnull().sum().to_dict()
            }
        )
    
    def _assess_consistency(self, df: pd.DataFrame, station_id: str) -> DataQualityMetric:
        """Assess data consistency"""
        consistency_issues = 0
        total_checks = 0
        
        # Check for duplicate records
        duplicates = df.duplicated().sum()
        consistency_issues += duplicates
        total_checks += len(df)
        
        # Check for inconsistent data types
        for column in df.columns:
            if df[column].dtype == 'object':
                # Check for mixed data types
                try:
                    pd.to_numeric(df[column], errors='raise')
                except:
                    consistency_issues += df[column].isnull().sum()
                    total_checks += len(df[column])
        
        # Check for logical inconsistencies
        if 'pm25' in df.columns and 'temperature' in df.columns:
            # PM2.5 and temperature should be positively correlated in general
            correlation = df['pm25'].corr(df['temperature'])
            if correlation < -0.8:  # Strong negative correlation might indicate issues
                consistency_issues += 1
            total_checks += 1
        
        consistency_ratio = 1.0 - (consistency_issues / max(total_checks, 1))
        
        # Determine status
        if consistency_ratio >= self.config.consistency_threshold:
            status = "good"
        elif consistency_ratio >= 0.7:
            status = "warning"
        else:
            status = "critical"
        
        return DataQualityMetric(
            metric_name="consistency",
            value=consistency_ratio,
            threshold=self.config.consistency_threshold,
            status=status,
            timestamp=datetime.now(),
            station_id=station_id,
            details={
                "duplicates": duplicates,
                "consistency_issues": consistency_issues,
                "total_checks": total_checks
            }
        )
    
    def _assess_accuracy(self, df: pd.DataFrame, station_id: str) -> DataQualityMetric:
        """Assess data accuracy"""
        accuracy_issues = 0
        total_checks = 0
        
        # Check for values outside expected ranges
        if 'pm25' in df.columns:
            pm25_issues = ((df['pm25'] < self.config.pm25_min) | 
                          (df['pm25'] > self.config.pm25_max)).sum()
            accuracy_issues += pm25_issues
            total_checks += len(df['pm25'])
        
        if 'temperature' in df.columns:
            temp_issues = ((df['temperature'] < self.config.temperature_min) | 
                          (df['temperature'] > self.config.temperature_max)).sum()
            accuracy_issues += temp_issues
            total_checks += len(df['temperature'])
        
        if 'humidity' in df.columns:
            humidity_issues = ((df['humidity'] < self.config.humidity_min) | 
                              (df['humidity'] > self.config.humidity_max)).sum()
            accuracy_issues += humidity_issues
            total_checks += len(df['humidity'])
        
        # Check for statistical outliers (using IQR method)
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            accuracy_issues += outliers
            total_checks += len(df[column])
        
        accuracy_ratio = 1.0 - (accuracy_issues / max(total_checks, 1))
        
        # Determine status
        if accuracy_ratio >= self.config.accuracy_threshold:
            status = "good"
        elif accuracy_ratio >= 0.7:
            status = "warning"
        else:
            status = "critical"
        
        return DataQualityMetric(
            metric_name="accuracy",
            value=accuracy_ratio,
            threshold=self.config.accuracy_threshold,
            status=status,
            timestamp=datetime.now(),
            station_id=station_id,
            details={
                "accuracy_issues": accuracy_issues,
                "total_checks": total_checks,
                "outlier_columns": [col for col in df.select_dtypes(include=[np.number]).columns]
            }
        )
    
    def _assess_validity(self, df: pd.DataFrame, station_id: str) -> DataQualityMetric:
        """Assess data validity"""
        validity_issues = 0
        total_checks = 0
        
        # Check for required columns
        required_columns = ['pm25', 'timestamp']
        for col in required_columns:
            if col not in df.columns:
                validity_issues += 1
            total_checks += 1
        
        # Check for valid timestamps
        if 'timestamp' in df.columns:
            try:
                pd.to_datetime(df['timestamp'])
            except:
                validity_issues += df['timestamp'].isnull().sum()
                total_checks += len(df['timestamp'])
        
        # Check for negative values where they shouldn't be
        for col in ['pm25', 'humidity']:
            if col in df.columns:
                negative_values = (df[col] < 0).sum()
                validity_issues += negative_values
                total_checks += len(df[col])
        
        validity_ratio = 1.0 - (validity_issues / max(total_checks, 1))
        
        # Determine status
        if validity_ratio >= 0.95:
            status = "good"
        elif validity_ratio >= 0.8:
            status = "warning"
        else:
            status = "critical"
        
        return DataQualityMetric(
            metric_name="validity",
            value=validity_ratio,
            threshold=0.95,
            status=status,
            timestamp=datetime.now(),
            station_id=station_id,
            details={
                "validity_issues": validity_issues,
                "total_checks": total_checks,
                "missing_required_columns": [col for col in required_columns if col not in df.columns]
            }
        )
    
    def _assess_timeliness(self, df: pd.DataFrame, station_id: str) -> DataQualityMetric:
        """Assess data timeliness"""
        if 'timestamp' not in df.columns:
            return DataQualityMetric(
                metric_name="timeliness",
                value=0.0,
                threshold=0.9,
                status="critical",
                timestamp=datetime.now(),
                station_id=station_id,
                details={"error": "No timestamp column found"}
            )
        
        # Convert timestamps
        try:
            timestamps = pd.to_datetime(df['timestamp'])
        except:
            return DataQualityMetric(
                metric_name="timeliness",
                value=0.0,
                threshold=0.9,
                status="critical",
                timestamp=datetime.now(),
                station_id=station_id,
                details={"error": "Invalid timestamp format"}
            )
        
        # Check for data freshness
        now = datetime.now()
        time_diffs = [(now - ts).total_seconds() / 3600 for ts in timestamps]  # Hours
        
        # Data should be within last 24 hours
        fresh_data = sum(1 for diff in time_diffs if diff <= 24)
        timeliness_ratio = fresh_data / len(time_diffs)
        
        # Determine status
        if timeliness_ratio >= 0.9:
            status = "good"
        elif timeliness_ratio >= 0.7:
            status = "warning"
        else:
            status = "critical"
        
        return DataQualityMetric(
            metric_name="timeliness",
            value=timeliness_ratio,
            threshold=0.9,
            status=status,
            timestamp=datetime.now(),
            station_id=station_id,
            details={
                "fresh_data_count": fresh_data,
                "total_data_count": len(time_diffs),
                "oldest_data_hours": max(time_diffs) if time_diffs else 0,
                "newest_data_hours": min(time_diffs) if time_diffs else 0
            }
        )
    
    def _calculate_overall_quality_score(self, metrics: List[DataQualityMetric]) -> float:
        """Calculate overall data quality score"""
        if not metrics:
            return 0.0
        
        # Weighted average of all metrics
        weights = {
            "completeness": 0.25,
            "consistency": 0.20,
            "accuracy": 0.25,
            "validity": 0.15,
            "timeliness": 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric in metrics:
            weight = weights.get(metric.metric_name, 0.1)
            weighted_sum += metric.value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_status(self, overall_score: float, metrics: List[DataQualityMetric]) -> str:
        """Determine overall quality status"""
        # Check for critical issues
        critical_metrics = [m for m in metrics if m.status == "critical"]
        if critical_metrics:
            return "critical"
        
        # Check for warning issues
        warning_metrics = [m for m in metrics if m.status == "warning"]
        if warning_metrics and overall_score < 0.8:
            return "warning"
        
        # Good quality
        if overall_score >= 0.9:
            return "good"
        elif overall_score >= 0.7:
            return "warning"
        else:
            return "critical"
    
    async def _detect_anomalies(self, df: pd.DataFrame, station_id: str) -> List[AnomalyDetection]:
        """Detect anomalies in the data"""
        anomalies = []
        
        # Select numeric columns for anomaly detection
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            return anomalies
        
        # Prepare data for anomaly detection
        numeric_data = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        if len(numeric_data) < 10:  # Need minimum data for anomaly detection
            return anomalies
        
        try:
            # Fit anomaly detector
            scaled_data = self.scaler.fit_transform(numeric_data)
            anomaly_scores = self.anomaly_detector.fit_predict(scaled_data)
            decision_scores = self.anomaly_detector.decision_function(scaled_data)
            
            # Identify anomalies
            for i, (score, decision) in enumerate(zip(anomaly_scores, decision_scores)):
                if score == -1:  # Anomaly detected
                    # Find the most anomalous variable
                    max_anomaly_idx = np.argmax(np.abs(scaled_data[i]))
                    variable = numeric_columns[max_anomaly_idx]
                    value = numeric_data.iloc[i, max_anomaly_idx]
                    
                    # Determine severity
                    if decision < -0.5:
                        severity = "critical"
                    elif decision < -0.3:
                        severity = "high"
                    elif decision < -0.1:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    # Create anomaly detection
                    anomaly = AnomalyDetection(
                        timestamp=datetime.now(),
                        station_id=station_id,
                        variable=variable,
                        value=value,
                        anomaly_score=abs(decision),
                        is_anomaly=True,
                        severity=severity,
                        description=f"Anomalous {variable} value: {value:.2f}",
                        suggested_action=self._get_anomaly_action(variable, severity)
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def _get_anomaly_action(self, variable: str, severity: str) -> str:
        """Get suggested action for anomaly"""
        if severity == "critical":
            return f"Immediate investigation required for {variable} anomaly"
        elif severity == "high":
            return f"Review {variable} data and sensor calibration"
        elif severity == "medium":
            return f"Monitor {variable} trends for potential issues"
        else:
            return f"Note {variable} anomaly for future reference"
    
    def _generate_recommendations(
        self, 
        quality_report: DataQualityReport, 
        anomalies: List[AnomalyDetection]
    ) -> List[str]:
        """Generate recommendations based on quality assessment"""
        recommendations = []
        
        # Quality-based recommendations
        for metric in quality_report.quality_metrics:
            if metric.status == "critical":
                if metric.metric_name == "completeness":
                    recommendations.append("Critical: High missing data rate. Check data collection systems.")
                elif metric.metric_name == "consistency":
                    recommendations.append("Critical: Data consistency issues detected. Review data processing pipeline.")
                elif metric.metric_name == "accuracy":
                    recommendations.append("Critical: Data accuracy problems. Verify sensor calibration.")
                elif metric.metric_name == "validity":
                    recommendations.append("Critical: Invalid data detected. Check data validation rules.")
                elif metric.metric_name == "timeliness":
                    recommendations.append("Critical: Data is not timely. Check data transmission systems.")
            
            elif metric.status == "warning":
                if metric.metric_name == "completeness":
                    recommendations.append("Warning: Some missing data detected. Monitor data collection.")
                elif metric.metric_name == "consistency":
                    recommendations.append("Warning: Minor consistency issues. Review data processing.")
                elif metric.metric_name == "accuracy":
                    recommendations.append("Warning: Some accuracy issues. Consider sensor maintenance.")
                elif metric.metric_name == "validity":
                    recommendations.append("Warning: Some invalid data. Review validation rules.")
                elif metric.metric_name == "timeliness":
                    recommendations.append("Warning: Data freshness concerns. Check transmission delays.")
        
        # Anomaly-based recommendations
        critical_anomalies = [a for a in anomalies if a.severity == "critical"]
        high_anomalies = [a for a in anomalies if a.severity == "high"]
        
        if critical_anomalies:
            recommendations.append(f"Critical: {len(critical_anomalies)} critical anomalies detected. Immediate attention required.")
        
        if high_anomalies:
            recommendations.append(f"High: {len(high_anomalies)} high-severity anomalies detected. Review sensor performance.")
        
        # Overall recommendations
        if quality_report.overall_quality_score < 0.7:
            recommendations.append("Overall data quality is poor. Comprehensive system review recommended.")
        elif quality_report.overall_quality_score < 0.9:
            recommendations.append("Data quality is acceptable but could be improved.")
        else:
            recommendations.append("Data quality is excellent. Continue current monitoring practices.")
        
        return recommendations
    
    def _load_quality_data(self):
        """Load historical quality data from file"""
        try:
            quality_path = Path(self.config.data_quality_file)
            if quality_path.exists():
                with open(quality_path, 'r') as f:
                    data = json.load(f)
                
                # Load quality metrics
                self.quality_metrics = []
                for item in data.get("quality_metrics", []):
                    metric = DataQualityMetric(
                        metric_name=item["metric_name"],
                        value=item["value"],
                        threshold=item["threshold"],
                        status=item["status"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        station_id=item.get("station_id"),
                        details=item.get("details", {})
                    )
                    self.quality_metrics.append(metric)
                
                # Load quality reports
                self.quality_reports = []
                for item in data.get("quality_reports", []):
                    report = DataQualityReport(
                        report_id=item["report_id"],
                        timestamp=datetime.fromisoformat(item["timestamp"]),
                        station_id=item["station_id"],
                        overall_quality_score=item["overall_quality_score"],
                        quality_metrics=[],  # Will be loaded separately
                        anomalies=[],  # Will be loaded separately
                        recommendations=item["recommendations"],
                        status=item["status"]
                    )
                    self.quality_reports.append(report)
                
                self.logger.info(f"Loaded {len(self.quality_metrics)} quality metrics and {len(self.quality_reports)} reports")
            else:
                self.quality_metrics = []
                self.quality_reports = []
                self.logger.info("No quality data file found, starting with empty data")
        except Exception as e:
            self.logger.error(f"Failed to load quality data: {e}")
            self.quality_metrics = []
            self.quality_reports = []
    
    def _save_quality_data(self):
        """Save quality data to file"""
        try:
            quality_path = Path(self.config.data_quality_file)
            quality_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "quality_metrics": [m.to_dict() for m in self.quality_metrics],
                "quality_reports": [r.to_dict() for r in self.quality_reports]
            }
            
            with open(quality_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save quality data: {e}")
    
    def get_quality_summary(self, station_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get quality summary for a station"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent metrics
        recent_metrics = [
            m for m in self.quality_metrics 
            if m.timestamp >= cutoff_time and m.station_id == station_id
        ]
        
        # Filter recent reports
        recent_reports = [
            r for r in self.quality_reports 
            if r.timestamp >= cutoff_time and r.station_id == station_id
        ]
        
        if not recent_metrics:
            return {"status": "no_data", "message": "No recent quality data available"}
        
        # Calculate summary statistics
        overall_scores = [r.overall_quality_score for r in recent_reports]
        avg_quality_score = np.mean(overall_scores) if overall_scores else 0.0
        
        # Count statuses
        status_counts = {}
        for metric in recent_metrics:
            status_counts[metric.status] = status_counts.get(metric.status, 0) + 1
        
        return {
            "status": "success",
            "station_id": station_id,
            "time_period_hours": hours,
            "average_quality_score": avg_quality_score,
            "status_counts": status_counts,
            "total_metrics": len(recent_metrics),
            "total_reports": len(recent_reports),
            "last_update": max([m.timestamp for m in recent_metrics]).isoformat() if recent_metrics else None
        }
