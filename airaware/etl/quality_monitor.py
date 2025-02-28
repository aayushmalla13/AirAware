"""Advanced data quality monitoring and alerting for OpenAQ ETL pipeline."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from .data_validator import DataQualityMetrics

logger = logging.getLogger(__name__)


class QualityAlert(BaseModel):
    """Represents a data quality alert."""
    alert_id: str
    timestamp: datetime
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "missing_data", "poor_quality", "outliers", "freshness"
    station_id: Optional[int] = None
    message: str
    details: Dict = Field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


class QualityThresholds(BaseModel):
    """Quality monitoring thresholds."""
    min_quality_score: float = 0.7
    max_missing_data_hours: int = 6
    max_outlier_rate: float = 0.05  # 5%
    min_daily_records: int = 20
    staleness_threshold_hours: int = 2
    consecutive_missing_limit: int = 3


class QualityMonitor:
    """Advanced data quality monitoring with alerting."""
    
    def __init__(self, alerts_file: str = "data/artifacts/quality_alerts.json"):
        self.alerts_file = Path(alerts_file)
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)
        self.thresholds = QualityThresholds()
        
    def assess_data_quality(self, 
                           station_id: int, 
                           df: pd.DataFrame,
                           previous_metrics: Optional[DataQualityMetrics] = None) -> List[QualityAlert]:
        """Assess data quality and generate alerts."""
        alerts = []
        
        if df.empty:
            alerts.append(QualityAlert(
                alert_id=f"no_data_{station_id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                timestamp=datetime.now(),
                severity="high",
                category="missing_data",
                station_id=station_id,
                message=f"No data available for station {station_id}",
                details={"record_count": 0}
            ))
            return alerts
        
        # Check data freshness
        if 'datetime_utc' in df.columns and not df['datetime_utc'].empty:
            latest_measurement = df['datetime_utc'].max()
            hours_since_latest = (datetime.now() - latest_measurement.replace(tzinfo=None)).total_seconds() / 3600
            
            if hours_since_latest > self.thresholds.staleness_threshold_hours:
                alerts.append(QualityAlert(
                    alert_id=f"stale_data_{station_id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    timestamp=datetime.now(),
                    severity="medium" if hours_since_latest < 24 else "high",
                    category="freshness",
                    station_id=station_id,
                    message=f"Stale data detected: {hours_since_latest:.1f} hours since last measurement",
                    details={"hours_since_latest": hours_since_latest, "latest_measurement": latest_measurement.isoformat()}
                ))
        
        # Check for excessive missing values
        if 'value' in df.columns:
            missing_count = df['value'].isnull().sum()
            missing_rate = missing_count / len(df) if len(df) > 0 else 1.0
            
            if missing_rate > 0.2:  # 20% missing values
                alerts.append(QualityAlert(
                    alert_id=f"missing_values_{station_id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    timestamp=datetime.now(),
                    severity="medium" if missing_rate < 0.5 else "high",
                    category="missing_data",
                    station_id=station_id,
                    message=f"High missing value rate: {missing_rate:.1%}",
                    details={"missing_count": int(missing_count), "total_count": len(df), "missing_rate": missing_rate}
                ))
        
        # Check for excessive outliers
        if 'value' in df.columns and not df['value'].empty:
            pm25_values = df['value'].dropna()
            outliers = ((pm25_values < 0) | (pm25_values > 500)).sum()  # Extreme outliers
            outlier_rate = outliers / len(pm25_values) if len(pm25_values) > 0 else 0
            
            if outlier_rate > self.thresholds.max_outlier_rate:
                alerts.append(QualityAlert(
                    alert_id=f"outliers_{station_id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    timestamp=datetime.now(),
                    severity="medium",
                    category="outliers",
                    station_id=station_id,
                    message=f"High outlier rate detected: {outlier_rate:.1%}",
                    details={"outlier_count": int(outliers), "outlier_rate": outlier_rate}
                ))
        
        # Check for quality degradation compared to previous metrics
        if previous_metrics:
            from .data_validator import OpenAQDataValidator
            validator = OpenAQDataValidator()
            current_metrics = validator.validate_data_quality(df, [station_id])
            
            quality_drop = previous_metrics.quality_score - current_metrics.quality_score
            if quality_drop > 0.2:  # 20% quality drop
                alerts.append(QualityAlert(
                    alert_id=f"quality_drop_{station_id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                    timestamp=datetime.now(),
                    severity="medium",
                    category="poor_quality",
                    station_id=station_id,
                    message=f"Quality score degraded by {quality_drop:.2f}",
                    details={
                        "previous_score": previous_metrics.quality_score,
                        "current_score": current_metrics.quality_score,
                        "quality_drop": quality_drop
                    }
                ))
        
        # Check daily record count
        daily_records = len(df)
        if daily_records < self.thresholds.min_daily_records:
            alerts.append(QualityAlert(
                alert_id=f"low_volume_{station_id}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                timestamp=datetime.now(),
                severity="low",
                category="missing_data",
                station_id=station_id,
                message=f"Low daily record count: {daily_records}",
                details={"daily_records": daily_records, "expected_minimum": self.thresholds.min_daily_records}
            ))
        
        return alerts
    
    def save_alerts(self, alerts: List[QualityAlert]) -> None:
        """Save alerts to persistent storage."""
        if not alerts:
            return
            
        # Load existing alerts
        existing_alerts = self.load_alerts()
        
        # Add new alerts
        all_alerts = existing_alerts + alerts
        
        # Keep only last 1000 alerts to prevent file bloat
        if len(all_alerts) > 1000:
            all_alerts = all_alerts[-1000:]
        
        # Save to file
        alerts_data = {
            "alerts": [alert.model_dump() for alert in all_alerts],
            "last_updated": datetime.now().isoformat(),
            "total_alerts": len(all_alerts)
        }
        
        with open(self.alerts_file, 'w') as f:
            json.dump(alerts_data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(alerts)} new quality alerts")
    
    def load_alerts(self, days_back: int = 7) -> List[QualityAlert]:
        """Load recent alerts from storage."""
        if not self.alerts_file.exists():
            return []
        
        try:
            with open(self.alerts_file) as f:
                data = json.load(f)
            
            # Filter to recent alerts
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_alerts = []
            
            for alert_data in data.get("alerts", []):
                try:
                    alert = QualityAlert(**alert_data)
                    if alert.timestamp >= cutoff_date:
                        recent_alerts.append(alert)
                except Exception as e:
                    logger.warning(f"Failed to parse alert: {e}")
            
            return recent_alerts
            
        except Exception as e:
            logger.error(f"Failed to load alerts: {e}")
            return []
    
    def get_active_alerts(self, station_id: Optional[int] = None, severity: Optional[str] = None) -> List[QualityAlert]:
        """Get currently active (unresolved) alerts."""
        alerts = self.load_alerts()
        
        # Filter by criteria
        filtered = [a for a in alerts if not a.resolved]
        
        if station_id is not None:
            filtered = [a for a in filtered if a.station_id == station_id]
        
        if severity:
            filtered = [a for a in filtered if a.severity == severity]
        
        return sorted(filtered, key=lambda x: x.timestamp, reverse=True)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        alerts = self.load_alerts(days_back=30)  # Look back further for resolution
        
        for alert in alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_timestamp = datetime.now()
                self.save_alerts([])  # Trigger save of updated alerts
                logger.info(f"Resolved alert {alert_id}")
                return True
        
        return False
    
    def get_quality_summary(self, days_back: int = 7) -> Dict:
        """Get quality monitoring summary."""
        alerts = self.load_alerts(days_back)
        
        # Count by severity
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        category_counts = {}
        station_counts = {}
        
        for alert in alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            category_counts[alert.category] = category_counts.get(alert.category, 0) + 1
            if alert.station_id:
                station_counts[alert.station_id] = station_counts.get(alert.station_id, 0) + 1
        
        active_alerts = [a for a in alerts if not a.resolved]
        
        return {
            "period_days": days_back,
            "total_alerts": len(alerts),
            "active_alerts": len(active_alerts),
            "resolved_alerts": len(alerts) - len(active_alerts),
            "by_severity": severity_counts,
            "by_category": category_counts,
            "by_station": station_counts,
            "most_recent": alerts[0].timestamp.isoformat() if alerts else None
        }
    
    def generate_quality_report(self, days_back: int = 7) -> str:
        """Generate human-readable quality monitoring report."""
        summary = self.get_quality_summary(days_back)
        active_alerts = self.get_active_alerts()
        
        report = f"""
Data Quality Monitoring Report
==============================
Period: Last {days_back} days

Summary:
- Total Alerts: {summary['total_alerts']}
- Active Alerts: {summary['active_alerts']}
- Resolved Alerts: {summary['resolved_alerts']}

By Severity:
- Critical: {summary['by_severity'].get('critical', 0)}
- High: {summary['by_severity'].get('high', 0)}
- Medium: {summary['by_severity'].get('medium', 0)}
- Low: {summary['by_severity'].get('low', 0)}

By Category:
"""
        for category, count in summary['by_category'].items():
            report += f"- {category.replace('_', ' ').title()}: {count}\n"
        
        if active_alerts:
            report += f"\nActive Alerts ({len(active_alerts)}):\n"
            for alert in active_alerts[:10]:  # Show top 10
                report += f"- [{alert.severity.upper()}] {alert.message} (Station {alert.station_id})\n"
        
        return report


