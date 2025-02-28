"""Data lineage tracking for ETL pipeline."""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class DataLineageEvent:
    """Represents a single data lineage event."""
    event_id: str
    timestamp: datetime
    event_type: str  # "extract", "transform", "load", "validate"
    source: str
    target: str
    operation: str
    metadata: Dict[str, Any]
    parent_event_id: Optional[str] = None
    station_id: Optional[int] = None
    record_count: Optional[int] = None
    data_quality_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class DataLineageTracker:
    """Tracks data lineage through the ETL pipeline."""
    
    def __init__(self, lineage_file: str = "data/artifacts/data_lineage.jsonl"):
        self.lineage_file = Path(lineage_file)
        self.lineage_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_events: List[DataLineageEvent] = []
    
    def track_extraction(self, 
                        source: str, 
                        station_id: int, 
                        record_count: int,
                        metadata: Optional[Dict] = None) -> str:
        """Track data extraction event."""
        event = DataLineageEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type="extract",
            source=source,
            target=f"raw/openaq/station_{station_id}",
            operation="api_fetch",
            metadata=metadata or {},
            station_id=station_id,
            record_count=record_count
        )
        
        self._save_event(event)
        return event.event_id
    
    def track_transformation(self, 
                           source: str, 
                           target: str, 
                           operation: str,
                           record_count: int,
                           parent_event_id: Optional[str] = None,
                           metadata: Optional[Dict] = None) -> str:
        """Track data transformation event."""
        event = DataLineageEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type="transform",
            source=source,
            target=target,
            operation=operation,
            metadata=metadata or {},
            parent_event_id=parent_event_id,
            record_count=record_count
        )
        
        self._save_event(event)
        return event.event_id
    
    def track_validation(self, 
                        source: str, 
                        quality_score: float,
                        station_id: int,
                        parent_event_id: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> str:
        """Track data validation event."""
        event = DataLineageEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type="validate",
            source=source,
            target="quality_metrics",
            operation="quality_assessment",
            metadata=metadata or {},
            parent_event_id=parent_event_id,
            station_id=station_id,
            data_quality_score=quality_score
        )
        
        self._save_event(event)
        return event.event_id
    
    def track_load(self, 
                  source: str, 
                  target: str, 
                  record_count: int,
                  parent_event_id: Optional[str] = None,
                  metadata: Optional[Dict] = None) -> str:
        """Track data loading event."""
        event = DataLineageEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type="load",
            source=source,
            target=target,
            operation="parquet_write",
            metadata=metadata or {},
            parent_event_id=parent_event_id,
            record_count=record_count
        )
        
        self._save_event(event)
        return event.event_id
    
    def _save_event(self, event: DataLineageEvent):
        """Save event to lineage file."""
        with open(self.lineage_file, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')
    
    def get_lineage_for_station(self, station_id: int, days_back: int = 7) -> List[DataLineageEvent]:
        """Get lineage events for a specific station."""
        if not self.lineage_file.exists():
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        events = []
        
        with open(self.lineage_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    event_time = datetime.fromisoformat(data['timestamp'])
                    
                    if (event_time >= cutoff_date and 
                        data.get('station_id') == station_id):
                        events.append(self._dict_to_event(data))
                        
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        
        return sorted(events, key=lambda x: x.timestamp)
    
    def get_lineage_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get summary of lineage events."""
        if not self.lineage_file.exists():
            return {"total_events": 0}
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        events_by_type = {}
        events_by_station = {}
        total_records = 0
        total_events = 0
        
        with open(self.lineage_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    event_time = datetime.fromisoformat(data['timestamp'])
                    
                    if event_time >= cutoff_date:
                        total_events += 1
                        event_type = data.get('event_type', 'unknown')
                        events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
                        
                        station_id = data.get('station_id')
                        if station_id:
                            events_by_station[station_id] = events_by_station.get(station_id, 0) + 1
                        
                        record_count = data.get('record_count', 0)
                        if record_count:
                            total_records += record_count
                            
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        
        return {
            "period_days": days_back,
            "total_events": total_events,
            "total_records": total_records,
            "events_by_type": events_by_type,
            "events_by_station": events_by_station,
            "avg_records_per_event": total_records / total_events if total_events > 0 else 0
        }
    
    def _dict_to_event(self, data: Dict[str, Any]) -> DataLineageEvent:
        """Convert dictionary back to DataLineageEvent."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return DataLineageEvent(**data)
    
    def generate_lineage_report(self, station_id: Optional[int] = None, days_back: int = 7) -> str:
        """Generate human-readable lineage report."""
        if station_id:
            events = self.get_lineage_for_station(station_id, days_back)
            title = f"Data Lineage Report - Station {station_id}"
        else:
            summary = self.get_lineage_summary(days_back)
            title = "Data Lineage Summary Report"
        
        report = f"{title}\n{'=' * len(title)}\n"
        
        if station_id and events:
            report += f"Period: Last {days_back} days\n"
            report += f"Total Events: {len(events)}\n\n"
            
            for event in events[-10:]:  # Show last 10 events
                report += f"[{event.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                report += f"{event.event_type.upper()}: {event.operation}\n"
                report += f"  {event.source} → {event.target}\n"
                if event.record_count:
                    report += f"  Records: {event.record_count}\n"
                if event.data_quality_score:
                    report += f"  Quality: {event.data_quality_score:.3f}\n"
                report += "\n"
        
        elif not station_id:
            summary = self.get_lineage_summary(days_back)
            report += f"Period: Last {days_back} days\n"
            report += f"Total Events: {summary['total_events']}\n"
            report += f"Total Records: {summary['total_records']:,}\n\n"
            
            report += "Events by Type:\n"
            for event_type, count in summary['events_by_type'].items():
                report += f"  {event_type}: {count}\n"
            
            report += "\nEvents by Station:\n"
            for station_id, count in summary['events_by_station'].items():
                report += f"  Station {station_id}: {count}\n"
        
        return report
