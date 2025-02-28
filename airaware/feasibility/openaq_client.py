"""OpenAQ v3 API client for PM₂.₅ station discovery and data coverage analysis.

Updated for OpenAQ v3 API structure with correct endpoint patterns.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from ..utils.api_helpers import APIClient, create_client

logger = logging.getLogger(__name__)

# Kathmandu Valley coordinates
KATHMANDU_LAT = 27.7172
KATHMANDU_LON = 85.3240


class StationInfo(BaseModel):
    """Information about an OpenAQ PM₂.₅ monitoring station."""
    station_id: int
    station_name: str
    latitude: float
    longitude: float
    distance_km: float
    pm25_sensor_ids: List[int] = Field(default_factory=list)
    first_measurement: Optional[datetime] = None
    last_measurement: Optional[datetime] = None
    total_measurements: int = 0
    data_span_days: int = 0
    missingness_pct: float = 0.0
    data_quality_score: float = 0.0


class CoverageAnalysis(BaseModel):
    """Analysis of data coverage for a station."""
    station_id: int
    sensor_id: int
    date_range: Tuple[datetime, datetime]
    expected_measurements: int
    actual_measurements: int
    missing_measurements: int
    missingness_pct: float
    coverage_score: float


class LocalDataSummary(BaseModel):
    """Summary of local OpenAQ data from manifest."""
    stations_found: List[int] = Field(default_factory=list)
    date_ranges: Dict[int, Tuple[datetime, datetime]] = Field(default_factory=dict)
    total_measurements: Dict[int, int] = Field(default_factory=dict)
    data_quality: Dict[int, float] = Field(default_factory=dict)


class OpenAQClient:
    """OpenAQ v3 API client with local data integration."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_local: bool = False,
        data_manifest_path: Optional[Path] = None,
        max_workers: int = 5,
    ) -> None:
        """Initialize OpenAQ client.
        
        Args:
            api_key: OpenAQ v3 API key
            use_local: Whether to use only local data
            data_manifest_path: Path to local data manifest
            max_workers: Maximum concurrent workers for parallel requests
        """
        self.use_local = use_local
        self.max_workers = max_workers
        self.data_manifest_path = data_manifest_path or Path("data/artifacts/local_data_manifest.json")
        
        # Initialize API client only if not using local-only mode
        if not use_local and api_key:
            # Enable caching for API responses (24-hour TTL)
            from ..utils.api_helpers import CacheConfig
            cache_config = CacheConfig(
                enabled=True,
                ttl_hours=24,
                cache_dir=Path("data/artifacts/.openaq_cache")
            )
            self.client = create_client("openaq", api_key=api_key, cache_config=cache_config)
        else:
            self.client = None
            
        if use_local:
            logger.info("OpenAQ client initialized in local-only mode")
        elif not api_key:
            logger.warning("No OpenAQ API key provided - local mode only")
    
    def _load_local_manifest(self) -> Dict:
        """Load local data manifest."""
        try:
            with open(self.data_manifest_path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Local manifest not found at {self.data_manifest_path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load local manifest: {e}")
            return {}
    
    def _analyze_local_data(self) -> LocalDataSummary:
        """Analyze local OpenAQ data from manifest with enhanced detection."""
        manifest = self._load_local_manifest()
        summary = LocalDataSummary()
        
        if not manifest.get("files"):
            return summary
        
        # Find OpenAQ-related files with enhanced pattern matching
        openaq_files = [
            f for f in manifest["files"]
            if ("openaq" in f.get("dataset_types", []) or 
                "openaq" in f.get("path", "").lower() or
                "nepal" in f.get("path", "").lower() or
                "sensor" in f.get("path", "").lower() or
                "measurement" in f.get("path", "").lower())
        ]
        
        logger.info(f"Found {len(openaq_files)} OpenAQ-related files in local data")
        
        # Enhanced station detection from various file patterns
        station_data = {}
        
        for file_info in openaq_files:
            path = file_info.get("path", "")
            file_size_mb = file_info.get("size_mb", 0)
            validation = file_info.get("schema_validation", {})
            
            # Multiple patterns for station detection
            station_id = None
            
            # Pattern 1: "station_123" in path
            if "station_" in path:
                try:
                    station_part = [p for p in path.split("/") if "station_" in p][0]
                    station_id = int(station_part.replace("station_", ""))
                except (ValueError, IndexError):
                    pass
            
            # Pattern 2: Nepal data files (assume single station per file)
            elif "nepal" in path.lower():
                # For Nepal data, use file naming pattern or default station IDs
                if "sensors" in path:
                    station_id = 1001  # Default Nepal sensor station
                elif "locations" in path:
                    station_id = 1002  # Default Nepal location station  
                elif "measurements" in path:
                    station_id = 1003  # Default Nepal measurement station
            
            if station_id:
                if station_id not in station_data:
                    station_data[station_id] = {
                        "measurements": 0,
                        "files": [],
                        "total_size_mb": 0
                    }
                
                station_data[station_id]["files"].append(path)
                station_data[station_id]["total_size_mb"] += file_size_mb
                
                # Extract measurement counts
                if validation.get("validated"):
                    row_count = validation.get("num_rows", 0)
                    station_data[station_id]["measurements"] += row_count
                elif "measurement" in path.lower():
                    # Estimate measurements from file size (rough heuristic)
                    estimated_rows = int(file_size_mb * 1000)  # ~1000 rows per MB
                    station_data[station_id]["measurements"] += estimated_rows
        
        # Populate summary
        for station_id, data in station_data.items():
            summary.stations_found.append(station_id)
            summary.total_measurements[station_id] = data["measurements"]
            
            # Calculate basic quality score based on file size and count
            file_count = len(data["files"])
            size_score = min(1.0, data["total_size_mb"] / 10.0)  # Normalize by 10MB
            summary.data_quality[station_id] = (size_score + min(1.0, file_count / 3.0)) / 2.0
        
        logger.info(f"Enhanced analysis: {len(summary.stations_found)} stations, {sum(summary.total_measurements.values()):,} total measurements")
        return summary
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points in kilometers."""
        from math import radians, sin, cos, sqrt, atan2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        # Earth's radius in kilometers
        r = 6371
        return r * c
    
    def get_locations_pm25(
        self,
        lat: float = KATHMANDU_LAT,
        lon: float = KATHMANDU_LON,
        radius_km: float = 25.0,
        limit: int = 1000,
    ) -> List[StationInfo]:
        """Get PM₂.₅ monitoring stations within radius of coordinates.
        
        Args:
            lat: Latitude of center point
            lon: Longitude of center point 
            radius_km: Search radius in kilometers
            limit: Maximum number of locations to fetch
            
        Returns:
            List of PM₂.₅ stations within radius
        """
        if self.use_local:
            return self._get_local_stations(lat, lon, radius_km)
        
        if not self.client:
            raise ValueError("No API client available and not in local mode")
        
        logger.info(f"Searching for PM₂.₅ stations within {radius_km}km of ({lat:.4f}, {lon:.4f})")
        
        # Get locations with PM2.5 parameter
        params = {
            "limit": limit,
            "coordinates": f"{lat},{lon}",
            "radius": int(radius_km * 1000),  # Convert to meters
            "parameters_id": 2,  # PM2.5 parameter ID
        }
        
        response = self.client.get("locations", params=params)
        
        if not response.success:
            raise Exception(f"Failed to fetch locations: {response.error}")
        
        results = response.data.get("results", [])
        stations = []
        
        for location in results:
            # Extract PM2.5 sensors
            pm25_sensors = [
                sensor for sensor in location.get("sensors", [])
                if sensor.get("parameter", {}).get("id") == 2
            ]
            
            if not pm25_sensors:
                continue
            
            # Calculate distance
            loc_coords = location.get("coordinates", {})
            if not loc_coords:
                continue
                
            distance = self._calculate_distance(
                lat, lon,
                loc_coords.get("latitude", 0),
                loc_coords.get("longitude", 0)
            )
            
            if distance > radius_km:
                continue
            
            station = StationInfo(
                station_id=location.get("id"),
                station_name=location.get("name", "Unknown"),
                latitude=loc_coords.get("latitude"),
                longitude=loc_coords.get("longitude"),
                distance_km=round(distance, 2),
                pm25_sensor_ids=[s.get("id") for s in pm25_sensors],
            )
            
            stations.append(station)
        
        # Sort by distance
        stations.sort(key=lambda s: s.distance_km)
        
        logger.info(f"Found {len(stations)} PM₂.₅ stations within {radius_km}km")
        return stations
    
    def _get_local_stations(
        self,
        lat: float,
        lon: float, 
        radius_km: float,
    ) -> List[StationInfo]:
        """Get stations from local data analysis."""
        summary = self._analyze_local_data()
        stations = []
        
        # Create mock station info from local data
        # In practice, you'd need coordinate info in the manifest or file metadata
        for i, station_id in enumerate(summary.stations_found):
            # Generate mock coordinates within radius for demonstration
            # In real implementation, extract from actual data files
            mock_lat = lat + (i * 0.01 - 0.05)  # Spread around center
            mock_lon = lon + (i * 0.01 - 0.05)
            
            distance = self._calculate_distance(lat, lon, mock_lat, mock_lon)
            
            if distance <= radius_km:
                station = StationInfo(
                    station_id=station_id,
                    station_name=f"Local Station {station_id}",
                    latitude=mock_lat,
                    longitude=mock_lon,
                    distance_km=round(distance, 2),
                    pm25_sensor_ids=[station_id * 100],  # Mock sensor ID
                    total_measurements=summary.total_measurements.get(station_id, 0),
                )
                stations.append(station)
        
        stations.sort(key=lambda s: s.distance_km)
        logger.info(f"Found {len(stations)} stations in local data within {radius_km}km")
        return stations
    
    def _find_sensor_location(self, sensor_id: int) -> Optional[int]:
        """Find which location a sensor belongs to (OpenAQ v3 compatibility)."""
        try:
            # In v3, we need to search through locations to find the sensor
            # This is a simplified approach - in practice would cache this mapping
            params = {
                'coordinates': f'{KATHMANDU_LAT},{KATHMANDU_LON}',
                'radius': 50000,  # 50km radius
                'limit': 50
            }
            
            response = self.client.get(f"{self.base_url}/locations", params=params)
            response.raise_for_status()
            
            locations = response.json().get('results', [])
            
            for location in locations:
                location_id = location.get('id')
                if location_id:
                    # Check if this location has the sensor
                    sensors_response = self.client.get(f"{self.base_url}/locations/{location_id}/sensors")
                    if sensors_response.status_code == 200:
                        sensors_data = sensors_response.json()
                        for sensor in sensors_data.get('results', []):
                            if sensor.get('id') == sensor_id:
                                return location_id
            return None
            
        except Exception as e:
            logger.warning(f"Failed to find location for sensor {sensor_id}: {e}")
            return None

    def get_sensor_coverage(
        self,
        sensor_id: int,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> CoverageAnalysis:
        """Analyze data coverage for a specific sensor.
        
        Args:
            sensor_id: Sensor ID to analyze
            start_date: Start date for analysis (default: 16 months ago)
            end_date: End date for analysis (default: now)
            
        Returns:
            Coverage analysis results
        """
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=16*30)  # ~16 months
        
        if self.use_local:
            return self._analyze_local_sensor_coverage(sensor_id, start_date, end_date)
        
        if not self.client:
            raise ValueError("No API client available and not in local mode")
        
        logger.info(f"Analyzing coverage for sensor {sensor_id} from {start_date.date()} to {end_date.date()}")
        
        # Get measurements count via API
        params = {
            "sensors_id": sensor_id,
            "date_from": start_date.isoformat(),
            "date_to": end_date.isoformat(),
            "limit": 1,  # We only need the count
        }
        
        response = self.client.get("measurements", params=params)
        
        if not response.success:
            logger.warning(f"Failed to get measurements for sensor {sensor_id}: {response.error}")
            # Return empty analysis
            return CoverageAnalysis(
                station_id=0,
                sensor_id=sensor_id,
                date_range=(start_date, end_date),
                expected_measurements=0,
                actual_measurements=0,
                missing_measurements=0,
                missingness_pct=100.0,
                coverage_score=0.0,
            )
        
        # Get total count from metadata
        meta = response.data.get("meta", {})
        actual_count = meta.get("found", 0)
        
        # Calculate expected measurements (hourly data)
        total_hours = int((end_date - start_date).total_seconds() / 3600)
        missing_count = max(0, total_hours - actual_count)
        missingness_pct = (missing_count / total_hours * 100) if total_hours > 0 else 100
        
        # Calculate coverage score (1.0 - missingness as fraction)
        coverage_score = max(0, 1.0 - missingness_pct / 100)
        
        return CoverageAnalysis(
            station_id=0,  # Would need separate call to get station
            sensor_id=sensor_id,
            date_range=(start_date, end_date),
            expected_measurements=total_hours,
            actual_measurements=actual_count,
            missing_measurements=missing_count,
            missingness_pct=round(missingness_pct, 2),
            coverage_score=round(coverage_score, 3),
        )
    
    def _analyze_local_sensor_coverage(
        self,
        sensor_id: int,
        start_date: datetime,
        end_date: datetime,
    ) -> CoverageAnalysis:
        """Analyze sensor coverage from local data."""
        summary = self._analyze_local_data()
        
        # Get measurements count from local summary
        actual_count = summary.total_measurements.get(sensor_id, 0)
        
        # Calculate expected (assuming hourly data)
        total_hours = int((end_date - start_date).total_seconds() / 3600)
        missing_count = max(0, total_hours - actual_count)
        missingness_pct = (missing_count / total_hours * 100) if total_hours > 0 else 100
        coverage_score = max(0, 1.0 - missingness_pct / 100)
        
        return CoverageAnalysis(
            station_id=0,
            sensor_id=sensor_id,
            date_range=(start_date, end_date),
            expected_measurements=total_hours,
            actual_measurements=actual_count,
            missing_measurements=missing_count,
            missingness_pct=round(missingness_pct, 2),
            coverage_score=round(coverage_score, 3),
        )
    
    def analyze_stations_parallel(
        self,
        stations: List[StationInfo],
        coverage_months: int = 16,
    ) -> List[StationInfo]:
        """Analyze multiple stations in parallel with enhanced concurrency.
        
        Args:
            stations: List of stations to analyze
            coverage_months: Number of months to analyze for coverage
            
        Returns:
            List of stations with updated coverage information
        """
        if not stations:
            return []
        
        # Enhance parallelism: process up to 5 stations concurrently for faster analysis
        effective_workers = min(self.max_workers, len(stations), 5)
        logger.info(f"Analyzing {len(stations)} stations in parallel (workers={effective_workers})")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=coverage_months * 30)
        
        def analyze_station_batch(station_batch: List[StationInfo]) -> List[StationInfo]:
            """Analyze a batch of stations for better memory efficiency."""
            results = []
            for station in station_batch:
                if not station.pm25_sensor_ids:
                    results.append(station)
                    continue
                
                # Analyze first PM2.5 sensor
                sensor_id = station.pm25_sensor_ids[0]
                
                try:
                    coverage = self.get_sensor_coverage(sensor_id, start_date, end_date)
                    
                    # Update station info
                    station.first_measurement = coverage.date_range[0]
                    station.last_measurement = coverage.date_range[1]
                    station.total_measurements = coverage.actual_measurements
                    station.data_span_days = (coverage.date_range[1] - coverage.date_range[0]).days
                    station.missingness_pct = coverage.missingness_pct
                    station.data_quality_score = coverage.coverage_score
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze station {station.station_id}: {e}")
                    station.data_quality_score = 0.0
                    station.missingness_pct = 100.0
                
                results.append(station)
            
            return results
        
        # Process stations in parallel with batching for efficiency
        updated_stations = []
        batch_size = max(1, len(stations) // effective_workers)
        
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            # Create batches
            station_batches = [
                stations[i:i + batch_size] 
                for i in range(0, len(stations), batch_size)
            ]
            
            # Submit batch tasks
            future_to_batch = {
                executor.submit(analyze_station_batch, batch): batch
                for batch in station_batches
            }
            
            # Collect results
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    updated_stations.extend(batch_results)
                except Exception as e:
                    original_batch = future_to_batch[future]
                    logger.error(f"Failed to process station batch of {len(original_batch)} stations: {e}")
                    # Add original stations without updates
                    updated_stations.extend(original_batch)
        
        # Sort by quality score (best first)
        updated_stations.sort(key=lambda s: s.data_quality_score, reverse=True)
        
        logger.info(f"Completed enhanced parallel analysis of {len(updated_stations)} stations")
        return updated_stations
    
    def get_best_stations(
        self,
        lat: float = KATHMANDU_LAT,
        lon: float = KATHMANDU_LON,
        radius_km: float = 25.0,
        min_quality_score: float = 0.7,
        max_missingness_pct: float = 30.0,
        min_data_span_days: int = 16 * 30,  # ~16 months
    ) -> List[StationInfo]:
        """Get best PM₂.₅ stations meeting quality criteria.
        
        Args:
            lat: Center latitude
            lon: Center longitude
            radius_km: Search radius
            min_quality_score: Minimum data quality score (0-1)
            max_missingness_pct: Maximum missingness percentage
            min_data_span_days: Minimum data span in days
            
        Returns:
            List of stations meeting criteria, sorted by quality
        """
        # Get stations in area
        stations = self.get_locations_pm25(lat, lon, radius_km)
        
        if not stations:
            logger.warning("No PM₂.₅ stations found in search area")
            return []
        
        # Analyze all stations
        analyzed_stations = self.analyze_stations_parallel(stations)
        
        # Filter by quality criteria
        good_stations = [
            station for station in analyzed_stations
            if (
                station.data_quality_score >= min_quality_score and
                station.missingness_pct <= max_missingness_pct and
                station.data_span_days >= min_data_span_days
            )
        ]
        
        logger.info(
            f"Found {len(good_stations)}/{len(analyzed_stations)} stations meeting quality criteria"
        )
        
        return good_stations
    
    def get_measurements(
        self,
        station_id: Optional[int] = None,
        sensor_id: Optional[int] = None,
        parameter: str = "pm25",
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict]:
        """Get measurements from OpenAQ API.
        
        Args:
            station_id: Station ID to filter by
            sensor_id: Sensor ID to filter by
            parameter: Parameter to get (default: pm25)
            date_from: Start date
            date_to: End date
            limit: Maximum number of measurements
            
        Returns:
            List of measurement records
        """
        if self.use_local:
            return self._get_local_measurements(station_id, sensor_id, parameter, date_from, date_to)
        
        if not self.client:
            raise ValueError("No API client available and not in local mode")
        
        logger.info(f"Fetching measurements for station/sensor {station_id}/{sensor_id}")
        
        # Build parameters
        params = {
            "limit": limit,
            "parameters_id": 2 if parameter == "pm25" else None,  # PM2.5 parameter ID
        }
        
        if station_id:
            params["locations_id"] = station_id
        if sensor_id:
            params["sensors_id"] = sensor_id
        if date_from:
            params["date_from"] = date_from.isoformat()
        if date_to:
            params["date_to"] = date_to.isoformat()
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        # Use the correct OpenAQ v3 sensors endpoint for historical data
        if sensor_id:
            # Use sensors/{id}/hours endpoint for hourly data
            endpoint = f"sensors/{sensor_id}/hours"
            response = self.client.get(endpoint, params=params)
        elif station_id:
            # For station-level data, we need to get sensor IDs first
            # This is a fallback - prefer using sensor_id directly
            logger.warning(f"Station-level queries not fully supported in v3. Use sensor_id instead.")
            return []
        else:
            raise ValueError("Either station_id or sensor_id must be provided")
        
        if not response.success:
            logger.warning(f"Failed to get measurements: {response.error}")
            return []
        
        results = response.data.get("results", [])
        
        # Transform OpenAQ v3 sensors format to expected format
        measurements = []
        for result in results:
            # Extract the hourly data
            period = result.get("period", {})
            datetime_from = period.get("datetimeFrom", {}).get("utc")
            datetime_to = period.get("datetimeTo", {}).get("utc")
            
            if datetime_from:
                # Use the start of the hour as the timestamp
                measurements.append({
                    "value": result.get("value"),
                    "unit": result.get("parameter", {}).get("units", "µg/m³"),
                    "datetime": datetime_from,
                    "coordinates": result.get("coordinates"),
                    "sensor_id": sensor_id,
                    "station_id": station_id,
                    "quality": "valid" if not result.get("flagInfo", {}).get("hasFlags") else "flagged",
                    "summary": result.get("summary", {}),
                    "coverage": result.get("coverage", {})
                })
        
        logger.info(f"Retrieved {len(measurements)} measurements")
        return measurements
    
    def _get_local_measurements(
        self,
        station_id: Optional[int] = None,
        sensor_id: Optional[int] = None,
        parameter: str = "pm25",
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[Dict]:
        """Get measurements from local data."""
        logger.info("Using local measurements data (not implemented yet)")
        return []  # Placeholder for local data retrieval
