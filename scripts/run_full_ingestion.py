#!/usr/bin/env python3
"""
Full-Scale OpenAQ Data Ingestion Script for Nepal, India, and China

This script fetches ALL available air quality data from OpenAQ API for:
- Nepal (ID: 145)
- India (ID: 9) 
- China (ID: 10)

Expected to collect millions of measurements across thousands of sensors.
Estimated runtime: 4-8 hours depending on API performance.

Usage:
    python run_full_ingestion.py

Output:
    - Multiple parquet files with all locations, sensors, and measurements
    - Comprehensive logging of the entire process
    - Progress tracking and estimated completion times
"""

import os
import sys
import time
import json
import logging
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import configuration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
try:
    from config import *
except ImportError:
    # Fallback values if config import fails
    API_KEY = "9a1e83c1ac07424296a0e90aa95153fe13c7c8b8996d5f4d712c95484d2d1588"
    OUTPUT_DIR = "data/full_data"
    TARGET_COUNTRIES = {"Nepal": 145, "India": 9, "China": 10}
    DATE_FROM = "2024-01-01T00:00:00Z"
    DATE_TO = None
    REQUEST_DELAY = 0.3
    BATCH_SIZE = 75000
    MAX_WORKERS = 3


class FullScaleOpenAQIngester:
    """Enhanced OpenAQ API data ingestion for full-scale data collection"""
    
    def __init__(self, api_key: str = None, output_dir: str = None):
        """
        Initialize the full-scale OpenAQ ingester.
        
        Args:
            api_key: OpenAQ API key (uses config if None)
            output_dir: Directory to save output files (uses config if None)
        """
        self.api_key = api_key or API_KEY
        self.base_url = "https://api.openaq.org/v3"
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Country IDs for Nepal, India, and China (in specified order)
        self.target_countries = TARGET_COUNTRIES
        
        # Setup headers
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "AirQuality4-Ingestion/1.0"
        }
        
        # Setup logging
        self.setup_logging()
        
        # Rate limiting and performance settings from config
        self.request_delay = REQUEST_DELAY
        self.batch_size = BATCH_SIZE
        self.max_workers = MAX_WORKERS
        
        # Progress tracking
        self.stats = {
            'total_locations': 0,
            'total_sensors': 0,
            'total_measurements': 0,
            'countries_processed': 0,
            'start_time': None,
            'current_country': None,
            'current_sensor': 0,
            'batch_count': 0
        }
        self.lock = threading.Lock()
        
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f"full_ingestion_{timestamp}.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def make_request(self, endpoint: str, params: Optional[Dict] = None, retries: int = 3) -> Optional[Dict]:
        """
        Make a request to the OpenAQ API with enhanced error handling and retries.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            retries: Number of retry attempts
            
        Returns:
            Response JSON or None if failed
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(retries + 1):
            try:
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    params=params, 
                    timeout=30
                )
                response.raise_for_status()
                
                # Rate limiting
                time.sleep(self.request_delay)
                
                return response.json()
                
            except requests.exceptions.Timeout:
                if attempt < retries:
                    self.logger.warning(f"Timeout on attempt {attempt + 1} for {url}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                self.logger.error(f"Final timeout for {url} after {retries + 1} attempts")
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    if attempt < retries:
                        wait_time = 60 * (attempt + 1)  # Increase wait time
                        self.logger.warning(f"Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                elif e.response.status_code >= 500:  # Server error
                    if attempt < retries:
                        self.logger.warning(f"Server error {e.response.status_code}, retrying...")
                        time.sleep(5 * (attempt + 1))
                        continue
                self.logger.error(f"HTTP error for {url}: {e}")
                
            except requests.exceptions.RequestException as e:
                if attempt < retries:
                    self.logger.warning(f"Request error on attempt {attempt + 1}: {e}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                self.logger.error(f"Final request error for {url}: {e}")
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode JSON from {url}: {e}")
                break
                
        return None
            
    def get_all_locations_for_country(self, country_id: int, country_name: str) -> List[Dict]:
        """
        Get all locations for a specific country with pagination.
        
        Args:
            country_id: Country ID
            country_name: Country name for logging
            
        Returns:
            List of location dictionaries
        """
        locations = []
        page = 1
        limit = 1000
        
        self.logger.info(f"🌍 Starting location collection for {country_name} (ID: {country_id})...")
        
        while True:
            params = {
                "countries_id": country_id,
                "page": page,
                "limit": limit
            }
            
            response = self.make_request("locations", params)
            if not response:
                self.logger.warning(f"Failed to get locations page {page} for {country_name}")
                break
                
            results = response.get("results", [])
            if not results:
                break
                
            locations.extend(results)
            self.logger.info(f"  📍 Page {page}: {len(results)} locations collected")
            
            # Check if we have more pages
            if len(results) < limit:
                break
                
            page += 1
            
        with self.lock:
            self.stats['total_locations'] += len(locations)
            
        self.logger.info(f"✅ {country_name}: {len(locations)} total locations collected")
        return locations
        
    def extract_sensors_from_locations(self, locations: List[Dict], country_name: str) -> List[Dict]:
        """
        Extract sensor information from location data with enhanced metadata.
        
        Args:
            locations: List of location dictionaries
            country_name: Country name for context
            
        Returns:
            List of sensor dictionaries with location context
        """
        sensors = []
        
        self.logger.info(f"🔬 Extracting sensors from {len(locations)} locations in {country_name}...")
        
        for location in locations:
            location_sensors = location.get("sensors", [])
            for sensor in location_sensors:
                sensor_info = {
                    # Sensor details
                    "sensor_id": sensor.get("id"),
                    "sensor_name": sensor.get("name"),
                    "parameter": sensor.get("parameter", {}),
                    
                    # Location context
                    "location_id": location.get("id"),
                    "location_name": location.get("name"),
                    "location_locality": location.get("locality"),
                    "timezone": location.get("timezone"),
                    
                    # Geographic info
                    "country": location.get("country", {}),
                    "coordinates": location.get("coordinates", {}),
                    "bounds": location.get("bounds", []),
                    
                    # Metadata
                    "owner": location.get("owner", {}),
                    "provider": location.get("provider", {}),
                    "instruments": location.get("instruments", []),
                    "licenses": location.get("licenses", []),
                    
                    # Temporal info
                    "datetime_first": location.get("datetimeFirst"),
                    "datetime_last": location.get("datetimeLast"),
                    
                    # Flags
                    "is_mobile": location.get("isMobile", False),
                    "is_monitor": location.get("isMonitor", False)
                }
                sensors.append(sensor_info)
                
        with self.lock:
            self.stats['total_sensors'] += len(sensors)
            
        self.logger.info(f"✅ {country_name}: {len(sensors)} sensors extracted")
        return sensors
        
    def get_sensor_measurements_batch(self, sensor_batch: List[Dict], country_name: str) -> List[Dict]:
        """
        Get measurements for a batch of sensors concurrently.
        
        Args:
            sensor_batch: List of sensor info dictionaries
            country_name: Country name for logging
            
        Returns:
            List of all measurements from the batch
        """
        all_measurements = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all sensor measurement tasks
            future_to_sensor = {
                executor.submit(
                    self.get_sensor_measurements_single,
                    sensor_info,
                    country_name
                ): sensor_info for sensor_info in sensor_batch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_sensor):
                sensor_info = future_to_sensor[future]
                try:
                    measurements = future.result()
                    all_measurements.extend(measurements)
                    
                    with self.lock:
                        self.stats['current_sensor'] += 1
                        self.stats['total_measurements'] += len(measurements)
                        
                    # Log progress
                    if self.stats['current_sensor'] % 10 == 0:
                        self.log_progress()
                        
                except Exception as exc:
                    sensor_id = sensor_info.get('sensor_id', 'unknown')
                    self.logger.error(f"Sensor {sensor_id} generated exception: {exc}")
                    
        return all_measurements
        
    def get_sensor_measurements_single(self, sensor_info: Dict, country_name: str,
                                     date_from: str = None,
                                     date_to: Optional[str] = None) -> List[Dict]:
        """
        Get all measurements for a single sensor.
        
        Args:
            sensor_info: Sensor information dictionary
            country_name: Country name for context
            date_from: Start date (ISO format)
            date_to: End date (ISO format), defaults to current time
            
        Returns:
            List of measurement dictionaries
        """
        sensor_id = sensor_info.get("sensor_id")
        if not sensor_id:
            return []
            
        # Use configuration date range if not provided
        if date_from is None:
            date_from = DATE_FROM
        if date_to is None:
            date_to = datetime.now(timezone.utc).isoformat()
            
        measurements = []
        page = 1
        limit = 1000
        
        while True:
            params = {
                "date_from": date_from,
                "date_to": date_to,
                "page": page,
                "limit": limit
            }
            
            response = self.make_request(f"sensors/{sensor_id}/measurements", params)
            if not response:
                break
                
            results = response.get("results", [])
            if not results:
                break
                
            # Add sensor context to each measurement
            for measurement in results:
                measurement.update({
                    "sensor_id": sensor_id,
                    "sensor_name": sensor_info.get("sensor_name"),
                    "location_id": sensor_info.get("location_id"),
                    "location_name": sensor_info.get("location_name"),
                    "country_name": sensor_info.get("country", {}).get("name"),
                    "country_code": sensor_info.get("country", {}).get("code"),
                    "coordinates_lat": sensor_info.get("coordinates", {}).get("latitude"),
                    "coordinates_lon": sensor_info.get("coordinates", {}).get("longitude"),
                    "timezone": sensor_info.get("timezone"),
                    "provider_name": sensor_info.get("provider", {}).get("name"),
                    "ingestion_timestamp": datetime.now(timezone.utc).isoformat()
                })
                
            measurements.extend(results)
            
            # Check if we have more pages
            if len(results) < limit:
                break
                
            page += 1
            
        return measurements
        
    def save_batch_to_parquet(self, data: List[Dict], filename: str, description: str):
        """
        Save data batch to parquet file with compression and optimization.
        
        Args:
            data: List of dictionaries to save
            filename: Output filename
            description: Description for logging
        """
        if not data:
            self.logger.warning(f"⚠️  No data to save for {description}")
            return
            
        try:
            df = pd.DataFrame(data)
            output_path = self.output_dir / filename
            
            # Save with compression for space efficiency
            df.to_parquet(
                output_path, 
                index=False,
                compression='snappy',
                engine='pyarrow'
            )
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"💾 Saved {len(data):,} {description} records to {filename}")
            self.logger.info(f"   📊 Size: {file_size_mb:.1f} MB, Columns: {len(df.columns)}, Shape: {df.shape}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save {description} to parquet: {e}")
            
    def log_progress(self):
        """Log current progress and estimated completion time"""
        if not self.stats['start_time']:
            return
            
        elapsed = time.time() - self.stats['start_time']
        elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"
        
        self.logger.info(f"📈 PROGRESS UPDATE:")
        self.logger.info(f"   ⏱️  Runtime: {elapsed_str}")
        self.logger.info(f"   🌍 Countries: {self.stats['countries_processed']}/3")
        self.logger.info(f"   📍 Locations: {self.stats['total_locations']:,}")
        self.logger.info(f"   🔬 Sensors: {self.stats['total_sensors']:,}")
        self.logger.info(f"   📊 Measurements: {self.stats['total_measurements']:,}")
        self.logger.info(f"   💾 Batches saved: {self.stats['batch_count']}")
        
        if self.stats['current_country']:
            self.logger.info(f"   🔄 Current: {self.stats['current_country']} (sensor {self.stats['current_sensor']})")
            
    def run_full_ingestion(self):
        """Run the complete full-scale data ingestion process"""
        self.stats['start_time'] = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.logger.info("🚀 STARTING FULL-SCALE OPENAQ DATA INGESTION")
        self.logger.info("=" * 80)
        self.logger.info(f"🎯 Target countries: {list(self.target_countries.keys())}")
        self.logger.info(f"📂 Output directory: {self.output_dir}")
        self.logger.info(f"⚙️  Batch size: {self.batch_size:,} measurements")
        self.logger.info(f"🔧 Max workers: {self.max_workers}")
        self.logger.info("=" * 80)
        
        # Master data collections
        all_locations = []
        all_sensors = []
        measurement_batches = []
        current_batch = []
        
        try:
            # Phase 1: Collect all locations and sensors
            self.logger.info("📋 PHASE 1: COLLECTING LOCATIONS AND SENSORS")
            
            for country_name, country_id in self.target_countries.items():
                self.stats['current_country'] = country_name
                
                # Get locations
                locations = self.get_all_locations_for_country(country_id, country_name)
                all_locations.extend(locations)
                
                # Extract sensors
                sensors = self.extract_sensors_from_locations(locations, country_name)
                all_sensors.extend(sensors)
                
                self.stats['countries_processed'] += 1
                self.log_progress()
                
            # Save locations and sensors
            self.save_batch_to_parquet(
                all_locations, 
                f"all_locations_{timestamp}.parquet", 
                "locations"
            )
            self.save_batch_to_parquet(
                all_sensors, 
                f"all_sensors_{timestamp}.parquet", 
                "sensors"
            )
            
            # Phase 2: Collect measurements from all sensors
            self.logger.info("📊 PHASE 2: COLLECTING MEASUREMENTS")
            self.logger.info(f"🔬 Processing {len(all_sensors):,} sensors total...")
            
            # Process sensors in batches to manage memory and save intermediate results
            sensor_batch_size = 50  # Process 50 sensors at a time
            
            for i in range(0, len(all_sensors), sensor_batch_size):
                sensor_batch = all_sensors[i:i + sensor_batch_size]
                batch_num = (i // sensor_batch_size) + 1
                total_batches = (len(all_sensors) + sensor_batch_size - 1) // sensor_batch_size
                
                self.logger.info(f"🔄 Processing sensor batch {batch_num}/{total_batches} ({len(sensor_batch)} sensors)")
                
                # Get measurements for this batch of sensors
                batch_measurements = self.get_sensor_measurements_batch(
                    sensor_batch, 
                    "Multi-country"
                )
                
                current_batch.extend(batch_measurements)
                
                # Save when batch gets large enough
                if len(current_batch) >= self.batch_size:
                    batch_filename = f"measurements_batch_{self.stats['batch_count']:04d}_{timestamp}.parquet"
                    self.save_batch_to_parquet(
                        current_batch,
                        batch_filename,
                        f"measurements (batch {self.stats['batch_count']})"
                    )
                    measurement_batches.append(batch_filename)
                    current_batch = []
                    self.stats['batch_count'] += 1
                    
                self.log_progress()
                
            # Save final batch if there's remaining data
            if current_batch:
                final_filename = f"measurements_final_{timestamp}.parquet"
                self.save_batch_to_parquet(
                    current_batch,
                    final_filename,
                    "measurements (final batch)"
                )
                measurement_batches.append(final_filename)
                self.stats['batch_count'] += 1
                
            # Phase 3: Create summary report
            self.create_ingestion_summary(timestamp, measurement_batches)
            
        except KeyboardInterrupt:
            self.logger.warning("⚠️  Ingestion interrupted by user")
            self.create_ingestion_summary(timestamp, measurement_batches, interrupted=True)
        except Exception as e:
            self.logger.error(f"❌ Fatal error during ingestion: {e}")
            raise
        finally:
            self.log_final_statistics()
            
    def create_ingestion_summary(self, timestamp: str, measurement_files: List[str], interrupted: bool = False):
        """Create a comprehensive summary of the ingestion process"""
        summary = {
            "ingestion_info": {
                "timestamp": timestamp,
                "start_time": datetime.fromtimestamp(self.stats['start_time']).isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_hours": (time.time() - self.stats['start_time']) / 3600,
                "status": "interrupted" if interrupted else "completed",
                "target_countries": self.target_countries
            },
            "data_summary": {
                "total_locations": self.stats['total_locations'],
                "total_sensors": self.stats['total_sensors'],
                "total_measurements": self.stats['total_measurements'],
                "batch_count": self.stats['batch_count']
            },
            "output_files": {
                "locations": f"all_locations_{timestamp}.parquet",
                "sensors": f"all_sensors_{timestamp}.parquet",
                "measurement_batches": measurement_files
            }
        }
        
        # Save summary as JSON
        summary_file = self.output_dir / f"ingestion_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"📋 Ingestion summary saved to {summary_file}")
        
    def log_final_statistics(self):
        """Log final ingestion statistics"""
        total_time = time.time() - self.stats['start_time']
        
        self.logger.info("🏁 FULL-SCALE INGESTION COMPLETED!")
        self.logger.info("=" * 80)
        self.logger.info(f"⏱️  Total runtime: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        self.logger.info(f"🌍 Countries processed: {self.stats['countries_processed']}/3")
        self.logger.info(f"📍 Total locations: {self.stats['total_locations']:,}")
        self.logger.info(f"🔬 Total sensors: {self.stats['total_sensors']:,}")
        self.logger.info(f"📊 Total measurements: {self.stats['total_measurements']:,}")
        self.logger.info(f"💾 Data batches created: {self.stats['batch_count']}")
        
        if self.stats['total_measurements'] > 0:
            rate = self.stats['total_measurements'] / total_time
            self.logger.info(f"⚡ Average rate: {rate:.1f} measurements/second")
            
        self.logger.info(f"📂 All data saved in: {self.output_dir}")
        self.logger.info("=" * 80)


def main():
    """Main execution function"""
    print("🌍 OpenAQ Full-Scale Data Ingestion for Nepal, India & China")
    print("=" * 60)
    print(f"📅 Date range: {DATE_FROM} to present")
    print(f"📊 Expected output: Measurements from 2024-2025 (reduced dataset)")
    print("💾 Data will be saved in compressed parquet format")
    print("=" * 60)
    
    # Confirm before starting
    response = input("Do you want to proceed? (yes/no): ").lower().strip()
    if response not in ['yes', 'y']:
        print("❌ Ingestion cancelled by user")
        return
        
    # Create ingester and run
    print("🚀 Starting optimized ingestion for 2024+ data...")
    ingester = FullScaleOpenAQIngester()
    ingester.run_full_ingestion()


if __name__ == "__main__":
    main()