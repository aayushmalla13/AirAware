#!/usr/bin/env python3
"""
Nepal-Specific OpenAQ Data Ingestion Script

This script focuses exclusively on fetching air quality data from all locations 
and sensors in Nepal with enhanced logging and error handling.

Features:
- Real-time logging of location and sensor processing
- Robust error handling with retry mechanisms
- Proper data partitioning
- Progress tracking and recovery capabilities
- Memory-efficient batch processing

Author: Generated for AirQuality4 project
Date: 2025-09-22
"""

import requests
import pandas as pd
import logging
import time
import json
import gc
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class NepalOpenAQIngester:
    """Nepal-focused OpenAQ API data ingestion class with enhanced logging"""
    
    def __init__(self, api_key: str, output_dir: str = "data/nepal_data"):
        """
        Initialize the Nepal OpenAQ ingester.
        
        Args:
            api_key: OpenAQ API key
            output_dir: Directory to save output files
        """
        self.api_key = api_key
        self.base_url = "https://api.openaq.org/v3"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nepal-specific configuration
        self.nepal_country_id = 145
        self.nepal_country_code = "NP"
        
        # Setup headers
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Performance settings
        self.request_delay = 0.5  # 0.5 seconds between requests
        self.max_retries = 3
        self.timeout_seconds = 30
        self.batch_size = 50000  # Measurements per file
        
        # Threading lock for thread-safe logging
        self.log_lock = threading.Lock()
        
        # Progress tracking
        self.progress = {
            'locations_processed': 0,
            'sensors_processed': 0,
            'measurements_collected': 0,
            'errors_encountered': 0,
            'start_time': None
        }
        
        # Setup enhanced logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup enhanced logging with real-time progress"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f"nepal_ingestion_{timestamp}.log"
        
        # Create custom formatter for detailed logging
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("🇳🇵 " + "="*80)
        self.logger.info("🇳🇵 NEPAL OPENAQ DATA INGESTION STARTED")
        self.logger.info("🇳🇵 " + "="*80)
        self.logger.info(f"📁 Output Directory: {self.output_dir}")
        self.logger.info(f"📋 Log File: {log_file}")
        
    def log_progress(self, message: str, level: str = "INFO"):
        """Thread-safe logging with progress information"""
        with self.log_lock:
            if level.upper() == "INFO":
                self.logger.info(f"📊 {message}")
            elif level.upper() == "ERROR":
                self.logger.error(f"❌ {message}")
            elif level.upper() == "WARNING":
                self.logger.warning(f"⚠️  {message}")
            elif level.upper() == "SUCCESS":
                self.logger.info(f"✅ {message}")
        
    def make_request_with_retry(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a request to the OpenAQ API with retry logic and detailed error handling.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            Response JSON or None if failed after all retries
        """
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                self.log_progress(f"API Request: {endpoint} (attempt {attempt + 1}/{self.max_retries})", "INFO")
                
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    params=params, 
                    timeout=self.timeout_seconds
                )
                response.raise_for_status()
                
                # Rate limiting
                time.sleep(self.request_delay)
                
                return response.json()
                
            except requests.exceptions.Timeout:
                self.log_progress(f"Request timeout for {endpoint} (attempt {attempt + 1})", "WARNING")
                self.progress['errors_encountered'] += 1
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = 60
                    self.log_progress(f"Rate limited! Waiting {wait_time} seconds...", "WARNING")
                    time.sleep(wait_time)
                else:
                    self.log_progress(f"HTTP Error {e.response.status_code} for {endpoint}: {e}", "ERROR")
                    self.progress['errors_encountered'] += 1
                    
            except requests.exceptions.ConnectionError:
                self.log_progress(f"Connection error for {endpoint} (attempt {attempt + 1})", "WARNING")
                self.progress['errors_encountered'] += 1
                time.sleep(5)  # Wait before retry
                
            except json.JSONDecodeError as e:
                self.log_progress(f"JSON decode error for {endpoint}: {e}", "ERROR")
                self.progress['errors_encountered'] += 1
                
            except Exception as e:
                self.log_progress(f"Unexpected error for {endpoint}: {e}", "ERROR")
                self.progress['errors_encountered'] += 1
                
            # Wait before retry
            if attempt < self.max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                time.sleep(wait_time)
        
        self.log_progress(f"Failed to fetch data from {endpoint} after {self.max_retries} attempts", "ERROR")
        return None
        
    def get_nepal_locations(self) -> List[Dict]:
        """
        Get all locations in Nepal with detailed logging.
        
        Returns:
            List of location dictionaries
        """
        locations = []
        page = 1
        limit = 1000
        
        self.log_progress("🏢 Starting to fetch all locations in Nepal...")
        
        while True:
            params = {
                "countries_id": self.nepal_country_id,
                "page": page,
                "limit": limit
            }
            
            self.log_progress(f"📍 Fetching locations page {page}...")
            response = self.make_request_with_retry("locations", params)
            
            if not response:
                self.log_progress(f"Failed to fetch locations page {page}", "ERROR")
                break
                
            results = response.get("results", [])
            if not results:
                self.log_progress(f"No more locations found (page {page})", "INFO")
                break
                
            locations.extend(results)
            self.progress['locations_processed'] += len(results)
            
            # Log details about each location
            for location in results:
                location_name = location.get("name", "Unknown")
                locality = location.get("locality", "Unknown")
                sensor_count = len(location.get("sensors", []))
                self.log_progress(
                    f"📍 Location: {location_name} | Locality: {locality} | Sensors: {sensor_count}"
                )
            
            self.log_progress(f"✅ Page {page}: {len(results)} locations added (Total: {len(locations)})")
            
            # Check if we have more pages
            if len(results) < limit:
                break
                
            page += 1
            
        self.log_progress(f"🎯 TOTAL NEPAL LOCATIONS: {len(locations)}", "SUCCESS")
        return locations
        
    def extract_sensors_from_locations(self, locations: List[Dict]) -> List[Dict]:
        """
        Extract sensor information from location data with detailed logging.
        
        Args:
            locations: List of location dictionaries
            
        Returns:
            List of sensor dictionaries with location context
        """
        sensors = []
        
        self.log_progress("🔍 Extracting sensors from locations...")
        
        for location in locations:
            location_id = location.get("id")
            location_name = location.get("name", "Unknown")
            locality = location.get("locality", "Unknown")
            location_sensors = location.get("sensors", [])
            
            self.log_progress(f"🏢 Processing location: {location_name} ({locality}) - {len(location_sensors)} sensors")
            
            for sensor in location_sensors:
                sensor_id = sensor.get("id")
                sensor_name = sensor.get("name", "Unknown")
                parameter = sensor.get("parameter", {})
                parameter_name = parameter.get("name", "Unknown")
                
                sensor_info = {
                    "sensor_id": sensor_id,
                    "sensor_name": sensor_name,
                    "parameter": parameter,
                    "parameter_name": parameter_name,
                    "location_id": location_id,
                    "location_name": location_name,
                    "location_locality": locality,
                    "country": location.get("country", {}),
                    "coordinates": location.get("coordinates", {}),
                    "timezone": location.get("timezone"),
                    "owner": location.get("owner", {}),
                    "provider": location.get("provider", {})
                }
                sensors.append(sensor_info)
                
                self.log_progress(f"  🔬 Sensor: {sensor_id} | {sensor_name} | Parameter: {parameter_name}")
                
        self.progress['sensors_processed'] = len(sensors)
        self.log_progress(f"🎯 TOTAL NEPAL SENSORS: {len(sensors)}", "SUCCESS")
        return sensors
        
    def get_sensor_measurements(self, sensor_info: Dict, 
                              date_from: str = "2024-01-01T00:00:00Z",
                              date_to: Optional[str] = None) -> List[Dict]:
        """
        Get all measurements for a specific sensor with detailed logging.
        
        Args:
            sensor_info: Sensor information dictionary
            date_from: Start date (ISO format)
            date_to: End date (ISO format), defaults to current time
            
        Returns:
            List of measurement dictionaries
        """
        sensor_id = sensor_info.get("sensor_id")
        sensor_name = sensor_info.get("sensor_name", "Unknown")
        location_name = sensor_info.get("location_name", "Unknown")
        parameter_name = sensor_info.get("parameter_name", "Unknown")
        
        if not sensor_id:
            self.log_progress(f"⚠️  Skipping sensor with no ID: {sensor_name}", "WARNING")
            return []
            
        if date_to is None:
            date_to = datetime.now(timezone.utc).isoformat()
            
        measurements = []
        page = 1
        limit = 1000
        
        self.log_progress(f"📊 Fetching measurements for sensor {sensor_id}")
        self.log_progress(f"  🔬 Sensor: {sensor_name} | Location: {location_name} | Parameter: {parameter_name}")
        
        while True:
            params = {
                "date_from": date_from,
                "date_to": date_to,
                "page": page,
                "limit": limit
            }
            
            response = self.make_request_with_retry(f"sensors/{sensor_id}/measurements", params)
            if not response:
                self.log_progress(f"❌ Failed to fetch measurements for sensor {sensor_id}", "ERROR")
                break
                
            results = response.get("results", [])
            if not results:
                break
                
            # Add sensor context to each measurement
            for measurement in results:
                measurement.update({
                    "sensor_id": sensor_id,
                    "sensor_name": sensor_name,
                    "parameter_name": parameter_name,
                    "location_id": sensor_info.get("location_id"),
                    "location_name": location_name,
                    "location_locality": sensor_info.get("location_locality"),
                    "country_name": "Nepal",
                    "country_code": "NP",
                })
                
            measurements.extend(results)
            self.progress['measurements_collected'] += len(results)
            
            self.log_progress(f"  📈 Page {page}: {len(results)} measurements | Total: {len(measurements)}")
            
            # Check if we have more pages
            if len(results) < limit:
                break
                
            page += 1
            
        self.log_progress(f"✅ Sensor {sensor_id} complete: {len(measurements)} measurements", "SUCCESS")
        return measurements
        
    def save_to_parquet_partitioned(self, data: List[Dict], data_type: str, timestamp: str):
        """
        Save data to partitioned parquet files.
        
        Args:
            data: List of dictionaries to save
            data_type: Type of data (locations, sensors, measurements)
            timestamp: Timestamp for file naming
        """
        if not data:
            self.log_progress(f"⚠️  No {data_type} data to save", "WARNING")
            return
            
        try:
            df = pd.DataFrame(data)
            
            # Create partitioned directory structure
            partition_dir = self.output_dir / data_type
            partition_dir.mkdir(exist_ok=True)
            
            filename = f"nepal_{data_type}_{timestamp}.parquet"
            output_path = partition_dir / filename
            
            df.to_parquet(output_path, index=False, compression='snappy')
            
            self.log_progress(f"💾 Saved {len(data)} {data_type} records to {output_path}", "SUCCESS")
            self.log_progress(f"  📊 Columns: {list(df.columns)}")
            self.log_progress(f"  📏 Shape: {df.shape}")
            
            # Memory cleanup
            del df
            gc.collect()
            
        except Exception as e:
            self.log_progress(f"❌ Failed to save {data_type} to parquet: {e}", "ERROR")
            
    def print_progress_summary(self):
        """Print a summary of current progress"""
        elapsed = time.time() - self.progress['start_time'] if self.progress['start_time'] else 0
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        self.log_progress("📊 " + "="*60, "INFO")
        self.log_progress("📊 PROGRESS SUMMARY", "INFO")
        self.log_progress("📊 " + "="*60, "INFO")
        self.log_progress(f"⏱️  Elapsed Time: {elapsed_str}", "INFO")
        self.log_progress(f"🏢 Locations Processed: {self.progress['locations_processed']}", "INFO")
        self.log_progress(f"🔬 Sensors Processed: {self.progress['sensors_processed']}", "INFO")
        self.log_progress(f"📊 Measurements Collected: {self.progress['measurements_collected']:,}", "INFO")
        self.log_progress(f"❌ Errors Encountered: {self.progress['errors_encountered']}", "INFO")
        self.log_progress("📊 " + "="*60, "INFO")
        
    def run_nepal_ingestion(self):
        """Run the complete Nepal data ingestion process"""
        self.progress['start_time'] = time.time()
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Step 1: Get all Nepal locations
            self.log_progress("🚀 STEP 1: Fetching Nepal locations...", "INFO")
            locations = self.get_nepal_locations()
            
            if not locations:
                self.log_progress("❌ No locations found for Nepal. Aborting.", "ERROR")
                return
                
            # Save locations
            self.save_to_parquet_partitioned(locations, "locations", timestamp)
            
            # Step 2: Extract sensors from locations
            self.log_progress("🚀 STEP 2: Extracting sensors from locations...", "INFO")
            sensors = self.extract_sensors_from_locations(locations)
            
            if not sensors:
                self.log_progress("❌ No sensors found in Nepal locations. Aborting.", "ERROR")
                return
                
            # Save sensors
            self.save_to_parquet_partitioned(sensors, "sensors", timestamp)
            
            # Step 3: Get measurements for each sensor
            self.log_progress("🚀 STEP 3: Collecting measurements from all sensors...", "INFO")
            self.log_progress(f"🎯 Total sensors to process: {len(sensors)}", "INFO")
            
            all_measurements = []
            batch_count = 0
            
            for i, sensor_info in enumerate(sensors, 1):
                self.log_progress(f"🔄 Processing sensor {i}/{len(sensors)}", "INFO")
                
                measurements = self.get_sensor_measurements(sensor_info)
                all_measurements.extend(measurements)
                
                # Save measurements in batches to manage memory
                if len(all_measurements) >= self.batch_size:
                    batch_count += 1
                    batch_timestamp = f"{timestamp}_batch_{batch_count:03d}"
                    self.save_to_parquet_partitioned(all_measurements, "measurements", batch_timestamp)
                    all_measurements = []  # Reset for next batch
                    
                # Print progress every 10 sensors
                if i % 10 == 0:
                    self.print_progress_summary()
                    
            # Save final measurements batch
            if all_measurements:
                batch_count += 1
                batch_timestamp = f"{timestamp}_batch_{batch_count:03d}"
                self.save_to_parquet_partitioned(all_measurements, "measurements", batch_timestamp)
                
            # Final summary
            self.log_progress("🎉 " + "="*80, "SUCCESS")
            self.log_progress("🎉 NEPAL DATA INGESTION COMPLETED SUCCESSFULLY!", "SUCCESS")
            self.log_progress("🎉 " + "="*80, "SUCCESS")
            self.print_progress_summary()
            
        except KeyboardInterrupt:
            self.log_progress("🛑 Ingestion interrupted by user", "WARNING")
            self.print_progress_summary()
        except Exception as e:
            self.log_progress(f"💥 Fatal error during ingestion: {e}", "ERROR")
            self.print_progress_summary()
            raise


def main():
    """Main function for Nepal ingestion"""
    # Configuration
    API_KEY = "9a1e83c1ac07424296a0e90aa95153fe13c7c8b8996d5f4d712c95484d2d1588"
    OUTPUT_DIR = "data/nepal_data"
    
    # Create ingester and run
    ingester = NepalOpenAQIngester(API_KEY, OUTPUT_DIR)
    ingester.run_nepal_ingestion()


if __name__ == "__main__":
    main()