#!/usr/bin/env python3
"""
OpenAQ Data Ingestion Script

This script fetches air quality data from the OpenAQ API for Nepal, India, and China,
including all their locations, sensors, and measurement data, then saves it in parquet format.

Country IDs:
- Nepal: 145
- India: 9  
- China: 10

Author: Generated for AirQuality4 project
Date: 2025-09-22
"""

import requests
import pandas as pd
import logging
import time
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional


class OpenAQIngester:
    """OpenAQ API data ingestion class"""
    
    def __init__(self, api_key: str, output_dir: str = "data"):
        """
        Initialize the OpenAQ ingester.
        
        Args:
            api_key: OpenAQ API key
            output_dir: Directory to save output files
        """
        self.api_key = api_key
        self.base_url = "https://api.openaq.org/v3"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Country IDs for Nepal, India, and China
        self.target_countries = {
            "Nepal": 145,
            "India": 9,
            "China": 10
        }
        
        # Setup headers
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # Setup logging
        self.setup_logging()
        
        # Rate limiting
        self.request_delay = 1.0  # 1 second between requests
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"openaq_ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a request to the OpenAQ API with error handling.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            Response JSON or None if failed
        """
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed for {url}: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON response from {url}: {e}")
            return None
            
    def get_all_locations(self, country_id: int, country_name: str) -> List[Dict]:
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
        
        self.logger.info(f"Fetching locations for {country_name} (ID: {country_id})...")
        
        while True:
            params = {
                "countries_id": country_id,
                "page": page,
                "limit": limit
            }
            
            response = self.make_request("locations", params)
            if not response:
                break
                
            results = response.get("results", [])
            if not results:
                break
                
            locations.extend(results)
            self.logger.info(f"  Page {page}: {len(results)} locations fetched")
            
            # Check if we have more pages
            meta = response.get("meta", {})
            if len(results) < limit:
                break
                
            page += 1
            
        self.logger.info(f"Total locations for {country_name}: {len(locations)}")
        return locations
        
    def extract_sensors_from_locations(self, locations: List[Dict]) -> List[Dict]:
        """
        Extract sensor information from location data.
        
        Args:
            locations: List of location dictionaries
            
        Returns:
            List of sensor dictionaries with location context
        """
        sensors = []
        
        for location in locations:
            location_sensors = location.get("sensors", [])
            for sensor in location_sensors:
                sensor_info = {
                    "sensor_id": sensor.get("id"),
                    "sensor_name": sensor.get("name"),
                    "parameter": sensor.get("parameter", {}),
                    "location_id": location.get("id"),
                    "location_name": location.get("name"),
                    "location_locality": location.get("locality"),
                    "country": location.get("country", {}),
                    "coordinates": location.get("coordinates", {}),
                    "timezone": location.get("timezone"),
                    "owner": location.get("owner", {}),
                    "provider": location.get("provider", {})
                }
                sensors.append(sensor_info)
                
        return sensors
        
    def get_sensor_measurements(self, sensor_id: int, sensor_info: Dict, 
                              date_from: str = "2015-01-01T00:00:00Z",
                              date_to: Optional[str] = None) -> List[Dict]:
        """
        Get all measurements for a specific sensor.
        
        Args:
            sensor_id: Sensor ID
            sensor_info: Sensor information dictionary
            date_from: Start date (ISO format)
            date_to: End date (ISO format), defaults to current time
            
        Returns:
            List of measurement dictionaries
        """
        if date_to is None:
            date_to = datetime.now(timezone.utc).isoformat()
            
        measurements = []
        page = 1
        limit = 1000
        
        self.logger.info(f"Fetching measurements for sensor {sensor_id} ({sensor_info.get('sensor_name', 'Unknown')})...")
        
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
                })
                
            measurements.extend(results)
            self.logger.info(f"  Page {page}: {len(results)} measurements fetched")
            
            # Check if we have more pages
            if len(results) < limit:
                break
                
            page += 1
            
        self.logger.info(f"Total measurements for sensor {sensor_id}: {len(measurements)}")
        return measurements
        
    def save_to_parquet(self, data: List[Dict], filename: str, description: str):
        """
        Save data to parquet file.
        
        Args:
            data: List of dictionaries to save
            filename: Output filename
            description: Description for logging
        """
        if not data:
            self.logger.warning(f"No data to save for {description}")
            return
            
        try:
            df = pd.DataFrame(data)
            output_path = self.output_dir / filename
            df.to_parquet(output_path, index=False)
            self.logger.info(f"Saved {len(data)} {description} records to {output_path}")
            self.logger.info(f"  Columns: {list(df.columns)}")
            self.logger.info(f"  Shape: {df.shape}")
        except Exception as e:
            self.logger.error(f"Failed to save {description} to parquet: {e}")
            
    def run_full_ingestion(self):
        """Run the complete data ingestion process"""
        self.logger.info("Starting OpenAQ data ingestion...")
        self.logger.info(f"Target countries: {list(self.target_countries.keys())}")
        
        all_locations = []
        all_sensors = []
        all_measurements = []
        
        # Step 1: Get all locations for each country
        for country_name, country_id in self.target_countries.items():
            locations = self.get_all_locations(country_id, country_name)
            all_locations.extend(locations)
            
            # Extract sensors from locations
            sensors = self.extract_sensors_from_locations(locations)
            all_sensors.extend(sensors)
            
        # Save locations and sensors
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_to_parquet(all_locations, f"locations_{timestamp}.parquet", "locations")
        self.save_to_parquet(all_sensors, f"sensors_{timestamp}.parquet", "sensors")
        
        # Step 2: Get measurements for each sensor
        self.logger.info(f"Starting measurement collection for {len(all_sensors)} sensors...")
        
        for i, sensor_info in enumerate(all_sensors, 1):
            sensor_id = sensor_info.get("sensor_id")
            if not sensor_id:
                continue
                
            self.logger.info(f"Processing sensor {i}/{len(all_sensors)}: {sensor_id}")
            
            measurements = self.get_sensor_measurements(sensor_id, sensor_info)
            all_measurements.extend(measurements)
            
            # Save measurements periodically to avoid memory issues
            if len(all_measurements) >= 100000:  # Save every 100k measurements
                batch_filename = f"measurements_batch_{timestamp}_{i}.parquet"
                self.save_to_parquet(all_measurements, batch_filename, f"measurements (batch {i})")
                all_measurements = []  # Reset for next batch
                
        # Save final measurements
        if all_measurements:
            self.save_to_parquet(all_measurements, f"measurements_final_{timestamp}.parquet", "measurements (final)")
            
        self.logger.info("OpenAQ data ingestion completed!")


def main():
    """Main function"""
    # Configuration
    API_KEY = "9a1e83c1ac07424296a0e90aa95153fe13c7c8b8996d5f4d712c95484d2d1588"
    OUTPUT_DIR = "data"
    
    # Create ingester and run
    ingester = OpenAQIngester(API_KEY, OUTPUT_DIR)
    ingester.run_full_ingestion()


if __name__ == "__main__":
    main()