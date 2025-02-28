#!/usr/bin/env python3
"""
Test script for OpenAQ ingestion - runs a limited test version
"""

import sys
sys.path.append('/home/babin/Babin/Personal/AirQuality4/openaq_ingestion')

from ingest_openaq_data import OpenAQIngester
import logging

def test_ingestion():
    """Test the ingestion with limited data"""
    API_KEY = "9a1e83c1ac07424296a0e90aa95153fe13c7c8b8996d5f4d712c95484d2d1588"
    OUTPUT_DIR = "../data/test_data"
    
    # Create ingester
    ingester = OpenAQIngester(API_KEY, OUTPUT_DIR)
    
    # Test with just Nepal first (smaller dataset)
    ingester.target_countries = {"Nepal": 145}
    
    # Modify to limit number of sensors processed
    original_run = ingester.run_full_ingestion
    
    def limited_run():
        """Limited version for testing"""
        ingester.logger.info("Starting LIMITED OpenAQ data ingestion test...")
        
        all_locations = []
        all_sensors = []
        all_measurements = []
        
        # Step 1: Get locations for Nepal only
        for country_name, country_id in ingester.target_countries.items():
            locations = ingester.get_all_locations(country_id, country_name)
            all_locations.extend(locations)
            
            # Extract sensors from locations
            sensors = ingester.extract_sensors_from_locations(locations)
            all_sensors.extend(sensors)
            
        # Save locations and sensors
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ingester.save_to_parquet(all_locations, f"test_locations_{timestamp}.parquet", "locations")
        ingester.save_to_parquet(all_sensors, f"test_sensors_{timestamp}.parquet", "sensors")
        
        # Step 2: Get measurements for only first 2 sensors (for testing)
        limited_sensors = all_sensors[:2]
        ingester.logger.info(f"Testing measurement collection for {len(limited_sensors)} sensors...")
        
        for i, sensor_info in enumerate(limited_sensors, 1):
            sensor_id = sensor_info.get("sensor_id")
            if not sensor_id:
                continue
                
            ingester.logger.info(f"Processing sensor {i}/{len(limited_sensors)}: {sensor_id}")
            
            # Limit measurements to recent data only
            measurements = ingester.get_sensor_measurements(
                sensor_id, 
                sensor_info,
                date_from="2024-01-01T00:00:00Z"  # Only 2024+ data for testing
            )
            all_measurements.extend(measurements)
            
        # Save test measurements
        if all_measurements:
            ingester.save_to_parquet(all_measurements, f"test_measurements_{timestamp}.parquet", "measurements (test)")
            
        ingester.logger.info("Test ingestion completed!")
        
    # Run limited test
    limited_run()

if __name__ == "__main__":
    test_ingestion()