#!/usr/bin/env python3
"""
NASA Earthdata PM2.5 ETL Pipeline
Downloads real PM2.5 data from NASA MERRA2_CNN_HAQAST_PM25 dataset
"""

import os
import requests
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NASAEarthdataPM25ETL:
    """NASA Earthdata PM2.5 ETL Pipeline"""
    
    def __init__(self):
        self.earthdata_token = os.getenv('EARTHDATA_TOKEN')
        if not self.earthdata_token:
            raise ValueError("EARTHDATA_TOKEN not found in environment variables")
        
        self.headers = {
            'Authorization': f'Bearer {self.earthdata_token}',
            'Content-Type': 'application/json'
        }
        
        # Nepal bounding box
        self.nepal_bbox = [80.0, 26.0, 89.0, 31.0]  # [west, south, east, north]
        
        # Major cities in Nepal for PM2.5 extraction
        self.nepal_cities = {
            'kathmandu': {'lat': 27.7172, 'lon': 85.3240, 'name': 'Kathmandu'},
            'pokhara': {'lat': 28.2096, 'lon': 83.9856, 'name': 'Pokhara'},
            'biratnagar': {'lat': 26.4525, 'lon': 87.2718, 'name': 'Biratnagar'},
            'lalitpur': {'lat': 27.6710, 'lon': 85.3258, 'name': 'Lalitpur'},
            'bhaktapur': {'lat': 27.6710, 'lon': 85.4298, 'name': 'Bhaktapur'}
        }
    
    def search_granules(self, start_date: str, end_date: str) -> List[Dict]:
        """Search for NASA PM2.5 granules in date range"""
        logger.info(f"🔍 Searching NASA granules from {start_date} to {end_date}")
        
        url = 'https://cmr.earthdata.nasa.gov/search/granules.json'
        params = {
            'short_name': 'MERRA2_CNN_HAQAST_PM25',
            'page_size': 2000,
            'temporal': f'{start_date}T00:00:00Z,{end_date}T23:59:59Z',
            'bounding_box': f'{self.nepal_bbox[0]},{self.nepal_bbox[1]},{self.nepal_bbox[2]},{self.nepal_bbox[3]}'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            granules = data.get('feed', {}).get('entry', [])
            
            logger.info(f"✅ Found {len(granules)} granules")
            return granules
            
        except Exception as e:
            logger.error(f"❌ Error searching granules: {e}")
            return []
    
    def download_granule(self, granule_url: str, output_path: str) -> bool:
        """Download a single granule"""
        try:
            logger.info(f"📥 Downloading: {granule_url}")
            
            response = requests.get(granule_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"✅ Downloaded: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return False
    
    def extract_pm25_data(self, netcdf_path: str) -> pd.DataFrame:
        """Extract PM2.5 data from NetCDF file"""
        logger.info(f"🔧 Extracting PM2.5 data from {netcdf_path}")
        
        try:
            # Open NetCDF file
            ds = xr.open_dataset(netcdf_path)
            
            # Extract PM2.5 variable (usually named 'PM25' or similar)
            pm25_var = None
            for var_name in ds.data_vars:
                if 'pm25' in var_name.lower() or 'pm2.5' in var_name.lower():
                    pm25_var = var_name
                    break
            
            if pm25_var is None:
                # Try common PM2.5 variable names
                possible_names = ['PM25', 'pm25', 'PM2.5', 'pm2_5', 'PM2_5']
                for name in possible_names:
                    if name in ds.data_vars:
                        pm25_var = name
                        break
            
            if pm25_var is None:
                logger.error(f"❌ PM2.5 variable not found in {netcdf_path}")
                logger.info(f"Available variables: {list(ds.data_vars)}")
                return pd.DataFrame()
            
            logger.info(f"📊 Using PM2.5 variable: {pm25_var}")
            
            # Extract data for Nepal cities
            city_data = []
            
            for city_key, city_info in self.nepal_cities.items():
                try:
                    # Find nearest grid point to city
                    lat_idx = np.argmin(np.abs(ds.lat.values - city_info['lat']))
                    lon_idx = np.argmin(np.abs(ds.lon.values - city_info['lon']))
                    
                    # Extract PM2.5 time series for this city
                    pm25_series = ds[pm25_var].isel(lat=lat_idx, lon=lon_idx)
                    
                    # Convert to DataFrame
                    city_df = pm25_series.to_dataframe().reset_index()
                    city_df['city'] = city_info['name']
                    city_df['city_key'] = city_key
                    city_df['latitude'] = city_info['lat']
                    city_df['longitude'] = city_info['lon']
                    city_df['pm25'] = city_df[pm25_var]
                    
                    # Keep only necessary columns
                    city_df = city_df[['time', 'city', 'city_key', 'latitude', 'longitude', 'pm25']]
                    city_df = city_df.rename(columns={'time': 'datetime_utc'})
                    
                    city_data.append(city_df)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error extracting data for {city_info['name']}: {e}")
                    continue
            
            if city_data:
                combined_df = pd.concat(city_data, ignore_index=True)
                logger.info(f"✅ Extracted {len(combined_df)} PM2.5 records")
                return combined_df
            else:
                logger.error("❌ No city data extracted")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"❌ Error extracting PM2.5 data: {e}")
            return pd.DataFrame()
    
    def download_and_process_granules(self, granules: List[Dict], output_dir: str) -> pd.DataFrame:
        """Download and process multiple granules"""
        logger.info(f"🔄 Processing {len(granules)} granules")
        
        all_data = []
        successful_downloads = 0
        
        for i, granule in enumerate(granules):
            try:
                # Get download URL
                download_links = granule.get('links', [])
                download_url = None
                
                for link in download_links:
                    if link.get('rel') == 'http://esipfed.org/ns/fedsearch/1.1/data#':
                        download_url = link.get('href')
                        break
                
                if not download_url:
                    logger.warning(f"⚠️ No download URL found for granule {i+1}")
                    continue
                
                # Download granule
                granule_filename = f"granule_{i+1:04d}.nc4"
                granule_path = os.path.join(output_dir, granule_filename)
                
                if self.download_granule(download_url, granule_path):
                    # Extract PM2.5 data
                    pm25_data = self.extract_pm25_data(granule_path)
                    
                    if not pm25_data.empty:
                        all_data.append(pm25_data)
                        successful_downloads += 1
                    
                    # Clean up downloaded file to save space
                    os.remove(granule_path)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error processing granule {i+1}: {e}")
                continue
        
        logger.info(f"✅ Successfully processed {successful_downloads}/{len(granules)} granules")
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data = combined_data.sort_values('datetime_utc')
            logger.info(f"📊 Total PM2.5 records: {len(combined_data)}")
            return combined_data
        else:
            logger.error("❌ No data extracted from any granules")
            return pd.DataFrame()
    
    def run_etl(self, start_date: str, end_date: str, output_path: str) -> bool:
        """Run the complete ETL pipeline"""
        logger.info(f"🚀 Starting NASA Earthdata PM2.5 ETL")
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        logger.info(f"📍 Coverage: Nepal ({len(self.nepal_cities)} cities)")
        
        try:
            # Create output directory
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Search for granules
            granules = self.search_granules(start_date, end_date)
            
            if not granules:
                logger.error("❌ No granules found")
                return False
            
            # Download and process granules
            pm25_data = self.download_and_process_granules(granules, output_dir)
            
            if pm25_data.empty:
                logger.error("❌ No PM2.5 data extracted")
                return False
            
            # Add metadata
            pm25_data['data_source'] = 'nasa_earthdata_merra2_cnn_haqast'
            pm25_data['parameter'] = 'pm25'
            pm25_data['unit'] = 'µg/m³'
            pm25_data['quality'] = 'satellite_reanalysis'
            
            # Save to Parquet
            pm25_data.to_parquet(output_path, index=False)
            
            logger.info(f"✅ ETL completed successfully!")
            logger.info(f"📁 Data saved to: {output_path}")
            logger.info(f"📊 Records: {len(pm25_data):,}")
            logger.info(f"📅 Date range: {pm25_data['datetime_utc'].min()} to {pm25_data['datetime_utc'].max()}")
            logger.info(f"🏙️ Cities: {pm25_data['city'].nunique()}")
            logger.info(f"📈 PM2.5 range: {pm25_data['pm25'].min():.1f} to {pm25_data['pm25'].max():.1f} µg/m³")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ETL failed: {e}")
            return False

def main():
    """Main function"""
    print("🌍 NASA Earthdata PM2.5 ETL Pipeline")
    print("=" * 50)
    
    try:
        etl = NASAEarthdataPM25ETL()
        
        # Download 3 months of data (January-March 2024)
        start_date = "2024-01-01"
        end_date = "2024-03-31"
        output_path = "data/interim/nasa_pm25_nepal.parquet"
        
        success = etl.run_etl(start_date, end_date, output_path)
        
        if success:
            print(f"\n✅ NASA PM2.5 ETL completed successfully!")
            print(f"📁 Data saved to: {output_path}")
        else:
            print(f"\n❌ NASA PM2.5 ETL failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
