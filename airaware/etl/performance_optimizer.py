"""Performance optimization utilities for ERA5/IMERG ETL pipelines."""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PerformanceMetrics(BaseModel):
    """Performance metrics for ETL operations."""
    operation: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    records_processed: int
    data_size_mb: float
    throughput_records_per_second: float
    memory_peak_mb: Optional[float] = None


class ETLPerformanceOptimizer:
    """Performance optimizer for meteorological data ETL."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
    
    def optimize_parquet_writing(self, df: pd.DataFrame, output_path: Path, 
                                compression: str = 'snappy') -> Dict:
        """Optimized Parquet writing with performance monitoring."""
        start_time = datetime.now()
        
        try:
            # Convert to PyArrow table for better performance
            table = pa.Table.from_pandas(df)
            
            # Optimize schema
            schema = self._optimize_schema(table.schema)
            table = table.cast(schema)
            
            # Write with optimized settings
            pq.write_table(
                table,
                output_path,
                compression=compression,
                use_dictionary=True,  # Better compression for repeated values
                row_group_size=10000,  # Optimal for time series data
                write_statistics=True,  # Enable statistics for faster queries
                use_compliant_nested_type=False  # Better compatibility
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Calculate metrics
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            throughput = len(df) / duration if duration > 0 else 0
            
            metrics = PerformanceMetrics(
                operation="parquet_write",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                records_processed=len(df),
                data_size_mb=file_size_mb,
                throughput_records_per_second=throughput
            )
            
            self.metrics_history.append(metrics)
            
            logger.info(f"Parquet write: {len(df):,} records in {duration:.2f}s "
                       f"({throughput:.0f} rec/s, {file_size_mb:.1f} MB)")
            
            return {
                "success": True,
                "duration_seconds": duration,
                "file_size_mb": file_size_mb,
                "throughput": throughput
            }
            
        except Exception as e:
            logger.error(f"Parquet write failed: {e}")
            return {"success": False, "error": str(e)}
    
    def optimize_netcdf_processing(self, file_path: Path, bbox: Dict,
                                  variables: List[str]) -> Dict:
        """Optimized NetCDF processing with memory management."""
        start_time = datetime.now()
        
        try:
            import xarray as xr
            
            # Open with chunking for memory efficiency
            with xr.open_dataset(file_path, chunks={'time': 24}) as ds:
                # Efficient spatial subsetting
                bbox_data = ds[variables].sel(
                    latitude=slice(bbox['north'], bbox['south']),
                    longitude=slice(bbox['west'], bbox['east'])
                )
                
                # Compute spatial means efficiently
                bbox_means = bbox_data.mean(dim=['latitude', 'longitude'])
                
                # Load into memory only when needed
                result = bbox_means.load()
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Estimate data size
                data_size_mb = file_path.stat().st_size / (1024 * 1024)
                
                logger.info(f"NetCDF processing: {data_size_mb:.1f} MB in {duration:.2f}s")
                
                return {
                    "success": True,
                    "duration_seconds": duration,
                    "data_size_mb": data_size_mb,
                    "variables_processed": len(variables)
                }
                
        except Exception as e:
            logger.error(f"NetCDF processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def optimize_hdf5_processing(self, file_path: Path, bbox: Dict) -> Dict:
        """Optimized HDF5 processing for IMERG data."""
        start_time = datetime.now()
        
        try:
            import h5py
            import numpy as np
            
            with h5py.File(file_path, 'r') as f:
                # Read data efficiently
                precip_data = f['Grid/precipitationCal'][:]
                lat_bounds = f['Grid/lat_bnds'][:]
                lon_bounds = f['Grid/lon_bnds'][:]
                
                # Efficient spatial indexing
                lat_mask = (
                    (lat_bounds[:, 0] <= bbox['north']) & 
                    (lat_bounds[:, 1] >= bbox['south'])
                )
                lon_mask = (
                    (lon_bounds[:, 0] <= bbox['east']) & 
                    (lon_bounds[:, 1] >= bbox['west'])
                )
                
                # Extract only relevant data
                bbox_precip = precip_data[np.ix_(lat_mask, lon_mask)]
                
                # Efficient mean calculation
                valid_data = bbox_precip[bbox_precip >= 0]
                mean_precip = np.mean(valid_data) if len(valid_data) > 0 else 0.0
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                data_size_mb = file_path.stat().st_size / (1024 * 1024)
                
                logger.info(f"HDF5 processing: {data_size_mb:.1f} MB in {duration:.2f}s")
                
                return {
                    "success": True,
                    "duration_seconds": duration,
                    "data_size_mb": data_size_mb,
                    "mean_precipitation": mean_precip,
                    "valid_pixels": len(valid_data)
                }
                
        except Exception as e:
            logger.error(f"HDF5 processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _optimize_schema(self, schema: pa.Schema) -> pa.Schema:
        """Optimize PyArrow schema for better performance."""
        optimized_fields = []
        
        for field in schema:
            name = field.name
            dtype = field.type
            
            # Optimize numeric types
            if pa.types.is_floating(dtype):
                # Use float32 for most meteorological data (sufficient precision)
                if name in ['u10', 'v10', 'wind_speed', 'wind_direction', 
                           't2m_celsius', 'precipitation_mm_hourly']:
                    optimized_fields.append(pa.field(name, pa.float32()))
                else:
                    optimized_fields.append(field)
            elif pa.types.is_integer(dtype):
                # Use appropriate integer sizes
                if name in ['station_id', 'sensor_id']:
                    optimized_fields.append(pa.field(name, pa.int32()))
                else:
                    optimized_fields.append(field)
            else:
                optimized_fields.append(field)
        
        return pa.schema(optimized_fields)
    
    def get_performance_summary(self, operation_type: Optional[str] = None) -> Dict:
        """Get performance summary for operations."""
        if operation_type:
            metrics = [m for m in self.metrics_history if m.operation == operation_type]
        else:
            metrics = self.metrics_history
        
        if not metrics:
            return {"no_data": True}
        
        durations = [m.duration_seconds for m in metrics]
        throughputs = [m.throughput_records_per_second for m in metrics]
        
        return {
            "total_operations": len(metrics),
            "avg_duration_seconds": sum(durations) / len(durations),
            "min_duration_seconds": min(durations),
            "max_duration_seconds": max(durations),
            "avg_throughput": sum(throughputs) / len(throughputs),
            "total_records_processed": sum(m.records_processed for m in metrics),
            "total_data_size_mb": sum(m.data_size_mb for m in metrics)
        }
    
    def optimize_batch_processing(self, date_list: List[datetime], 
                                  batch_size: int = 5) -> List[List[datetime]]:
        """Optimize batch processing for multiple dates."""
        # Group dates into optimal batches
        batches = []
        for i in range(0, len(date_list), batch_size):
            batch = date_list[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Optimized processing: {len(date_list)} dates into {len(batches)} batches")
        return batches
    
    def estimate_processing_time(self, num_records: int, 
                               operation: str = "parquet_write") -> float:
        """Estimate processing time based on historical performance."""
        relevant_metrics = [m for m in self.metrics_history if m.operation == operation]
        
        if not relevant_metrics:
            # Default estimates (records per second)
            default_rates = {
                "parquet_write": 10000,
                "netcdf_processing": 5000,
                "hdf5_processing": 3000
            }
            rate = default_rates.get(operation, 5000)
            return num_records / rate
        
        # Use average throughput from history
        avg_throughput = sum(m.throughput_records_per_second for m in relevant_metrics) / len(relevant_metrics)
        return num_records / avg_throughput if avg_throughput > 0 else 60  # fallback
    
    def clear_metrics_history(self):
        """Clear performance metrics history."""
        self.metrics_history.clear()
        logger.info("Performance metrics history cleared")


class MemoryMonitor:
    """Memory usage monitoring for ETL operations."""
    
    def __init__(self):
        self.peak_memory_mb = 0
        self.current_memory_mb = 0
    
    def start_monitoring(self):
        """Start memory monitoring."""
        try:
            import psutil
            process = psutil.Process()
            self.current_memory_mb = process.memory_info().rss / (1024 * 1024)
            self.peak_memory_mb = self.current_memory_mb
            logger.debug(f"Memory monitoring started: {self.current_memory_mb:.1f} MB")
        except ImportError:
            logger.warning("psutil not available - memory monitoring disabled")
    
    def update_peak_memory(self):
        """Update peak memory usage."""
        try:
            import psutil
            process = psutil.Process()
            current = process.memory_info().rss / (1024 * 1024)
            self.current_memory_mb = current
            
            if current > self.peak_memory_mb:
                self.peak_memory_mb = current
                
        except ImportError:
            pass
    
    def get_memory_stats(self) -> Dict:
        """Get current memory statistics."""
        return {
            "current_memory_mb": self.current_memory_mb,
            "peak_memory_mb": self.peak_memory_mb
        }


