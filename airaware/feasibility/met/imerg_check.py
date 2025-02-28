"""IMERG precipitation data availability checker (optional)."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# IMERG data URLs and parameters
IMERG_BASE_URL = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDL.07"
KATHMANDU_BBOX = {
    "north": 27.8,
    "south": 27.6, 
    "east": 85.5,
    "west": 85.1,
}


class IMERGValidationResult(BaseModel):
    """Result of IMERG data validation."""
    date_checked: datetime
    bbox: Dict[str, float]
    data_available: bool = False
    file_url: Optional[str] = None
    file_size_mb: Optional[float] = None
    access_successful: bool = False
    validation_errors: List[str] = Field(default_factory=list)
    response_time_ms: Optional[float] = None


class IMERGChecker:
    """IMERG precipitation data availability checker."""
    
    def __init__(
        self,
        earthdata_token: Optional[str] = None,
        use_local: bool = False,
    ) -> None:
        """Initialize IMERG checker.
        
        Args:
            earthdata_token: NASA Earthdata bearer token
            use_local: Whether to use only local data
        """
        self.earthdata_token = earthdata_token
        self.use_local = use_local
        
        # Setup session with authentication
        self.session = requests.Session()
        if earthdata_token:
            self.session.headers.update({
                "Authorization": f"Bearer {earthdata_token}",
                "User-Agent": "AirAware-Feasibility/1.0",
            })
        
        if use_local:
            logger.info("IMERG checker initialized in local-only mode")
        elif not earthdata_token:
            logger.warning("No Earthdata token provided - local mode only")
    
    def _construct_imerg_url(self, date: datetime) -> str:
        """Construct IMERG file URL for given date."""
        year = date.year
        month = date.month
        day = date.day
        
        # IMERG filename pattern: 3B-DAY.MS.MRG.3IMERG.YYYYMMDD-S000000-E235959.VXXX.nc4
        filename = f"3B-DAY.MS.MRG.3IMERG.{date.strftime('%Y%m%d')}-S000000-E235959.V07B.nc4"
        
        url = f"{IMERG_BASE_URL}/{year:04d}/{month:02d}/{filename}"
        return url
    
    def check_imerg_day(
        self,
        date: datetime,
        bbox: Optional[Dict[str, float]] = None,
    ) -> IMERGValidationResult:
        """Check IMERG data availability for a specific day.
        
        Args:
            date: Date to check
            bbox: Bounding box (for future spatial subsetting)
            
        Returns:
            Validation result
        """
        bbox = bbox or KATHMANDU_BBOX
        
        if self.use_local:
            return self._check_local_imerg_day(date, bbox)
        
        if not self.earthdata_token:
            return IMERGValidationResult(
                date_checked=date,
                bbox=bbox,
                validation_errors=["No Earthdata token provided"],
            )
        
        logger.info(f"Checking IMERG data for {date.date()}")
        
        # Construct file URL
        file_url = self._construct_imerg_url(date)
        
        try:
            # Make HEAD request to check availability
            start_time = datetime.now()
            response = self.session.head(file_url, timeout=30)
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                # File exists and accessible
                file_size_mb = None
                if 'content-length' in response.headers:
                    file_size_mb = int(response.headers['content-length']) / (1024 * 1024)
                
                result = IMERGValidationResult(
                    date_checked=date,
                    bbox=bbox,
                    data_available=True,
                    file_url=file_url,
                    file_size_mb=round(file_size_mb, 2) if file_size_mb else None,
                    access_successful=True,
                    response_time_ms=round(response_time_ms, 1),
                )
                
                logger.info(f"IMERG data available for {date.date()}")
                return result
                
            elif response.status_code == 401:
                return IMERGValidationResult(
                    date_checked=date,
                    bbox=bbox,
                    validation_errors=["Authentication failed - check Earthdata token"],
                    response_time_ms=round(response_time_ms, 1),
                )
                
            elif response.status_code == 404:
                return IMERGValidationResult(
                    date_checked=date,
                    bbox=bbox,
                    validation_errors=[f"IMERG data not found for {date.date()}"],
                    response_time_ms=round(response_time_ms, 1),
                )
                
            else:
                return IMERGValidationResult(
                    date_checked=date,
                    bbox=bbox,
                    validation_errors=[f"HTTP {response.status_code}: {response.reason}"],
                    response_time_ms=round(response_time_ms, 1),
                )
                
        except requests.exceptions.Timeout:
            return IMERGValidationResult(
                date_checked=date,
                bbox=bbox,
                validation_errors=["Request timeout"],
            )
            
        except requests.exceptions.ConnectionError:
            return IMERGValidationResult(
                date_checked=date,
                bbox=bbox,
                validation_errors=["Connection error"],
            )
            
        except Exception as e:
            return IMERGValidationResult(
                date_checked=date,
                bbox=bbox,
                validation_errors=[f"Unexpected error: {e}"],
            )
    
    def _check_local_imerg_day(
        self,
        date: datetime,
        bbox: Dict[str, float],
    ) -> IMERGValidationResult:
        """Check IMERG data from local files (placeholder)."""
        # In a real implementation, this would check local IMERG files
        # For now, return a placeholder result
        
        return IMERGValidationResult(
            date_checked=date,
            bbox=bbox,
            data_available=False,
            validation_errors=["Local IMERG check not implemented"],
        )
    
    def check_imerg_availability(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sample_days: int = 3,
    ) -> List[IMERGValidationResult]:
        """Check IMERG availability over a date range.
        
        Args:
            start_date: Start date (default: 7 days ago)
            end_date: End date (default: now)
            sample_days: Number of sample days to check
            
        Returns:
            List of validation results
        """
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=7)
        
        logger.info(f"Checking IMERG availability from {start_date.date()} to {end_date.date()}")
        
        # Sample dates across the range
        total_days = (end_date - start_date).days
        if total_days <= sample_days:
            sample_dates = [start_date + timedelta(days=i) for i in range(total_days)]
        else:
            step = total_days // sample_days
            sample_dates = [start_date + timedelta(days=i * step) for i in range(sample_days)]
        
        results = []
        for date in sample_dates:
            try:
                result = self.check_imerg_day(date)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to check IMERG for {date.date()}: {e}")
        
        success_count = sum(1 for r in results if r.data_available and r.access_successful)
        logger.info(f"IMERG availability check: {success_count}/{len(results)} days successful")
        
        return results
    
    def get_availability_summary(
        self,
        results: List[IMERGValidationResult],
    ) -> Dict[str, any]:
        """Generate summary of IMERG availability results.
        
        Args:
            results: List of validation results
            
        Returns:
            Summary dictionary
        """
        if not results:
            return {
                "total_days_checked": 0,
                "successful_days": 0,
                "availability_pct": 0.0,
                "avg_response_time_ms": None,
                "common_errors": [],
            }
        
        successful = [r for r in results if r.data_available and r.access_successful]
        response_times = [r.response_time_ms for r in results if r.response_time_ms is not None]
        
        # Collect all errors
        all_errors = []
        for r in results:
            all_errors.extend(r.validation_errors)
        
        # Count error types
        error_counts = {}
        for error in all_errors:
            error_counts[error] = error_counts.get(error, 0) + 1
        
        # Get most common errors
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "total_days_checked": len(results),
            "successful_days": len(successful),
            "availability_pct": round(len(successful) / len(results) * 100, 1),
            "avg_response_time_ms": round(sum(response_times) / len(response_times), 1) if response_times else None,
            "common_errors": [{"error": error, "count": count} for error, count in common_errors],
        }

