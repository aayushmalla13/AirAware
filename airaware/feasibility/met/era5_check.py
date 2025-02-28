"""ERA5 reanalysis data availability checker for Kathmandu Valley."""

import json
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import xarray as xr
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Kathmandu Valley bounding box
KATHMANDU_BBOX = {
    "north": 27.8,
    "south": 27.6, 
    "east": 85.5,
    "west": 85.1,
}

# Required ERA5 variables for air quality modeling
REQUIRED_VARIABLES = {
    "u10": "10 metre U wind component",
    "v10": "10 metre V wind component", 
    "t2m": "2 metre temperature",
    "blh": "Boundary layer height",
}

# Mapping to ERA5 CDS API variable names
ERA5_CDS_VARIABLES = {
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "t2m": "2m_temperature", 
    "blh": "boundary_layer_height"
}


class ERA5ValidationResult(BaseModel):
    """Result of ERA5 data validation."""
    date_checked: datetime
    bbox: Dict[str, float]
    variables_requested: List[str]
    variables_found: List[str]
    missing_variables: List[str]
    time_dimension_ok: bool
    expected_time_steps: int
    actual_time_steps: int
    data_shape: Optional[Tuple[int, ...]] = None
    file_size_mb: Optional[float] = None
    validation_errors: List[str] = Field(default_factory=list)
    ok_vars: bool = False
    ok_time24: bool = False
    overall_ok: bool = False


class LocalERA5Summary(BaseModel):
    """Summary of local ERA5 data."""
    files_found: List[str] = Field(default_factory=list)
    date_coverage: List[datetime] = Field(default_factory=list)
    variables_available: List[str] = Field(default_factory=list)
    total_size_mb: float = 0.0


class ERA5Checker:
    """ERA5 reanalysis data availability checker."""
    
    def __init__(
        self,
        cds_api_key: Optional[str] = None,
        use_local: bool = False,
        data_manifest_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """Initialize ERA5 checker.
        
        Args:
            cds_api_key: Copernicus CDS API key (token only)
            use_local: Whether to use only local data
            data_manifest_path: Path to local data manifest
            cache_dir: Directory for temporary downloads
        """
        self.cds_api_key = cds_api_key
        self.use_local = use_local
        self.data_manifest_path = data_manifest_path or Path("data/artifacts/local_data_manifest.json")
        self.cache_dir = cache_dir or Path("data/artifacts/.era5_cache")
        
        # Setup CDS API client
        if not use_local and cds_api_key:
            self._setup_cds_client()
        else:
            self.cds_client = None
            
        if use_local:
            logger.info("ERA5 checker initialized in local-only mode")
        elif not cds_api_key:
            logger.warning("No CDS API key provided - local mode only")
    
    def _setup_cds_client(self) -> None:
        """Setup CDS API client with error handling for common issues."""
        try:
            import cdsapi
            
            # Create client with explicit configuration
            self.cds_client = cdsapi.Client(
                key=self.cds_api_key,
                verify=True,
                timeout=300,
                retry_max=3,
            )
            
            logger.info("CDS API client initialized successfully")
            
        except ImportError:
            raise ImportError(
                "cdsapi package not found. Install with: pip install cdsapi"
            )
        except Exception as e:
            if "licence" in str(e).lower():
                raise ValueError(
                    "ERA5 licence not accepted. Visit https://cds.climate.copernicus.eu/cdsapp#!/terms/licence-to-use-copernicus-products and accept the licence terms."
                )
            elif "key" in str(e).lower() or "auth" in str(e).lower():
                raise ValueError(
                    f"CDS API authentication failed: {e}. Check your CDSAPI_KEY token."
                )
            else:
                raise RuntimeError(f"Failed to setup CDS client: {e}")
    
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
    
    def _analyze_local_era5_data(self) -> LocalERA5Summary:
        """Analyze local ERA5 data from manifest."""
        manifest = self._load_local_manifest()
        summary = LocalERA5Summary()
        
        if not manifest.get("files"):
            return summary
        
        # Find ERA5-related files
        era5_files = [
            f for f in manifest["files"]
            if "era5" in f.get("dataset_types", []) or "era5" in f.get("path", "").lower()
        ]
        
        logger.info(f"Found {len(era5_files)} ERA5-related files in local data")
        
        total_size = 0
        variables_found = set()
        
        for file_info in era5_files:
            file_path = file_info.get("path", "")
            file_size = file_info.get("size_mb", 0)
            
            summary.files_found.append(file_path)
            total_size += file_size
            
            # Try to extract variables from schema validation if available
            validation = file_info.get("schema_validation", {})
            if validation.get("validated"):
                columns = validation.get("columns", [])
                for var in REQUIRED_VARIABLES.keys():
                    if var in columns:
                        variables_found.add(var)
            
            # Try to extract date from filename
            try:
                if "era5" in file_path and any(c.isdigit() for c in file_path):
                    # Look for date patterns in filename
                    parts = file_path.split("/")
                    for part in parts:
                        if len(part) >= 8 and part.isdigit():
                            date_str = part[:8]  # YYYYMMDD
                            date_obj = datetime.strptime(date_str, "%Y%m%d")
                            summary.date_coverage.append(date_obj)
                            break
            except Exception:
                continue
        
        summary.variables_available = list(variables_found)
        summary.total_size_mb = total_size
        
        # Remove duplicate dates and sort
        summary.date_coverage = sorted(list(set(summary.date_coverage)))
        
        logger.info(f"Local ERA5 summary: {len(summary.files_found)} files, {total_size:.1f} MB")
        return summary
    
    def check_era5_day(
        self,
        date: datetime,
        bbox: Optional[Dict[str, float]] = None,
        variables: Optional[List[str]] = None,
    ) -> ERA5ValidationResult:
        """Check ERA5 data availability for a specific day.
        
        Args:
            date: Date to check
            bbox: Bounding box dict with north, south, east, west
            variables: List of variables to check
            
        Returns:
            Validation result
        """
        bbox = bbox or KATHMANDU_BBOX
        variables = variables or list(REQUIRED_VARIABLES.keys())
        
        if self.use_local:
            return self._check_local_era5_day(date, bbox, variables)
        
        if not self.cds_client:
            raise ValueError("No CDS client available and not in local mode")
        
        logger.info(f"Checking ERA5 data for {date.date()} via CDS API")
        
        # Create temporary download
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temp_file = self.cache_dir / f"era5_test_{date.strftime('%Y%m%d')}.nc"
        
        try:
            # Download single day of data
            self._download_era5_day(date, bbox, variables, temp_file)
            
            # Validate downloaded file
            result = self._validate_era5_file(temp_file, date, bbox, variables)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to check ERA5 data for {date.date()}: {e}")
            
            return ERA5ValidationResult(
                date_checked=date,
                bbox=bbox,
                variables_requested=variables,
                variables_found=[],
                missing_variables=variables,
                time_dimension_ok=False,
                expected_time_steps=24,
                actual_time_steps=0,
                validation_errors=[str(e)],
                ok_vars=False,
                ok_time24=False,
                overall_ok=False,
            )
        
        finally:
            # Cleanup temporary file
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
    
    def _check_local_era5_day(
        self,
        date: datetime,
        bbox: Dict[str, float],
        variables: List[str],
    ) -> ERA5ValidationResult:
        """Check ERA5 data from local files."""
        summary = self._analyze_local_era5_data()
        
        # Check if date is covered in local data
        date_covered = date.date() in [d.date() for d in summary.date_coverage]
        
        # Check variable availability
        variables_found = [v for v in variables if v in summary.variables_available]
        missing_variables = [v for v in variables if v not in variables_found]
        
        # Assume hourly data if date is covered
        time_steps = 24 if date_covered else 0
        
        result = ERA5ValidationResult(
            date_checked=date,
            bbox=bbox,
            variables_requested=variables,
            variables_found=variables_found,
            missing_variables=missing_variables,
            time_dimension_ok=time_steps == 24,
            expected_time_steps=24,
            actual_time_steps=time_steps,
            ok_vars=len(missing_variables) == 0,
            ok_time24=time_steps == 24,
            overall_ok=len(missing_variables) == 0 and time_steps == 24,
        )
        
        if not date_covered:
            result.validation_errors.append(f"Date {date.date()} not found in local data")
        
        if missing_variables:
            result.validation_errors.append(f"Missing variables: {missing_variables}")
        
        logger.info(f"Local ERA5 check for {date.date()}: ok_vars={result.ok_vars}, ok_time24={result.ok_time24}")
        return result
    
    def _download_era5_day(
        self,
        date: datetime,
        bbox: Dict[str, float],
        variables: List[str],
        output_file: Path,
    ) -> None:
        """Download single day of ERA5 data via CDS API."""
        
        # Convert short variable names to CDS API names
        cds_variables = [ERA5_CDS_VARIABLES.get(var, var) for var in variables]
        
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': cds_variables,
            'year': str(date.year),
            'month': f"{date.month:02d}",
            'day': f"{date.day:02d}",
            'time': [f"{h:02d}:00" for h in range(24)],  # All 24 hours
            'area': [
                bbox['north'],
                bbox['west'], 
                bbox['south'],
                bbox['east'],
            ],
        }
        
        logger.info(f"Downloading ERA5 data for {date.date()} to {output_file}")
        
        try:
            self.cds_client.retrieve(
                'reanalysis-era5-single-levels',
                request,
                str(output_file)
            )
            
            logger.info(f"Successfully downloaded ERA5 data to {output_file}")
            
        except Exception as e:
            if "licence" in str(e).lower():
                raise ValueError(
                    "ERA5 licence not accepted. Visit your CDS profile and accept the licence terms."
                )
            elif "key" in str(e).lower() or "authentication" in str(e).lower():
                raise ValueError(
                    "CDS API authentication failed. Check your CDSAPI_KEY token."
                )
            elif "request" in str(e).lower():
                raise ValueError(f"Invalid CDS request: {e}")
            else:
                raise RuntimeError(f"CDS download failed: {e}")
    
    def _validate_era5_file(
        self,
        file_path: Path,
        date: datetime,
        bbox: Dict[str, float],
        expected_variables: List[str],
    ) -> ERA5ValidationResult:
        """Validate ERA5 NetCDF file."""
        
        try:
            # Load with xarray
            ds = xr.open_dataset(file_path)
            
            # Check variables
            available_vars = list(ds.data_vars.keys())
            variables_found = [v for v in expected_variables if v in available_vars]
            missing_variables = [v for v in expected_variables if v not in available_vars]
            
            # Check time dimension
            time_steps = len(ds.time) if 'time' in ds.dims else 0
            
            # Get file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Validation checks
            ok_vars = len(missing_variables) == 0
            ok_time24 = time_steps == 24
            
            errors = []
            if not ok_vars:
                errors.append(f"Missing variables: {missing_variables}")
            if not ok_time24:
                errors.append(f"Expected 24 time steps, got {time_steps}")
            
            result = ERA5ValidationResult(
                date_checked=date,
                bbox=bbox,
                variables_requested=expected_variables,
                variables_found=variables_found,
                missing_variables=missing_variables,
                time_dimension_ok=ok_time24,
                expected_time_steps=24,
                actual_time_steps=time_steps,
                data_shape=tuple(ds.dims.values()) if hasattr(ds, 'dims') else None,
                file_size_mb=round(file_size_mb, 2),
                validation_errors=errors,
                ok_vars=ok_vars,
                ok_time24=ok_time24,
                overall_ok=ok_vars and ok_time24,
            )
            
            ds.close()
            
            logger.info(f"ERA5 validation: ok_vars={ok_vars}, ok_time24={ok_time24}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to validate ERA5 file {file_path}: {e}")
            
            return ERA5ValidationResult(
                date_checked=date,
                bbox=bbox,
                variables_requested=expected_variables,
                variables_found=[],
                missing_variables=expected_variables,
                time_dimension_ok=False,
                expected_time_steps=24,
                actual_time_steps=0,
                validation_errors=[f"File validation failed: {e}"],
                ok_vars=False,
                ok_time24=False,
                overall_ok=False,
            )
    
    def check_era5_availability(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sample_days: int = 3,
    ) -> List[ERA5ValidationResult]:
        """Check ERA5 availability over a date range.
        
        Args:
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: now)
            sample_days: Number of sample days to check
            
        Returns:
            List of validation results
        """
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        logger.info(f"Checking ERA5 availability from {start_date.date()} to {end_date.date()}")
        
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
                result = self.check_era5_day(date)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to check ERA5 for {date.date()}: {e}")
        
        success_count = sum(1 for r in results if r.overall_ok)
        logger.info(f"ERA5 availability check: {success_count}/{len(results)} days successful")
        
        return results
