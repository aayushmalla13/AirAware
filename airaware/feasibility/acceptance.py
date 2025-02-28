"""Acceptance criteria validation for AirAware feasibility assessment."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .openaq_client import OpenAQClient, StationInfo
from .met.era5_check import ERA5Checker, ERA5ValidationResult
from .met.imerg_check import IMERGChecker, IMERGValidationResult

logger = logging.getLogger(__name__)

# Kathmandu Valley coordinates for feasibility checks
KATHMANDU_LAT = 27.7172
KATHMANDU_LON = 85.3240


class AcceptanceCriteria(BaseModel):
    """Acceptance criteria for AirAware feasibility."""
    
    # OpenAQ PM₂.₅ station requirements
    min_pm25_stations: int = Field(default=3, ge=1)
    max_distance_km: float = Field(default=25.0, gt=0)
    min_data_span_months: int = Field(default=16, ge=1)
    max_missingness_pct: float = Field(default=30.0, ge=0, le=100)
    min_quality_score: float = Field(default=0.7, ge=0, le=1)
    
    # ERA5 requirements
    required_era5_variables: List[str] = Field(default=["u10", "v10", "t2m", "blh"])
    require_hourly_era5: bool = Field(default=True)
    
    # IMERG requirements (optional)
    require_imerg: bool = Field(default=False)
    min_imerg_availability_pct: float = Field(default=80.0, ge=0, le=100)
    
    # Geographic constraints
    target_lat: float = Field(default=KATHMANDU_LAT)
    target_lon: float = Field(default=KATHMANDU_LON)


class FeasibilityResult(BaseModel):
    """Result of comprehensive feasibility assessment."""
    
    # Overall results
    overall_pass: bool = False
    assessment_timestamp: datetime = Field(default_factory=datetime.now)
    criteria_used: AcceptanceCriteria
    
    # OpenAQ results
    openaq_pass: bool = False
    pm25_stations_found: int = 0
    pm25_stations_qualified: int = 0
    best_stations: List[StationInfo] = Field(default_factory=list)
    openaq_issues: List[str] = Field(default_factory=list)
    
    # ERA5 results
    era5_pass: bool = False
    era5_variables_available: List[str] = Field(default_factory=list)
    era5_variables_missing: List[str] = Field(default_factory=list)
    era5_hourly_ok: bool = False
    era5_sample_results: List[ERA5ValidationResult] = Field(default_factory=list)
    era5_issues: List[str] = Field(default_factory=list)
    
    # IMERG results (optional)
    imerg_pass: bool = True  # Default pass if not required
    imerg_availability_pct: float = 0.0
    imerg_sample_results: List[IMERGValidationResult] = Field(default_factory=list)
    imerg_issues: List[str] = Field(default_factory=list)
    
    # Summary
    summary_message: str = ""
    recommendations: List[str] = Field(default_factory=list)


class FeasibilityValidator:
    """Validates data availability against acceptance criteria."""
    
    def __init__(
        self,
        openaq_api_key: Optional[str] = None,
        cds_api_key: Optional[str] = None,
        earthdata_token: Optional[str] = None,
        use_local: bool = False,
    ) -> None:
        """Initialize feasibility validator.
        
        Args:
            openaq_api_key: OpenAQ v3 API key
            cds_api_key: Copernicus CDS API key
            earthdata_token: NASA Earthdata token
            use_local: Whether to use only local data
        """
        self.use_local = use_local
        
        # Initialize clients
        self.openaq_client = OpenAQClient(
            api_key=openaq_api_key,
            use_local=use_local,
        )
        
        self.era5_checker = ERA5Checker(
            cds_api_key=cds_api_key,
            use_local=use_local,
        )
        
        self.imerg_checker = IMERGChecker(
            earthdata_token=earthdata_token,
            use_local=use_local,
        )
        
        logger.info(f"Feasibility validator initialized (use_local={use_local})")
    
    def validate_openaq_requirements(
        self,
        criteria: AcceptanceCriteria,
    ) -> Dict[str, any]:
        """Validate OpenAQ PM₂.₅ station requirements.
        
        Args:
            criteria: Acceptance criteria
            
        Returns:
            Validation results dictionary
        """
        logger.info("Validating OpenAQ PM₂.₅ station requirements...")
        
        try:
            # Get best stations meeting criteria
            stations = self.openaq_client.get_best_stations(
                lat=criteria.target_lat,
                lon=criteria.target_lon,
                radius_km=criteria.max_distance_km,
                min_quality_score=criteria.min_quality_score,
                max_missingness_pct=criteria.max_missingness_pct,
                min_data_span_days=criteria.min_data_span_months * 30,
            )
            
            qualified_stations = len(stations)
            openaq_pass = qualified_stations >= criteria.min_pm25_stations
            
            issues = []
            if not openaq_pass:
                issues.append(
                    f"Only {qualified_stations}/{criteria.min_pm25_stations} qualifying PM₂.₅ stations found"
                )
            
            # Check individual station quality
            for station in stations[:5]:  # Report top 5
                if station.missingness_pct > criteria.max_missingness_pct:
                    issues.append(
                        f"Station {station.station_id}: {station.missingness_pct:.1f}% missingness "
                        f"(limit: {criteria.max_missingness_pct}%)"
                    )
                
                if station.data_span_days < criteria.min_data_span_months * 30:
                    issues.append(
                        f"Station {station.station_id}: {station.data_span_days} days data "
                        f"(required: {criteria.min_data_span_months * 30})"
                    )
            
            return {
                "pass": openaq_pass,
                "stations_found": len(stations),
                "stations_qualified": qualified_stations,
                "best_stations": stations[:criteria.min_pm25_stations],
                "issues": issues,
            }
            
        except Exception as e:
            logger.error(f"OpenAQ validation failed: {e}")
            return {
                "pass": False,
                "stations_found": 0,
                "stations_qualified": 0,
                "best_stations": [],
                "issues": [f"OpenAQ validation error: {e}"],
            }
    
    def validate_era5_requirements(
        self,
        criteria: AcceptanceCriteria,
        sample_days: int = 3,
    ) -> Dict[str, any]:
        """Validate ERA5 meteorological data requirements.
        
        Args:
            criteria: Acceptance criteria
            sample_days: Number of days to sample for validation
            
        Returns:
            Validation results dictionary
        """
        logger.info("Validating ERA5 meteorological data requirements...")
        
        try:
            # Check recent ERA5 availability
            end_date = datetime.now() - timedelta(days=2)  # ERA5 has ~2 day lag
            start_date = end_date - timedelta(days=30)
            
            results = self.era5_checker.check_era5_availability(
                start_date=start_date,
                end_date=end_date,
                sample_days=sample_days,
            )
            
            if not results:
                return {
                    "pass": False,
                    "variables_available": [],
                    "variables_missing": criteria.required_era5_variables,
                    "hourly_ok": False,
                    "sample_results": [],
                    "issues": ["No ERA5 validation results obtained"],
                }
            
            # Analyze results
            successful_results = [r for r in results if r.overall_ok]
            success_rate = len(successful_results) / len(results)
            
            # Check variable availability across all samples
            all_variables_found = set()
            hourly_ok_count = 0
            
            for result in results:
                all_variables_found.update(result.variables_found)
                if result.ok_time24:
                    hourly_ok_count += 1
            
            variables_available = list(all_variables_found)
            variables_missing = [
                v for v in criteria.required_era5_variables 
                if v not in variables_available
            ]
            
            hourly_ok = hourly_ok_count >= len(results) // 2  # Majority should have 24 hours
            
            # Overall pass criteria
            era5_pass = (
                len(variables_missing) == 0 and
                (not criteria.require_hourly_era5 or hourly_ok) and
                success_rate >= 0.5  # At least half of samples successful
            )
            
            issues = []
            if variables_missing:
                issues.append(f"Missing ERA5 variables: {variables_missing}")
            
            if criteria.require_hourly_era5 and not hourly_ok:
                issues.append(f"ERA5 hourly requirement not met ({hourly_ok_count}/{len(results)} samples)")
            
            if success_rate < 0.5:
                issues.append(f"Low ERA5 success rate: {success_rate:.1%}")
            
            return {
                "pass": era5_pass,
                "variables_available": variables_available,
                "variables_missing": variables_missing,
                "hourly_ok": hourly_ok,
                "sample_results": results,
                "issues": issues,
            }
            
        except Exception as e:
            logger.error(f"ERA5 validation failed: {e}")
            return {
                "pass": False,
                "variables_available": [],
                "variables_missing": criteria.required_era5_variables,
                "hourly_ok": False,
                "sample_results": [],
                "issues": [f"ERA5 validation error: {e}"],
            }
    
    def validate_imerg_requirements(
        self,
        criteria: AcceptanceCriteria,
        sample_days: int = 3,
    ) -> Dict[str, any]:
        """Validate IMERG precipitation data requirements.
        
        Args:
            criteria: Acceptance criteria
            sample_days: Number of days to sample
            
        Returns:
            Validation results dictionary
        """
        if not criteria.require_imerg:
            return {
                "pass": True,
                "availability_pct": 0.0,
                "sample_results": [],
                "issues": [],
            }
        
        logger.info("Validating IMERG precipitation data requirements...")
        
        try:
            # Check recent IMERG availability
            end_date = datetime.now() - timedelta(days=1)  # IMERG has ~1 day lag
            start_date = end_date - timedelta(days=14)
            
            results = self.imerg_checker.check_imerg_availability(
                start_date=start_date,
                end_date=end_date,
                sample_days=sample_days,
            )
            
            if not results:
                return {
                    "pass": False,
                    "availability_pct": 0.0,
                    "sample_results": [],
                    "issues": ["No IMERG validation results obtained"],
                }
            
            # Calculate availability percentage
            successful_count = sum(1 for r in results if r.data_available and r.access_successful)
            availability_pct = (successful_count / len(results)) * 100
            
            imerg_pass = availability_pct >= criteria.min_imerg_availability_pct
            
            issues = []
            if not imerg_pass:
                issues.append(
                    f"IMERG availability {availability_pct:.1f}% < required {criteria.min_imerg_availability_pct}%"
                )
            
            # Collect common errors
            all_errors = []
            for result in results:
                all_errors.extend(result.validation_errors)
            
            if all_errors:
                error_counts = {}
                for error in all_errors:
                    error_counts[error] = error_counts.get(error, 0) + 1
                
                common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:2]
                for error, count in common_errors:
                    issues.append(f"Common IMERG issue: {error} ({count} times)")
            
            return {
                "pass": imerg_pass,
                "availability_pct": availability_pct,
                "sample_results": results,
                "issues": issues,
            }
            
        except Exception as e:
            logger.error(f"IMERG validation failed: {e}")
            return {
                "pass": False,
                "availability_pct": 0.0,
                "sample_results": [],
                "issues": [f"IMERG validation error: {e}"],
            }
    
    def run_full_assessment(
        self,
        criteria: Optional[AcceptanceCriteria] = None,
    ) -> FeasibilityResult:
        """Run comprehensive feasibility assessment.
        
        Args:
            criteria: Acceptance criteria (uses defaults if None)
            
        Returns:
            Complete feasibility results
        """
        if criteria is None:
            criteria = AcceptanceCriteria()
        
        logger.info("Starting comprehensive feasibility assessment...")
        
        # Initialize result
        result = FeasibilityResult(criteria_used=criteria)
        
        # Validate OpenAQ requirements
        openaq_results = self.validate_openaq_requirements(criteria)
        result.openaq_pass = openaq_results["pass"]
        result.pm25_stations_found = openaq_results["stations_found"]
        result.pm25_stations_qualified = openaq_results["stations_qualified"]
        result.best_stations = openaq_results["best_stations"]
        result.openaq_issues = openaq_results["issues"]
        
        # Validate ERA5 requirements
        era5_results = self.validate_era5_requirements(criteria)
        result.era5_pass = era5_results["pass"]
        result.era5_variables_available = era5_results["variables_available"]
        result.era5_variables_missing = era5_results["variables_missing"]
        result.era5_hourly_ok = era5_results["hourly_ok"]
        result.era5_sample_results = era5_results["sample_results"]
        result.era5_issues = era5_results["issues"]
        
        # Validate IMERG requirements
        imerg_results = self.validate_imerg_requirements(criteria)
        result.imerg_pass = imerg_results["pass"]
        result.imerg_availability_pct = imerg_results["availability_pct"]
        result.imerg_sample_results = imerg_results["sample_results"]
        result.imerg_issues = imerg_results["issues"]
        
        # Overall assessment
        result.overall_pass = (
            result.openaq_pass and 
            result.era5_pass and 
            result.imerg_pass
        )
        
        # Generate summary and recommendations
        result.summary_message = self._generate_summary(result)
        result.recommendations = self._generate_recommendations(result)
        
        # Log final result
        status = "PASS" if result.overall_pass else "FAIL"
        logger.info(f"Feasibility assessment complete: {status}")
        
        return result
    
    def _generate_summary(self, result: FeasibilityResult) -> str:
        """Generate summary message for feasibility result."""
        parts = []
        
        if result.overall_pass:
            parts.append("✅ FEASIBILITY ASSESSMENT: PASS")
            parts.append(f"Found {result.pm25_stations_qualified} qualifying PM₂.₅ stations")
            parts.append(f"ERA5 variables available: {len(result.era5_variables_available)}")
            if result.criteria_used.require_imerg:
                parts.append(f"IMERG availability: {result.imerg_availability_pct:.1f}%")
        else:
            parts.append("❌ FEASIBILITY ASSESSMENT: FAIL")
            
            if not result.openaq_pass:
                parts.append(f"Insufficient PM₂.₅ stations: {result.pm25_stations_qualified}/{result.criteria_used.min_pm25_stations}")
            
            if not result.era5_pass:
                if result.era5_variables_missing:
                    parts.append(f"Missing ERA5 variables: {result.era5_variables_missing}")
                if not result.era5_hourly_ok:
                    parts.append("ERA5 hourly data requirement not met")
            
            if not result.imerg_pass and result.criteria_used.require_imerg:
                parts.append(f"IMERG availability too low: {result.imerg_availability_pct:.1f}%")
        
        return " | ".join(parts)
    
    def _generate_recommendations(self, result: FeasibilityResult) -> List[str]:
        """Generate recommendations based on feasibility results."""
        recommendations = []
        
        if result.overall_pass:
            recommendations.append("Proceed with AirAware development using identified data sources")
            recommendations.append("Consider downloading historical data for model training")
            recommendations.append("Set up automated data ingestion pipelines")
        else:
            if not result.openaq_pass:
                recommendations.append("Expand search radius or lower quality requirements for PM₂.₅ stations")
                recommendations.append("Consider alternative air quality data sources")
            
            if not result.era5_pass:
                if result.era5_variables_missing:
                    recommendations.append("Verify ERA5 licence acceptance and API key configuration")
                recommendations.append("Check CDS API access and try alternative meteorological data")
            
            if not result.imerg_pass and result.criteria_used.require_imerg:
                recommendations.append("Check Earthdata token configuration")
                recommendations.append("Consider making IMERG precipitation data optional")
        
        if not recommendations:
            recommendations.append("Review validation logs for detailed error information")
        
        return recommendations

