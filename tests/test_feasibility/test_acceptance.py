"""Tests for acceptance criteria validation."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from airaware.feasibility.acceptance import (
    AcceptanceCriteria,
    FeasibilityResult,
    FeasibilityValidator,
)
from airaware.feasibility.openaq_client import StationInfo


class TestAcceptanceCriteria:
    """Test AcceptanceCriteria model."""
    
    def test_default_criteria(self) -> None:
        """Test default acceptance criteria."""
        criteria = AcceptanceCriteria()
        
        assert criteria.min_pm25_stations == 3
        assert criteria.max_distance_km == 25.0
        assert criteria.min_data_span_months == 16
        assert criteria.max_missingness_pct == 30.0
        assert criteria.min_quality_score == 0.7
        assert criteria.required_era5_variables == ["u10", "v10", "t2m", "blh"]
        assert criteria.require_hourly_era5 is True
        assert criteria.require_imerg is False
    
    def test_custom_criteria(self) -> None:
        """Test custom acceptance criteria."""
        criteria = AcceptanceCriteria(
            min_pm25_stations=5,
            max_distance_km=30.0,
            max_missingness_pct=20.0,
            require_imerg=True,
            min_imerg_availability_pct=90.0,
        )
        
        assert criteria.min_pm25_stations == 5
        assert criteria.max_distance_km == 30.0
        assert criteria.max_missingness_pct == 20.0
        assert criteria.require_imerg is True
        assert criteria.min_imerg_availability_pct == 90.0


class TestFeasibilityResult:
    """Test FeasibilityResult model."""
    
    def test_default_result(self) -> None:
        """Test default feasibility result."""
        criteria = AcceptanceCriteria()
        result = FeasibilityResult(criteria_used=criteria)
        
        assert result.overall_pass is False
        assert result.openaq_pass is False
        assert result.era5_pass is False
        assert result.imerg_pass is True  # Default pass when not required
        assert result.pm25_stations_found == 0
        assert result.best_stations == []
        assert isinstance(result.assessment_timestamp, datetime)


class TestFeasibilityValidator:
    """Test FeasibilityValidator functionality."""
    
    def test_initialization_local_only(self) -> None:
        """Test validator initialization in local-only mode."""
        validator = FeasibilityValidator(use_local=True)
        
        assert validator.use_local is True
        assert validator.openaq_client is not None
        assert validator.era5_checker is not None
        assert validator.imerg_checker is not None
    
    def test_initialization_with_keys(self) -> None:
        """Test validator initialization with API keys."""
        validator = FeasibilityValidator(
            openaq_api_key="openaq_key",
            cds_api_key="cds_key",
            earthdata_token="earth_token",
        )
        
        assert validator.use_local is False
    
    @patch('airaware.feasibility.openaq_client.OpenAQClient.get_best_stations')
    def test_validate_openaq_requirements_success(self, mock_get_best: Mock) -> None:
        """Test successful OpenAQ validation."""
        # Mock successful station discovery
        mock_stations = [
            StationInfo(
                station_id=123,
                station_name="Station 1",
                latitude=27.7,
                longitude=85.3,
                distance_km=10.0,
                data_quality_score=0.85,
                missingness_pct=15.0,
                data_span_days=500,
            ),
            StationInfo(
                station_id=456,
                station_name="Station 2", 
                latitude=27.72,
                longitude=85.32,
                distance_km=12.0,
                data_quality_score=0.90,
                missingness_pct=10.0,
                data_span_days=600,
            ),
            StationInfo(
                station_id=789,
                station_name="Station 3",
                latitude=27.71,
                longitude=85.33,
                distance_km=8.0,
                data_quality_score=0.80,
                missingness_pct=20.0,
                data_span_days=480,
            ),
        ]
        mock_get_best.return_value = mock_stations
        
        validator = FeasibilityValidator(use_local=True)
        criteria = AcceptanceCriteria(min_pm25_stations=3)
        
        result = validator.validate_openaq_requirements(criteria)
        
        assert result["pass"] is True
        assert result["stations_found"] == 3
        assert result["stations_qualified"] == 3
        assert len(result["best_stations"]) == 3
        assert result["issues"] == []
    
    @patch('airaware.feasibility.openaq_client.OpenAQClient.get_best_stations')
    def test_validate_openaq_requirements_insufficient_stations(self, mock_get_best: Mock) -> None:
        """Test OpenAQ validation with insufficient stations."""
        # Mock insufficient stations
        mock_stations = [
            StationInfo(
                station_id=123,
                station_name="Station 1",
                latitude=27.7,
                longitude=85.3,
                distance_km=10.0,
                data_quality_score=0.85,
                missingness_pct=15.0,
                data_span_days=500,
            )
        ]
        mock_get_best.return_value = mock_stations
        
        validator = FeasibilityValidator(use_local=True)
        criteria = AcceptanceCriteria(min_pm25_stations=3)
        
        result = validator.validate_openaq_requirements(criteria)
        
        assert result["pass"] is False
        assert result["stations_qualified"] == 1
        assert len(result["issues"]) > 0
        assert "1/3 qualifying" in result["issues"][0]
    
    @patch('airaware.feasibility.openaq_client.OpenAQClient.get_best_stations')
    def test_validate_openaq_requirements_error_handling(self, mock_get_best: Mock) -> None:
        """Test OpenAQ validation error handling."""
        # Mock exception
        mock_get_best.side_effect = Exception("API Error")
        
        validator = FeasibilityValidator(use_local=True)
        criteria = AcceptanceCriteria()
        
        result = validator.validate_openaq_requirements(criteria)
        
        assert result["pass"] is False
        assert result["stations_found"] == 0
        assert len(result["issues"]) > 0
        assert "API Error" in result["issues"][0]
    
    @patch('airaware.feasibility.met.era5_check.ERA5Checker.check_era5_availability')
    def test_validate_era5_requirements_success(self, mock_check: Mock) -> None:
        """Test successful ERA5 validation."""
        from airaware.feasibility.met.era5_check import ERA5ValidationResult
        
        # Mock successful ERA5 results
        mock_results = [
            ERA5ValidationResult(
                date_checked=datetime(2024, 12, 1),
                bbox={},
                variables_requested=["u10", "v10", "t2m", "blh"],
                variables_found=["u10", "v10", "t2m", "blh"],
                missing_variables=[],
                time_dimension_ok=True,
                expected_time_steps=24,
                actual_time_steps=24,
                ok_vars=True,
                ok_time24=True,
                overall_ok=True,
            )
        ]
        mock_check.return_value = mock_results
        
        validator = FeasibilityValidator(use_local=True)
        criteria = AcceptanceCriteria()
        
        result = validator.validate_era5_requirements(criteria)
        
        assert result["pass"] is True
        assert result["variables_available"] == ["u10", "v10", "t2m", "blh"]
        assert result["variables_missing"] == []
        assert result["hourly_ok"] is True
        assert result["issues"] == []
    
    @patch('airaware.feasibility.met.era5_check.ERA5Checker.check_era5_availability')
    def test_validate_era5_requirements_missing_variables(self, mock_check: Mock) -> None:
        """Test ERA5 validation with missing variables."""
        from airaware.feasibility.met.era5_check import ERA5ValidationResult
        
        # Mock results with missing variables
        mock_results = [
            ERA5ValidationResult(
                date_checked=datetime(2024, 12, 1),
                bbox={},
                variables_requested=["u10", "v10", "t2m", "blh"],
                variables_found=["u10", "v10"],
                missing_variables=["t2m", "blh"],
                time_dimension_ok=True,
                expected_time_steps=24,
                actual_time_steps=24,
                ok_vars=False,
                ok_time24=True,
                overall_ok=False,
            )
        ]
        mock_check.return_value = mock_results
        
        validator = FeasibilityValidator(use_local=True)
        criteria = AcceptanceCriteria()
        
        result = validator.validate_era5_requirements(criteria)
        
        assert result["pass"] is False
        assert result["variables_missing"] == ["t2m", "blh"]
        assert len(result["issues"]) > 0
        assert "Missing ERA5 variables" in result["issues"][0]
    
    @patch('airaware.feasibility.met.imerg_check.IMERGChecker.check_imerg_availability')
    def test_validate_imerg_requirements_not_required(self, mock_check: Mock) -> None:
        """Test IMERG validation when not required."""
        validator = FeasibilityValidator(use_local=True)
        criteria = AcceptanceCriteria(require_imerg=False)
        
        result = validator.validate_imerg_requirements(criteria)
        
        assert result["pass"] is True
        assert result["availability_pct"] == 0.0
        assert result["issues"] == []
        # Should not call the checker
        mock_check.assert_not_called()
    
    @patch('airaware.feasibility.met.imerg_check.IMERGChecker.check_imerg_availability')
    def test_validate_imerg_requirements_success(self, mock_check: Mock) -> None:
        """Test successful IMERG validation."""
        from airaware.feasibility.met.imerg_check import IMERGValidationResult
        
        # Mock successful IMERG results
        mock_results = [
            IMERGValidationResult(
                date_checked=datetime(2024, 12, 1),
                bbox={},
                data_available=True,
                access_successful=True,
            ),
            IMERGValidationResult(
                date_checked=datetime(2024, 12, 2),
                bbox={},
                data_available=True,
                access_successful=True,
            ),
        ]
        mock_check.return_value = mock_results
        
        validator = FeasibilityValidator(use_local=True)
        criteria = AcceptanceCriteria(
            require_imerg=True,
            min_imerg_availability_pct=80.0,
        )
        
        result = validator.validate_imerg_requirements(criteria)
        
        assert result["pass"] is True
        assert result["availability_pct"] == 100.0
        assert result["issues"] == []
    
    @patch('airaware.feasibility.acceptance.FeasibilityValidator.validate_openaq_requirements')
    @patch('airaware.feasibility.acceptance.FeasibilityValidator.validate_era5_requirements')
    @patch('airaware.feasibility.acceptance.FeasibilityValidator.validate_imerg_requirements')
    def test_run_full_assessment_pass(
        self,
        mock_imerg: Mock,
        mock_era5: Mock,
        mock_openaq: Mock,
    ) -> None:
        """Test full assessment with passing results."""
        # Mock all validations passing
        mock_openaq.return_value = {
            "pass": True,
            "stations_found": 5,
            "stations_qualified": 3,
            "best_stations": [],
            "issues": [],
        }
        
        mock_era5.return_value = {
            "pass": True,
            "variables_available": ["u10", "v10", "t2m", "blh"],
            "variables_missing": [],
            "hourly_ok": True,
            "sample_results": [],
            "issues": [],
        }
        
        mock_imerg.return_value = {
            "pass": True,
            "availability_pct": 95.0,
            "sample_results": [],
            "issues": [],
        }
        
        validator = FeasibilityValidator(use_local=True)
        result = validator.run_full_assessment()
        
        assert result.overall_pass is True
        assert result.openaq_pass is True
        assert result.era5_pass is True
        assert result.imerg_pass is True
        assert "PASS" in result.summary_message
        assert len(result.recommendations) > 0
    
    @patch('airaware.feasibility.acceptance.FeasibilityValidator.validate_openaq_requirements')
    @patch('airaware.feasibility.acceptance.FeasibilityValidator.validate_era5_requirements')
    @patch('airaware.feasibility.acceptance.FeasibilityValidator.validate_imerg_requirements')
    def test_run_full_assessment_fail(
        self,
        mock_imerg: Mock,
        mock_era5: Mock,
        mock_openaq: Mock,
    ) -> None:
        """Test full assessment with failing results."""
        # Mock validations failing
        mock_openaq.return_value = {
            "pass": False,
            "stations_found": 1,
            "stations_qualified": 1,
            "best_stations": [],
            "issues": ["Insufficient stations"],
        }
        
        mock_era5.return_value = {
            "pass": False,
            "variables_available": ["u10", "v10"],
            "variables_missing": ["t2m", "blh"],
            "hourly_ok": False,
            "sample_results": [],
            "issues": ["Missing variables"],
        }
        
        mock_imerg.return_value = {
            "pass": True,
            "availability_pct": 0.0,
            "sample_results": [],
            "issues": [],
        }
        
        validator = FeasibilityValidator(use_local=True)
        result = validator.run_full_assessment()
        
        assert result.overall_pass is False
        assert result.openaq_pass is False
        assert result.era5_pass is False
        assert "FAIL" in result.summary_message
        assert len(result.recommendations) > 0
    
    def test_generate_summary_pass(self) -> None:
        """Test summary generation for passing assessment."""
        criteria = AcceptanceCriteria()
        result = FeasibilityResult(
            criteria_used=criteria,
            overall_pass=True,
            openaq_pass=True,
            era5_pass=True,
            imerg_pass=True,
            pm25_stations_qualified=5,
            era5_variables_available=["u10", "v10", "t2m", "blh"],
        )
        
        validator = FeasibilityValidator(use_local=True)
        summary = validator._generate_summary(result)
        
        assert "PASS" in summary
        assert "5 qualifying PM₂.₅ stations" in summary
        assert "4" in summary  # 4 ERA5 variables
    
    def test_generate_summary_fail(self) -> None:
        """Test summary generation for failing assessment."""
        criteria = AcceptanceCriteria()
        result = FeasibilityResult(
            criteria_used=criteria,
            overall_pass=False,
            openaq_pass=False,
            era5_pass=False,
            pm25_stations_qualified=1,
            era5_variables_missing=["t2m", "blh"],
            era5_hourly_ok=False,
        )
        
        validator = FeasibilityValidator(use_local=True)
        summary = validator._generate_summary(result)
        
        assert "FAIL" in summary
        assert "1/3" in summary  # Insufficient stations
        assert "Missing ERA5 variables" in summary
        assert "hourly data requirement not met" in summary
    
    def test_generate_recommendations_pass(self) -> None:
        """Test recommendations for passing assessment."""
        criteria = AcceptanceCriteria()
        result = FeasibilityResult(
            criteria_used=criteria,
            overall_pass=True,
        )
        
        validator = FeasibilityValidator(use_local=True)
        recommendations = validator._generate_recommendations(result)
        
        assert len(recommendations) > 0
        assert any("Proceed with AirAware development" in rec for rec in recommendations)
    
    def test_generate_recommendations_fail(self) -> None:
        """Test recommendations for failing assessment."""
        criteria = AcceptanceCriteria()
        result = FeasibilityResult(
            criteria_used=criteria,
            overall_pass=False,
            openaq_pass=False,
            era5_pass=False,
        )
        
        validator = FeasibilityValidator(use_local=True)
        recommendations = validator._generate_recommendations(result)
        
        assert len(recommendations) > 0
        assert any("Expand search radius" in rec for rec in recommendations)
        assert any("ERA5 licence" in rec for rec in recommendations)


