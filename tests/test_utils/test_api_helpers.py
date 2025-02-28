"""Tests for API helper utilities."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from airaware.utils.api_helpers import (
    APIClient,
    APIError,
    APIResponse,
    AuthenticationError,
    CacheConfig,
    RateLimitError,
    RetryConfig,
    create_client,
)


class TestAPIResponse:
    """Test APIResponse model."""
    
    def test_success_response(self) -> None:
        """Test successful response creation."""
        response = APIResponse(
            success=True,
            data={"test": "data"},
            status_code=200,
            response_time_ms=150.0,
        )
        
        assert response.success is True
        assert response.data == {"test": "data"}
        assert response.status_code == 200
        assert response.response_time_ms == 150.0
        assert response.from_cache is False
    
    def test_error_response(self) -> None:
        """Test error response creation."""
        response = APIResponse(
            success=False,
            error="Test error",
            status_code=404,
        )
        
        assert response.success is False
        assert response.error == "Test error"
        assert response.status_code == 404
        assert response.data is None


class TestRetryConfig:
    """Test RetryConfig model."""
    
    def test_default_config(self) -> None:
        """Test default retry configuration."""
        config = RetryConfig()
        
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
        assert config.jitter is True
    
    def test_custom_config(self) -> None:
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
            backoff_factor=3.0,
            jitter=False,
        )
        
        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_factor == 3.0
        assert config.jitter is False


class TestCacheConfig:
    """Test CacheConfig model."""
    
    def test_default_config(self) -> None:
        """Test default cache configuration."""
        config = CacheConfig()
        
        assert config.enabled is True
        assert config.cache_dir == Path("data/artifacts/.api_cache")
        assert config.ttl_hours == 24


class TestAPIClient:
    """Test APIClient functionality."""
    
    def test_initialization(self) -> None:
        """Test client initialization."""
        client = APIClient(
            base_url="https://api.example.com",
            api_key="test_key",
        )
        
        assert client.base_url == "https://api.example.com"
        assert client.api_key == "test_key"
        assert client.rate_limit_delay == 0.5
        assert "X-API-Key" in client.session.headers
        assert client.session.headers["X-API-Key"] == "test_key"
    
    def test_cache_key_generation(self) -> None:
        """Test cache key generation."""
        client = APIClient("https://api.example.com")
        
        key1 = client._get_cache_key("GET", "/test", {"param": "value"})
        key2 = client._get_cache_key("GET", "/test", {"param": "value"})
        key3 = client._get_cache_key("GET", "/test", {"param": "different"})
        
        assert key1 == key2  # Same request = same key
        assert key1 != key3  # Different params = different key
        assert len(key1) == 32  # MD5 hash length
    
    def test_delay_calculation(self) -> None:
        """Test retry delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            backoff_factor=2.0,
            max_delay=10.0,
            jitter=False,
        )
        client = APIClient("https://api.example.com", retry_config=config)
        
        assert client._calculate_delay(0) == 1.0  # 1.0 * 2^0
        assert client._calculate_delay(1) == 2.0  # 1.0 * 2^1
        assert client._calculate_delay(2) == 4.0  # 1.0 * 2^2
        assert client._calculate_delay(10) == 10.0  # Capped at max_delay
    
    @patch('requests.Session.request')
    def test_successful_request(self, mock_request: Mock) -> None:
        """Test successful API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_request.return_value = mock_response
        
        client = APIClient("https://api.example.com")
        response = client.get("/test")
        
        assert response.success is True
        assert response.data == {"data": "test"}
        assert response.status_code == 200
        assert response.from_cache is False
    
    @patch('requests.Session.request')
    def test_http_error_handling(self, mock_request: Mock) -> None:
        """Test HTTP error handling."""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_request.return_value = mock_response
        
        client = APIClient("https://api.example.com")
        response = client.get("/nonexistent")
        
        assert response.success is False
        assert "404" in response.error
    
    @patch('requests.Session.request')
    def test_rate_limit_handling(self, mock_request: Mock) -> None:
        """Test rate limit error handling."""
        # Mock 429 response
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.text = "Rate Limited"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_request.return_value = mock_response
        
        config = RetryConfig(max_retries=1)  # Quick failure
        client = APIClient("https://api.example.com", retry_config=config)
        response = client.get("/test")
        
        assert response.success is False
        assert "429" in response.error or "Rate" in response.error
    
    @patch('requests.Session.request')
    def test_authentication_error(self, mock_request: Mock) -> None:
        """Test authentication error handling."""
        # Mock 401 response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = requests.HTTPError()
        mock_request.return_value = mock_response
        
        client = APIClient("https://api.example.com")
        response = client.get("/test")
        
        assert response.success is False
        assert response.status_code == 401
    
    def test_cache_operations(self) -> None:
        """Test cache save and load operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_config = CacheConfig(
                enabled=True,
                cache_dir=Path(temp_dir),
                ttl_hours=1,
            )
            
            client = APIClient("https://api.example.com", cache_config=cache_config)
            
            # Create test response
            response = APIResponse(
                success=True,
                data={"test": "cached_data"},
                status_code=200,
            )
            
            # Save to cache
            cache_key = "test_key"
            client._save_to_cache(cache_key, response)
            
            # Load from cache
            cached_response = client._load_from_cache(cache_key)
            
            assert cached_response is not None
            assert cached_response.success is True
            assert cached_response.data == {"test": "cached_data"}
            assert cached_response.from_cache is True
    
    def test_cache_disabled(self) -> None:
        """Test behavior when cache is disabled."""
        cache_config = CacheConfig(enabled=False)
        client = APIClient("https://api.example.com", cache_config=cache_config)
        
        response = APIResponse(success=True, data={"test": "data"})
        cache_key = "test_key"
        
        # Save should do nothing
        client._save_to_cache(cache_key, response)
        
        # Load should return None
        cached_response = client._load_from_cache(cache_key)
        assert cached_response is None
    
    def test_clear_cache(self) -> None:
        """Test cache clearing functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_config = CacheConfig(cache_dir=Path(temp_dir))
            client = APIClient("https://api.example.com", cache_config=cache_config)
            
            # Create some cache files
            cache_file1 = Path(temp_dir) / "test1.json"
            cache_file2 = Path(temp_dir) / "test2.json"
            
            cache_file1.write_text('{"test": 1}')
            cache_file2.write_text('{"test": 2}')
            
            # Clear cache
            removed_count = client.clear_cache()
            
            assert removed_count == 2
            assert not cache_file1.exists()
            assert not cache_file2.exists()
    
    def test_get_cache_stats(self) -> None:
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_config = CacheConfig(cache_dir=Path(temp_dir))
            client = APIClient("https://api.example.com", cache_config=cache_config)
            
            # Create test cache file
            cache_file = Path(temp_dir) / "test.json"
            cache_file.write_text('{"test": "data"}')
            
            stats = client.get_cache_stats()
            
            assert stats["enabled"] is True
            assert stats["total_files"] == 1
            assert stats["total_size_mb"] > 0


class TestCreateClient:
    """Test create_client helper function."""
    
    def test_openaq_client(self) -> None:
        """Test OpenAQ client creation."""
        client = create_client("openaq", api_key="test_key")
        
        assert client.base_url == "https://api.openaq.org/v3"
        assert client.api_key == "test_key"
        assert client.rate_limit_delay == 0.5
    
    def test_cds_client(self) -> None:
        """Test CDS client creation."""
        client = create_client("cds", api_key="test_key")
        
        assert client.base_url == "https://cds.climate.copernicus.eu/api"
        assert client.rate_limit_delay == 1.0
    
    def test_earthdata_client(self) -> None:
        """Test Earthdata client creation."""
        client = create_client("earthdata", api_key="test_token")
        
        assert client.base_url == "https://urs.earthdata.nasa.gov"
        assert client.rate_limit_delay == 0.1
    
    def test_unknown_service(self) -> None:
        """Test error for unknown service."""
        with pytest.raises(ValueError, match="Unknown service"):
            create_client("unknown_service")
    
    def test_custom_config(self) -> None:
        """Test client creation with custom configuration."""
        client = create_client(
            "openaq",
            api_key="test_key",
            rate_limit_delay=2.0,
        )
        
        assert client.rate_limit_delay == 2.0

