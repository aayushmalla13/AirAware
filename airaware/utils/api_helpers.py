"""Common API utilities for AirAware with robust error handling and rate limiting."""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union
import hashlib
import random

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API-related errors."""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass


class APIResponse(BaseModel):
    """Standardized API response wrapper."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    from_cache: bool = False


class RetryConfig(BaseModel):
    """Configuration for retry logic."""
    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=1.0, gt=0)
    max_delay: float = Field(default=60.0, gt=0)
    backoff_factor: float = Field(default=2.0, gt=1)
    jitter: bool = Field(default=True)


class CacheConfig(BaseModel):
    """Configuration for response caching."""
    enabled: bool = Field(default=True)
    cache_dir: Path = Field(default=Path("data/artifacts/.api_cache"))
    ttl_hours: int = Field(default=24, ge=1)


class APIClient:
    """Base API client with retry logic, rate limiting, and caching."""
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        rate_limit_delay: float = 0.5,
    ) -> None:
        """Initialize API client.
        
        Args:
            base_url: Base URL for the API
            api_key: API key for authentication
            retry_config: Retry configuration
            cache_config: Cache configuration
            rate_limit_delay: Minimum delay between requests in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.retry_config = retry_config or RetryConfig()
        self.cache_config = cache_config or CacheConfig()
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0
        
        # Setup cache directory
        if self.cache_config.enabled:
            self.cache_config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup session with default headers
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})
        
        self.session.headers.update({
            "User-Agent": "AirAware-Feasibility/1.0",
            "Accept": "application/json",
        })
    
    def _get_cache_key(self, method: str, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for request."""
        cache_data = {
            "method": method.upper(),
            "url": url,
            "params": params or {},
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self.cache_config.cache_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached response is still valid."""
        if not cache_path.exists():
            return False
        
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return cache_age < timedelta(hours=self.cache_config.ttl_hours)
    
    def _load_from_cache(self, cache_key: str) -> Optional[APIResponse]:
        """Load response from cache if valid with hit tracking."""
        if not self.cache_config.enabled:
            self._track_cache_miss()
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        if not self._is_cache_valid(cache_path):
            self._track_cache_miss()
            return None
        
        try:
            with open(cache_path) as f:
                cached_data = json.load(f)
            
            response = APIResponse(**cached_data)
            response.from_cache = True
            self._track_cache_hit()
            logger.debug(f"Cache hit for key: {cache_key}")
            return response
        
        except Exception as e:
            logger.warning(f"Failed to load cache for key {cache_key}: {e}")
            self._track_cache_miss()
            return None
    
    def _track_cache_hit(self) -> None:
        """Track cache hit for statistics."""
        if not hasattr(self, '_cache_hits'):
            self._cache_hits = 0
        self._cache_hits += 1
    
    def _track_cache_miss(self) -> None:
        """Track cache miss for statistics.""" 
        if not hasattr(self, '_cache_misses'):
            self._cache_misses = 0
        self._cache_misses += 1
    
    def _save_to_cache(self, cache_key: str, response: APIResponse) -> None:
        """Save response to cache."""
        if not self.cache_config.enabled or response.from_cache:
            return
        
        try:
            cache_path = self._get_cache_path(cache_key)
            
            # Don't cache error responses
            if not response.success:
                return
            
            with open(cache_path, 'w') as f:
                json.dump(response.model_dump(), f, indent=2, default=str)
            
            logger.debug(f"Cached response for key: {cache_key}")
        
        except Exception as e:
            logger.warning(f"Failed to cache response for key {cache_key}: {e}")
    
    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt with exponential backoff."""
        delay = self.retry_config.base_delay * (self.retry_config.backoff_factor ** attempt)
        delay = min(delay, self.retry_config.max_delay)
        
        if self.retry_config.jitter:
            # Add random jitter (±25%)
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay = max(0, delay + jitter)
        
        return delay
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        timeout: float = 30.0,
    ) -> APIResponse:
        """Make HTTP request with retry logic and error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Check cache first
        cache_key = self._get_cache_key(method, url, params)
        cached_response = self._load_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Rate limiting
                self._wait_for_rate_limit()
                
                # Make request
                start_time = time.time()
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=data,
                    timeout=timeout,
                )
                response_time_ms = (time.time() - start_time) * 1000
                self.last_request_time = time.time()
                
                # Handle HTTP errors
                if response.status_code == 401:
                    raise AuthenticationError(f"Authentication failed: {response.text}")
                elif response.status_code == 429:
                    raise RateLimitError(f"Rate limit exceeded: {response.text}")
                elif response.status_code >= 400:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    if attempt == self.retry_config.max_retries:
                        raise APIError(error_msg)
                    logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")
                    last_exception = APIError(error_msg)
                    continue
                
                # Parse response
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    response_data = {"raw_text": response.text}
                
                api_response = APIResponse(
                    success=True,
                    data=response_data,
                    status_code=response.status_code,
                    response_time_ms=response_time_ms,
                )
                
                # Cache successful response
                self._save_to_cache(cache_key, api_response)
                
                logger.debug(f"Request successful: {method} {url} ({response_time_ms:.1f}ms)")
                return api_response
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = e
                if attempt == self.retry_config.max_retries:
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Network error on attempt {attempt + 1}: {e}, retrying in {delay:.1f}s")
                time.sleep(delay)
                
            except RateLimitError as e:
                last_exception = e
                if attempt == self.retry_config.max_retries:
                    break
                
                # Longer delay for rate limits
                delay = 60 + self._calculate_delay(attempt)
                logger.warning(f"Rate limited on attempt {attempt + 1}, waiting {delay:.1f}s")
                time.sleep(delay)
                
            except AuthenticationError as e:
                # Don't retry auth errors
                return APIResponse(
                    success=False,
                    error=str(e),
                    status_code=401,
                )
                
            except Exception as e:
                last_exception = e
                if attempt == self.retry_config.max_retries:
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}, retrying in {delay:.1f}s")
                time.sleep(delay)
        
        # All retries failed
        error_msg = f"Request failed after {self.retry_config.max_retries + 1} attempts: {last_exception}"
        logger.error(error_msg)
        
        return APIResponse(
            success=False,
            error=error_msg,
            status_code=getattr(last_exception, 'response', {}).get('status_code'),
        )
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        timeout: float = 30.0,
    ) -> APIResponse:
        """Make GET request."""
        return self._make_request("GET", endpoint, params=params, timeout=timeout)
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        timeout: float = 30.0,
    ) -> APIResponse:
        """Make POST request."""
        return self._make_request("POST", endpoint, params=params, data=data, timeout=timeout)
    
    def clear_cache(self) -> int:
        """Clear all cached responses.
        
        Returns:
            Number of cache files removed
        """
        if not self.cache_config.enabled or not self.cache_config.cache_dir.exists():
            return 0
        
        cache_files = list(self.cache_config.cache_dir.glob("*.json"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        removed_count = len(cache_files)
        logger.info(f"Cleared {removed_count} cached responses")
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics with hit rate tracking."""
        if not self.cache_config.enabled or not self.cache_config.cache_dir.exists():
            return {"enabled": False}
        
        cache_files = list(self.cache_config.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        # Check valid vs expired
        valid_count = sum(1 for f in cache_files if self._is_cache_valid(f))
        expired_count = len(cache_files) - valid_count
        
        # Calculate cache efficiency metrics
        if hasattr(self, '_cache_hits'):
            total_requests = self._cache_hits + self._cache_misses
            hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        else:
            hit_rate = 0
            self._cache_hits = 0
            self._cache_misses = 0
        
        return {
            "enabled": True,
            "total_files": len(cache_files),
            "valid_files": valid_count,
            "expired_files": expired_count,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "ttl_hours": self.cache_config.ttl_hours,
            "hit_rate_pct": round(hit_rate, 1),
            "cache_hits": getattr(self, '_cache_hits', 0),
            "cache_misses": getattr(self, '_cache_misses', 0),
        }
    
    def cleanup_expired_cache(self) -> int:
        """Remove expired cache files and return count of removed files."""
        if not self.cache_config.enabled or not self.cache_config.cache_dir.exists():
            return 0
        
        cache_files = list(self.cache_config.cache_dir.glob("*.json"))
        removed_count = 0
        
        for cache_file in cache_files:
            if not self._is_cache_valid(cache_file):
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove expired cache file {cache_file}: {e}")
        
        logger.info(f"Cleaned up {removed_count} expired cache files")
        return removed_count


def create_client(
    service: str,
    api_key: Optional[str] = None,
    **kwargs,
) -> APIClient:
    """Create API client for specific service.
    
    Args:
        service: Service name ('openaq', 'cds', 'earthdata')
        api_key: API key for the service
        **kwargs: Additional configuration
        
    Returns:
        Configured API client
    """
    base_urls = {
        "openaq": "https://api.openaq.org/v3",
        "cds": "https://cds.climate.copernicus.eu/api",
        "earthdata": "https://urs.earthdata.nasa.gov",
    }
    
    if service not in base_urls:
        raise ValueError(f"Unknown service: {service}. Available: {list(base_urls.keys())}")
    
    base_url = base_urls[service]
    
    # Service-specific defaults
    if service == "openaq":
        kwargs.setdefault("rate_limit_delay", 0.5)  # 2 requests/second
    elif service == "cds":
        kwargs.setdefault("rate_limit_delay", 1.0)   # 1 request/second
    elif service == "earthdata":
        kwargs.setdefault("rate_limit_delay", 0.1)   # 10 requests/second
    
    return APIClient(base_url=base_url, api_key=api_key, **kwargs)
