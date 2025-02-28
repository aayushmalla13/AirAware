"""Circuit breaker pattern for robust API error handling."""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception,
                 fallback_function: Optional[Callable] = None):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.fallback_function = fallback_function


class CircuitBreaker:
    """Circuit breaker for API calls with automatic recovery."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker {self.name} attempting recovery")
            else:
                if self.config.fallback_function:
                    logger.warning(f"Circuit breaker {self.name} is OPEN, using fallback")
                    return self.config.fallback_function(*args, **kwargs)
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 2:  # Need 2 successes to fully recover
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} recovered to CLOSED")
        
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name} failed during recovery, back to OPEN")
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name} opened after {self.failure_count} failures")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "success_count": self.success_count,
            "failure_threshold": self.config.failure_threshold,
            "recovery_timeout": self.config.recovery_timeout
        }


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services."""
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.breakers:
            if config is None:
                config = CircuitBreakerConfig()
            self.breakers[name] = CircuitBreaker(name, config)
        
        return self.breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: breaker.get_status() for name, breaker in self.breakers.items()}
    
    def reset_breaker(self, name: str) -> bool:
        """Manually reset a circuit breaker."""
        if name in self.breakers:
            breaker = self.breakers[name]
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            breaker.last_failure_time = None
            logger.info(f"Manually reset circuit breaker {name}")
            return True
        return False


# Global circuit breaker manager
circuit_manager = CircuitBreakerManager()


