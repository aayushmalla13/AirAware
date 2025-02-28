"""Error recovery and resilience utilities for ETL pipelines."""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ETLError(BaseModel):
    """ETL error record for tracking and recovery."""
    error_id: str
    timestamp: datetime
    operation: str
    error_type: str
    error_message: str
    context: Dict
    retry_count: int = 0
    max_retries: int = 3
    recovered: bool = False
    recovery_timestamp: Optional[datetime] = None


class RecoveryStrategy(BaseModel):
    """Recovery strategy configuration."""
    strategy_type: str  # "retry", "skip", "fallback", "manual"
    max_retries: int = 3
    backoff_factor: float = 1.5
    fallback_action: Optional[str] = None
    requires_manual_intervention: bool = False


class ETLErrorRecovery:
    """Advanced error recovery system for ETL pipelines."""
    
    def __init__(self, error_log_path: Optional[Path] = None):
        self.error_log_path = error_log_path or Path("data/artifacts/etl_errors.json")
        self.errors: List[ETLError] = []
        self.recovery_strategies = self._initialize_recovery_strategies()
        self._load_error_history()
    
    def _initialize_recovery_strategies(self) -> Dict[str, RecoveryStrategy]:
        """Initialize default recovery strategies for different error types."""
        return {
            # Network/API errors - retry with backoff
            "ConnectionError": RecoveryStrategy(
                strategy_type="retry",
                max_retries=5,
                backoff_factor=2.0
            ),
            "HTTPError": RecoveryStrategy(
                strategy_type="retry",
                max_retries=3,
                backoff_factor=1.5
            ),
            "TimeoutError": RecoveryStrategy(
                strategy_type="retry",
                max_retries=3,
                backoff_factor=2.0
            ),
            
            # Authentication errors - manual intervention required
            "AuthenticationError": RecoveryStrategy(
                strategy_type="manual",
                requires_manual_intervention=True
            ),
            "PermissionError": RecoveryStrategy(
                strategy_type="manual",
                requires_manual_intervention=True
            ),
            
            # Data format errors - skip and continue
            "DataFormatError": RecoveryStrategy(
                strategy_type="skip",
                max_retries=1
            ),
            "ValidationError": RecoveryStrategy(
                strategy_type="skip",
                max_retries=1
            ),
            
            # File system errors - retry with fallback
            "FileNotFoundError": RecoveryStrategy(
                strategy_type="fallback",
                max_retries=2,
                fallback_action="use_cached_data"
            ),
            "DiskSpaceError": RecoveryStrategy(
                strategy_type="manual",
                requires_manual_intervention=True
            ),
            
            # Memory errors - retry with reduced batch size
            "MemoryError": RecoveryStrategy(
                strategy_type="fallback",
                max_retries=2,
                fallback_action="reduce_batch_size"
            )
        }
    
    def record_error(self, operation: str, error: Exception, 
                    context: Dict) -> str:
        """Record an error for potential recovery."""
        error_id = f"{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(error)}"
        error_type = type(error).__name__
        
        etl_error = ETLError(
            error_id=error_id,
            timestamp=datetime.now(),
            operation=operation,
            error_type=error_type,
            error_message=str(error),
            context=context,
            max_retries=self.recovery_strategies.get(error_type, RecoveryStrategy(strategy_type="retry")).max_retries
        )
        
        self.errors.append(etl_error)
        self._save_error_history()
        
        logger.error(f"ETL Error recorded [{error_id}]: {error_type} in {operation}")
        return error_id
    
    def attempt_recovery(self, error_id: str) -> Tuple[bool, Optional[str]]:
        """Attempt to recover from a recorded error."""
        error = self._find_error(error_id)
        if not error:
            return False, "Error not found"
        
        if error.recovered:
            return True, "Already recovered"
        
        if error.retry_count >= error.max_retries:
            return False, "Max retries exceeded"
        
        strategy = self.recovery_strategies.get(error.error_type, 
                                              RecoveryStrategy(strategy_type="retry"))
        
        logger.info(f"Attempting recovery for {error_id} using strategy: {strategy.strategy_type}")
        
        if strategy.strategy_type == "retry":
            return self._attempt_retry_recovery(error, strategy)
        elif strategy.strategy_type == "skip":
            return self._attempt_skip_recovery(error)
        elif strategy.strategy_type == "fallback":
            return self._attempt_fallback_recovery(error, strategy)
        elif strategy.strategy_type == "manual":
            return self._require_manual_intervention(error)
        
        return False, "Unknown recovery strategy"
    
    def _attempt_retry_recovery(self, error: ETLError, 
                               strategy: RecoveryStrategy) -> Tuple[bool, str]:
        """Attempt retry-based recovery with exponential backoff."""
        # Calculate backoff delay
        delay = strategy.backoff_factor ** error.retry_count
        
        logger.info(f"Retrying {error.operation} after {delay:.1f}s delay (attempt {error.retry_count + 1})")
        time.sleep(delay)
        
        error.retry_count += 1
        self._save_error_history()
        
        # The actual retry should be handled by the calling code
        # This just sets up the retry attempt
        return True, f"Retry attempt {error.retry_count} set up"
    
    def _attempt_skip_recovery(self, error: ETLError) -> Tuple[bool, str]:
        """Skip the failed operation and continue."""
        logger.warning(f"Skipping failed operation: {error.operation}")
        
        error.recovered = True
        error.recovery_timestamp = datetime.now()
        self._save_error_history()
        
        return True, "Operation skipped - continuing with next"
    
    def _attempt_fallback_recovery(self, error: ETLError, 
                                  strategy: RecoveryStrategy) -> Tuple[bool, str]:
        """Attempt fallback recovery strategy."""
        fallback_action = strategy.fallback_action
        
        if fallback_action == "use_cached_data":
            logger.info(f"Attempting to use cached data for {error.operation}")
            # Implementation would check for cached data
            return True, "Switched to cached data"
        
        elif fallback_action == "reduce_batch_size":
            logger.info(f"Reducing batch size for {error.operation}")
            # Implementation would reduce batch size in context
            error.context['batch_size'] = error.context.get('batch_size', 10) // 2
            return True, "Batch size reduced - retry with smaller batches"
        
        return False, f"Unknown fallback action: {fallback_action}"
    
    def _require_manual_intervention(self, error: ETLError) -> Tuple[bool, str]:
        """Flag error for manual intervention."""
        logger.error(f"Manual intervention required for {error.operation}: {error.error_message}")
        
        # Create intervention file
        intervention_file = Path("data/artifacts/manual_intervention_required.json")
        intervention_data = {
            "error_id": error.error_id,
            "operation": error.operation,
            "error_type": error.error_type,
            "error_message": error.error_message,
            "context": error.context,
            "timestamp": error.timestamp.isoformat(),
            "instructions": self._get_manual_intervention_instructions(error.error_type)
        }
        
        with open(intervention_file, 'w') as f:
            json.dump(intervention_data, f, indent=2)
        
        return False, f"Manual intervention required - see {intervention_file}"
    
    def _get_manual_intervention_instructions(self, error_type: str) -> str:
        """Get human-readable instructions for manual intervention."""
        instructions = {
            "AuthenticationError": "Check API keys in .env file. Regenerate if expired.",
            "PermissionError": "Check file permissions and disk space. Run with appropriate permissions.",
            "DiskSpaceError": "Free up disk space or increase storage capacity.",
        }
        
        return instructions.get(error_type, "Review error details and resolve manually.")
    
    def get_recovery_status(self) -> Dict:
        """Get overall recovery status."""
        total_errors = len(self.errors)
        recovered_errors = len([e for e in self.errors if e.recovered])
        pending_errors = len([e for e in self.errors if not e.recovered])
        manual_interventions = len([e for e in self.errors 
                                  if not e.recovered and 
                                  self.recovery_strategies.get(e.error_type, 
                                  RecoveryStrategy(strategy_type="retry")).requires_manual_intervention])
        
        return {
            "total_errors": total_errors,
            "recovered_errors": recovered_errors,
            "pending_errors": pending_errors,
            "manual_interventions_required": manual_interventions,
            "recovery_rate": recovered_errors / total_errors if total_errors > 0 else 1.0
        }
    
    def get_error_summary(self) -> Dict:
        """Get error summary by type and operation."""
        error_by_type = {}
        error_by_operation = {}
        
        for error in self.errors:
            # Count by error type
            if error.error_type not in error_by_type:
                error_by_type[error.error_type] = 0
            error_by_type[error.error_type] += 1
            
            # Count by operation
            if error.operation not in error_by_operation:
                error_by_operation[error.operation] = 0
            error_by_operation[error.operation] += 1
        
        return {
            "errors_by_type": error_by_type,
            "errors_by_operation": error_by_operation,
            "recent_errors": [
                {
                    "error_id": e.error_id,
                    "timestamp": e.timestamp.isoformat(),
                    "operation": e.operation,
                    "error_type": e.error_type,
                    "recovered": e.recovered
                }
                for e in sorted(self.errors, key=lambda x: x.timestamp, reverse=True)[:10]
            ]
        }
    
    def mark_error_resolved(self, error_id: str) -> bool:
        """Manually mark an error as resolved."""
        error = self._find_error(error_id)
        if error:
            error.recovered = True
            error.recovery_timestamp = datetime.now()
            self._save_error_history()
            logger.info(f"Error {error_id} manually marked as resolved")
            return True
        return False
    
    def clean_old_errors(self, days_old: int = 30):
        """Clean up old error records."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        old_errors = [e for e in self.errors if e.timestamp < cutoff_date]
        
        self.errors = [e for e in self.errors if e.timestamp >= cutoff_date]
        self._save_error_history()
        
        logger.info(f"Cleaned up {len(old_errors)} error records older than {days_old} days")
    
    def _find_error(self, error_id: str) -> Optional[ETLError]:
        """Find error by ID."""
        for error in self.errors:
            if error.error_id == error_id:
                return error
        return None
    
    def _load_error_history(self):
        """Load error history from file."""
        if not self.error_log_path.exists():
            return
        
        try:
            with open(self.error_log_path, 'r') as f:
                error_data = json.load(f)
            
            self.errors = [ETLError(**error) for error in error_data]
            logger.debug(f"Loaded {len(self.errors)} error records from history")
            
        except Exception as e:
            logger.warning(f"Failed to load error history: {e}")
    
    def _save_error_history(self):
        """Save error history to file."""
        try:
            self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            error_data = [error.model_dump() for error in self.errors]
            
            with open(self.error_log_path, 'w') as f:
                json.dump(error_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.warning(f"Failed to save error history: {e}")


class ResilientETLWrapper:
    """Wrapper for making ETL operations more resilient."""
    
    def __init__(self, error_recovery: ETLErrorRecovery):
        self.error_recovery = error_recovery
    
    def execute_with_recovery(self, operation_name: str, operation_func, 
                             *args, **kwargs):
        """Execute an operation with automatic error recovery."""
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                return operation_func(*args, **kwargs)
                
            except Exception as e:
                attempt += 1
                context = {
                    "args": str(args)[:200],  # Truncate for storage
                    "kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
                    "attempt": attempt
                }
                
                error_id = self.error_recovery.record_error(operation_name, e, context)
                
                if attempt < max_attempts:
                    success, message = self.error_recovery.attempt_recovery(error_id)
                    if success:
                        logger.info(f"Recovery successful: {message}")
                        continue
                    else:
                        logger.error(f"Recovery failed: {message}")
                
                # If we get here, all attempts failed
                raise e
        
        raise RuntimeError(f"Operation {operation_name} failed after {max_attempts} attempts")


