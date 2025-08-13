"""
Error handling and exception management for the neuromorphic system.
Provides centralized error handling, logging, and recovery mechanisms.
"""

import functools
import logging
import sys
import traceback
from typing import Any, Callable, Dict, Optional, Type, Union

import numpy as np


class NeuromorphicError(Exception):
    """Base exception for neuromorphic system errors."""

    pass


class NetworkError(NeuromorphicError):
    """Network-related errors."""

    pass


class NeuronError(NeuromorphicError):
    """Neuron model errors."""

    pass


class SynapseError(NeuromorphicError):
    """Synapse model errors."""

    pass


class SimulationError(NeuromorphicError):
    """Simulation execution errors."""

    pass


class ResourceError(NeuromorphicError):
    """Resource limitation errors."""

    pass


class ValidationError(NeuromorphicError):
    """Input validation errors."""

    pass


class GPUError(NeuromorphicError):
    """GPU-related errors."""

    pass


class ErrorHandler:
    """Centralized error handling and recovery."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error handler.

        Args:
            logger: Logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        self.error_count = {}
        self.max_retries = 3
        self.recovery_strategies = {}

    def handle_error(
        self, error: Exception, context: Dict[str, Any], severity: str = "ERROR"
    ) -> bool:
        """
        Handle an error with appropriate logging and recovery.

        Args:
            error: The exception that occurred
            context: Context information about the error
            severity: Error severity level

        Returns:
            True if error was handled, False if it should be re-raised
        """
        error_type = type(error).__name__
        self.error_count[error_type] = self.error_count.get(error_type, 0) + 1

        # Log the error with context
        self.logger.log(
            getattr(logging, severity, logging.ERROR),
            f"{error_type}: {str(error)}",
            extra={"context": context, "traceback": traceback.format_exc()},
        )

        # Try recovery strategy if available
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error, context)
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")
                return False

        # Default: don't handle (will be re-raised)
        return False

    def register_recovery_strategy(
        self, error_type: Type[Exception], strategy: Callable[[Exception, Dict], bool]
    ):
        """
        Register a recovery strategy for a specific error type.

        Args:
            error_type: Type of exception to handle
            strategy: Recovery function that returns True if recovered
        """
        self.recovery_strategies[error_type.__name__] = strategy

    def get_error_statistics(self) -> Dict[str, int]:
        """Get error occurrence statistics."""
        return self.error_count.copy()

    def reset_error_counts(self):
        """Reset error counters."""
        self.error_count.clear()


def safe_execution(
    func: Optional[Callable] = None,
    *,
    default_return: Any = None,
    error_types: tuple = (Exception,),
    max_retries: int = 1,
    logger: Optional[logging.Logger] = None,
):
    """
    Decorator for safe function execution with error handling.

    Args:
        func: Function to wrap
        default_return: Default value to return on error
        error_types: Tuple of exception types to catch
        max_retries: Maximum number of retry attempts
        logger: Logger for error reporting
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(max_retries):
                try:
                    return f(*args, **kwargs)
                except error_types as e:
                    last_error = e
                    if logger:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}"
                        )

                    if attempt < max_retries - 1:
                        # Exponential backoff for retries
                        import time

                        time.sleep(0.1 * (2**attempt))

            # All attempts failed
            if logger:
                logger.error(f"All {max_retries} attempts failed: {last_error}")

            if default_return is not None:
                return default_return
            else:
                raise last_error

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def validate_numeric(
    value: Union[float, int, np.ndarray],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
    name: str = "value",
) -> Union[float, int, np.ndarray]:
    """
    Validate numeric values with comprehensive checks.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        allow_nan: Whether to allow NaN values
        allow_inf: Whether to allow infinite values
        name: Name of the value for error messages

    Returns:
        Validated value

    Raises:
        ValidationError: If validation fails
    """
    # Check for NaN
    if not allow_nan:
        if isinstance(value, np.ndarray):
            if np.any(np.isnan(value)):
                raise ValidationError(f"{name} contains NaN values")
        elif np.isnan(value):
            raise ValidationError(f"{name} is NaN")

    # Check for Inf
    if not allow_inf:
        if isinstance(value, np.ndarray):
            if np.any(np.isinf(value)):
                raise ValidationError(f"{name} contains infinite values")
        elif np.isinf(value):
            raise ValidationError(f"{name} is infinite")

    # Check bounds
    if min_val is not None:
        if isinstance(value, np.ndarray):
            if np.any(value < min_val):
                raise ValidationError(f"{name} contains values below {min_val}")
        elif value < min_val:
            raise ValidationError(f"{name} ({value}) is below minimum {min_val}")

    if max_val is not None:
        if isinstance(value, np.ndarray):
            if np.any(value > max_val):
                raise ValidationError(f"{name} contains values above {max_val}")
        elif value > max_val:
            raise ValidationError(f"{name} ({value}) is above maximum {max_val}")

    return value


def validate_shape(
    array: np.ndarray,
    expected_shape: Optional[tuple] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    name: str = "array",
) -> np.ndarray:
    """
    Validate array shape.

    Args:
        array: Array to validate
        expected_shape: Expected shape (None for any)
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        name: Name for error messages

    Returns:
        Validated array

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(array, np.ndarray):
        raise ValidationError(f"{name} must be a numpy array")

    # Check exact shape
    if expected_shape is not None and array.shape != expected_shape:
        raise ValidationError(
            f"{name} has shape {array.shape}, expected {expected_shape}"
        )

    # Check dimensions
    ndims = array.ndim
    if min_dims is not None and ndims < min_dims:
        raise ValidationError(f"{name} has {ndims} dimensions, minimum is {min_dims}")

    if max_dims is not None and ndims > max_dims:
        raise ValidationError(f"{name} has {ndims} dimensions, maximum is {max_dims}")

    return array


class NumericalStabilizer:
    """Utilities for numerical stability."""

    EPSILON = 1e-10
    MAX_EXP = 10.0
    MIN_LOG = 1e-10

    @staticmethod
    def safe_exp(x: np.ndarray) -> np.ndarray:
        """Compute exponential with overflow protection."""
        return np.exp(
            np.clip(x, -NumericalStabilizer.MAX_EXP, NumericalStabilizer.MAX_EXP)
        )

    @staticmethod
    def safe_log(x: np.ndarray) -> np.ndarray:
        """Compute logarithm with underflow protection."""
        return np.log(np.maximum(x, NumericalStabilizer.MIN_LOG))

    @staticmethod
    def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Safe division preventing divide-by-zero."""
        safe_denom = np.where(
            np.abs(denominator) < NumericalStabilizer.EPSILON,
            np.sign(denominator) * NumericalStabilizer.EPSILON,
            denominator,
        )
        return numerator / safe_denom

    @staticmethod
    def safe_sqrt(x: np.ndarray) -> np.ndarray:
        """Compute square root with negative value protection."""
        return np.sqrt(np.maximum(x, 0))

    @staticmethod
    def clip_gradients(gradients: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
        """Clip gradients to prevent explosion."""
        norm = np.linalg.norm(gradients)
        if norm > max_norm:
            gradients = gradients * (max_norm / norm)
        return gradients


class RecoveryStrategies:
    """Common recovery strategies for errors."""

    @staticmethod
    def reset_to_default(error: Exception, context: Dict) -> bool:
        """Reset to default state on error."""
        if "default_state" in context:
            # Reset to default state
            for key, value in context["default_state"].items():
                if "object" in context:
                    setattr(context["object"], key, value)
            return True
        return False

    @staticmethod
    def retry_with_smaller_step(error: Exception, context: Dict) -> bool:
        """Retry operation with smaller step size."""
        if "step_size" in context:
            context["step_size"] *= 0.5
            if context["step_size"] > 1e-6:
                return True
        return False

    @staticmethod
    def fallback_to_cpu(error: Exception, context: Dict) -> bool:
        """Fallback from GPU to CPU computation."""
        if isinstance(error, GPUError) and "use_gpu" in context:
            context["use_gpu"] = False
            return True
        return False

    @staticmethod
    def reduce_precision(error: Exception, context: Dict) -> bool:
        """Reduce numerical precision to avoid overflow."""
        if "dtype" in context:
            if context["dtype"] == np.float64:
                context["dtype"] = np.float32
                return True
            elif context["dtype"] == np.float32:
                context["dtype"] = np.float16
                return True
        return False


# Global error handler instance
global_error_handler = ErrorHandler()


def setup_error_handling(log_file: Optional[str] = None):
    """
    Setup global error handling for the neuromorphic system.

    Args:
        log_file: Optional log file path
    """
    # Configure logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # Register default recovery strategies
    global_error_handler.register_recovery_strategy(
        GPUError, RecoveryStrategies.fallback_to_cpu
    )
    global_error_handler.register_recovery_strategy(
        MemoryError, RecoveryStrategies.reduce_precision
    )
    global_error_handler.register_recovery_strategy(
        SimulationError, RecoveryStrategies.retry_with_smaller_step
    )

    # Set up exception hook for uncaught exceptions
    def exception_hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        global_error_handler.handle_error(
            exc_value,
            {
                "type": exc_type.__name__,
                "traceback": traceback.format_tb(exc_traceback),
            },
            severity="CRITICAL",
        )

    sys.excepthook = exception_hook
