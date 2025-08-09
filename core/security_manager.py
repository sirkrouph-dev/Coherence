"""
Security manager for the neuromorphic programming system.
Provides input validation, sanitization, and secure operations.
"""

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


class SecurityManager:
    """Security utilities for neuromorphic system."""
    
    # File operation limits
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_LOG_SIZE = 50 * 1024 * 1024    # 50MB
    ALLOWED_EXTENSIONS = {'.json', '.npz', '.npy', '.txt', '.csv', '.log'}
    
    # Input limits
    MAX_STRING_LENGTH = 10000
    MAX_ARRAY_SIZE = 1_000_000
    
    @staticmethod
    def validate_network_input(
        value: Any,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        dtype: type = float
    ) -> Any:
        """
        Validate and sanitize network inputs.
        
        Args:
            value: Input value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            dtype: Expected data type
            
        Returns:
            Validated and sanitized value
            
        Raises:
            ValueError: If input is invalid
        """
        try:
            # Convert to desired type
            if dtype == float:
                value = float(value)
            elif dtype == int:
                value = int(value)
            elif dtype == str:
                value = str(value)
                # Sanitize string input
                if len(value) > SecurityManager.MAX_STRING_LENGTH:
                    raise ValueError(f"String too long: {len(value)} > {SecurityManager.MAX_STRING_LENGTH}")
                # Remove potentially dangerous characters
                value = re.sub(r'[<>\"\'`]', '', value)
            elif dtype == bool:
                value = bool(value)
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
                
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid input type: {e}")
        
        # Check bounds for numeric types
        if dtype in (float, int):
            if min_val is not None and value < min_val:
                raise ValueError(f"Value {value} below minimum {min_val}")
            if max_val is not None and value > max_val:
                raise ValueError(f"Value {value} above maximum {max_val}")
        
        return value
    
    @staticmethod
    def validate_array_input(
        array: Union[np.ndarray, List],
        shape: Optional[tuple] = None,
        dtype: Optional[np.dtype] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None
    ) -> np.ndarray:
        """
        Validate array inputs.
        
        Args:
            array: Input array
            shape: Expected shape (None for any)
            dtype: Expected dtype (None for any)
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated numpy array
            
        Raises:
            ValueError: If array is invalid
        """
        # Convert to numpy array
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        
        # Check size limit
        if array.size > SecurityManager.MAX_ARRAY_SIZE:
            raise ValueError(f"Array too large: {array.size} > {SecurityManager.MAX_ARRAY_SIZE}")
        
        # Check shape if specified
        if shape is not None and array.shape != shape:
            raise ValueError(f"Invalid array shape: expected {shape}, got {array.shape}")
        
        # Check dtype if specified
        if dtype is not None:
            array = array.astype(dtype)
        
        # Check value bounds
        if min_val is not None or max_val is not None:
            if min_val is None:
                min_val = -np.inf
            if max_val is None:
                max_val = np.inf
            
            if np.any(array < min_val) or np.any(array > max_val):
                raise ValueError(f"Array values outside bounds [{min_val}, {max_val}]")
        
        # Check for NaN or Inf
        if np.any(np.isnan(array)):
            raise ValueError("Array contains NaN values")
        if np.any(np.isinf(array)):
            raise ValueError("Array contains infinite values")
        
        return array
    
    @staticmethod
    def safe_exp(x: np.ndarray, max_val: float = 10.0) -> np.ndarray:
        """
        Compute exponential with overflow protection.
        
        Args:
            x: Input array
            max_val: Maximum value to prevent overflow
            
        Returns:
            Exponential of input with clipping
        """
        return np.exp(np.clip(x, -max_val, max_val))
    
    @staticmethod
    def safe_divide(
        numerator: np.ndarray,
        denominator: np.ndarray,
        epsilon: float = 1e-10
    ) -> np.ndarray:
        """
        Safe division preventing divide-by-zero.
        
        Args:
            numerator: Numerator array
            denominator: Denominator array
            epsilon: Small value to prevent division by zero
            
        Returns:
            Result of safe division
        """
        # Add epsilon to prevent division by zero
        safe_denominator = np.where(
            np.abs(denominator) < epsilon,
            np.sign(denominator) * epsilon,
            denominator
        )
        return numerator / safe_denominator
    
    @staticmethod
    def validate_file_path(base_dir: str, filename: str) -> str:
        """
        Validate file paths to prevent traversal attacks.
        
        Args:
            base_dir: Base directory for file operations
            filename: Requested filename
            
        Returns:
            Safe absolute path
            
        Raises:
            ValueError: If path is unsafe
        """
        # Check for path traversal attempts BEFORE basename
        if '..' in filename or '../' in filename or '..\\' in filename:
            raise ValueError(f"Path traversal detected: {filename}")
        
        # Convert to Path objects for safer handling
        base_path = Path(base_dir).resolve()
        
        # Remove any path separators from filename
        safe_name = os.path.basename(filename)
        
        # Additional check after basename
        if '..' in safe_name or os.path.isabs(safe_name):
            raise ValueError(f"Invalid filename: {filename}")
        
        # Check file extension
        extension = Path(safe_name).suffix.lower()
        if extension and extension not in SecurityManager.ALLOWED_EXTENSIONS:
            raise ValueError(f"File extension {extension} not allowed")
        
        # Construct full path
        full_path = (base_path / safe_name).resolve()
        
        # Ensure the resolved path is within base directory
        try:
            full_path.relative_to(base_path)
        except ValueError:
            raise ValueError("Path traversal detected")
        
        return str(full_path)
    
    @staticmethod
    def validate_file_size(file_path: str, max_size: Optional[int] = None) -> bool:
        """
        Check if file size is within limits.
        
        Args:
            file_path: Path to file
            max_size: Maximum allowed size in bytes
            
        Returns:
            True if file size is acceptable
            
        Raises:
            ValueError: If file is too large
        """
        if max_size is None:
            max_size = SecurityManager.MAX_FILE_SIZE
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > max_size:
                raise ValueError(f"File too large: {file_size} > {max_size}")
        
        return True
    
    @staticmethod
    def sanitize_log_message(message: str) -> str:
        """
        Sanitize log messages to prevent injection attacks.
        
        Args:
            message: Log message to sanitize
            
        Returns:
            Sanitized message
        """
        # Limit message length
        if len(message) > SecurityManager.MAX_STRING_LENGTH:
            message = message[:SecurityManager.MAX_STRING_LENGTH] + "..."
        
        # Remove control characters
        message = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', message)
        
        # Escape special characters
        message = message.replace('\n', '\\n').replace('\r', '\\r')
        
        return message
    
    @staticmethod
    def mask_sensitive_data(data: Dict[str, Any], sensitive_keys: List[str]) -> Dict[str, Any]:
        """
        Mask sensitive data in dictionaries.
        
        Args:
            data: Dictionary potentially containing sensitive data
            sensitive_keys: List of keys to mask
            
        Returns:
            Dictionary with masked sensitive values
        """
        masked_data = data.copy()
        
        for key in sensitive_keys:
            if key in masked_data:
                # Replace sensitive value with masked version
                value = str(masked_data[key])
                if len(value) > 4:
                    masked_data[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                else:
                    masked_data[key] = '*' * len(value)
        
        return masked_data
    
    @staticmethod
    def generate_safe_id(prefix: str = "") -> str:
        """
        Generate a safe unique identifier.
        
        Args:
            prefix: Optional prefix for the ID
            
        Returns:
            Safe unique identifier
        """
        import uuid
        unique_id = str(uuid.uuid4())
        
        if prefix:
            # Sanitize prefix
            prefix = re.sub(r'[^a-zA-Z0-9_-]', '', prefix)[:20]
            return f"{prefix}_{unique_id}"
        
        return unique_id


class ResourceLimiter:
    """Resource usage limiter for preventing DoS attacks."""
    
    def __init__(
        self,
        max_memory_mb: int = 1024,
        max_cpu_percent: float = 80.0,
        max_gpu_memory_mb: int = 2048
    ):
        """
        Initialize resource limiter.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_cpu_percent: Maximum CPU usage percentage
            max_gpu_memory_mb: Maximum GPU memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.max_gpu_memory_mb = max_gpu_memory_mb
        
        # Track resource usage
        self.current_memory_mb = 0
        self.current_cpu_percent = 0
        self.current_gpu_memory_mb = 0
    
    def check_memory_allocation(self, size_bytes: int) -> bool:
        """
        Check if memory allocation is within limits.
        
        Args:
            size_bytes: Size of allocation in bytes
            
        Returns:
            True if allocation is allowed
            
        Raises:
            MemoryError: If allocation would exceed limits
        """
        size_mb = size_bytes / (1024 * 1024)
        
        if self.current_memory_mb + size_mb > self.max_memory_mb:
            raise MemoryError(
                f"Memory allocation would exceed limit: "
                f"{self.current_memory_mb + size_mb:.2f}MB > {self.max_memory_mb}MB"
            )
        
        return True
    
    def update_memory_usage(self, delta_mb: float):
        """Update tracked memory usage."""
        self.current_memory_mb = max(0, self.current_memory_mb + delta_mb)
    
    def check_array_allocation(self, shape: tuple, dtype: np.dtype) -> bool:
        """
        Check if array allocation is within limits.
        
        Args:
            shape: Shape of array to allocate
            dtype: Data type of array
            
        Returns:
            True if allocation is allowed
            
        Raises:
            MemoryError: If allocation would exceed limits
        """
        # Calculate size in bytes
        num_elements = np.prod(shape)
        bytes_per_element = np.dtype(dtype).itemsize
        size_bytes = num_elements * bytes_per_element
        
        return self.check_memory_allocation(size_bytes)
    
    def get_usage_stats(self) -> Dict[str, float]:
        """Get current resource usage statistics."""
        import psutil
        
        # Get system resource usage
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        stats = {
            "memory_used_mb": memory_info.used / (1024 * 1024),
            "memory_percent": memory_info.percent,
            "cpu_percent": cpu_percent,
            "tracked_memory_mb": self.current_memory_mb,
        }
        
        # Add GPU stats if available
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            stats["gpu_memory_used_mb"] = mempool.used_bytes() / (1024 * 1024)
        except:
            stats["gpu_memory_used_mb"] = 0
        
        return stats


class RateLimiter:
    """Rate limiter for preventing abuse of operations."""
    
    def __init__(self, max_calls: int = 100, time_window: float = 60.0):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_times = []
    
    def check_rate_limit(self) -> bool:
        """
        Check if operation is within rate limit.
        
        Returns:
            True if operation is allowed
            
        Raises:
            RuntimeError: If rate limit exceeded
        """
        import time
        current_time = time.time()
        
        # Remove old calls outside time window
        self.call_times = [
            t for t in self.call_times
            if current_time - t < self.time_window
        ]
        
        # Check if limit exceeded
        if len(self.call_times) >= self.max_calls:
            raise RuntimeError(
                f"Rate limit exceeded: {self.max_calls} calls per {self.time_window}s"
            )
        
        # Record this call
        self.call_times.append(current_time)
        return True
