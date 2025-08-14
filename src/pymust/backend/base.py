"""
Abstract base backend for PyMUST array operations.

Defines the interface that all backend implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Union, Optional, Tuple
import numpy as np


class BaseBackend(ABC):
    """Abstract base class for PyMUST computation backends."""
    
    def __init__(self):
        self._name = self.__class__.__name__.lower().replace('backend', '')
        self._double_precision = False  # Default to single precision
    
    @property
    def name(self) -> str:
        """Backend name identifier."""
        return self._name
    
    def set_precision(self, precision: str) -> None:
        """Set precision mode for this backend.
        
        Args:
            precision: "single" or "double"
        """
        if precision not in ["single", "double"]:
            raise ValueError("Precision must be 'single' or 'double'")
        self._double_precision = (precision == "double")
    
    @property
    def precision(self) -> str:
        """Get current precision mode."""
        return "double" if self._double_precision else "single"
    
    # Dynamic precision properties
    @property  
    def float_type(self) -> Any:
        """Get current floating point type based on precision."""
        return self.float64 if self._double_precision else self.float32
    
    @property
    def complex_type(self) -> Any:
        """Get current complex type based on precision."""
        return self.complex128 if self._double_precision else self.complex64
    
    @property
    def int_type(self) -> Any:
        """Get current integer type based on precision.""" 
        return self.int64 if self._double_precision else self.int32
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are available."""
        pass
    
    @abstractmethod
    def is_array(self, obj: Any) -> bool:
        """Check if object is a valid array for this backend."""
        pass
    
    @abstractmethod
    def to_backend(self, array: Any) -> Any:
        """Convert array to backend-specific format."""
        pass
    
    @abstractmethod
    def from_backend(self, array: Any, target_type: Optional[type] = None) -> Any:
        """Convert from backend format to target type if specified."""
        pass
    
    @abstractmethod
    def get_array_module(self) -> Any:
        """Get the main array module (numpy, torch, etc.)."""
        pass
    
    def __getattr__(self, name: str) -> Any:
        """
        Automatic function mapping to backend operations.
        
        This method enables automatic routing of function calls to the appropriate
        backend implementation. Subclasses should override this to provide
        backend-specific function mapping.
        """
        # Try to get from array module first
        try:
            array_module = self.get_array_module()
            if hasattr(array_module, name):
                return getattr(array_module, name)
        except ImportError:
            pass
        
        # If not found, raise AttributeError
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def ensure_backend_array(self, *arrays: Any) -> Union[Any, Tuple[Any, ...]]:
        """
        Ensure all input arrays are converted to backend format.
        
        Returns single array if single input, tuple of arrays if multiple inputs.
        """
        converted = []
        for array in arrays:
            if not self.is_array(array):
                converted.append(self.to_backend(array))
            else:
                converted.append(array)
        
        if len(converted) == 1:
            return converted[0]
        return tuple(converted)
    
    def auto_return_type(self, result: Any, reference_input: Any) -> Any:
        """
        Automatically convert result to match input type.
        
        Implements "type in, type out" behavior.
        """
        if reference_input is not None and not self.is_array(reference_input):
            # Input was not from this backend, convert result to match
            return self.from_backend(result, type(reference_input))
        return result
    
    def mysinc(self, x: Any) -> Any:
        """
        Specialized sinc function for this backend."""
        eps = 1e-16
        return self.sin(self.abs(x) + eps) / (self.abs(x) + eps)

    def to_numpy(self, array: Any) -> np.ndarray:
        """
        Convert input array to a NumPy array.
        """
        return self.from_backend(array, np.ndarray)