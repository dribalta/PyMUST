"""
NumPy backend implementation for PyMUST.

This is the default backend using NumPy and SciPy for all array operations.
"""

import numpy as np
from typing import Any, Optional
from .base import BaseBackend


class NumpyBackend(BaseBackend):
    """NumPy-based backend implementation."""
    
    def __init__(self):
        super().__init__()
        self._numpy = None
        self._scipy = None
        self._scipy_signal = None
        self._scipy_fft = None
    
    def is_available(self) -> bool:
        """Check if NumPy backend dependencies are available."""
        try:
            import numpy
            return True
        except ImportError:
            return False
    
    def is_array(self, obj: Any) -> bool:
        """Check if object is a NumPy array."""
        return isinstance(obj, np.ndarray)
    
    def to_backend(self, array: Any) -> np.ndarray:
        """Convert array to NumPy format."""
        if isinstance(array, np.ndarray):
            return array
        return np.asarray(array)
    
    def from_backend(self, array: np.ndarray, target_type: Optional[type] = None) -> Any:
        """Convert from NumPy array to target type if specified."""
        if target_type is None:
            return array
        
        # Handle conversion to other array types
        if hasattr(target_type, '__module__'):
            if 'torch' in target_type.__module__:
                # Convert to PyTorch tensor if torch is available
                try:
                    import torch
                    return torch.from_numpy(array)
                except ImportError:
                    pass
        
        return array
    
    def get_array_module(self) -> Any:
        """Get NumPy module."""
        if self._numpy is None:
            import numpy as np
            self._numpy = np
        return self._numpy

    # Backend-specific optimized operations can be defined here
    def sinc(self, x: np.ndarray) -> np.ndarray:
        """Optimized sinc function for NumPy."""
        eps = 1e-16
        return np.sin(np.abs(x) + eps) / (np.abs(x) + eps)
    
    def zeros_like(self, array: np.ndarray, **kwargs) -> np.ndarray:
        """Create zeros array with same shape and dtype."""
        return np.zeros_like(array, **kwargs)
    
    def ones_like(self, array: np.ndarray, **kwargs) -> np.ndarray:
        """Create ones array with same shape and dtype."""
        return np.ones_like(array, **kwargs)