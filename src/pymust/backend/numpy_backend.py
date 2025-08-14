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
    
    ##################################
    #   Abstract method redefiniton
    ##################################
    
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
        if target_type is None or target_type is np.ndarray:
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
    
    #########################################
    #   Numpy-specific methods redefiniton   
    #########################################
    
    def flatten(self, array: Any, order: str = 'C') -> np.ndarray:
        """Flatten array with order support."""
        if hasattr(array, 'flatten'):
            return array.flatten(order=order)
        return np.asarray(array).flatten(order=order)