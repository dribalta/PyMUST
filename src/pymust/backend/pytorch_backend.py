"""
PyTorch backend implementation for PyMUST.

Optional backend that uses PyTorch tensors for GPU acceleration and automatic differentiation.
Uses lazy loading to avoid importing PyTorch unless explicitly requested.
"""

import warnings
from typing import Any, Optional
from .base import BaseBackend


class PyTorchBackend(BaseBackend):
    """PyTorch-based backend implementation with lazy loading."""
    
    def __init__(self):
        super().__init__()
        self._torch = None
        self._torch_fft = None
        self._available = None
    
    def _lazy_import_torch(self):
        """Lazy import PyTorch modules."""
        if self._torch is None:
            try:
                import torch
                self._torch = torch
                self._torch_fft = torch.fft
            except ImportError:
                raise ImportError(
                    "PyTorch backend requested but PyTorch is not installed. "
                    "Install with: pip install torch"
                )
    
    def is_available(self) -> bool:
        """Check if PyTorch backend dependencies are available."""
        if self._available is None:
            try:
                import torch
                self._available = True
            except ImportError:
                self._available = False
        return self._available
    
    def is_array(self, obj: Any) -> bool:
        """Check if object is a PyTorch tensor."""
        if not self.is_available():
            return False
        
        self._lazy_import_torch()
        return isinstance(obj, self._torch.Tensor)
    
    def to_backend(self, array: Any) -> Any:
        """Convert array to PyTorch tensor with numpy compatibility."""
        self._lazy_import_torch()
        
        if isinstance(array, self._torch.Tensor):
            return self._add_numpy_compatibility(array)
        
        # Convert numpy arrays and other types to torch tensors
        if hasattr(array, '__array__'):
            # Handle numpy arrays and array-like objects
            import numpy as np
            if isinstance(array, np.ndarray):
                tensor = self._torch.from_numpy(array)
            else:
                tensor = self._torch.tensor(array)
        else:
            tensor = self._torch.tensor(array)
        
        return self._add_numpy_compatibility(tensor)
    
    def _add_numpy_compatibility(self, tensor):
        """Add numpy-compatible methods to PyTorch tensors."""
        if hasattr(tensor, '_numpy_compat_added'):
            return tensor
            
        import numpy as np
        
        # Add astype method
        def astype_method(dtype):
            dtype_map = {
                np.float32: self._torch.float32,
                np.float64: self._torch.float64,
                np.int32: self._torch.int32,
                np.int64: self._torch.int64,
                np.bool_: self._torch.bool,
                np.complex64: self._torch.complex64,
                np.complex128: self._torch.complex128
            }
            torch_dtype = dtype_map.get(dtype, dtype)
            result = tensor.to(torch_dtype)
            return self._add_numpy_compatibility(result)
        
        tensor.astype = astype_method
        tensor._numpy_compat_added = True
        return tensor
    
    def from_backend(self, array: Any, target_type: Optional[type] = None) -> Any:
        """Convert from PyTorch tensor to target type if specified."""
        self._lazy_import_torch()
        
        if not isinstance(array, self._torch.Tensor):
            return array
        
        if target_type is None:
            return array
        
        # Convert to numpy if requested
        if hasattr(target_type, '__module__'):
            if 'numpy' in target_type.__module__:
                return array.detach().cpu().numpy()
        
        return array
    
    def get_array_module(self) -> Any:
        """Get PyTorch module."""
        self._lazy_import_torch()
        return self._torch
    
    def __getattr__(self, name: str) -> Any:
        """Route function calls to appropriate PyTorch modules."""
        self._lazy_import_torch()
        
        # Handle FFT operations
        if name.startswith('fft'):
            if hasattr(self._torch_fft, name):
                return getattr(self._torch_fft, name)
        
        # Handle array creation function specifically
        if name == 'array':
            return lambda data, **kwargs: self.to_backend(data)
        
        # Handle common factory functions to ensure compatibility
        if name in ['zeros', 'ones', 'zeros_like', 'ones_like', 'empty', 'full', 'linspace', 'arange']:
            original_func = getattr(self._torch, name) if hasattr(self._torch, name) else None
            if original_func:
                def factory_wrapper(*args, **kwargs):
                    if 'dtype' not in kwargs and name in ['zeros', 'ones', 'empty', 'full']:
                        kwargs['dtype'] = self._torch.float32
                    result = original_func(*args, **kwargs)
                    return self._add_numpy_compatibility(result)
                return factory_wrapper
        
        # Handle functions that need scalar tensor conversion
        if name in ['isinf', 'isnan', 'isfinite']:
            def scalar_safe_func(x, *args, **kwargs):
                # Convert scalar to tensor if needed
                if not hasattr(x, 'shape'):  # scalar
                    x = self._torch.tensor(x)
                elif not self._torch.is_tensor(x):  # numpy array or list
                    x = self.to_backend(x)
                return getattr(self._torch, name)(x, *args, **kwargs)
            return scalar_safe_func
        
        # Handle min/max functions that return named tuples in PyTorch
        if name in ['min', 'max']:
            def minmax_wrapper(input_tensor, *args, **kwargs):
                input_tensor = self.to_backend(input_tensor)
                result = getattr(self._torch, name)(input_tensor, *args, **kwargs)
                # If it's a named tuple (has .values), return just the values
                if hasattr(result, 'values'):
                    return result.values
                return result
            return minmax_wrapper
        
        # Handle array comparison functions
        if name == 'array_equal':
            def array_equal_wrapper(a, b):
                a = self.to_backend(a)
                b = self.to_backend(b)
                return self._torch.equal(a, b)
            return array_equal_wrapper
        
        # Handle dtype properties
        if name in ['float32', 'float64', 'int32', 'int64', 'bool', 'complex64', 'complex128']:
            return getattr(self._torch, name)
        
        # Handle standard torch operations
        if hasattr(self._torch, name):
            attr = getattr(self._torch, name)
            
            # For functions that need to preserve gradients, return as-is
            # For factory functions, ensure they create tensors with appropriate properties
            if callable(attr) and name in ['zeros', 'ones', 'empty', 'randn', 'rand']:
                def factory_wrapper(*args, **kwargs):
                    # Set default dtype if not specified
                    if 'dtype' not in kwargs:
                        kwargs['dtype'] = self._torch.float32
                    return attr(*args, **kwargs)
                return factory_wrapper
            
            return attr
        
        # Fallback to base class
        return super().__getattr__(name)
    
    # PyTorch-specific optimized operations
    def sinc(self, x: Any) -> Any:
        """Optimized sinc function for PyTorch."""
        self._lazy_import_torch()
        x = self.to_backend(x)
        eps = 1e-16
        return self._torch.sin(self._torch.abs(x) + eps) / (self._torch.abs(x) + eps)
    
    def zeros_like(self, array: Any, **kwargs) -> Any:
        """Create zeros tensor with same shape and dtype."""
        self._lazy_import_torch()
        array = self.to_backend(array)
        return self._torch.zeros_like(array, **kwargs)
    
    def ones_like(self, array: Any, **kwargs) -> Any:
        """Create ones tensor with same shape and dtype."""
        self._lazy_import_torch()
        array = self.to_backend(array)
        return self._torch.ones_like(array, **kwargs)
    
    def to_device(self, array: Any, device: str = "cpu") -> Any:
        """Move tensor to specified device (CPU/GPU)."""
        self._lazy_import_torch()
        array = self.to_backend(array)
        return array.to(device)
    
    def device_available(self, device: str = "cuda") -> bool:
        """Check if device is available."""
        if not self.is_available():
            return False
        
        self._lazy_import_torch()
        if device == "cuda":
            return self._torch.cuda.is_available()
        elif device == "mps":
            return self._torch.backends.mps.is_available()
        return True  # CPU always available