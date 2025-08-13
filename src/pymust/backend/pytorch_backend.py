"""
Clean PyTorch backend implementation with clear separation of concerns.

This approach separates the different types of function handling for better maintainability.
"""

from typing import Any, Optional
from .base import BaseBackend
import numpy as np


class PyTorchBackend(BaseBackend):
    """Clean PyTorch-based backend with clear function categorization."""
    
    def __init__(self):
        super().__init__()
        self._torch = None
        self._torch_fft = None
        self._available = None
        self._lazy_import_torch()
    
    def _lazy_import_torch(self):
        """Lazy import PyTorch modules."""
        if self._torch is None:
            try:
                import torch
                self._torch = torch
                self._torch_fft = torch.fft
                self._available = True
            except ImportError:
                raise ImportError(
                    "PyTorch backend requested but PyTorch is not installed. "
                    "Install with: pip install torch"
                )
    
    ##################################
    #   Abstract method redefiniton
    ##################################
    
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
        return isinstance(obj, self._torch.Tensor)
    
    def to_backend(self, array: Any) -> Any:
        """Convert array to PyTorch tensor."""
        if isinstance(array, self._torch.Tensor):
            return array
        
        # Convert numpy arrays and other types to torch tensors
        if hasattr(array, '__array__'):
            if isinstance(array, np.ndarray):
                return self._torch.from_numpy(array)
            else:
                return self._torch.tensor(array)
        
        return self._torch.tensor(array)
    
    def from_backend(self, array: Any, target_type: Optional[type] = None) -> Any:
        """Convert from PyTorch tensor to target type if specified."""
        if not isinstance(array, self._torch.Tensor):
            return array
        
        if target_type is None:
            return array
        
        if hasattr(target_type, '__module__'):
            if 'numpy' in target_type.__module__:
                return array.detach().cpu().numpy()
        
        return array
    
    def get_array_module(self) -> Any:
        """Get PyTorch module."""
        return self._torch
    
    ###############################################
    #   Torch-specific method recovery functions   
    ###############################################
    
    def _convert_to_tensor_if_array(self, arg: Any) -> Any:
        """Convert argument to tensor only if it's an array-like object."""
        if isinstance(arg, (str, type(None))):
            return arg
        elif isinstance(arg, (int, float, complex, list)):
            return self._torch.tensor(arg)
        elif hasattr(arg, '__array__') or hasattr(arg, 'shape'):
            return self.to_backend(arg)
        else:
            return arg
    
    def _extract_scalar_if_needed(self, result: Any) -> Any:
        """Extract scalar value from 0-d tensor if needed."""
        if hasattr(result, 'numel') and result.numel() == 1 and result.ndim == 0:
            return result.item()
        return result
    
    def _create_tensor_wrapper(self, torch_func, convert_args: bool = True) -> callable:
        """Create a wrapper that auto-converts arguments to tensors."""
        def wrapper(*args, **kwargs):
            if convert_args:
                tensor_args = [self._convert_to_tensor_if_array(arg) for arg in args]
                result = torch_func(*tensor_args, **kwargs)
            else:
                result = torch_func(*args, **kwargs)
            
            return self._extract_scalar_if_needed(result)
        
        return wrapper
    
    # Define function categories for different handling strategies
    @property
    def _shape_functions(self):
        """Functions that take shape arguments that shouldn't be tensorized."""
        return {'reshape', 'view', 'zeros', 'ones', 'empty', 'randn', 'rand'}
    
    @property
    def _alias_mappings(self):
        """Function name aliases."""
        return {
            'arcsin': 'asin',
            'arccos': 'acos', 
            'arctan': 'atan',
            'mod': 'remainder',
        }
    
    @property 
    def _special_functions(self):
        """Functions that need completely custom implementations."""
        return {'min', 'max', 'power', 'prod', 'isscalar', 'array', 'concatenate'}
    
    def __getattr__(self, name: str) -> Any:
        """Route function calls with appropriate handling."""
        
        # 1. Handle aliases first
        actual_method = self._alias_mappings.get(name, name)
        
        # 2. Handle special cases
        if name in self._special_functions:
            return self._get_special_function(name)
        
        # 3. Handle FFT operations
        if name.startswith('fft') and hasattr(self._torch_fft, name):
            torch_func = getattr(self._torch_fft, name)
            return self._create_tensor_wrapper(torch_func)
        
        # 4. Handle shape functions (no arg conversion)
        if actual_method in self._shape_functions and hasattr(self._torch, actual_method):
            torch_func = getattr(self._torch, actual_method)
            if callable(torch_func):
                return self._get_shape_function_wrapper(name, torch_func)
            return torch_func
        
        # 5. Handle standard torch functions (with arg conversion)
        if hasattr(self._torch, actual_method):
            torch_func = getattr(self._torch, actual_method)
            if callable(torch_func):
                return self._create_tensor_wrapper(torch_func)
            return torch_func
        
        # 6. Fallback to base class
        return super().__getattr__(name)
    
    def _get_shape_function_wrapper(self, name: str, torch_func: callable) -> callable:
        """Get wrapper for functions that take shape arguments."""
        def shape_wrapper(*args, **kwargs):
            if name in ['zeros', 'ones', 'empty', 'randn', 'rand']:
                # Factory functions - keep shape args as-is, set default dtype
                if 'dtype' not in kwargs:
                    kwargs['dtype'] = self._torch.float32
                return torch_func(*args, **kwargs)
            elif name in ['reshape', 'view']:
                # Reshape functions - convert first arg to tensor, keep shape as-is
                if args:
                    tensor_arg = self.to_backend(args[0])
                    return torch_func(tensor_arg, *args[1:], **kwargs)
                return torch_func(*args, **kwargs)
            else:
                # Generic shape function
                return torch_func(*args, **kwargs)
        
        return shape_wrapper
    
    def _get_special_function(self, name: str) -> callable:
        """Get special function implementations."""
        
        if name == 'array':
            return lambda data, **kwargs: self.to_backend(data)
        
        elif name in ['min', 'max']:
            torch_func = getattr(self._torch, name)
            def minmax_wrapper(input_tensor, *args, **kwargs):
                input_tensor = self.to_backend(input_tensor)
                result = torch_func(input_tensor, *args, **kwargs)
                if hasattr(result, 'values') and not callable(result.values):
                    return result.values
                return result
            return minmax_wrapper
        
        elif name == 'power':
            def power_wrapper(base, exponent):
                base = self._torch.tensor(float(base)) if not hasattr(base, 'shape') else self.to_backend(base)
                exponent = self._torch.tensor(float(exponent)) if not hasattr(exponent, 'shape') else self.to_backend(exponent)
                result = self._torch.pow(base, exponent)
                return self._extract_scalar_if_needed(result)
            return power_wrapper
        
        elif name == 'prod':
            def prod_wrapper(x):
                if isinstance(x, self._torch.Size):
                    x = self._torch.tensor(list(x))
                elif not hasattr(x, 'shape'):
                    x = self._torch.tensor(x)
                else:
                    x = self.to_backend(x)
                result = self._torch.prod(x)
                return self._extract_scalar_if_needed(result)
            return prod_wrapper
        
        elif name == 'isscalar':
            def isscalar_wrapper(obj):
                if self._torch.is_tensor(obj):
                    return obj.numel() == 1 and obj.ndim == 0
                return not hasattr(obj, '__len__') and not hasattr(obj, 'shape')
            return isscalar_wrapper
        
        elif name == 'concatenate':
            def concatenate_wrapper(tensors, axis=0, **kwargs):
                # Ensure tensors is a sequence
                if not isinstance(tensors, (list, tuple)):
                    if hasattr(tensors, 'shape'):  # Single tensor
                        tensors = [tensors]
                    else:  # Convert single item to list
                        tensors = [self.to_backend(tensors)]
                else:
                    # Convert all tensors in the sequence
                    tensors = [self.to_backend(t) for t in tensors]
                
                # Use torch.cat (concatenate equivalent)
                result = self._torch.cat(tensors, dim=axis, **kwargs)
                return result
            return concatenate_wrapper
        
        else:
            raise AttributeError(f"Special function '{name}' not implemented")
    
    #########################################
    #   Torch-specific methods redefiniton   
    #########################################
    
    @property
    def integer(self):
        return self._torch.int64

    @property
    def floating(self):
        return self._torch.float64

    def astype(self, tensor: Any, dtype: Any) -> Any:
        """Convert tensor to specified dtype (NumPy compatibility)."""
        tensor = self.to_backend(tensor)
        return tensor.to(dtype)
    
    def to_device(self, array: Any, device: str = "cpu") -> Any:
        """Move tensor to specified device (CPU/GPU)."""
        array = self.to_backend(array)
        return array.to(device)
    
    def device_available(self, device: str = "cuda") -> bool:
        """Check if device is available."""
        if not self.is_available():
            return False
        
        if device == "cuda":
            return self._torch.cuda.is_available()
        elif device == "mps":
            return self._torch.backends.mps.is_available()
        return True

    def flatten(self, tensor, order='C'):
        """
        Flatten the tensor to 1D (NumPy compatibility).
        Order is not supported but keep it for compatibility.
        """
        tensor = self.to_backend(tensor)
        return tensor.flatten()

    def array_equal(self, array1: Any, array2: Any) -> bool:
        """Check if two arrays are equal (NumPy compatibility)."""
        array1 = self.to_backend(array1)
        array2 = self.to_backend(array2)
        return self._torch.equal(array1, array2)

    def issubdtype(self, dtype, general_dtype):
        """Check if dtype is a subtype of general_dtype (NumPy compatibility)."""
        if general_dtype == self.integer:
            return dtype in [self.int8, self.int16, self.int32, self.int64]
        elif general_dtype == self.floating:
            return dtype in [self.float16, self.float32, self.float64]
        elif general_dtype == self.complex:
            return dtype in [self.complex64, self.complex128]
        return False