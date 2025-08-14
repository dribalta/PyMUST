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
                # For non-numpy arrays, use appropriate dtype
                if hasattr(array, 'dtype') and 'complex' in str(array.dtype):
                    return self._torch.tensor(array, dtype=self.complex_type)
                else:
                    return self._torch.tensor(array, dtype=self.float_type)
        
        # Handle scalar values
        if isinstance(array, complex) or (isinstance(array, list) and any(isinstance(x, complex) for x in array)):
            return self._torch.tensor(array, dtype=self.complex_type)
        else:
            return self._torch.tensor(array, dtype=self.float_type)
    
    def from_backend(self, array: Any, target_type: Optional[type] = None) -> Any:
        """Convert from PyTorch tensor to target type if specified."""
        if target_type is None or array is self._torch.Tensor:
            return array
        
        if hasattr(target_type, '__module__'):
            if 'numpy' in target_type.__module__:
                return array.detach().cpu().numpy()
        
        return array
    
    def get_array_module(self) -> Any:
        """Get PyTorch module."""
        return self._torch
    
    ###############################################
    #   Efficient Parameter Conversion Helpers   
    ###############################################
    
    def _to_scalar(self, value: Any) -> Any:
        """Convert tensor to scalar if needed, otherwise return as-is."""
        return value.item() if hasattr(value, 'item') else value
    
    def _to_int(self, value: Any) -> int:
        """Convert to integer, handling tensors efficiently."""
        if hasattr(value, 'item'):
            return int(value.item())
        return int(value)
    
    def _ensure_tensor(self, value: Any, dtype: Any = None) -> Any:
        """Convert to tensor only if not already one."""
        if isinstance(value, self._torch.Tensor):
            return value
        if dtype is None:
            return self.to_backend(value)
        return self._torch.tensor(value, dtype=dtype)
    
    def _extract_scalar_if_needed(self, result: Any) -> Any:
        """Extract scalar value from 0-d tensor if needed."""
        if hasattr(result, 'numel') and result.numel() == 1 and result.ndim == 0:
            return result.item()
        return result
    
    def _convert_to_tensor_if_array(self, arg: Any) -> Any:
        """Convert argument to tensor only if it's an array-like object."""
        if isinstance(arg, (str, type(None))):
            return arg
        elif isinstance(arg, (int, float, complex, list)):
            if isinstance(arg, complex) or (isinstance(arg, list) and any(isinstance(x, complex) for x in arg)):
                return self._torch.tensor(arg, dtype=self.complex_type)
            else:
                return self._torch.tensor(arg, dtype=self.float_type)
        elif hasattr(arg, '__array__') or hasattr(arg, 'shape'):
            return self.to_backend(arg)
        else:
            return arg
    
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
        return {'reshape', 'view', 'zeros', 'ones', 'empty', 'randn', 'rand', 'swapaxes'}
    
    @property
    def _alias_mappings(self):
        """Function name aliases."""
        return {
            'arcsin': 'asin',
            'arccos': 'acos', 
            'arctan': 'atan',
            'mod': 'remainder',
        }
    
    
    def __getattr__(self, name: str) -> Any:
        """Route function calls with appropriate handling."""
        
        # 1. Handle aliases first
        actual_method = self._alias_mappings.get(name, name)
        
        # 2. Handle FFT operations
        if name.startswith('fft') and hasattr(self._torch_fft, name):
            torch_func = getattr(self._torch_fft, name)
            return self._create_tensor_wrapper(torch_func)
        
        # 3. Handle shape functions (no arg conversion)
        if actual_method in self._shape_functions and hasattr(self._torch, actual_method):
            torch_func = getattr(self._torch, actual_method)
            if callable(torch_func):
                return self._get_shape_function_wrapper(name, torch_func)
            return torch_func
        
        # 4. Handle standard torch functions (with arg conversion)
        if hasattr(self._torch, actual_method):
            torch_func = getattr(self._torch, actual_method)
            if callable(torch_func):
                return self._create_tensor_wrapper(torch_func)
            return torch_func
        
        # 5. Fallback to base class
        return super().__getattr__(name)
    
    def _get_shape_function_wrapper(self, name: str, torch_func: callable) -> callable:
        """Get wrapper for functions that take shape arguments."""
        def shape_wrapper(*args, **kwargs):
            if name in ['zeros', 'ones', 'empty', 'randn', 'rand']:
                # Factory functions - ensure shape args are integers
                if args:
                    # Convert shape arguments to integers
                    shape_args = []
                    for arg in args:
                        if hasattr(arg, '__iter__') and not isinstance(arg, str):
                            # Handle tuples/lists of shape
                            shape_args.append(tuple(int(x.item() if hasattr(x, 'item') else x) for x in arg))
                        else:
                            # Handle single integer
                            shape_args.append(int(arg.item() if hasattr(arg, 'item') else arg))
                    
                    # Use precision-aware default dtype
                    if 'dtype' not in kwargs:
                        kwargs['dtype'] = self.float_type
                    return torch_func(*shape_args, **kwargs)
                else:
                    if 'dtype' not in kwargs:
                        kwargs['dtype'] = self.float_type
                    return torch_func(*args, **kwargs)
            elif name in ['reshape', 'view', 'swapaxes']:
                # Reshape functions - convert first arg to tensor, keep shape as-is
                if args:
                    tensor_arg = self.to_backend(args[0])
                    return torch_func(tensor_arg, *args[1:], **kwargs)
                return torch_func(*args, **kwargs)
            else:
                # Generic shape function
                return torch_func(*args, **kwargs)
        
        return shape_wrapper
    
    
    ###############################################
    #   Direct Method Implementations            
    ###############################################
    
    def linspace(self, start, stop, num, **kwargs):
        """Generate evenly spaced numbers over specified interval."""
        # Extract scalars directly, no double conversion
        start = self._to_scalar(start)
        stop = self._to_scalar(stop)
        num = self._to_int(num)
        
        # Use precision-aware default dtype if not specified
        if 'dtype' not in kwargs:
            kwargs['dtype'] = self.float_type
        
        return self._torch.linspace(start, stop, num, **kwargs)
    
    def sum(self, input_tensor, axis=None, **kwargs):
        """Sum array elements over given axis."""
        input_tensor = self._ensure_tensor(input_tensor)
        
        # Handle axis parameter directly (NumPy uses axis, PyTorch uses dim)
        if axis is not None:
            kwargs['dim'] = self._to_int(axis)
        
        return self._torch.sum(input_tensor, **kwargs)
    
    def moveaxis(self, tensor, source, destination):
        """Move axes of tensor to new positions."""
        tensor = self._ensure_tensor(tensor)
        source = self._to_int(source)
        destination = self._to_int(destination)
        
        return self._torch.moveaxis(tensor, source, destination)
    
    def power(self, base, exponent):
        """Element-wise power operation."""
        # Handle base
        if not isinstance(base, self._torch.Tensor):
            if isinstance(base, (int, float, complex)):
                base = self._torch.tensor(float(base), dtype=self.float_type)
            else:
                base = self._ensure_tensor(base)
        
        # Handle exponent  
        if not isinstance(exponent, self._torch.Tensor):
            if isinstance(exponent, (int, float, complex)):
                exponent = self._torch.tensor(float(exponent), dtype=self.float_type)
            else:
                exponent = self._ensure_tensor(exponent)
        
        result = self._torch.pow(base, exponent)
        return self._extract_scalar_if_needed(result)
    
    def prod(self, x):
        """Product of array elements."""
        if isinstance(x, self._torch.Size):
            x = self._torch.tensor(list(x), dtype=self.float_type)
        else:
            x = self._ensure_tensor(x, dtype=self.float_type)
        
        result = self._torch.prod(x)
        return self._extract_scalar_if_needed(result)
    
    def array(self, data, **kwargs):
        """Create array from data."""
        return self.to_backend(data)
    
    def concatenate(self, tensors, axis=0, **kwargs):
        """Join sequence of tensors along existing axis."""
        # Ensure tensors is a sequence
        if not isinstance(tensors, (list, tuple)):
            if hasattr(tensors, 'shape'):  # Single tensor
                tensors = [tensors]
            else:  # Convert single item to list
                tensors = [self._ensure_tensor(tensors)]
        else:
            # Convert all tensors in the sequence
            tensors = [self._ensure_tensor(t) for t in tensors]
        
        # Use torch.cat (concatenate equivalent)
        return self._torch.cat(tensors, dim=axis, **kwargs)
    
    def isscalar(self, obj):
        """Check if object is scalar."""
        if self._torch.is_tensor(obj):
            return obj.numel() == 1 and obj.ndim == 0
        return not hasattr(obj, '__len__') and not hasattr(obj, 'shape')
    
    def min(self, input_tensor, *args, **kwargs):
        """Minimum values along axis."""
        input_tensor = self._ensure_tensor(input_tensor)
        result = self._torch.min(input_tensor, *args, **kwargs)
        # PyTorch min returns named tuple with .values for axis operations
        if hasattr(result, 'values') and not callable(result.values):
            return result.values
        return result
    
    def max(self, input_tensor, *args, **kwargs):
        """Maximum values along axis."""
        input_tensor = self._ensure_tensor(input_tensor)
        result = self._torch.max(input_tensor, *args, **kwargs)
        # PyTorch max returns named tuple with .values for axis operations
        if hasattr(result, 'values') and not callable(result.values):
            return result.values
        return result

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