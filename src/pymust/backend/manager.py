"""
Backend manager for PyMUST.

Handles backend registration, configuration, and global state management.
Provides the main API for switching between different computation backends.
"""

import warnings
from typing import Dict, Optional, Any, Union, Tuple
from .base import BaseBackend
from .numpy_backend import NumpyBackend


class BackendManager:
    """Global backend manager for PyMUST operations."""
    
    def __init__(self):
        self._backends: Dict[str, BaseBackend] = {}
        self._current_backend: Optional[BaseBackend] = None
        self._default_backend_name = "numpy"
        
        # Register default backends
        self._register_default_backends()
        
        # Set default backend
        self.set(self._default_backend_name)
    
    def _register_default_backends(self):
        """Register built-in backends."""
        # Always register NumPy backend
        self.register("numpy", NumpyBackend())
        
        # Don't pre-register PyTorch backend - lazy load it only when requested
    
    def register(self, name: str, backend: BaseBackend):
        """Register a new backend."""
        if not isinstance(backend, BaseBackend):
            raise TypeError(f"Backend must inherit from BaseBackend, got {type(backend)}")
        
        self._backends[name.lower()] = backend
    
    def available_backends(self) -> Dict[str, bool]:
        """Get list of available backends and their availability status."""
        backends = {name: backend.is_available() for name, backend in self._backends.items()}
        
        # Note: PyTorch backend is lazy-loaded, so it won't appear here until first used
        return backends
    
    def set(self, backend_name: str) -> None:
        """Set the active backend."""
        backend_name = backend_name.lower()
        
        # Handle lazy loading of PyTorch backend
        if backend_name == "pytorch" and backend_name not in self._backends:
            from .pytorch_backend import PyTorchBackend
            pytorch_backend = PyTorchBackend()
            if pytorch_backend.is_available():
                self.register("pytorch", pytorch_backend)
        
        if backend_name not in self._backends:
            available = list(self._backends.keys())
            raise ValueError(f"Backend '{backend_name}' not found. Available: {available}")
        
        backend = self._backends[backend_name]
        if not backend.is_available():
            warnings.warn(
                f"Backend '{backend_name}' is not available. "
                f"Required dependencies may not be installed."
            )
            return
        
        self._current_backend = backend
    
    def get_current(self) -> BaseBackend:
        """Get the currently active backend."""
        if self._current_backend is None:
            self.set(self._default_backend_name)
        return self._current_backend
    
    def get_backend_name(self) -> str:
        """Get the name of the currently active backend."""
        return self.get_current().name
    
    def auto_detect_backend(self, *arrays: Any) -> Optional[str]:
        """
        Auto-detect the appropriate backend based on input array types.
        
        Returns the backend name that can handle the input arrays, or None if
        multiple backends are suitable.
        """
        detected_backends = set()
        
        for array in arrays:
            for name, backend in self._backends.items():
                if backend.is_available() and backend.is_array(array):
                    detected_backends.add(name)
        
        if len(detected_backends) == 1:
            return detected_backends.pop()
        
        return None
    
    def ensure_compatible_arrays(self, *arrays: Any) -> Union[Any, Tuple[Any, ...]]:
        """
        Ensure all arrays are compatible with current backend.
        
        Returns converted arrays in backend-compatible format.
        """
        backend = self.get_current()
        return backend.ensure_backend_array(*arrays)
    
    def __getattr__(self, name: str) -> Any:
        """Route attribute access to current backend."""
        return getattr(self.get_current(), name)
    
    def __call__(self, func_name: str, *args, **kwargs) -> Any:
        """
        Execute function with automatic type handling.
        
        This method provides "type in, type out" behavior by detecting input types
        and converting results to match.
        """
        backend = self.get_current()
        
        # Detect reference input for type preservation
        reference_input = None
        for arg in args:
            if hasattr(arg, '__array__') or hasattr(arg, 'shape'):
                reference_input = arg
                break
        
        # Get function from backend
        func = getattr(backend, func_name)
        
        # Execute with backend arrays
        converted_args = backend.ensure_backend_array(*args)
        if isinstance(converted_args, tuple):
            result = func(*converted_args, **kwargs)
        else:
            result = func(converted_args, **kwargs)
        
        # Auto-convert result to match input type
        return backend.auto_return_type(result, reference_input)


# Global backend manager instance
_backend_manager = BackendManager()


def set_backend(backend_name: str) -> None:
    """Set the global backend for PyMUST operations."""
    _backend_manager.set(backend_name)


def get_backend() -> BaseBackend:
    """Get the current global backend."""
    return _backend_manager.get_current()


def get_backend_name() -> str:
    """Get the name of the current backend."""
    return _backend_manager.get_backend_name()


def available_backends() -> Dict[str, bool]:
    """Get available backends and their status."""
    return _backend_manager.available_backends()


def backend_call(func_name: str, *args, **kwargs) -> Any:
    """Execute function with current backend and automatic type handling."""
    return _backend_manager(func_name, *args, **kwargs)


def get_backend_function(func_name: str) -> Any:
    """Get function from current backend."""
    return getattr(_backend_manager.get_current(), func_name)