"""
PyMUST Backend System

Provides backend abstraction for PyMUST to support multiple array computation libraries
(NumPy, PyTorch) with automatic type handling and lazy loading of optional dependencies.

Usage:
    import pymust
    
    # Set backend (numpy is default)
    pymust.backend.set("pytorch")  # or "numpy"
    
    # Functions automatically use configured backend
    result = pymust.pfield(x, y, z, delays, param)
    
    # Check available backends
    print(pymust.backend.available())
"""

from .manager import (
    set_backend as set,
    get_backend,
    get_backend_name as current,
    available_backends as available,
    backend_call,
    get_backend_function,
    set_precision,
    get_precision
)

# Expose main API
__all__ = [
    'set',
    'current', 
    'available',
    'get_backend',
    'backend_call',
    'get_backend_function',
    'set_precision',
    'get_precision'
]