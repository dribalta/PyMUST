"""
Helper utilities for backend migration.

Provides convenience functions to ease migration from direct numpy usage
to backend abstraction.
"""

from typing import Any
from .manager import get_backend


def get_array_module():
    """Get the array module for the current backend."""
    return get_backend().get_array_module()


def backend_function(func_name: str):
    """Get a function from the current backend."""
    return getattr(get_backend(), func_name)


def ensure_array(*arrays: Any):
    """Ensure arrays are compatible with current backend."""
    backend = get_backend()
    return backend.ensure_backend_array(*arrays)


# Common array operations that are frequently used
def zeros(*args, **kwargs):
    """Create zeros array using current backend."""
    return get_backend().zeros(*args, **kwargs)


def ones(*args, **kwargs):
    """Create ones array using current backend."""
    return get_backend().ones(*args, **kwargs)


def array(*args, **kwargs):
    """Create array using current backend."""
    return get_backend().array(*args, **kwargs)


def arange(*args, **kwargs):
    """Create range array using current backend."""
    return get_backend().arange(*args, **kwargs)


def linspace(*args, **kwargs):
    """Create linearly spaced array using current backend."""
    return get_backend().linspace(*args, **kwargs)


def pi():
    """Get pi constant from current backend."""
    return get_backend().pi


def inf():
    """Get infinity constant from current backend."""
    return get_backend().inf


def isnan(*args, **kwargs):
    """Check for NaN values using current backend."""
    return get_backend().isnan(*args, **kwargs)


def isinf(*args, **kwargs):
    """Check for infinite values using current backend."""
    return get_backend().isinf(*args, **kwargs)


def isscalar(*args, **kwargs):
    """Check if value is scalar using current backend."""
    return get_backend().isscalar(*args, **kwargs)


def abs(*args, **kwargs):
    """Absolute value using current backend."""
    return get_backend().abs(*args, **kwargs)


def sqrt(*args, **kwargs):
    """Square root using current backend."""
    return get_backend().sqrt(*args, **kwargs)


def sin(*args, **kwargs):
    """Sine function using current backend."""
    return get_backend().sin(*args, **kwargs)


def cos(*args, **kwargs):
    """Cosine function using current backend."""
    return get_backend().cos(*args, **kwargs)


def exp(*args, **kwargs):
    """Exponential function using current backend."""
    return get_backend().exp(*args, **kwargs)


def log(*args, **kwargs):
    """Natural logarithm using current backend."""
    return get_backend().log(*args, **kwargs)