#!/usr/bin/env python3
"""
Test script for PyMUST backend system.

Tests backend switching, type preservation, and basic functionality.
"""

import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pymust
import numpy as np

def test_backend_availability():
    """Test which backends are available."""
    print("Available backends:")
    backends = pymust.backend.available()
    for name, available in backends.items():
        status = "[OK]" if available else "[NO]"
        print(f"  {status} {name}")
    print()

def test_numpy_backend():
    """Test NumPy backend functionality."""
    print("Testing NumPy backend...")
    pymust.backend.set("numpy")
    print(f"Current backend: {pymust.backend.current()}")
    
    # Test basic operations
    backend = pymust.backend.get_backend()
    x = backend.array([1, 2, 3, 4, 5])
    y = backend.sin(x)
    print(f"sin([1,2,3,4,5]) = {y}")
    
    # Test type checking
    print(f"Is numpy array: {backend.is_array(x)}")
    print()

def test_pytorch_backend():
    """Test PyTorch backend if available."""
    backends = pymust.backend.available()
    if not backends.get("pytorch", False):
        print("PyTorch backend not available (PyTorch not installed)")
        print()
        return
    
    print("Testing PyTorch backend...")
    pymust.backend.set("pytorch")
    print(f"Current backend: {pymust.backend.current()}")
    
    # Test basic operations
    backend = pymust.backend.get_backend()
    x = backend.array([1, 2, 3, 4, 5])
    y = backend.sin(x)
    print(f"sin([1,2,3,4,5]) = {y}")
    print(f"Type: {type(y)}")
    
    # Test type checking
    print(f"Is torch tensor: {backend.is_array(x)}")
    print()

def test_type_preservation():
    """Test type in, type out behavior."""
    print("Testing type preservation...")
    
    # Start with numpy
    pymust.backend.set("numpy")
    np_array = np.array([1.0, 2.0, 3.0])
    
    # Test utils functions with type preservation
    result = pymust.utils.mysinc(np_array)
    print(f"NumPy input type: {type(np_array)}")
    print(f"Result type: {type(result)}")
    
    # Test with PyTorch if available
    backends = pymust.backend.available()
    if backends.get("pytorch", False):
        pymust.backend.set("pytorch")
        try:
            import torch
            torch_tensor = torch.tensor([1.0, 2.0, 3.0])
            
            # The backend should handle torch tensors appropriately
            backend = pymust.backend.get_backend()
            result = backend.sin(torch_tensor)
            print(f"PyTorch input type: {type(torch_tensor)}")
            print(f"Result type: {type(result)}")
        except ImportError:
            print("PyTorch not available for type testing")
    
    print()

def test_backend_switching():
    """Test switching between backends."""
    print("Testing backend switching...")
    
    # Test switching to numpy
    pymust.backend.set("numpy")
    print(f"Switched to: {pymust.backend.current()}")
    
    # Test switching to pytorch if available
    backends = pymust.backend.available()
    if backends.get("pytorch", False):
        pymust.backend.set("pytorch")
        print(f"Switched to: {pymust.backend.current()}")
        
        # Switch back to numpy
        pymust.backend.set("numpy")
        print(f"Switched back to: {pymust.backend.current()}")
    
    print()

if __name__ == "__main__":
    print("PyMUST Backend System Test")
    print("=" * 40)
    
    try:
        test_backend_availability()
        test_numpy_backend()
        test_pytorch_backend()
        test_type_preservation()
        test_backend_switching()
        
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)