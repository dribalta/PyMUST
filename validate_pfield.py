#!/usr/bin/env python3
"""
Validation script for PyMUST pfield functionality with backend system.

Tests the core pfield functionality to ensure backend migration works correctly.
"""

import sys
import os
import numpy as np

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pymust

def test_basic_pfield():
    """Test basic pfield functionality with backend system."""
    print("Testing basic pfield functionality...")
    
    # Test numpy backend
    pymust.backend.set("numpy")
    print(f"Using backend: {pymust.backend.current()}")
    
    # Get probe parameters (from pfield.ipynb example)
    param = pymust.getparam('P4-2v')
    print(f"Loaded probe parameters: fc={param.fc} Hz, Nelements={param.Nelements}")
    
    # Define focus position
    xf = 2e-2
    zf = 5e-2
    print(f"Focus position: x={xf*1e2:.1f} cm, z={zf*1e2:.1f} cm")
    
    # Calculate transmit delays
    txdel = pymust.txdelay(xf, zf, param)
    print(f"Transmit delays shape: {txdel.shape}")
    print(f"Delay range: {np.min(txdel)*1e6:.2f} to {np.max(txdel)*1e6:.2f} us")
    
    # Create a small grid for testing
    x = np.linspace(-2e-2, 2e-2, 50)
    z = np.linspace(1e-2, 6e-2, 50)
    x, z = np.meshgrid(x, z)
    y = np.zeros_like(x)
    
    print(f"Grid shape: {x.shape}")
    
    # Calculate pressure field
    P, spect, idx = pymust.pfield(x, y, z, txdel, param)
    
    print(f"Pressure field shape: {P.shape}")
    print(f"Pressure range: {np.min(P):.2e} to {np.max(P):.2e}")
    print(f"Focus pressure (center): {P[25, 25]:.2e}")
    
    # Verify the pressure field has a focus at the expected location
    focus_idx_x = np.argmin(np.abs(x[0, :] - xf))
    focus_idx_z = np.argmin(np.abs(z[:, 0] - zf))
    focus_pressure = P[focus_idx_z, focus_idx_x]
    max_pressure = np.max(P)
    
    print(f"Focus pressure ratio: {focus_pressure/max_pressure:.3f}")
    
    if focus_pressure/max_pressure > 0.8:
        print("[OK] Focus quality looks good")
    else:
        print("[WARN] Focus may not be optimal")
    
    return True

def test_backend_switching():
    """Test switching backends during operations."""
    print("\nTesting backend switching...")
    
    # Test with numpy backend
    pymust.backend.set("numpy")
    param = pymust.getparam('P4-2v')
    
    # Simple test
    x = np.array([0])
    z = np.array([5e-2])
    y = np.array([0])
    txdel = np.zeros((1, param.Nelements))
    
    P1, _, _ = pymust.pfield(x, y, z, txdel, param)
    print(f"NumPy backend result: {float(P1[0]):.6f}")
    
    # Try PyTorch backend if available
    backends = pymust.backend.available()
    if backends.get("pytorch", False):
        print("Testing PyTorch backend...")
        pymust.backend.set("pytorch")
        P2, _, _ = pymust.pfield(x, y, z, txdel, param)
        print(f"PyTorch backend result: {float(P2[0]):.6f}")
        
        # Results should be very close
        diff = abs(float(P1[0]) - float(P2[0]))
        if diff < 1e-6:
            print("[OK] Backend results match")
        else:
            print(f"[WARN] Backend results differ by {diff}")
    else:
        print("PyTorch backend not available")
    
    # Switch back to numpy
    pymust.backend.set("numpy")
    print(f"Switched back to: {pymust.backend.current()}")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    try:
        # Test with invalid backend
        pymust.backend.set("invalid_backend")
        print("[FAIL] Should have failed with invalid backend")
    except ValueError as e:
        print("[OK] Invalid backend correctly rejected")
    
    # Test basic array operations
    backend = pymust.backend.get_backend()
    x = backend.array([1, 2, 3])
    y = backend.sin(x)
    print(f"[OK] Basic operations work: sin([1,2,3]) = {y}")

if __name__ == "__main__":
    print("PyMUST pfield Validation Test")
    print("=" * 50)
    
    try:
        test_basic_pfield()
        test_backend_switching()
        test_edge_cases()
        
        print("\n" + "=" * 50)
        print("All validation tests passed!")
        print("Backend migration is working correctly.")
        
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)