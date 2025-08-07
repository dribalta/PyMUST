#!/usr/bin/env python3
"""
Test script to verify the backend abstraction updates worked correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all updated modules can be imported."""
    print("Testing imports...")
    
    try:
        from pymust import pfield3, genscat, txdelay3
        print("✓ All modules imported successfully")
        
        # Check functions exist
        functions_to_check = [
            (pfield3, 'pfield3'),
            (genscat, 'genscat'), 
            (txdelay3, 'txdelay3')
        ]
        
        for module, func_name in functions_to_check:
            if hasattr(module, func_name):
                print(f"✓ {func_name} function available")
            else:
                print(f"✗ {func_name} function not found")
                print(f"  Available functions: {[name for name in dir(module) if not name.startswith('_')]}")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backend_usage():
    """Test that backend functions are accessible."""
    print("\nTesting backend usage...")
    
    try:
        from pymust.backend import get_backend
        backend = get_backend()
        print(f"✓ Backend available: {backend.__name__}")
        
        # Test basic backend functions
        test_funcs = ['array', 'zeros', 'ones', 'sqrt', 'sin', 'cos']
        for func_name in test_funcs:
            if hasattr(backend, func_name):
                print(f"✓ {func_name} available")
            else:
                print(f"✗ {func_name} not available")
        
        return True
    except Exception as e:
        print(f"✗ Backend test failed: {e}")
        return False

def check_file_updates():
    """Check that numpy was properly replaced with backend calls."""
    print("\nChecking file updates...")
    
    files_to_check = [
        'src/pymust/pfield3.py',
        'src/pymust/genscat.py', 
        'src/pymust/txdelay3.py'
    ]
    
    all_good = True
    
    for file_path in files_to_check:
        print(f"\nChecking {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for backend import
        if 'from .backend import get_backend' in content:
            print("✓ Backend import found")
        else:
            print("✗ Backend import missing")
            all_good = False
        
        # Check for remaining numpy imports
        if 'import numpy as np' in content:
            print("✗ Found 'import numpy as np' - should be removed")
            all_good = False
        else:
            print("✓ No numpy imports found")
        
        # Count backend usage
        backend_calls = content.count('backend.')
        get_backend_calls = content.count('get_backend()')
        
        print(f"✓ Backend calls: {backend_calls}")
        print(f"✓ get_backend() calls: {get_backend_calls}")
        
        if backend_calls > 0 and get_backend_calls > 0:
            print("✓ Backend abstraction implemented")
        else:
            print("✗ Backend abstraction may be incomplete")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("Backend Abstraction Update Test")
    print("=" * 40)
    
    success = True
    success &= check_file_updates()
    success &= test_backend_usage()
    success &= test_imports()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! Backend abstraction update successful.")
    else:
        print("✗ Some tests failed. Please review the output above.")