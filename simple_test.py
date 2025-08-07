#!/usr/bin/env python3
"""Simple test to verify backend updates work."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic functionality of updated modules."""
    print("Testing basic functionality...")
    
    try:
        # Test backend import
        from pymust.backend import get_backend
        backend = get_backend()
        print(f"Backend: {type(backend).__module__}.{type(backend).__name__}")
        
        # Test creating a simple array
        arr = backend.array([1, 2, 3])
        print(f"Simple array test: {arr}")
        
        # Test importing the updated modules  
        print("\nImporting updated modules...")
        import pymust.pfield3 as pfield3_module
        import pymust.genscat as genscat_module
        import pymust.txdelay3 as txdelay3_module
        
        print("pfield3 imported successfully")
        print("genscat imported successfully") 
        print("txdelay3 imported successfully")
        
        # Check if functions are defined by looking at the module source
        print("\nChecking function availability...")
        
        # Check pfield3 functions
        pfield3_funcs = [name for name in dir(pfield3_module) if not name.startswith('_')]
        print(f"pfield3 functions/variables: {pfield3_funcs}")
        
        # Check genscat functions  
        genscat_funcs = [name for name in dir(genscat_module) if not name.startswith('_')]
        print(f"genscat functions/variables: {genscat_funcs}")
        
        # Check txdelay3 functions
        txdelay3_funcs = [name for name in dir(txdelay3_module) if not name.startswith('_')]
        print(f"txdelay3 functions/variables: {txdelay3_funcs}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Simple Backend Update Test")
    print("=" * 30)
    
    if test_basic_functionality():
        print("\n" + "=" * 30)
        print("SUCCESS: Backend abstraction is working!")
    else:
        print("\n" + "=" * 30)
        print("FAILED: Issues found with backend abstraction.")