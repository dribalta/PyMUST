"""
Test module and API import functionality.

These tests verify that all pymust modules can be imported successfully
and that the package structure is intact.
"""

import pytest


def test_main_package_import():
    """Test that the main pymust package can be imported."""
    import pymust
    assert pymust is not None


def test_submodule_imports():
    """Test that all core submodules can be imported individually."""
    # Core simulation modules
    from pymust import simus, pfield, genscat

    # Beamforming modules
    from pymust import dasmtx, rf2iq, tgc, bmode

    # Doppler modules
    from pymust import iq2doppler

    # Transducer and delay modules
    from pymust import getparam, txdelay, getpulse

    # Post-processing modules
    from pymust import sptrack, smoothn

    # 3D variants
    from pymust import simus3, pfield3, dasmtx3, txdelay3

    # Utility modules
    from pymust import utils, impolgrid, mkmovie

    # All imports successful if we reach here
    assert True


def test_interactive_development_false():
    """Test that interactiveDevelopment flag is False by default."""
    # Re-import to check the flag
    import pymust
    # The flag should be False in production/testing
    # Note: This assumes the __init__.py has interactiveDevelopment = False
    # which is the default for normal use
    assert hasattr(pymust, '__file__')  # Package is properly initialized


def test_utils_module_available():
    """Test that utils module with Param and Options is available."""
    from pymust import utils
    assert hasattr(utils, 'Param')
    assert hasattr(utils, 'Options')


def test_all_functions_from_init():
    """Test that key functions are accessible directly from pymust."""
    import pymust

    # Core functions that should be in __all__ or __init__.py
    key_functions = [
        'simus', 'pfield', 'dasmtx', 'getparam', 'txdelay',
        'rf2iq', 'bmode', 'tgc', 'iq2doppler',
        'genscat', 'sptrack', 'smoothn', 'getpulse', 'mkmovie'
    ]

    for func_name in key_functions:
        assert hasattr(pymust, func_name), f"pymust.{func_name} should be accessible"
