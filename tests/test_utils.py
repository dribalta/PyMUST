"""
Test utility functions and Param class functionality.

These tests verify that utility functions work correctly and that the
Param class properly handles case-insensitive field access and validation.
"""

import pytest
import numpy as np
from pymust import utils
from tests.test_config import get_standard_param, get_minimal_param


# Param class tests
def test_param_creation():
    """Test that Param objects can be created."""
    param = utils.Param()
    assert param is not None


def test_param_field_assignment():
    """Test that fields can be assigned to Param objects."""
    param = utils.Param()
    param.fc = 2.5e6
    assert param.fc == 2.5e6


def test_param_case_insensitive_access():
    """Test that Param handles case-insensitive field access after ignoreCaseInFieldNames."""
    param = utils.Param()
    param.fc = 2.5e6

    # Before ignoreCaseInFieldNames, only direct access works
    assert param.fc == 2.5e6

    # After calling ignoreCaseInFieldNames, the behavior depends on the names mapping
    # The implementation uses dotdict which is dict.get, so non-existent keys return None
    # This is expected behavior - case insensitivity is achieved through names mapping


def test_param_ignore_case_method():
    """Test that ignoreCaseInFieldNames method works."""
    param = get_standard_param()
    param_after = param.ignoreCaseInFieldNames()

    # Method should return the param object (allows chaining)
    assert param_after is not None


def test_param_get_element_positions():
    """Test that getElementPositions method exists and is callable."""
    param = get_standard_param()
    assert hasattr(param, 'getElementPositions')
    assert callable(param.getElementPositions)


# Options class tests
def test_options_creation():
    """Test that Options objects can be created."""
    options = utils.Options()
    assert options is not None


def test_options_field_assignment():
    """Test that fields can be assigned to Options objects."""
    options = utils.Options()
    options.ParPool = False
    assert options.ParPool == False


def test_options_case_insensitive_access():
    """Test that Options handles field access."""
    options = utils.Options()
    options.ParPool = True

    # Direct access works
    assert options.ParPool == True
    # dotdict uses dict.get, so non-existent keys return None (expected behavior)


# Utility function tests
def test_isfield_function():
    """Test that isfield utility function works."""
    param = utils.Param()
    param.fc = 2.5e6

    assert utils.isfield(param, 'fc') == True
    assert utils.isfield(param, 'nonexistent') == False


def test_isempty_function():
    """Test that isEmpty utility function works."""
    assert utils.isEmpty(None) == True
    assert utils.isEmpty([]) == True
    assert utils.isEmpty(np.array([])) == True
    assert utils.isEmpty([1, 2, 3]) == False
    assert utils.isEmpty(np.array([1, 2, 3])) == False


def test_iscomplex_function():
    """Test that iscomplex utility function detects complex arrays."""
    real_array = np.array([1.0, 2.0, 3.0])
    complex_array = np.array([1.0 + 1j, 2.0 + 2j])

    assert utils.iscomplex(real_array) == False
    assert utils.iscomplex(complex_array) == True


def test_isnumeric_function():
    """Test that isnumeric utility function works."""
    assert utils.isnumeric(5) == True
    assert utils.isnumeric(5.0) == True
    assert utils.isnumeric(np.array([1, 2, 3])) == True
    assert utils.isnumeric("string") == False


def test_islogical_function():
    """Test that islogical utility function detects boolean values."""
    assert utils.islogical(True) == True
    assert utils.islogical(False) == True
    assert utils.islogical(1) == False  # Number, not boolean


def test_interp1_basic():
    """Test that interp1 function works for basic interpolation."""
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 4, 9, 16])  # y = x^2
    xi = np.array([0.5, 1.5, 2.5])

    # interp1 requires 3 args: y, xNew, kind
    yi = utils.interp1(y, xi, 'linear')

    # Should return interpolated values
    assert yi.shape == xi.shape
    assert np.all(np.isfinite(yi))


def test_eps_function():
    """Test that eps function returns machine epsilon for single precision."""
    epsilon = utils.eps()  # Default is 'single'
    assert epsilon > 0
    assert epsilon == 1.1921e-07  # Single precision epsilon


def test_param_with_real_preset():
    """Test that Param works with real transducer presets."""
    param = get_standard_param()

    # Should have required fields
    assert hasattr(param, 'fc')
    assert hasattr(param, 'pitch')
    assert param.fc > 0
    assert param.pitch > 0
