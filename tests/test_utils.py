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
    """Test that Param handles case-sensitive access before normalization and case-insensitive after."""
    param = utils.Param()

    # Set field with non-canonical case (FC instead of fc)
    param.FC = 2.5e6

    # Before ignoreCaseInFieldNames, accessing with wrong case returns None (dict.get behavior)
    assert param.FC == 2.5e6  # Original case works
    assert param.fc is None   # Canonical case returns None (field doesn't exist yet with that case)

    # Call ignoreCaseInFieldNames to normalize field names
    param.ignoreCaseInFieldNames()

    # After normalization, field is accessible with canonical case
    assert param.fc == 2.5e6      # Now canonical case works
    assert 'fc' in param           # Canonical key exists
    assert 'FC' not in param       # Original non-canonical key is gone


def test_param_ignore_case_method():
    """Test that ignoreCaseInFieldNames normalizes mixed-case fields to canonical form."""
    param = utils.Param()

    # Set fields with various non-canonical cases
    param.FC = 3.5e6        # Should become 'fc'
    param.PITCH = 0.3e-3    # Should become 'pitch'
    param.Width = 0.27e-3   # Should become 'width'

    # Verify original mixed-case fields exist
    assert param.FC == 3.5e6
    assert param.PITCH == 0.3e-3
    assert param.Width == 0.27e-3

    # Call ignoreCaseInFieldNames
    param_after = param.ignoreCaseInFieldNames()

    # Method should return the param object (allows chaining)
    assert param_after is param

    # Verify all fields normalized to canonical form
    assert param.fc == 3.5e6
    assert param.pitch == 0.3e-3
    assert param.width == 0.27e-3

    # Verify original mixed-case keys are deleted
    assert 'FC' not in param
    assert 'PITCH' not in param
    assert 'Width' not in param

    # Verify canonical keys exist
    assert 'fc' in param
    assert 'pitch' in param
    assert 'width' in param


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
    """Test that Options handles case-sensitive access before normalization and case-insensitive after."""
    options = utils.Options()

    # Set field with non-canonical case (parpool instead of ParPool)
    options.parpool = True

    # Before ignoreCaseInFieldNames, accessing with wrong case returns None (dict.get behavior)
    assert options.parpool == True  # Original case works
    assert options.ParPool is None  # Canonical case returns None (field doesn't exist yet with that case)

    # Call ignoreCaseInFieldNames to normalize field names
    options.ignoreCaseInFieldNames()

    # After normalization, field is accessible with canonical case
    assert options.ParPool == True   # Now canonical case works
    assert 'ParPool' in options      # Canonical key exists
    assert 'parpool' not in options  # Original non-canonical key is gone


def test_options_ignore_case_method():
    """Test that ignoreCaseInFieldNames normalizes mixed-case Options fields to canonical form."""
    options = utils.Options()

    # Set fields with various non-canonical cases
    options.parpool = False           # Should become 'ParPool'
    options.waitbar = True            # Should become 'WaitBar'
    options.dbthresh = -60            # Should become 'dBThresh'

    # Verify original mixed-case fields exist
    assert options.parpool == False
    assert options.waitbar == True
    assert options.dbthresh == -60

    # Call ignoreCaseInFieldNames
    options_after = options.ignoreCaseInFieldNames()

    # Method should return the options object (allows chaining)
    assert options_after is options

    # Verify all fields normalized to canonical form
    assert options.ParPool == False
    assert options.WaitBar == True
    assert options.dBThresh == -60

    # Verify original mixed-case keys are deleted
    assert 'parpool' not in options
    assert 'waitbar' not in options
    assert 'dbthresh' not in options

    # Verify canonical keys exist
    assert 'ParPool' in options
    assert 'WaitBar' in options
    assert 'dBThresh' in options


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


def test_param_with_real_preset():
    """Test that Param works with real transducer presets."""
    param = get_standard_param()

    # Should have required fields
    assert hasattr(param, 'fc')
    assert hasattr(param, 'pitch')
    assert param.fc > 0
    assert param.pitch > 0
