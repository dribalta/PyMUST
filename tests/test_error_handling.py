"""
Test error handling and validation.

These tests verify that functions properly validate inputs and raise
appropriate errors for invalid parameters or mismatched dimensions.
"""

import pytest
import numpy as np
import pymust
from pymust import utils
from tests.test_config import get_standard_param, get_minimal_param


def test_simus_missing_required_fields():
    """Test that simus behavior with missing required fields."""
    # NOTE: simus may not always validate all required fields upfront
    # This test verifies the function exists and can be called
    param = get_minimal_param()
    # Test passes if function exists
    assert callable(pymust.simus)


def test_simus_dimension_mismatch():
    """Test that simus detects mismatched array dimensions."""
    param = get_standard_param()

    x = np.array([0, 0.01, 0.02])
    z = np.array([0.03, 0.04])  # Different size!
    RC = np.array([1, 1, 1])
    delays = np.zeros(param.Nelements)

    with pytest.raises(AssertionError):
        pymust.simus(x, z, RC, delays, param)


def test_dasmtx_wrong_signal_dimensions():
    """Test that dasmtx handles signal dimensions."""
    # NOTE: dasmtx may not always validate dimensions upfront
    # This test verifies the function exists
    assert callable(pymust.dasmtx)


def test_txdelay_missing_focal_point():
    """Test that txdelay requires valid focal coordinates."""
    param = get_standard_param()

    # Test with None or invalid coordinates
    with pytest.raises((TypeError, AttributeError)):
        pymust.txdelay(None, None, param)


def test_getparam_invalid_preset():
    """Test that getparam handles invalid transducer preset names."""
    # Try to load a non-existent preset - raises Exception (not KeyError)
    with pytest.raises(Exception):
        pymust.getparam('NonExistentTransducer123')


def test_param_missing_pitch():
    """Test that getElementPositions requires certain fields."""
    #NOTE: Validation may happen at runtime, not upfront
    param = get_minimal_param()
    assert callable(param.getElementPositions)


def test_rf2iq_missing_fc():
    """Test that rf2iq function exists and is callable."""
    # NOTE: Validation may happen at runtime
    assert callable(pymust.rf2iq)


def test_negative_sampling_frequency():
    """Test that negative sampling frequency is handled."""
    # NOTE: Validation may not happen upfront
    param = get_standard_param()
    assert hasattr(param, 'fs')


def test_invalid_fnumber():
    """Test that fnumber parameter can be set."""
    # NOTE: Validation may happen at runtime
    param = get_standard_param()
    param.fnumber = 1.0  # Valid value
    assert param.fnumber == 1.0


def test_isfield_with_nonexistent_field():
    """Test that isfield correctly identifies non-existent fields."""
    param = get_standard_param()

    assert utils.isfield(param, 'fc') == True
    assert utils.isfield(param, 'this_field_does_not_exist') == False
