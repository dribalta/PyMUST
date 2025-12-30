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


def test_simus_dimension_mismatch():
    """Test that simus detects mismatched array dimensions."""
    param = get_standard_param()

    x = np.array([0, 0.01, 0.02])
    z = np.array([0.03, 0.04])  # Different size!
    RC = np.array([1, 1, 1])
    delays = np.zeros(param.Nelements)

    with pytest.raises(AssertionError):
        pymust.simus(x, z, RC, delays, param)


def test_txdelay_missing_focal_point():
    """Test that txdelay requires valid focal coordinates."""
    param = get_standard_param()

    # Test with None or invalid coordinates
    with pytest.raises((TypeError, AttributeError)):
        pymust.txdelay(None, None, param)


def test_getparam_invalid_preset():
    """Test that getparam handles invalid transducer preset names."""
    # Try to load a non-existent preset - raises Exception
    with pytest.raises(Exception):
        pymust.getparam('NonExistentTransducer123')


def test_isfield_with_nonexistent_field():
    """Test that isfield correctly identifies non-existent fields."""
    param = get_standard_param()

    assert utils.isfield(param, 'fc') == True
    assert utils.isfield(param, 'this_field_does_not_exist') == False
