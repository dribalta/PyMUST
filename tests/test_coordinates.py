"""
Test coordinate system and element position calculations.

These tests verify that element positions are calculated correctly
for both linear and convex arrays, and that coordinate systems
follow the expected conventions.
"""

import pytest
import numpy as np
from tests.test_config import get_standard_param, get_convex_param


def test_linear_array_element_positions():
    """Test that linear array elements have z=0."""
    param = get_standard_param()
    xe, ze, THe, h = param.getElementPositions()

    # Linear array should have all elements at z=0
    assert np.all(ze == 0), "Linear array elements should be at z=0"


def test_convex_array_element_positions():
    """Test that convex array elements have varying z coordinates."""
    param = get_convex_param()
    xe, ze, THe, h = param.getElementPositions()

    # Convex array should have varying z coordinates
    assert not np.all(ze == 0), "Convex array elements should have varying z"
    # Z coordinates should vary (not all the same)
    assert np.std(ze) > 0, "Convex array z coordinates should vary"


def test_element_count_matches():
    """Test that number of element positions matches Nelements."""
    param = get_standard_param()
    xe, ze, THe, h = param.getElementPositions()

    # getElementPositions returns (1, N) shaped arrays
    assert xe.shape[1] == param.Nelements
    assert ze.shape[1] == param.Nelements
    assert THe.shape[1] == param.Nelements


def test_element_positions_no_nan():
    """Test that element positions contain no NaN values."""
    param = get_standard_param()
    xe, ze, THe, h = param.getElementPositions()

    assert np.all(np.isfinite(xe)), "Element x positions should be finite"
    assert np.all(np.isfinite(ze)), "Element z positions should be finite"
    assert np.all(np.isfinite(THe)), "Element angles should be finite"


def test_element_positions_no_inf():
    """Test that element positions contain no infinite values."""
    param = get_standard_param()
    xe, ze, THe, h = param.getElementPositions()

    assert not np.any(np.isinf(xe)), "Element x positions should not be infinite"
    assert not np.any(np.isinf(ze)), "Element z positions should not be infinite"
    assert not np.any(np.isinf(THe)), "Element angles should not be infinite"


def test_linear_array_symmetry():
    """Test that linear array is symmetric around x=0."""
    param = get_standard_param()
    xe, ze, THe, h = param.getElementPositions()

    # For linear array, elements should be symmetric around x=0
    # The center should be at or near x=0
    assert np.abs(np.mean(xe)) < param.pitch, "Linear array should be centered near x=0"


def test_element_spacing_linear():
    """Test that linear array elements are spaced by pitch."""
    param = get_standard_param()
    xe, ze, THe, h = param.getElementPositions()

    # Calculate spacing between adjacent elements
    spacings = np.diff(xe)

    # All spacings should be equal to pitch (within numerical precision)
    expected_spacing = param.pitch
    assert np.allclose(spacings, expected_spacing, rtol=1e-10), \
        "Element spacing should equal pitch"


def test_element_angles_linear():
    """Test that linear array elements point in same direction (THe ≈ 0)."""
    param = get_standard_param()
    xe, ze, THe, h = param.getElementPositions()

    # For linear array, all elements should point in same direction (z-axis)
    # THe should be zero (or very close to it)
    assert np.allclose(THe, 0, atol=1e-10), "Linear array elements should point straight down"


def test_convex_array_angles_vary():
    """Test that convex array element angles vary."""
    param = get_convex_param()
    xe, ze, THe, h = param.getElementPositions()

    # For convex array, elements should have different angles
    assert not np.allclose(THe, 0), "Convex array elements should have varying angles"
    # Angles should be finite
    assert np.all(np.isfinite(THe)), "Convex array angles should be finite"
