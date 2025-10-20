"""
Test public API availability and function signatures.

These tests verify that the public API is accessible and functions
have the expected signatures and are callable.
"""

import inspect
import pytest
import pymust


def test_simus_is_callable():
    """Test that simus function exists and is callable."""
    assert callable(pymust.simus)


def test_pfield_is_callable():
    """Test that pfield function exists and is callable."""
    assert callable(pymust.pfield)


def test_dasmtx_is_callable():
    """Test that dasmtx function exists and is callable."""
    assert callable(pymust.dasmtx)


def test_getparam_is_callable():
    """Test that getparam function exists and is callable."""
    assert callable(pymust.getparam)


def test_txdelay_is_callable():
    """Test that txdelay function exists and is callable."""
    assert callable(pymust.txdelay)


def test_rf2iq_is_callable():
    """Test that rf2iq function exists and is callable."""
    assert callable(pymust.rf2iq)


def test_bmode_is_callable():
    """Test that bmode function exists and is callable."""
    assert callable(pymust.bmode)


def test_tgc_is_callable():
    """Test that tgc function exists and is callable."""
    assert callable(pymust.tgc)


def test_iq2doppler_is_callable():
    """Test that iq2doppler function exists and is callable."""
    assert callable(pymust.iq2doppler)


def test_sptrack_is_callable():
    """Test that sptrack function exists and is callable."""
    assert callable(pymust.sptrack)


def test_genscat_is_callable():
    """Test that genscat function exists and is callable."""
    assert callable(pymust.genscat)


def test_getpulse_is_callable():
    """Test that getpulse function exists and is callable."""
    assert callable(pymust.getpulse)


def test_smoothn_is_callable():
    """Test that smoothn function exists and is callable."""
    assert callable(pymust.smoothn)


def test_mkmovie_is_callable():
    """Test that mkmovie function exists and is callable."""
    assert callable(pymust.mkmovie)


def test_3d_variants_callable():
    """Test that 3D variant functions are callable."""
    assert callable(pymust.simus3)
    assert callable(pymust.pfield3)
    assert callable(pymust.dasmtx3)
    assert callable(pymust.txdelay3)


def test_txdelay_variants_callable():
    """Test that txdelay variant functions are callable."""
    assert callable(pymust.txdelayCircular)
    assert callable(pymust.txdelayPlane)
    assert callable(pymust.txdelayFocused)


def test_txdelay3_variants_callable():
    """Test that txdelay3 variant functions are callable."""
    assert callable(pymust.txdelay3Plane)
    assert callable(pymust.txdelay3Diverging)
    assert callable(pymust.txdelay3Focused)


def test_doppler_helper_functions():
    """Test that Doppler helper functions are available."""
    assert callable(pymust.getNyquistVelocity)
    assert callable(pymust.getDopplerColorMap)


def test_impolgrid_is_callable():
    """Test that impolgrid function exists and is callable."""
    assert callable(pymust.impolgrid)


def test_simus_accepts_varargs():
    """Test that simus accepts variable arguments (MATLAB-style)."""
    sig = inspect.signature(pymust.simus)
    # Should have *varargin in signature
    params = list(sig.parameters.values())
    # Check that it has VAR_POSITIONAL parameter
    has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    assert has_varargs, "simus should accept variable positional arguments"


def test_pfield_accepts_multiple_args():
    """Test that pfield has expected parameters."""
    sig = inspect.signature(pymust.pfield)
    params = list(sig.parameters.keys())
    # pfield should have x, y, z, delaysTX, param and optional args
    assert 'x' in params
    assert 'param' in params
