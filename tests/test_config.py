"""
Reusable test configuration and fixtures.

This module provides helper functions to generate common test parameters
and configurations used across the test suite.
"""

import pymust
from pymust import utils


def get_standard_param():
    """
    Returns standard P4-2v transducer parameters for testing.

    This is the most commonly used transducer in examples and provides
    a good baseline for testing linear array functionality.

    Returns:
        utils.Param: Configured P4-2v transducer parameters
    """
    return pymust.getparam('P4-2v')


def get_convex_param():
    """
    Returns C5-2V convex array parameters for testing.

    Used for testing curved array functionality and coordinate
    system calculations that differ from linear arrays.

    Returns:
        utils.Param: Configured C5-2V convex array parameters
    """
    return pymust.getparam('C5-2V')


def get_minimal_param():
    """
    Returns an empty Param object for error handling tests.

    This is useful for testing validation logic and error cases
    where required fields are missing.

    Returns:
        utils.Param: Empty parameter object
    """
    return utils.Param()
