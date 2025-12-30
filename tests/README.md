# PyMUST Test Suite

This directory contains the test suite for PyMUST, organized into fast unit tests and comprehensive computational validation tests.

## Overview

The test suite is designed with two priorities:
- **Fast CI/CD testing**: Quick validation of core functionality (~1-2 seconds)
- **Comprehensive validation**: Long-running computational tests for numerical accuracy

Tests use pytest markers to separate fast tests from computationally expensive ones, allowing flexible testing workflows.

## Test Categories

### Fast Tests (Default)
Run by default in CI/CD and during development. Complete in seconds.


### Long Tests (`@pytest.mark.long`)
Computationally intensive tests requiring significant processing time.

- ⚠️ **PLACEHOLDER** - Many tests are placeholders awaiting implementation
- Run explicitly via `pytest -m "long"` or `tox -e long`
- Include full simulation workflows, numerical validation, and baseline comparisons

## Test Structure

```
tests/
├── README.md                    # This file
├── __init__.py                  # Test package initialization
│
├── test_config.py              # 🔧 Shared fixtures and test configuration
│
├── test_imports.py             # ✅ Module import validation
├── test_api.py                 # ✅ Public API availability
├── test_utils.py               # ✅ Utility functions and Param class
├── test_coordinates.py         # ✅ Element position calculations
├── test_error_handling.py      # ✅ Input validation and error cases
│
├── test_integration.py         # ⚠️ PLACEHOLDER - Full workflow tests
└── test_numerical.py           # ⚠️ PLACEHOLDER - Numerical validation
```

## File Descriptions

### Helper & Configuration

#### `test_config.py` 🔧
Reusable test fixtures and configuration helpers.

**Purpose**: Centralized parameter generation for consistent testing
- `get_standard_param()` - P4-2v linear array (most common)
- `get_convex_param()` - C5-2V convex array
- `get_minimal_param()` - Empty Param for error testing

**Usage**: Import these helpers in your tests instead of creating params manually

### Fast Tests (Fully Implemented ✅)

#### `test_imports.py`
Validates package structure and module imports.

**Tests**:
- Main package import (`import pymust`)
- All submodule imports (simus, pfield, dasmtx, etc.)
- Public API accessibility

**When to Run**: Always - catches import errors and packaging issues

---

#### `test_api.py`
Verifies public API functions are callable and have expected signatures.

**Tests**:
- Core functions (simus, pfield, dasmtx, etc.) are callable
- 3D variants available (simus3, pfield3, etc.)
- Helper functions accessible (txdelay variants, Doppler helpers)
- Function signatures support MATLAB-style variable arguments

**When to Run**: Always - ensures API stability

---

#### `test_utils.py`
Tests utility functions and Param/Options classes.

**Tests**:
- Param object creation and field assignment
- Case-insensitive field access via `ignoreCaseInFieldNames()`
- Options class functionality
- Utility functions: `isfield`, `isEmpty`, `iscomplex`, `isnumeric`, `islogical`, `interp1`, `eps`
- Integration with real transducer presets

**When to Run**: Always - validates core infrastructure

---

#### `test_coordinates.py`
Validates coordinate system calculations for linear and convex arrays.

**Tests**:
- Linear array: elements at z=0, symmetric around x=0
- Convex array: varying z coordinates, varying element angles
- Element spacing matches `pitch` parameter
- No NaN or infinite values in positions
- Element count matches `Nelements`

**When to Run**: Always - ensures geometric calculations are correct

---

#### `test_error_handling.py`
Tests input validation and error cases.

**Tests**:
- Missing required fields (partial validation)
- Dimension mismatches in array inputs
- Invalid transducer preset names
- Field existence checking with `isfield`

**Note**: Some validation happens at runtime, not upfront, so tests verify callable behavior

**When to Run**: Always - catches invalid inputs early

### Long Tests (Mostly Placeholders ⚠️)

#### `test_integration.py` ⚠️ **PLACEHOLDER**
Full ultrasound simulation workflows from start to finish.

**Planned Tests** (Not Yet Implemented):
- Complete simulation: `getparam → txdelay → simus`
- Beamforming pipeline: RF generation → dasmtx → reconstruction
- Pressure field computation with realistic grids
- Doppler processing workflow (RF → I/Q → velocity estimation)
- Speckle tracking with known displacement
- 3D simulation workflows
- Parallel processing with ParPool
- Multi-line transmit (MLT) acquisition

**Status**: All tests currently skip with `pytest.skip("Placeholder...")`

**When to Implement**: After core functionality is stable and baseline data available

---

#### `test_numerical.py` ⚠️ **PLACEHOLDER**
Numerical validation against MATLAB MUST reference data.

**Planned Tests** (Not Yet Implemented):
- Baseline comparison for simus output
- Pressure field validation against analytical solutions
- Beamformed image quality metrics
- Doppler velocity estimation accuracy
- Numerical stability across parameter ranges
- Energy conservation in simulations

**Requirements**:
- Reference datasets from MATLAB MUST (`.npz` format)
- Baseline data loading/saving infrastructure
- Tolerance definitions for numerical differences (~1% RMS)

**Status**: Infrastructure functions defined but not implemented (`save_baseline_data`, `load_baseline_data`, `compare_arrays_with_tolerance`)

**When to Implement**: After baseline data generation from MATLAB MUST

## Running Tests

### Quick Development Testing
```bash
# Run all fast tests (excludes @pytest.mark.long)
pytest -m "not long"

# Run specific test file
pytest tests/test_utils.py

# Run specific test function
pytest tests/test_coordinates.py::test_linear_array_element_positions

# Run with verbose output
pytest -v tests/test_imports.py
```

### Long-Running Tests
```bash
# Run only long tests
pytest -m "long"

# Run long tests via tox
tox -e long
```

### Using Tox (Multi-Version Testing)
```bash
# Run fast tests on all Python versions
tox

# Run on specific Python version
tox -e py310

# Run long tests on specific version
tox -e long -- --maxfail=1
```

### CI/CD Default
```bash
# What runs in continuous integration (fast only)
pytest -m "not long"
```

## Adding New Tests

### Adding Tests to Existing Files

1. **Import necessary modules**:
```python
import pytest
import numpy as np
import pymust
from tests.test_config import get_standard_param
```

2. **Write test function** with descriptive name:
```python
def test_descriptive_name():
    """Clear docstring explaining what is tested."""
    param = get_standard_param()

    # Test implementation
    result = pymust.some_function(param)

    # Assertions
    assert result is not None
    assert np.all(np.isfinite(result))
```

3. **Use markers for long tests**:
```python
@pytest.mark.long
def test_expensive_computation():
    """This test takes significant time."""
    pytest.skip("Placeholder - not yet implemented")  # Remove when implementing
```

### When to Create New Test Files

Create a new test file when testing a distinct category:

```python
# tests/test_new_category.py
"""
Brief description of what this category tests.

Explain the scope and purpose clearly.
"""

import pytest
import pymust
from tests.test_config import get_standard_param

def test_something():
    """Individual test."""
    pass
```

### Using Test Fixtures

Instead of creating parameters in every test, use shared fixtures:

```python
# ✅ Good - Reuse fixtures
from tests.test_config import get_standard_param, get_convex_param

def test_with_linear_array():
    param = get_standard_param()  # P4-2v linear array
    # ... test code

def test_with_convex_array():
    param = get_convex_param()    # C5-2V convex array
    # ... test code

# ❌ Avoid - Creating params manually
def test_manual_param():
    param = pymust.getparam('P4-2v')  # Prefer fixtures
    # ... test code
```

### Test Marker Usage

```python
# Fast test (default - no marker needed)
def test_quick_validation():
    assert True

# Long computational test
@pytest.mark.long
def test_full_simulation():
    # Expensive computation here
    pass

# Multiple markers
@pytest.mark.long
@pytest.mark.integration
def test_complex_workflow():
    # Integration test that's also long
    pass
```

### Placeholder Pattern for Future Tests

```python
@pytest.mark.long
def test_future_feature():
    """
    Test description of what will be tested.

    TODO: Implement with specific grid size (100x100)
    TODO: Verify expected behavior
    TODO: Compare against baseline data
    """
    pytest.skip("Placeholder - not yet implemented")
```

## Notes

- **Small numerical differences** from MATLAB MUST are expected and acceptable
- **Placeholder tests** serve as documentation of planned validation
- **Baseline data** from MATLAB MUST is needed for numerical tests
- **Parallel processing** tests may not work on Windows
- Test coverage will expand as baseline data becomes available
