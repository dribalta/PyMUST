# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyMUST is a Python reimplementation of the MUST ultrasound toolbox for synthetic ultrasound image generation and reconstruction. It maintains close syntax compatibility with the original MATLAB version while providing Python implementations of ultrasound signal processing algorithms.

## Development Commands

### Installation and Setup
```bash
# Install in development mode with dependencies
pip install -e .

# Install directly from GitHub
pip install git+https://github.com/creatis-ULTIM/PyMUST.git

# Install from PyPI
pip install pymust

# Install with optional dependencies
pip install pymust[pytorch]    # GPU acceleration via PyTorch
pip install pymust[accelerated] # PyTorch + pyfftw for FFT acceleration
```

### Testing and Validation
```bash
# Run simple functionality test
python simple_test.py

# Run comprehensive backend system tests  
python test_backend.py

# Run backend update validation
python test_backend_updates.py

# Validate field computation (basic functionality)
python validate_pfield.py

# Compare with reference versions (if available)
python ../compare_versions.py

# Test specific field computations
python ../test_field.py      # 2D field tests
python ../test_field3.py     # 3D field tests
```

### Package Building
The project uses modern Python packaging with `pyproject.toml` and setuptools-scm for version management.

```bash
# Build package
python -m build

# Clean build artifacts
python setup.py clean --all
```

## Code Architecture

### Core Module Structure
The main package is located in `src/pymust/` and follows a modular architecture:

**Signal Processing Pipeline:**
- `simus.py` - Main simulation engine for ultrasound RF signal generation
- `pfield.py` - RMS acoustic pressure field calculations for 2D arrays
- `pfield3.py` - 3D acoustic field calculations
- `rf2iq.py` - RF to I/Q signal demodulation
- `bmode.py` - B-mode image generation from I/Q signals
- `iq2doppler.py` - Doppler processing from I/Q signals

**Array and Beam Management:**
- `getparam.py` - Transducer parameter definitions
- `txdelay.py` - Element delay calculations for 2D arrays
- `txdelay3.py` - Element delay calculations for 3D arrays
- `dasmtx.py` - Delay-and-Sum (DAS) matrix generation for 2D
- `dasmtx3.py` - DAS matrix generation for 3D

**Utilities and Processing:**
- `tgc.py` - Time Gain Compensation
- `genscat.py` - Scatterer generation
- `sptrack.py` - Speckle tracking for motion estimation
- `smoothn.py` - Smoothing algorithms
- `utils.py` - Common utilities and parameter classes
- `mkmovie.py` - Movie generation utilities

**Key Design Principles:**
- Maintains MATLAB-like function call syntax for compatibility
- Functions return maximum number of variables from MATLAB version
- Supports both 2D and 3D ultrasound simulations
- Modular design allows selective importing of functions

### Development Mode
The `__init__.py` includes an `interactiveDevelopment` flag that can be set to `True` for easier module reloading during development.

## Examples and Tutorials

**Examples Directory (`examples/`):**
- `quickstart_demo.ipynb` - Main getting started notebook
- `pfield3.ipynb` - 3D acoustics demonstration
- `rotatingDiskVelocitySynthetic.ipynb` - Motion estimation examples
- Various specialized demos for different PyMUST features

**Tutorials Directory (`tutorials/`):**
- Course materials for Universitat Pompeu Fabra
- Incomplete notebooks for practical sessions on ultrasound imaging

## Backend System

PyMUST now includes a flexible backend abstraction system that separates physics from math/solver implementations:

### Backend Configuration
```python
import pymust

# Set backend (numpy is default)
pymust.backend.set("numpy")      # Default backend
pymust.backend.set("pytorch")    # GPU-accelerated backend (optional)

# Check available backends
print(pymust.backend.available())

# Get current backend
print(pymust.backend.current())
```

### Supported Backends
- **NumPy** (default): Standard CPU-based computations using NumPy/SciPy
- **PyTorch** (optional): GPU-accelerated computations with automatic differentiation

### Type Preservation
The backend system implements "type in, type out" behavior:
- NumPy arrays → NumPy arrays
- PyTorch tensors → PyTorch tensors
- Automatic type conversion when switching backends

## Dependencies

Core dependencies defined in `pyproject.toml`:
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical computations (required)
- `scipy` - Scientific computing functions (required)

Optional dependencies:
- `torch` - PyTorch backend for GPU acceleration (install with: `pip install pymust[pytorch]`)
- `pyfftw` - Future FFT acceleration (install with: `pip install pymust[accelerated]`)

## Development Notes

### Version Management and Build
- The project uses setuptools-scm for automatic version management from git tags
- Both `setup.py` and `pyproject.toml` are present - `pyproject.toml` is the modern standard
- Build system requires setuptools>=61.0 and setuptools-scm>=8

### Testing Architecture
The repository includes multiple levels of testing:
- **Unit Tests**: `simple_test.py`, `test_backend.py`, `test_backend_updates.py`
- **Integration Tests**: `validate_pfield.py` validates core acoustics computations
- **Comparison Tests**: `../compare_versions.py` compares against reference implementations
- **Field Tests**: `../test_field.py`, `../test_field3.py` test specific acoustic field scenarios

### Backend System Testing
The backend abstraction system can be validated through:
- `test_backend.py`: Comprehensive backend switching and type preservation tests
- `simple_test.py`: Basic functionality verification for updated modules
- Manual testing: Import backends and verify `pymust.backend.available()` shows correct status

### Platform Considerations  
- Parallelization support may be limited on Windows platforms
- Small numerical differences from MATLAB version are expected
- GPU acceleration via PyTorch backend available with optional dependencies
- Cross-platform compatibility maintained for Windows, Linux, and macOS

### Development Environment
- Set `interactiveDevelopment = True` in `__init__.py` for easier module reloading during development
- Use virtual environments for testing different dependency combinations
- Test with both NumPy-only and PyTorch-enabled configurations

### Upcoming Features
- GPU acceleration and harmonic imaging features are planned for future releases
- Differentiable rendering capabilities under development