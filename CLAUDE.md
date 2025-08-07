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
```

### Package Building
The project uses modern Python packaging with `pyproject.toml` and setuptools-scm for version management.

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

- The project uses setuptools-scm for automatic version management from git tags
- Parallelization support may be limited on Windows platforms
- Small numerical differences from MATLAB version are expected
- GPU acceleration and harmonic imaging features are planned for future releases