# pylint: disable=unnecessary-lambda-assignment

import numpy as np
import tqdm

from ..utils import Param
from .methods import interpolate_spectrum

_EPS = np.finfo(np.float32).eps
mysinc = lambda x = None: np.sin(np.abs(x) + _EPS)/ (np.abs(x) + _EPS) # [NOTE: In MATLAB/numpy, sinc is sin(pi*x)/(pi*x)]

def simus(x_range: np.ndarray, z_range: np.ndarray, P_SPECT_grid: np.ndarray, x_scatterers: np.ndarray, z_scatterers: np.ndarray, interpolator_name: str, freqs: np.ndarray, RC: np.ndarray, param: Param, harmonic: bool = False, debug: bool = False, just_RF_spectrum: bool = False, lowResources: bool = False):
    """
    Simulates RF signals by interpolating a spectral grid onto scatterers.

    This function takes a pre-computed complex pressure spectrum on a 2D grid,
    interpolates it to the locations of specified scatterers (given by x_scatterers,
    z_scatterers) using the chosen method, and then computes the RF signals
    received by the transducer elements based on the re-emitted waves from these scatterers.

    Args:
        x_range (ndarray): 1D array of unique x-coordinates defining the grid.
        z_range (ndarray): 1D array of unique z-coordinates defining the grid.
        P_SPECT_grid (ndarray): 3D array (len(z_range), len(x_range), n_freq)
                                 of complex spectrum values on the grid.
        x_scatterers (ndarray): 1D array of x-coordinates for scatterer positions.
        z_scatterers (ndarray): 1D array of z-coordinates for scatterer positions.
                                Must have the same length as x_scatterers.
        interpolator_name (str): Name of the interpolation method registered in
                                 `methods.py` (e.g., 'linear', 'nearest', 'green').
        freqs (ndarray): 1D array of frequency values (Hz) corresponding to P_SPECT_grid's
                         last dimension.
        RC (ndarray | float | None): Reflection coefficients for scatterers.
                                     - ndarray: (n_scatterers,) or (n_scatterers, 1) array,
                                       where n_scatterers = len(x_scatterers).
                                     - float: Single coefficient applied to all.
                                     - None: Defaults to 1.0 for all scatterers.
        param (Param): PyMUST parameter object containing transducer and medium properties.
                       Requires attributes/methods: c, fc, Nelements, width,
                       attenuation (dB/cm/MHz), getElementPositions(), getProbeFunction().
                       May optionally use param.fs for IFFT.
        harmonic (bool): If True, adjusts the probe function frequency argument,
                         potentially simulating harmonic reception based on the
                         definition of getProbeFunction(). Defaults to False.
        debug (bool): If True, enables debug logging to track progress.
                      Defaults to False.
        just_RF_spectrum (bool): If True, skips the inverse FFT step and returns the
                                 RF spectrum directly. Defaults to False.
        lowResources (bool): If True, reduces memory usage by switching to complex64 precision.
                             Defaults to False.
    Returns:
        tuple[ndarray, ndarray]:
            - RF (ndarray): 2D array (n_time_samples, Nelements) of time-domain RF signals.
            - RF_SPECT (ndarray): 2D array (n_freq, Nelements) of complex spectrum at
                                  the transducer elements before IFFT.
    """
    # Define complex number type based on lowResources
    dtype_complex = np.complex64 if lowResources else np.complex128

    if debug: print(f"Starting RF simulation from spectral grid using '{interpolator_name}' interpolator.")

    # --- Input Validation ---
    n_scatterers = len(x_scatterers)
    n_freq = len(freqs)
    if debug: print(f"Processing {n_scatterers} scatterers.")

    if not isinstance(param, Param):
        raise TypeError("`param` must be a valid Param object.")
    required_attrs = ['c', 'fc', 'Nelements', 'width', 'attenuation']
    for attr in required_attrs:
        if not hasattr(param, attr):
            raise AttributeError(f"Parameter object 'param' missing required attribute: {attr}")

    # Ensure RC is a column vector matching scatterers
    if RC is None:
        RC = np.ones((n_scatterers, 1), dtype=np.float32)
    elif isinstance(RC, (int, float)):
        RC = np.full((n_scatterers, 1), RC, dtype=np.float32)
    elif isinstance(RC, np.ndarray):
        if RC.size == n_scatterers:
            RC = RC.reshape(-1, 1).astype(np.float32)
        else:
            raise ValueError(f"RC array shape mismatch. Expected size ({n_scatterers}), got {RC.size}")
    else:
        raise TypeError("RC must be None, a scalar, or a numpy array.")

    # --- 1. Interpolate the grid spectrum at the scatterer positions ---
    if debug: print("Calling interpolation dispatcher.")

    P_SPECT_scatterers = interpolate_spectrum(
        interpolator_name=interpolator_name,
        grid_values=P_SPECT_grid, # Pass the 3D grid
        x_scatterers=x_scatterers,
        z_scatterers=z_scatterers,
        x_range=x_range,
        z_range=z_range,
        param=param,
        freqs=freqs
    ).astype(dtype_complex)

    if debug: print(f"Interpolation complete. Interpolated spectrum shape: {P_SPECT_scatterers.shape}")

    del P_SPECT_grid # Free up memory from the potentially large grid

    # --- 2. Prepare for RF Signal Computation ---
    RF_SPECT = np.zeros((n_freq, param.Nelements), dtype=dtype_complex)

    # Re-emitted spectral pressure field at each scatterer location
    P_reemitted = RC * P_SPECT_scatterers  # Shape: (n_scatterers, n_freq)

    # --- 3. Compute Geometry and Propagation ---
    if debug: print("Calculating geometry and propagation paths.")
    xs = x_scatterers
    zs = z_scatterers

    # Get transducer element positions (xe, ze) and orientations (THe)
    xe, ze, THe, _ = param.getElementPositions() # Shapes (Nelements,)

    # Calculate distances and angles between scatterers and elements
    # Reshape xs, zs for broadcasting: (n_scatterers, 1) vs (1, Nelements)
    dxi = xs[:, np.newaxis] - xe[np.newaxis, :] # Shape (n_scatterers, Nelements)
    dzi = zs[:, np.newaxis] - ze[np.newaxis, :] # Shape (n_scatterers, Nelements)
    r = np.sqrt(dxi**2 + dzi**2).astype(np.float64) # Shape (n_scatterers, Nelements)

    # Angle relative to element normal (accounts for element orientation THe)
    angles_rel_z = np.arcsin(np.clip(dxi / (r + _EPS), -1.0, 1.0))
    Th = angles_rel_z - THe[np.newaxis, :] # Shape (n_scatterers, Nelements)
    sinTh = np.sin(Th)

    probeFunction = param.getProbeFunction()

    # --- 4. Compute Spectral Response at Transducer (Loop over Frequencies) ---
    if debug: print("Computing spectral response at transducer elements.")
    alpha_dB = param.attenuation

    for i, freq in (enumerate(freqs) if not debug else tqdm.tqdm(enumerate(freqs), total=n_freq, desc="Processing Frequencies")):
        kw = 2 * np.pi * freq / param.c # wavenumber for the current frequency.
        kwa = (alpha_dB / 8.69) * (freq / 1e6) * 1e2 #  attenuation-based wavenumber

        # Compute the Green's function propagation factor:
        EXP = np.exp(-kwa * r + 1j * np.mod(kw * r, 2 * np.pi)).astype(dtype_complex) / np.sqrt(r)

        # Directivity
        DIR_argument = kw * param.width / 2 * sinTh
        DIR = mysinc(DIR_argument)

        # Propagation
        propagation = EXP * DIR # Shape: (n_scatterers, Nelements)

        # Summation and Probe Response
        received_spectrum = P_reemitted[:, i].reshape(1, -1) @ propagation
        if not harmonic:
            probe_resp = probeFunction(2 * np.pi * freq)
        else:
            probe_resp = probeFunction(2 * np.pi * (freq - param.fc)) # For harmonic, filter around 2 times the fc
        RF_SPECT[i, :] = probe_resp * received_spectrum.flatten()

    if just_RF_spectrum:
        return None, RF_SPECT

    # --- 5. Reconstruct Time-Domain RF Signals (Inverse FFT) ---
    if debug: print("Performing Inverse FFT to get time-domain RF signals.")
    param.fs = 8 * param.fc

    nf = int(np.ceil(param.fs/(freqs[1] - freqs[0])))
    RF = np.fft.irfft(np.conj(RF_SPECT), n=nf, axis=0)
    RF = RF[: (nf + 1) // 2] # Take only the first half of the RF signal

    if debug: print(f"Simulation complete. RF signal shape: {RF.shape}")
    return RF, RF_SPECT
