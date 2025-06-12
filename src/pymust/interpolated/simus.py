# pylint: disable=unnecessary-lambda-assignment

import numpy as np
import tqdm

from ..utils import Param
from .methods import interpolate_spectrum

_EPS = np.finfo(np.float32).eps
mysinc = lambda x = None: np.sinc(x / np.pi) # [NOTE: In MATLAB/numpy, sinc is sin(pi*x)/(pi*x)]

def simus(x_range: np.ndarray, z_range: np.ndarray, P_SPECT_grid: np.ndarray,
          x_scatterers: np.ndarray, z_scatterers: np.ndarray, interpolator_name: str,
          freqs: np.ndarray, RC: np.ndarray, param: Param, harmonic: bool = False, IDX: np.ndarray = None,
          debug: bool = False, just_RF_spectrum: bool = False, doublePrecision: bool = False):
    """
    Simulate RF signals by interpolating a spectral grid onto scatterer positions.

    This function interpolates a pre-computed complex pressure spectrum (P_SPECT_grid)
    defined on a 2D (x, z) grid to the positions of specified scatterers, then computes
    the RF signals received by transducer elements due to the re-emitted waves from these
    scatterers.

    Args:
        x_range (np.ndarray): 1D array of x-coordinates defining the grid.
        z_range (np.ndarray): 1D array of z-coordinates defining the grid.
        P_SPECT_grid (np.ndarray): 3D array (len(z_range), len(x_range), n_freq[IDX]) of complex
            spectrum values on the grid.
        x_scatterers (np.ndarray): 1D array of x-coordinates for scatterer positions.
        z_scatterers (np.ndarray): 1D array of z-coordinates for scatterer positions.
        interpolator_name (str): Name of the interpolation method (e.g., 'linear', 'nearest').
        freqs (np.ndarray): 1D array of frequency values (Hz) for the "full spectrum" of P_SPECT_grid.
        RC (np.ndarray | float | None): Reflection coefficients for scatterers. Can be:
            - ndarray: shape (n_scatterers,) or (n_scatterers, 1)
            - float: single value for all scatterers
            - None: defaults to 1.0 for all scatterers
        param (Param): Parameter object with transducer and medium properties.
        
        harmonic (bool, optional): If True, simulates harmonic reception. Default is False.
        IDX (np.ndarray, optional): Boolean mask for frequency selection. Default is None (all True).
        debug (bool, optional): If True, prints debug information. Default is False.
        just_RF_spectrum (bool, optional): If True, returns only the RF spectrum (skips IFFT).
            Default is False.
        doublePrecision (bool, optional): If True, uses double precision for calculations.
            Default is False.

    Returns:
        tuple:
            RF (np.ndarray or None): Time-domain RF signals (n_time_samples, Nelements),
                or None if just_RF_spectrum is True.
            RF_SPECT (np.ndarray): Complex spectrum at the transducer elements
                (n_freq, Nelements).
    """
    assert isinstance(doublePrecision, bool), "doublePrecision must be a boolean."
    # Define complex number type based on doublePrecision
    dtype_complex = np.complex128 if doublePrecision else np.complex64

    if IDX is not None:
        assert isinstance(IDX, np.ndarray), "IDX must be a numpy array."
        assert IDX.shape == freqs.shape, "IDX must have the same shape as freqs."
        assert np.sum(IDX) == P_SPECT_grid.shape[2], "IDX must sum to the number of frequency points in P_SPECT_grid."
    else:
        IDX = np.ones(freqs.shape, dtype=bool)

    IDX_idx = np.where(IDX)[0]  # Get indices where IDX is True

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
        freqs=freqs[IDX]
    ).astype(dtype_complex)

    if debug: print(f"Interpolation complete. Interpolated spectrum shape: {P_SPECT_scatterers.shape}")

    del P_SPECT_grid # Free up memory from the potentially large grid

    # --- 2. Prepare for RF Signal Computation ---
    RF_SPECT = np.zeros((n_freq, param.Nelements), dtype=dtype_complex)

    # Re-emitted spectral pressure field at each scatterer location
    P_reemitted = RC * P_SPECT_scatterers  # Shape: (n_scatterers, n_freq[IDX])

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

    # Precompute and precast variables for performance
    if not doublePrecision:
        freqs = freqs.astype(np.float32)
        P_reemitted = P_reemitted.astype(np.complex64)
        r = r.astype(np.float32)
        alpha_dB = np.float32(alpha_dB)
    sqrt_r = np.sqrt(r) # same dtype as r

    for i, freq in (enumerate(freqs[IDX]) if not debug else tqdm.tqdm(enumerate(freqs[IDX]), total=n_freq, desc="Processing Frequencies")):
        kw = np.float32(2 * np.pi * freq / param.c) # wavenumber for the current frequency.
        kwa = (alpha_dB / 8.69) * (freq / 1e6) * 1e2 #  attenuation-based wavenumber

        # Compute the Green's function propagation factor:
        EXP = np.exp(-kwa * r + 1j * kw * r) / sqrt_r # Shape: (n_scatterers, Nelements)
        # EXP = np.exp(-kwa * r + 1j * np.mod(kw * r, 2 * np.pi)) * inv_sqrt_r # NOTE np.mod could be too slow

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
        RF_SPECT[IDX_idx[i], :] = probe_resp * received_spectrum.flatten()

    if just_RF_spectrum:
        if debug: print("Returning only the RF spectrum without IFFT.")
        return None, RF_SPECT

    # --- 5. Reconstruct Time-Domain RF Signals (Inverse FFT) ---
    if debug: print("Performing Inverse FFT to get time-domain RF signals.")
    param.fs = 8 * param.fc

    nf = int(np.ceil(param.fs/(freqs[1] - freqs[0])))
    RF = np.fft.irfft(np.conj(RF_SPECT), n=nf, axis=0)
    RF = RF[: (nf + 1) // 2] # Take only the first half of the RF signal

    if debug: print(f"Simulation complete. RF signal shape: {RF.shape}")
    return RF, RF_SPECT
