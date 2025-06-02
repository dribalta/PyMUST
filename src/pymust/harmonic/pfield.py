"""
This module computes the harmonic (non-linear) field
"""
from collections.abc import Iterable

import numpy as np
import scipy

from .. import utils
from .. import pfield as linear_pfield

def pfield(xbound: np.ndarray, zbound: np.ndarray, delaysTX: np.ndarray,
        param: utils.Param, options: utils.Options = None, * ,
        lowResources = False, DR = None, debug = False,
        auxiliary_returns: Iterable[str] = None):
    """
    Initial implementation of the harmonic (non-linear) field.
    """
    # Ensure auxiliary_returns is a set of str
    if isinstance(auxiliary_returns, str):
        auxiliary_returns = {auxiliary_returns}
    elif isinstance(auxiliary_returns, Iterable):
        auxiliary_returns = {item for item in auxiliary_returns if isinstance(item, str)}
    else:
        auxiliary_returns = set()

    # Assert that DR is a positive number
    if DR is not None: assert utils.isnumeric(DR) and DR > 0, "DR must be a positive number."
    # Define complex number type based on lowResources
    dtype_complex = np.complex64 if lowResources else np.complex128

    c = param.c
    lambda_ = param.c / param.fc

    _, _, IDX = linear_pfield(np.array([1e-6]), None, np.array([1e-6]), delaysTX, param)
    fs = param.f[IDX]

    if not isinstance(xbound, np.ndarray):
        xbound = np.array([-4e-2,4e-2]) # in m
    if not isinstance(zbound, np.ndarray):
        zbound = np.array([lambda_/2,10e-2]) # in m
    elif zbound[0] < lambda_/2:
        zbound[0] = lambda_/2

    Nx = np.ceil(2*range_matlab(xbound)/min(c/fs))
    Nz = np.ceil(2*range_matlab(zbound)/min(c/fs))
    Nx = round(Nx / 2) * 2 + 1
    Nz = round(Nz / 2) * 2 + 1

    if debug:
        print("DEBUG - Number of grid points in x:", Nx)
        print("DEBUG - of grid points in z:", Nz)

    # TODO precompute the needed RAM and check if it is too large

    # Save the number of grid points in the param structure
    param.Nx = int(Nx)
    param.Nz = int(Nz)
    param.xbound = xbound
    param.zbound = zbound

    x = np.linspace(min(xbound),max(xbound),Nx)
    z = np.linspace(min(zbound),max(zbound),Nz)
    dx = np.mean(np.diff(x)) # grid spacing in x (m)
    dz = np.mean(np.diff(z)) # grid spacing in z (m)
    X,Z = np.meshgrid(x,z)

    P0, P0_SPECT, linear_IDX = linear_pfield(X,None, Z,delaysTX,param,options=options if options else None)
    if "P0" not in auxiliary_returns: del P0  # Free memory if not needed

    # Adjust complex precision using dtype_complex
    P0_SPECT = P0_SPECT.astype(dtype_complex)
    IDX = np.concatenate((linear_IDX, np.zeros_like(linear_IDX))) # Extend the IDX to match the full spectrum - also negative frequencies
    f = np.concatenate((param.f, param.f + param.f[-1] + param.f[1] - param.f[0]))
    if "linear_IDX" not in auxiliary_returns: del linear_IDX  # Free memory if not needed

    P02_SPECT_compact = scipy.signal.fftconvolve(P0_SPECT,P0_SPECT, 'full', axes = 2) # Convolve to obtain P0^2
    if "P0_SPECT" not in auxiliary_returns: del P0_SPECT  # Free memory if not needed

    # Do a simulated convolution, with the indexes (and full spectra, also the negative, to obtain where are the active frequencies after convolution)
    IDX_extended= np.concatenate((np.zeros_like(IDX), IDX))
    IDX2 = scipy.signal.convolve(IDX_extended,IDX_extended, 'same')[-len(IDX):] > 0 #Indices where the convolution is non-zero
    fs = f[IDX2]
    if "IDX_extended" not in auxiliary_returns: del IDX_extended  # Free memory if not needed

    # Filter the spectrum  - I do this before receive, so I can decide which "source" points I want to keep
    # This was important because in the previous version, I was doing the convolution of the full spectra, hence there were low frequencies generated that would be filtered out later.
    ws_P02 = 2* np.pi * fs

    # if DR:
        # # Select the points above a certain number of dB (based on the DR)
        # IDX_P = np.any(to_decibel(np.abs(P02_SPECT_compact)) > - DR , axis = 2)
        # P02_SPECT_compact = P02_SPECT_compact[IDX_P, :].reshape(-1, P02_SPECT_compact.shape[2])
        # X = X[IDX_P].reshape(-1, 1)
        # Z = Z[IDX_P].reshape(-1, 1)

    D_kernel = np.sqrt((X-X.mean() + dx/2)**2+(Z-Z.mean() + dz/2)**2)
    P1_SPECT = np.zeros_like(P02_SPECT_compact, dtype=dtype_complex)

    for k, w  in enumerate(ws_P02): # NOTE Could be parallelized
        k_wave = w / c
        # Compute the Green's function
        G = (1j / 4 * scipy.special.hankel1(0, k_wave * D_kernel)).astype(dtype_complex)
        # Compute an attenuation-dependent term.
        kwa = param.attenuation / 8.69 * (w / (2 * np.pi)) / 1e6 * 1e2
        # Apply attenuation
        G *= np.exp(-kwa * D_kernel).astype(dtype_complex)
        # Convolve
        P1_conv = scipy.signal.fftconvolve(P02_SPECT_compact[:,:,k], G, mode='same').astype(dtype_complex)
        # Multiply by a frequency-dependent factor and scale by grid spacing.
        P1_SPECT[:,:,k] = (w / 2) ** 2 * dx * dz * P1_conv

    # if DR:
    #     # NOTE: if this creates RAM issues, we can use a sparse matrix (from scipy.sparse.coo_matrix)
    #     P1_SPECT_full = np.zeros([*IDX_P.shape, P02_SPECT_compact.shape[1]], dtype=dtype_complex)
    #     P1_SPECT_full[IDX_P, :] = P1_SPECT
    #     P1_SPECT = P1_SPECT_full

    P1 = np.linalg.norm(P1_SPECT, axis = 2)

    if not auxiliary_returns:
        return P1, P1_SPECT, IDX2, f

    extra_returns = {}
    for item in auxiliary_returns:
        if item in locals():
            extra_returns[item] = locals()[item]
    return P1, P1_SPECT, IDX2, f, extra_returns

def range_matlab(x):
    """Calculate the range (difference) between the maximum and minimum of an array."""
    return max(x) - min(x)

def to_decibel(x):
    """Convert linear scale to decibel scale."""
    return 10 * np.log10(np.abs(x)/np.max(x) + 1e-10)
