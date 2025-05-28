"""
This module computes the harmonic (non-linear) field
"""

import numpy as np
import scipy

from .. import utils
from .. import pfield as linear_pfield

def pfield(xbound, zbound, delaysTX: np.ndarray, param: utils.Param, options: utils.Options = None, lowResources = False, debug = False):
    """
    Initial implementation of the harmonic (non-linear) field.
    """
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

    _, P_SPECT, IDX = linear_pfield(X,None, Z,delaysTX,param,options=options if options else None)
    _ = None # free memory NOTE Test vs "del"

    # Adjust complex precision using dtype_complex
    P_SPECT = P_SPECT.astype(dtype_complex)

    # Compute the square
    L = P_SPECT.shape[2]
    fP0 = np.fft.fft(P_SPECT,int(L*1.5),2)
    fP02 = fP0**2
    P02 = np.fft.ifft(fP02, axis = 2).astype(dtype_complex)
    P02 = P02[:,:,-L:]
    fs = param.f[IDX] + param.f[IDX][0] +L/2 *param.df
    ws_P02 = 2* np.pi * fs

    D_kernel = np.sqrt((X-X.mean() + dx/2)**2+(Z-Z.mean() + dz/2)**2)
    spectP1 = np.zeros_like(P02, dtype=dtype_complex)
    for k, w  in enumerate(ws_P02):
        k_wave = w / c
        # Compute the Green's function
        G = (1j / 4 * scipy.special.hankel1(0, k_wave * D_kernel)).astype(dtype_complex)
        # Compute an attenuation-dependent term.
        kwa = param.attenuation / 8.69 * (w / (2 * np.pi)) / 1e6 * 1e2
        # Apply attenuation
        G *= np.exp(-kwa * D_kernel).astype(dtype_complex)
        # Convolution
        P1_conv = scipy.signal.fftconvolve(P02[:, :, k], G, mode='same').astype(dtype_complex)
        # Multiply by a frequency-dependent factor and scale by grid spacing.
        spectP1[:, :, k] = (w / 2) ** 2 * dx * dz * P1_conv

    P1 = np.linalg.norm(spectP1, axis = 2)

    # Double check
    IDX_2 = np.concatenate((np.zeros(IDX.shape[0]//2, dtype = bool), IDX, np.zeros(IDX.shape[0]//2, dtype = bool)))
    f = param.df * np.arange(len(IDX_2))
    return P1, spectP1, IDX_2, f

def range_matlab(x):
    return max(x) - min(x)
