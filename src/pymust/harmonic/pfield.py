"""
This module computes the harmonic (non-linear) field
"""
from collections.abc import Iterable

import numpy as np
import scipy

from .. import utils
from .. import pfield as linear_pfield
from .. import pfield3
# 

def pfield(bounds: np.ndarray, delaysTX: np.ndarray,
        param: utils.Param, options: utils.Options = None, * ,
        doublePrecision: bool = False, debug: bool = False,
        reducedKernel: bool = False, DR: int = 30,
        auxiliary_returns: Iterable[str] = None, is3D = False):
    """
    Initial implementation of the harmonic (non-linear) field.
    """
    if not is3D:
        if len(bounds) == 3:
            raise ValueError("Found 3 bounds for a 2D simulation. Is this an error?")
        xbound = bounds[0]
        zbound = bounds[1]
    else:
        xbound = bounds[0]
        ybound = bounds[1]
        zbound = bounds[2]

    # Ensure auxiliary_returns is a set of str
    if isinstance(auxiliary_returns, str):
        auxiliary_returns = {auxiliary_returns}
    elif isinstance(auxiliary_returns, Iterable):
        auxiliary_returns = {item for item in auxiliary_returns if isinstance(item, str)}
    else:
        auxiliary_returns = set()

    # Assert that reducedKernel is a bool
    assert isinstance(reducedKernel, bool), "reducedKernel must be a boolean."
    if reducedKernel:
        assert utils.isnumeric(DR) and DR > 0, "DR must be a positive integer."
    # Assert that doublePrecision is a bool
    assert isinstance(doublePrecision, bool), "doublePrecision must be a boolean."
    # Define complex number type based on doublePrecision
    dtype_complex = np.complex128 if doublePrecision else np.complex64

    c = param.c
    lambda_ = param.c / param.fc
    if not is3D:
        _, _, IDX = linear_pfield(np.array([1e-6]), None, np.array([1e-6]), delaysTX, param)
        fs = param.f[IDX]
    else:
        _, _, IDX = pfield3(np.array([1e-6]).reshape(1,1,1), 
                            np.array([1e-6]).reshape(1,1,1),
                            np.array([1e-6]).reshape(1,1,1),
                            delaysTX, param, options=options)
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
    if is3D:
        Ny = np.ceil(2*range_matlab(ybound)/min(c/fs))
        Ny = round(Ny / 2) * 2 + 1

    if debug:
        print("DEBUG - Number of grid points in x:", Nx)
        print("DEBUG - of grid points in z:", Nz)
        if is3D:
            print("DEBUG - of grid points in y:", Ny)


    # TODO precompute the needed RAM and check if it is too large

    # Save the number of grid points in the param structure
    param.Nx = int(Nx)
    param.Nz = int(Nz)
    param.xbound = xbound
    param.zbound = zbound
    if is3D:
        param.Ny = int(Ny)
        param.ybound = ybound

    x = np.linspace(min(xbound),max(xbound),Nx)
    z = np.linspace(min(zbound),max(zbound),Nz)
    dx = np.mean(np.diff(x)) # grid spacing in x (m)
    dz = np.mean(np.diff(z)) # grid spacing in z (m)
    if is3D:
        y = np.linspace(min(ybound),max(ybound),Ny)
        dy = np.mean(np.diff(y))
        X, Y, Z = np.meshgrid(x, y, z)
        if options is None:
            options = utils.Options()
        # Do it  by slices, to avoid memory issues
        if  Ny > 32:
            for k, _ in enumerate(y):
                if k != 0:
                    options.f = f.copy()
                P0_i, P0_SPECT_i, linear_IDX = pfield3(X[:, [k], :],Y[:, [k], :], Z[:, [k], :],delaysTX,param,options=options if options else None)
                if k == 0:
                    P0 = np.zeros((len(x), len(y), len(z)), dtype=dtype_complex)
                    P0_SPECT = np.zeros((len(x), len(y), len(z), len(param.f[linear_IDX])), dtype=dtype_complex)
                    f = param.f.copy()
                P0[:, k, :] = P0_i[:, 0, :]
                P0_SPECT[:,k, :, :] = P0_SPECT_i[:, 0, :, :]
        else:
            P0, P0_SPECT, linear_IDX = pfield3(X, Y, Z, delaysTX, param, options=options if options else None)

    else:
        X, Z = np.meshgrid(x, z)
        P0, P0_SPECT, linear_IDX = linear_pfield(X,None, Z,delaysTX,param,options=options if options else None)
    if "P0" not in auxiliary_returns: del P0  # Free memory if not needed

    if debug:
        print ("DEBUG - Finished computing P0")


    # Adjust complex precision using dtype_complex
    P0_SPECT = P0_SPECT.astype(dtype_complex)
    
    
    P0_SPECT /= np.max(np.abs(P0_SPECT))  # Normalize the spectrum to avoid overflow!! Warning.

    IDX = np.concatenate((linear_IDX, np.zeros_like(linear_IDX))) # Extend the IDX to match the full spectrum - also negative frequencies
    f = np.concatenate((param.f, param.f + param.f[-1] + param.f[1] - param.f[0]))
    if "linear_IDX" not in auxiliary_returns: del linear_IDX  # Free memory if not needed

    P02_SPECT_compact = scipy.signal.fftconvolve(P0_SPECT,P0_SPECT, 'full', axes = -1) # Convolve to obtain P0^2
    if "P0_SPECT" not in auxiliary_returns: del P0_SPECT  # Free memory if not needed

    # Do a simulated convolution, with the indexes (and full spectra, also the negative, to obtain where are the active frequencies after convolution)
    IDX_extended= np.concatenate((np.zeros_like(IDX), IDX))
    IDX2 = scipy.signal.convolve(IDX_extended,IDX_extended, 'same')[-len(IDX):] > 0 #Indices where the convolution is non-zero
    fs = f[IDX2]
    if "IDX_extended" not in auxiliary_returns: del IDX_extended  # Free memory if not needed

    # Filter the spectrum  - I do this before receive, so I can decide which "source" points I want to keep
    # This was important because in the previous version, I was doing the convolution of the full spectra, hence there were low frequencies generated that would be filtered out later.
    ws_P02 = 2* np.pi * fs


    if is3D:
        D_kernel = np.sqrt((X-X.mean() + dx/2)**2 + (Y-Y.mean() + dy/2)**2 + (Z-Z.mean() + dz/2)**2)
    else:
        D_kernel = np.sqrt((X-X.mean() + dx/2)**2 + (Z-Z.mean() + dz/2)**2)
    # If 3D
    # D_kernel += np.sqrt(D_kernel**2 + (Y-Y.mean() + dy/2)**2) # 3D distance kernel
    P1_SPECT = np.zeros_like(P02_SPECT_compact, dtype=dtype_complex)

    if debug:
        print ("DEBUG - Number of frequencies after filtering:", len(fs))

    pixel_size = dx * dz
    if is3D:
        pixel_size *= dy

    for k, w  in enumerate(ws_P02): # NOTE Could be parallelized
        if debug:
            print (f"New itertion {k}/{len(ws_P02)}, angular freq = {w}")

        if reducedKernel and not is3D:
            nPointsKeep = DR/(param.attenuation * dx *fs[k]/1e4) # dx is in m, fs is in Hz, attenuation is in dB/cm/MHz
            D_kernel_effective, kernel_xbound, kernel_ybound = reduceSizeKernel(D_kernel, nPointsKeep)
            xslice = slice(kernel_xbound[0], kernel_xbound[1])
            zslice = slice(kernel_ybound[0], kernel_ybound[1])
        else:
            D_kernel_effective = D_kernel
            xslice = slice(None)
            zslice = slice(None)
            yslice = slice(None) 

        k_wave = w / c
        # Compute the Green's function
        if not is3D:
            G = (1j / 4 * scipy.special.hankel1(0, k_wave * D_kernel_effective)).astype(dtype_complex)
        else:
            G  =  np.exp(1j * k_wave * D_kernel_effective) / (4 * np.pi * D_kernel_effective) # 3D Green's function
        G *= pixel_size
        kwa = param.attenuation / 8.69 * (w / (2 * np.pi)) / 1e6 * 1e2
        # Apply attenuation
        G *= np.exp(-kwa * D_kernel_effective)
        if is3D:
            # Reduce the size of the kernel if needed
            G = G[xslice, yslice, zslice]
        else:
            G =  G[xslice, zslice]
        # Convolve
        P1_conv = scipy.signal.fftconvolve(P02_SPECT_compact[...,k],G, mode='same')  #GB: Important, you don't  need to make the P02 smaller, but G!

        # Multiply by a frequency-dependent factor and scale by grid spacing.
        P1_SPECT[...,k] = (w / 2) ** 2 * P1_conv #  *dz if 3D ; Important, you don't  need to make the P02 smaller, but G!

    P1 = np.linalg.norm(P1_SPECT, axis = -1)

    if is3D:
        # If 3D, we need to compute the norm across the last three axes
        print(P1_SPECT.shape)
        norm = np.linalg.norm(P1_SPECT.reshape((-1, P1_SPECT.shape[-1])), axis = 0)
    else:
        norm = np.linalg.norm(P1_SPECT, axis = (0, 1))

    # Filter to the frequencies that are almost -zero
    norm_db = to_decibel(norm)
    filtered_freqs = norm_db > -60
    IDX2[IDX2] = filtered_freqs
    P1_SPECT = P1_SPECT[...,filtered_freqs]

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
    x = np.asarray(x)
    eps = np.finfo(x.dtype).eps
    return 20*np.log10(np.abs(x)/(np.max(np.abs(x)) + eps) + eps)

def reduceSizeKernel(kernel, nPointsKeep = 50):
    """
    Make the kernel smaller, by keeping only the central part.
    """
    if kernel.shape[0] <= nPointsKeep:
        return kernel, (0, kernel.shape[0]), (0, kernel.shape[1])
    else:
        mid_x = kernel.shape[0]//2
        mid_y = kernel.shape[1]//2
        xbound = (int(mid_x - nPointsKeep//2), int(mid_x + nPointsKeep//2))
        ybound = (int(mid_y - nPointsKeep//2), int(mid_y + nPointsKeep//2))
        return kernel[xbound[0]:xbound[1], ybound[0]:ybound[1]], xbound, ybound
