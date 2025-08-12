import numpy as np
import skimage, scipy
import logging

from collections.abc import Iterable
from .. import utils
from ..utils import Param
from . import pfield


_EPS = np.finfo(np.float32).eps
mysinc = lambda x = None: np.sin(np.abs(x) + _EPS)/ (np.abs(x) + _EPS) # [NOTE: In MATLAB/numpy, sinc is sin(pi*x)/(pi*x)]


def simus(bounds: np.ndarray, delaysTX: np.ndarray,
        RC: np.ndarray, scatter_coords: np.ndarray, 
        param: utils.Param, options: utils.Options = None, dtype_complex = np.complex64,
        debug: bool = False,  DR: int = 30,
        auxiliary_returns: Iterable[str] = None, is3D = False, P_SPECT_grid: np.ndarray = None, f = None, IDX2 = None, harmonic = True):
    """
    TODO: Add docstring for simus function.
    TODO: use automatic bounds detection if bounds is None. IDEA: Try a coarse grid using linear pfield.
    """
    # Arguments cleaning and validation
    if not all([P_SPECT_grid is None, f is None, IDX2 is None]) and not any([P_SPECT_grid is not None, f is not None, IDX2 is not None]):
        raise ValueError("P_SPECT_grid, f, and IDX2 must be either all None, or all given.")

    if not is3D:
        if len(bounds) == 3:
            raise ValueError("Found 3 bounds for a 2D simulation. Is this an error?")
        xbound = bounds[0]
        zbound = bounds[1]
    else:
        xbound = bounds[0]
        ybound = bounds[1]
        zbound = bounds[2]

    if P_SPECT_grid is None:
        _, P_SPECT_grid, IDX2, f = pfield(bounds, delaysTX, param, options, is3D = is3D)

    xbound_range = np.linspace(xbound[0], xbound[-1], num=P_SPECT_grid.shape[0])
    zbound_range = np.linspace(zbound[0], zbound[-1], num=P_SPECT_grid.shape[-2])
    if is3D:
        ybound_range = np.linspace(ybound[0], ybound[-1], num=P_SPECT_grid.shape[1])
        ranges = (xbound_range, ybound_range, zbound_range)
    else:
        ranges = (xbound_range, zbound_range)

    # Convert to numpy if not already
    if not isinstance(scatter_coords, np.ndarray):
        scatter_coords = np.array(scatter_coords, dtype=np.float64)
    if not isinstance(delaysTX, np.ndarray):
        delaysTX = np.array(delaysTX, dtype=np.float64)
    if not isinstance(RC, np.ndarray):
        RC = np.array(RC, dtype=np.float64)
    assert scatter_coords.ndim == 2, "scatter_coords must be a 2D array."
    RC = RC.reshape(-1, 1)
    assert RC.shape[0] == scatter_coords.shape[0], "RC must have the same number of elements as scatter_coords."


    RF_SPECT = np.zeros((len(f), delaysTX.size), dtype=np.complex64)

    # Precompute distances and other things
    center_freq = np.argmax(np.sum(np.abs(P_SPECT_grid), axis = (0, 1) if not is3D else (0, 1, 2)))
    phase_unwrapped = skimage.restoration.unwrap_phase(np.angle(P_SPECT_grid[:, :, center_freq]))
    grad_phase_unwrapped = np.gradient(phase_unwrapped) # TODO: this might not work if 3D
    grad_phase_unwrapped_normalized = grad_phase_unwrapped / np.linalg.norm(grad_phase_unwrapped, axis=0)
    logging.debug(f'Center frequency: {f[IDX2][center_freq]}')

    # Grad at scatter TODO: this can be much faster with the loops
    distanceFromClosestGridPoint = np.zeros(scatter_coords.shape[0])
    for i, coord in enumerate(scatter_coords):
        # Obtain the coordinates of the closest grid point
        if is3D:
            closest_grid_point = np.array( [np.argmin(np.abs(xbound_range - coord[0])),
                                           np.argmin(np.abs(ybound_range - coord[1])),
                                           np.argmin(np.abs(zbound_range - coord[2]))]
                                           )
            displacementFromClosestGridPoint = np.array([xbound_range[closest_grid_point[0]] - coord[0],
                                                        ybound_range[closest_grid_point[1]] - coord[1],
                                                        zbound_range[closest_grid_point[2]] - coord[2]])
        else:
            closest_grid_point = np.array([np.argmin(np.abs(xbound_range - coord[0])),
                                           np.argmin(np.abs(zbound_range - coord[1]))])
            displacementFromClosestGridPoint = np.array([xbound_range[closest_grid_point[0]] - coord[0],
                                                        zbound_range[closest_grid_point[1]] - coord[1]])
        # Get the gradient
        grad = grad_phase_unwrapped_normalized[:, *closest_grid_point]
        distanceFromClosestGridPoint[i] = np.dot(grad, displacementFromClosestGridPoint)


    # --- Compute Geometry and Propagation ---
    xs = scatter_coords[:, 0]
    zs = scatter_coords[:,-1]

    # Get transducer element positions (xe, ze) and orientations (THe)
    xe, ze, THe, _ = param.getElementPositions() # Shapes (Nelements,)

    # Calculate distances and angles between scatterers and elements
    # Reshape xs, zs for broadcasting: (n_scatterers, 1) vs (1, Nelements)
    dxi = xs[:, np.newaxis] - xe[np.newaxis, :] # Shape (n_scatterers, Nelements)
    dzi = zs[:, np.newaxis] - ze[np.newaxis, :] # Shape (n_scatterers, Nelements)
    r2 = dxi**2 + dzi**2
    if is3D:
        dyi = scatter_coords[:, 1][:, np.newaxis] - ye[np.newaxis, :]
        r2 += dyi**2
    r = np.sqrt(r2).astype(np.float64) # Shape (n_scatterers, Nelements)

    # Angle relative to element normal (accounts for element orientation THe)
    angles_rel_z = np.arcsin(np.clip(dxi / (r + _EPS), -1.0, 1.0))
    Th = angles_rel_z - THe[np.newaxis, :] # Shape (n_scatterers, Nelements)
    sinTh = np.sin(Th)

    probeFunction = param.getProbeFunction()
    alpha_dB = param.attenuation


    for k, kw in enumerate(2*np.pi*f[IDX2]):
        # STEP 1: Interpolate the field at the scatter coordinate
        # Interpolate the magnitude TODO: check why reversed
        norm_interpolator = scipy.interpolate.RegularGridInterpolator(ranges, np.abs(P_SPECT_grid[..., k]), method='linear')
        norm_interpolated = norm_interpolator(scatter_coords)
        # Interpolate the phase
        phase_interpolator = scipy.interpolate.RegularGridInterpolator(ranges, np.angle(P_SPECT_grid[..., k]),  method='nearest')
        phase_interpolated = phase_interpolator(scatter_coords)
          # Correct with the distance from the closest grid point
        phase_interpolated += distanceFromClosestGridPoint * kw / param.c
        P_SPECT_interp = norm_interpolated * np.exp(1j * phase_interpolated) # Slow as hell... maybe something faster

        # STEP 2: Compute the backpropagation matrix
        # If the frequency is not included, ignore it ...
        kw = kw / param.c # wavenumber for the current frequency.
        kwa = (alpha_dB / 8.69) * (kw / 1e6/ 2 /np.pi) * 1e2 #  attenuation-based wavenumber

        # Compute the Green's function propagation factor:
        # TODO: use fast evaluation as in normal simus (pfield)
        if not is3D:
            EXP = np.exp(-kwa * r + 1j * np.mod(kw * r, 2 * np.pi)).astype(dtype_complex) / np.sqrt(r)
        else:
            EXP = np.exp(-kwa * r + 1j * np.mod(kw * r, 2 * np.pi)).astype(dtype_complex) / (4 * np.pi * r)

        # Directivity
        DIR_argument = kw * param.width / 2 * sinTh
        DIR = mysinc(DIR_argument)

        # Propagation
        propagation = EXP * DIR # Shape: (n_scatterers, Nelements)

        # STEP 3: Compute and accumulate
        if not harmonic:
            probe_resp = probeFunction(kw)
        else:
            probe_resp = probeFunction(kw - 2 * np.pi*param.fc) # For harmonic, filter around 2 times the fc

        RF_SPECT[k, :] = probe_resp * ((RC * P_SPECT_interp).reshape(1, -1) @ propagation) # TODO: need to be filtered back by the probe function

    # Now, we need to compute the rfftt
    param.fs = 8 * param.fc # GB: not sure 

    nf = int(np.ceil(param.fs/(f[1] - f[0])))
    RF = np.fft.irfft(np.conj(RF_SPECT), n=nf, axis=0)
    RF = RF[: (nf + 1) // 2] # Take only the first half of the RF signal
    return RF, RF_SPECT
