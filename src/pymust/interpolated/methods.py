"""
Helper functions for the interpolation of the pressure field spectrum.

Provides interpolation methods and a central dispatcher function
`interpolate_spectrum` that handles argument validation, reshaping,
and calling the appropriate registered interpolation function.

Interpolation functions are registered using the @register_interpolator decorator.
"""
import numpy as np
from scipy.special import hankel1
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
import functools
import inspect

# --- Registry for Interpolation Methods ---

_INTERPOLATOR_REGISTRY = {}
_FLATTENED_INPUT_METHODS = set() # Keep track of methods needing flattened grid inputs

def register_interpolator(name=None, needs_flattened_grid=False):
    """
    Decorator to register an interpolation function.

    Args:
        name (str, optional): The name to register the function under.
                              If None, uses the function's __name__. Defaults to None.
        needs_flattened_grid (bool): If True, indicates that this interpolator
                                     expects grid_points and grid_values
                                     to be flattened (N, 2) and (N, n_freq).
                                     Defaults to False.
    """
    def decorator(func):
        reg_name = name if name is not None else func.__name__

        # Store the function itself
        _INTERPOLATOR_REGISTRY[reg_name] = func
        if needs_flattened_grid:
            _FLATTENED_INPUT_METHODS.add(reg_name)

        return func # Return the original function
    return decorator

def get_interpolator_func(name):
    """Retrieves the registered interpolation function by name."""
    if name not in _INTERPOLATOR_REGISTRY:
        raise ValueError(f"Unknown interpolator name: '{name}'. "
                         f"Available methods: {list(_INTERPOLATOR_REGISTRY.keys())}")
    return _INTERPOLATOR_REGISTRY[name]

def get_required_args(name):
    """Gets the list of required argument names for a registered interpolator."""
    func = get_interpolator_func(name)
    sig = inspect.signature(func)
    # Filter out args with default values, considering them optional for the caller
    required = [
        p.name for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
    ]
    return required

def _validate_and_prepare_inputs(interpolator_name, **kwargs):
    """
    Internal helper to validate inputs and prepare args for the target function.

    Performs shape checks and reshapes grid data if needed by the interpolator.

    Returns:
        tuple: (interpolator_function, call_args_dict)
    """
    required_args = get_required_args(interpolator_name)
    interp_func = get_interpolator_func(interpolator_name)
    needs_flattening = interpolator_name in _FLATTENED_INPUT_METHODS

    if "scatterers" in required_args:
        required_args.remove("scatterers") # scatterers will be prepared separately
    
    # --- Argument Presence Check ---
    missing_args = []
    for req_arg in required_args:
        if kwargs.get(req_arg, None) is None:
            missing_args.append(req_arg)
    if missing_args:
        raise ValueError(f"Interpolator '{interpolator_name}' requires argument(s): {', '.join(missing_args)}, but they were not provided (were None).")

    # --- Extract and Check Provided Arguments ---
    grid_points = kwargs.get('grid_points')
    grid_values = kwargs.get('grid_values')
    x_scatterers = kwargs.get('x_scatterers')
    z_scatterers = kwargs.get('z_scatterers')
    x_range = kwargs.get('x_range')
    z_range = kwargs.get('z_range')
    param = kwargs.get('param')
    freqs = kwargs.get('freqs')
    
    # --- Validate and Prepare Scatterer Positions ---
    if x_scatterers is None and z_scatterers is None:
        scatterers = None
    elif x_scatterers is None or z_scatterers is None:
        raise ValueError("Both x_scatterers and z_scatterers must be provided together.")
    else:
        if not isinstance(x_scatterers, np.ndarray) or not isinstance(z_scatterers, np.ndarray):
            raise TypeError("x_scatterers and z_scatterers must be numpy arrays.")
        if x_scatterers.ndim != 1 or z_scatterers.ndim != 1:
            raise ValueError("x_scatterers and z_scatterers must be 1D arrays.")
        if x_scatterers.shape[0] != z_scatterers.shape[0]:
            raise ValueError("x_scatterers and z_scatterers must have the same length.")
        scatterers = np.column_stack((z_scatterers, x_scatterers)) # Combine into (n_scatterers, 2)

    n_freq = n_x = n_z = None

    # Basic checks on provided arguments
    if freqs is not None:
        if not isinstance(freqs, np.ndarray):
            raise TypeError("freqs must be a numpy array.")
        if freqs.ndim != 1:
            freqs = freqs.flatten()  # Flatten freqs if it's not 1D
        n_freq = len(freqs)

    if x_range is not None:
        if not isinstance(x_range, np.ndarray) or x_range.ndim != 1:
            raise TypeError("x_range must be a 1D numpy array.")
        n_x = len(x_range)

    if z_range is not None:
        if not isinstance(z_range, np.ndarray) or z_range.ndim != 1:
            raise TypeError("z_range must be a 1D numpy array.")
        n_z = len(z_range)

    # Grid shape checks and potential flattening
    grid_points_flat = grid_values_flat = n_grid_points = None

    if grid_points is not None:
        if not isinstance(grid_points, np.ndarray):
            raise TypeError("grid_points must be a numpy array.")
        if grid_points.ndim == 3: # Assume (nz, nx, 2)
            if grid_points.shape[2] != 2:
                raise ValueError(f"3D grid_points must have shape (nz, nx, 2), got {grid_points.shape}")
            if n_z is not None and grid_points.shape[0] != n_z:
                raise ValueError(f"grid_points dimension 0 ({grid_points.shape[0]}) doesn't match len(z_range) ({n_z})")
            if n_x is not None and grid_points.shape[1] != n_x:
                raise ValueError(f"grid_points dimension 1 ({grid_points.shape[1]}) doesn't match len(x_range) ({n_x})")
            n_grid_points = grid_points.shape[0] * grid_points.shape[1]
        elif grid_points.ndim == 2: # Assume already flat (N, 2)
             if grid_points.shape[1] != 2:
                raise ValueError(f"2D grid_points must have shape (N, 2), got {grid_points.shape}")
             n_grid_points = grid_points.shape[0]
        else:
            raise ValueError("grid_points must be a 2D or 3D array.")

    if grid_values is not None:
        if not isinstance(grid_values, np.ndarray):
            raise TypeError("grid_values must be a numpy array.")
        if grid_values.ndim == 3: # Assume (nz, nx, n_freq)
            if n_z is not None and grid_values.shape[0] != n_z:
                raise ValueError(f"grid_values dimension 0 ({grid_values.shape[0]}) doesn't match len(z_range) ({n_z})")
            if n_x is not None and grid_values.shape[1] != n_x:
                raise ValueError(f"grid_values dimension 1 ({grid_values.shape[1]}) doesn't match len(x_range) ({n_x})")
            if n_freq is not None and grid_values.shape[2] != n_freq:
                raise ValueError(f"grid_values dimension 2 ({grid_values.shape[2]}) doesn't match len(freqs) ({n_freq})")
            current_n_grid_points = grid_values.shape[0] * grid_values.shape[1]
            if n_grid_points is not None and current_n_grid_points != n_grid_points:
                 raise ValueError(f"Mismatch in number of grid points between grid_points ({n_grid_points}) and grid_values ({current_n_grid_points})")
            n_grid_points = current_n_grid_points
        elif grid_values.ndim == 2: # Assume already flat (N, n_freq)
            if n_freq is not None and grid_values.shape[1] != n_freq:
                 raise ValueError(f"2D grid_values dimension 1 ({grid_values.shape[1]}) doesn't match len(freqs) ({n_freq})")
            current_n_grid_points = grid_values.shape[0]
            if n_grid_points is not None and current_n_grid_points != n_grid_points:
                 raise ValueError(f"Mismatch in number of grid points between grid_points ({n_grid_points}) and grid_values ({current_n_grid_points})")
            n_grid_points = current_n_grid_points
        else:
            raise ValueError("grid_values must be a 2D or 3D array.")

    # --- Prepare Arguments for the Specific Interpolator ---
    available_args = {
        # Use consistent names that target functions expect
        'grid_points': grid_points.reshape(n_grid_points, 2) if needs_flattening else grid_points, # Pass flat/original based on need
        'grid_values': grid_values.reshape(n_grid_points, -1) if needs_flattening else grid_values, # Pass flat/original based on need
        'scatterers': scatterers,
        'x_range': x_range,
        'z_range': z_range,
        'param': param,
        'freqs': freqs
    }

    # Select only the arguments required by the target function
    sig = inspect.signature(interp_func)
    call_args = {param_name: available_args[param_name] for param_name in sig.parameters}

    return interp_func, call_args


def interpolate_spectrum(interpolator_name, *, grid_points=None, grid_values=None, x_scatterers=None, z_scatterers=None, x_range=None, z_range=None, param=None, freqs=None):
    """
    Central dispatcher for grid interpolation.

    Validates inputs, reshapes grid data if necessary, and calls the
    specified interpolation function.

    Args:
        interpolator_name (str): Name of the registered interpolation method.
        grid_points (ndarray): Grid coordinates. Can be:
            - (nz, nx, 2) array.
            - (N, 2) array (pre-flattened).
            Defaults to None.
        grid_values (ndarray): Complex spectrum values on the grid. Can be:
            - (nz, nx, n_freq) array.
            - (N, n_freq) array (pre-flattened).
            Defaults to None.
        x_scatterers (ndarray): 1D array of x-coordinates for scatterer positions. Defaults to None.
        z_scatterers (ndarray): 1D array of z-coordinates for scatterer positions. Defaults to None.
        x_range (ndarray): 1D array of unique x-coordinates for the grid. Defaults to None.
        z_range (ndarray): 1D array of unique z-coordinates for the grid. Defaults to None.
        param (object): Parameter object (contents depend on interpolator needs). Defaults to None.
        freqs (ndarray): 1D array of frequency values (Hz). Defaults to None.

    Returns:
        ndarray: (n_scatterers, n_freq) complex spectrum interpolated at scatterer locations.

    Raises:
        ValueError: If interpolator name is unknown, required arguments are missing (None),
                    or shapes are inconsistent.
        TypeError: If inputs are not numpy arrays where expected.
    """
    
    # Validate inputs and prepare arguments for the specific function
    interp_func, call_args = _validate_and_prepare_inputs(
        interpolator_name,
        grid_points=grid_points,
        grid_values=grid_values,
        x_scatterers=x_scatterers,
        z_scatterers=z_scatterers,
        x_range=x_range,
        z_range=z_range,
        param=param,
        freqs=freqs
    )

    # Call the selected interpolation function
    interpolated_spectrum = interp_func(**call_args)

    return interpolated_spectrum


# --- Interpolation Methods ---

# A small epsilon to prevent division by zero or log(0)
_EPS = np.finfo(float).eps

@register_interpolator(name='linear')
def linear_interpolation(grid_values, scatterers, x_range, z_range):
    """
    Performs linear interpolation using RegularGridInterpolator for magnitude and phase separately.

    Args:
        grid_values (ndarray): (nz, nx, n_freq) complex spectrum on the grid points.
        scatterers (ndarray): (n_scatterers, 2) scatterer positions ([z, x]).
        x_range (array-like): 1D array of *unique* x-coordinates defining the grid.
        z_range (array-like): 1D array of *unique* z-coordinates defining the grid.

    Returns:
        ndarray: (n_scatterers, n_freq) complex spectrum interpolated at scatterer locations.
    """
    if grid_values.ndim != 3:
        raise ValueError(f"'linear' interpolator expects 3D grid_values (nz, nx, n_freq), got shape {grid_values.shape}")

    # 1. Linear interpolation for magnitude
    magnitude = np.abs(grid_values)
    interpolator_mag = RegularGridInterpolator(
        (z_range, x_range), magnitude,
        method='linear', bounds_error=False, fill_value=0.0
    )
    interpolated_magnitude = interpolator_mag(scatterers)

    # 2. Linear interpolation for phase
    # TODO: Check (un)/wrapped phase interpolation
    phase = np.angle(grid_values)
    interpolator_phase = RegularGridInterpolator(
        (z_range, x_range), phase,
        method='linear', bounds_error=False, fill_value=0.0
    )
    interpolated_phase = interpolator_phase(scatterers)

    # 3. Combine magnitude and phase
    interp_spectrum = interpolated_magnitude * np.exp(1j * interpolated_phase)

    return interp_spectrum

@register_interpolator(name='nearest', needs_flattened_grid=True)
def nearest_neighbor(grid_points, grid_values, scatterers):
    """
    Performs nearest neighbor interpolation using KDTree.

    Args:
        grid_points (ndarray): (N, 2) array of flattened grid coordinates ([z, x]).
        grid_values (ndarray): (N, n_freq) complex spectrum on the flattened grid points.
        scatterers (ndarray): (n_scatterers, 2) scatterer positions ([z, x]).

    Returns:
        ndarray: (n_scatterers, n_freq) complex spectrum interpolated at scatterer locations.
    """
    if grid_points.ndim != 2 or grid_values.ndim != 2:
         raise ValueError(f"'nearest' interpolator expects 2D flattened grid_points and grid_values, got shapes {grid_points.shape} and {grid_values.shape}")

    # Build a KD-tree for efficient nearest neighbor search
    tree = cKDTree(grid_points)

    # Find the index of the nearest grid point for each scatterer
    _, indices = tree.query(scatterers)

    # Assign the spectrum from the nearest grid point
    interp_spectrum = grid_values[indices, :]

    return interp_spectrum

# TODO search for nearest neighbor in 3D grid (z, x) and return the corresponding value
@register_interpolator(name='linear_mag_nearest_phase')
def linear_magnitude_nearest_phase(grid_points, grid_values, scatterers, x_range, z_range):
    """
    Interpolates magnitude linearly (using RegularGridInterpolator) and takes the phase
    from the nearest neighbor (using KDTree).

    Args:
        grid_points (ndarray): (nz, nx, 2) array of grid coordinates ([z, x]).
        grid_values (ndarray): (nz, nx, n_freq) complex spectrum on the grid points.
        scatterers (ndarray): (n_scatterers, 2) scatterer positions ([z, x]).
        x_range (array-like): 1D array of *unique* x-coordinates defining the grid.
        z_range (array-like): 1D array of *unique* z-coordinates defining the grid.

    Returns:
        ndarray: (n_scatterers, n_freq) complex spectrum interpolated at scatterer locations.
    """
    if grid_points.ndim != 3 or grid_values.ndim != 3:
        raise ValueError(f"'linear_mag_nearest_phase' expects 3D grid_points/values, got shapes {grid_points.shape} and {grid_values.shape}")

    nz, nx, n_freq = grid_values.shape
    if grid_points.shape != (nz, nx, 2):
        raise ValueError(f"Shape of 'grid_points' must be (nz, nx, 2), got {grid_points.shape}.")

    # Ensure scatterers are in the correct format (z, x) for RegularGridInterpolator
    scatterers_zx = scatterers[:, ::-1]  # Swap columns to [z, x]

    # 1. Linear interpolation for magnitude
    magnitude = np.abs(grid_values)
    interpolator_mag = RegularGridInterpolator(
        (z_range, x_range), magnitude,
        method='linear', bounds_error=False, fill_value=0.0
    )
    interpolated_magnitude = interpolator_mag(scatterers_zx)

    # 2. Nearest neighbor for phase
    # Flatten grid_points and grid_values for KDTree
    flat_grid_points = grid_points.reshape(-1, 2)  # (N, 2) where N = nz * nx
    flat_phase = np.angle(grid_values).reshape(-1, n_freq)  # (N, n_freq)
    tree = cKDTree(flat_grid_points)
    _, indices = tree.query(scatterers)
    interpolated_phase = flat_phase[indices, :]  # Phase from nearest grid point

    # 3. Combine magnitude and phase
    interp_spectrum = interpolated_magnitude * np.exp(1j * interpolated_phase)

    return interp_spectrum


@register_interpolator(name='green', needs_flattened_grid=True)
def green_function_interpolation(grid_points, grid_values, scatterers, x_range, z_range, param, freqs):
    """
    EXPERIMENTAL: Interpolate using the 2D free-space Green's function (Hankel)
    with the four closest *rectangular* grid neighbors.

    Expects FLATTENED grid_points (N, 2) and grid_values (N, n_freq).
    Requires x_range, z_range, param (for param.c), and freqs.

    Args:
        grid_points (ndarray): (N, 2) array of flattened grid coordinates ([z, x]).
        grid_values (ndarray): (N, n_freq) complex spectrum on the flattened grid points.
        scatterers (ndarray): (n_scatterers, 2) scatterer positions ([z, x]).
        x_range (array-like): 1D array of *unique* x-coordinates defining the grid.
        z_range (array-like): 1D array of *unique* z-coordinates defining the grid.
        param (object): Using `param.c` (speed of sound).
        freqs (array-like): 1D array of frequency values (Hz).

    Returns:
        ndarray: (n_scatterers, n_freq) complex spectrum interpolated at scatterer locations.
    """
    if grid_points.ndim != 2 or grid_values.ndim != 2:
         raise ValueError(f"'green' interpolator expects 2D flattened grid_points/values, got shapes {grid_points.shape} and {grid_values.shape}")

    n_x = len(x_range)
    n_z = len(z_range)
    n_scatterers, _ = scatterers.shape
    n_freq = grid_values.shape[1]

    if not hasattr(param, 'c'):
        raise AttributeError("Parameter object 'param' must have attribute 'c' (speed of sound).")
    if grid_values.shape[0] != n_x * n_z:
         raise ValueError("Shape of flattened 'grid_values' does not match dimensions from x_range and z_range.")

    # --- Pre-calculations ---
    w = 2 * np.pi * freqs
    k_waves = w / param.c  # Wavenumber, shape (n_freq,)

    interp_spectrum = np.zeros((n_scatterers, n_freq), dtype=grid_values.dtype)

    # Reshape grid values for easier indexing (z, x, freq) - needed to find neighbor values
    values_reshaped_3d = grid_values.reshape(n_z, n_x, n_freq)

    # --- Interpolation Loop ---
    for i, scatterer in enumerate(scatterers):
        x_s, z_s = scatterer[0], scatterer[1]

        # Find indices of the enclosing rectangle in the grid
        idx_x = np.searchsorted(x_range, x_s, side='right') - 1
        idx_z = np.searchsorted(z_range, z_s, side='right') - 1

        # Clip indices to be within valid range [0, n-2] to get 4 neighbors
        idx_x = np.clip(idx_x, 0, n_x - 2)
        idx_z = np.clip(idx_z, 0, n_z - 2)

        # Get coordinates of the four neighbors
        neighbor_coords = np.array([
            [x_range[idx_x],   z_range[idx_z]],
            [x_range[idx_x+1], z_range[idx_z]],
            [x_range[idx_x],   z_range[idx_z+1]],
            [x_range[idx_x+1], z_range[idx_z+1]]
        ])
        # Get values at the four neighbors using the 3D reshaped array
        neighbor_values = values_reshaped_3d[idx_z:idx_z+2, idx_x:idx_x+2, :].reshape(4, n_freq)

        # Calculate distances from scatterer to the four neighbors
        distances = np.sqrt(np.sum((neighbor_coords - scatterer)**2, axis=1)) # Shape (4,)

        # Calculate Green's function (Hankel function of the first kind, order 0)
        R = distances[:, None]  # Shape (4, 1)
        K = k_waves[None, :]    # Shape (1, n_freq)
        arg = K * R             # Shape (4, n_freq)
        G = (1j / 4) * hankel1(0, arg + _EPS) # Shape (4, n_freq)

        # Perform weighted sum: sum(G_i * Value_i) / sum(G_i)
        weighted_sum = np.sum(G * neighbor_values, axis=0) # Shape (n_freq,)
        sum_weights = np.sum(G, axis=0) + _EPS # Add epsilon to avoid division by zero
        interp_spectrum[i, :] = weighted_sum / sum_weights

    return interp_spectrum
