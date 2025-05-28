import numpy as np

from ..utils import Param
from ..interpolated.simus import simus as interpolated_simus

def simus(x_range: np.ndarray, z_range: np.ndarray, P_SPECT_grid: np.ndarray, x_scatterers: np.ndarray, z_scatterers: np.ndarray, interpolator_name: str, freqs: np.ndarray, RC: np.ndarray, param: Param, debug: bool = False, just_RF_spectrum: bool = False, lowResources: bool = False):
    """
    TODO: Add docstring for simus function.
    """
    return interpolated_simus(x_range=x_range, z_range=z_range, P_SPECT_grid=P_SPECT_grid, x_scatterers=x_scatterers, z_scatterers=z_scatterers, interpolator_name=interpolator_name, freqs=freqs, RC=RC, param=param, harmonic=True, debug=debug, just_RF_spectrum=just_RF_spectrum, lowResources=lowResources)
