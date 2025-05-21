import numpy as np

from ..interpolated.simus import simus as interpolated_simus

def simus(x_range, z_range, P_SPECT_grid, x_scatterers, z_scatterers, interpolator_name, freqs, RC, param, debug=False, just_RF_spectrum=False):
    """
    
    """
    return simus(x_range=x_range, z_range=z_range, P_SPECT_grid=P_SPECT_grid, x_scatterers=x_scatterers, z_scatterers=z_scatterers, interpolator_name=interpolator_name, freqs=freqs, RC=RC, param=param, harmonic=True, debug=debug, just_RF_spectrum=just_RF_spectrum)