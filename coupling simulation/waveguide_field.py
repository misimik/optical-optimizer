# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:33:11 2023

@author: Michal Mikolajczyk (michal@mikolajczyk.link)
"""

from n_air_ciddor import n_air_ciddor
from n_lt_dolev import n_lt_dolev
from n_ln_umemura import n_ln_umemura
from field_calc import field_calc

def waveguide_field(lambda_, width, thickness, temperature, dL):
    """
    Calculates E-field distribution in a Lithium Niobate waveguide for given
    dimensions, wavelength and temperature

    Parameters
    ----------
    lambda_ : float
        Wavelength in vacuum.
    thickness : float
        Thickness of the waveguide.
    width : float
        Width of the waveguide (ridge).
    temperature : float
        Temperature.
    dL : float
        Calculation resolution (length)

    Returns
    -------
    E : np.aray() of complex
        E-field distribution

    """
    n_ln = n_ln_umemura(lambda_*1e6, temperature)
    n_lt = n_lt_dolev(lambda_*1e6, temperature)
    n_air = n_air_ciddor(lambda_*1e6)
    
    E = field_calc( lambda_, thickness, width, n_ln, n_lt, n_air, dL )
    return E


