# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:21:02 2023

@author: Michal
"""
from numba import njit


@njit
def n_air_ciddor(lambda_):
    # Appl. Optics 35, 1566-1573 (1996)
    # for Standard air: dry air at 15 C, 101.325 kPa,
    # and with 450 ppm CO2 content.

    n = 1 + 0.05792105 / (238.0185 - lambda_**(-2)) +\
        0.00167917 / (57.362 - lambda_**(-2))

    return n
