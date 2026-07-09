# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:32:54 2023

@author: Michal
"""
import numpy as np
from numba import njit


@njit
def kappa_y(kappa, k, n1, n3, Ly):
    # definition of transcendental equation for kappa_y according to method
    # after Marcusse

    g4 = np.sqrt((n1**2 - n3**2) * k**2 - kappa**2)

    return np.tan(kappa * Ly) - 2 * kappa * g4 / (kappa**2 - g4**2)
