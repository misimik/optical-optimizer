# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:32:54 2023

@author: Michal
"""
import numpy as np
from numba import njit


@njit
def kappa_x(kappa, k, n1, n2, n3, Lx):
    # definition of transcendental equation for kappa_x according to method
    # after Marcusse

    g2 = np.sqrt((n1**2 - n2**2) * k**2 - kappa**2)
    g3 = np.sqrt((n1**2 - n3**2) * k**2 - kappa**2)

    return np.tan(kappa * Lx) - n1**2 * kappa *\
        (n3**2 * g2 + n2**2 * g3) /\
        (n3**2 * n2**2 * kappa**2 - n1**4 * g2 * g3)
