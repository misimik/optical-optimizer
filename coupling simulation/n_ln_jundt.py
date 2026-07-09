# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:49:18 2023

@author: Michal
"""
from numba import njit
import numpy as np


@njit
def n_ln_jundt_5(lambda_, T):
    # Jundt, Dieter H. (1997). "Temperature-dependent Sellmeier equation for
    # the index of refraction n e n_{e} in congruent lithium niobate".
    # Optics Letters. 22 (20): 1553–5.
    # doi:10.1364/OL.22.001553
    # 5% of MgO

    f = (T - 24.5)*(T + 570.82)
    a1 = 5.756
    a2 = 0.0983
    a3 = 0.2020
    a4 = 189.32
    a5 = 12.52
    a6 = 1.32e-2
    b1 = 2.860e-6
    b2 = 4.700e-8
    b3 = 6.113e-8
    b4 = 1.516e-4
    b5 = 0
    n = np.sqrt(a1 + b1*f + (a2+b2*f)/(lambda_**2 - (a3+b3*f)**2) +
                (a4 + b4*f)/(lambda_**2 - a5**2) - (a6+b5*f)*lambda_**2)

    return n

def n_ln_jundt_1(lambda_, T):
    # Jundt, Dieter H. (1997). "Temperature-dependent Sellmeier equation for
    # the index of refraction n e n_{e} in congruent lithium niobate".
    # Optics Letters. 22 (20): 1553–5.
    # doi:10.1364/OL.22.001553
    # 1% of MgO

    f = (T - 24.5)*(T + 570.82)
    a1 = 5.078
    a2 = 0.0964
    a3 = 0.2065
    a4 = 61.16
    a5 = 10.55
    a6 = 1.59e-2
    b1 = 4.677e-7
    b2 = 7.822e-8
    b3 = -2.653e-8
    b4 = 1.096e-4
    b5 = 0
    n = np.sqrt(a1 + b1*f + (a2+b2*f)/(lambda_**2 - (a3+b3*f)**2) +
                (a4 + b4*f)/(lambda_**2 - a5**2) - (a6+b5*f)*lambda_**2)
    return n