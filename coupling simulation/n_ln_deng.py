# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:49:18 2023

@author: Michal
"""
from numba import njit
import numpy as np


@njit
def n_ln_deng(lambda_, T):
    # Deng, L.H.; Gao, X.M.; Cao, Z.S.; Chen, W.D.; Yuan, Y.Q.; Zhang, W.J.;
    # Gong, Z.B. (2006). "Improvement to Sellmeier equation for periodically
    # poled LiNbO3 crystal using mid-infrared difference-frequency generation".
    # Optics Communications. 268 (1): 110–114. Bibcode:2006OptCo.268..110D.
    # doi:10.1016/j.optcom.2006.06.082.

    # n at 20 degree C
    f = (T - 24.5)*(T + 570.82)
    a1 = 5.39121
    a2 = 0.100473
    a3 = 0.20692
    a4 = 100
    a5 = 11.34927
    a6 = 1.544e-2
    b1 = 4.96827e-7
    b2 = 3.862e-8
    b3 = -0.89e-8
    b4 = 2.657e-5
    b5 = 9.62119e-10
    n = np.sqrt(a1 + b1*f + (a2+b2*f)/(lambda_**2 - (a3+b3*f)**2) +
                (a4 + b4*f)/(lambda_**2 - a5**2) - (a6+b5*f)*lambda_**2)

    return n
