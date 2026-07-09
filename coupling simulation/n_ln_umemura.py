# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:49:18 2023

@author: Michal
"""
from numba import njit
import numpy as np


@njit
def n_ln_umemura(lambda_, T):
    # APPLIED OPTICS / Vol. 53, No. 25 / 1 September 2014
    # 5% MgO doped CLN extraordinary

    # n at 20 degree C
    n = np.sqrt(4.54514 + 0.096471 / (lambda_**2 - 0.043763) -
                0.021502*lambda_**2)
    Dn = (0.4175 / lambda_**3 - 0.6643 / lambda_**2 + 0.9036/lambda_ +
          3.5332 - 0.0744 * lambda_) * (T - 20 + 0.00138 * (T - 20)**2) * 1e-5

    n = n + Dn

    return n
