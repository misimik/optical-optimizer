# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:56:05 2023

@author: Michal
"""

import numpy as np
from scipy.optimize import fsolve
from kappa_x import kappa_x
from kappa_y import kappa_y


def beta_calc(lambda_, d, b, n1, n2, n3, dL):

    k = 2 * np.pi / lambda_

    zeros_x = 2 * np.pi / (d * n1)
    zeros_y = 2 * np.pi / (b * n1)

    kax = fsolve(lambda kappa: kappa_x(kappa, k, n1, n2, n3, d), zeros_x)
    kay = fsolve(lambda kappa: kappa_y(kappa, k, n1, n3, b), zeros_y)

    beta = np.sqrt(n1**2 * k**2 - kax**2 - kay ** 2)

    return beta
