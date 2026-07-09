# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:56:05 2023

@author: Michal
"""

import numpy as np
from scipy.optimize import fsolve
from kappa_x import kappa_x
from kappa_y import kappa_y


def beta_field_calc(lambda_, d, b, n1, n2, n3, dL):

    k = 2 * np.pi / lambda_

    zeros_x = 2 * np.pi / (d * n1)
    zeros_y = 2 * np.pi / (b * n1)

    kax = fsolve(lambda kappa: kappa_x(kappa, k, n1, n2, n3, d), zeros_x)
    kay = fsolve(lambda kappa: kappa_y(kappa, k, n1, n3, b), zeros_y)

    g3 = np.sqrt((n1 ** 2 - n3 ** 2) * k ** 2 - kax ** 2)
    g5 = np.sqrt((n1 ** 2 - n3 ** 2) * k ** 2 - kay ** 2)

    xi = (1 / kax) * np.arctan(- (n3 / n1) ** 2 * kax / g3)
    eta = (1 / kay) * np.arctan(- g5 / kay)
    beta = np.sqrt(n1**2 * k**2 - kax**2 - kay ** 2)

    f = 0.002654418729345            # sqrt(eps0/mu0)

    # Meshgrid for WG cross section
    X, Y = np.meshgrid(np.arange(-d, 0, dL), np.arange(0, b, dL))
    dA = dL ** 2                             # surface element

    Ex = (1j / kax / beta) * (n1**2 * k**2 - kax**2) * \
        np.sin(kax * (X + xi)) * np.cos(kay * (Y + eta))
    Hy = 1j * f * n1**2 * (k / kax) *\
        np.sin(kax * (X + xi)) * np.cos(kay * (Y + eta))
    Sz = Ex * Hy.conj() / 2

    Anorm = np.sqrt(np.sum(np.sum(Sz)) * dA)**-1

    Ex *= Anorm
    Hy *= Anorm
    Sz = Ex * Hy.conjugate() / 2

    return beta, Ex, Sz
