# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 15:56:05 2023

@author: Michal
"""

import numpy as np
from scipy.optimize import fsolve
from kappa_x import kappa_x
from kappa_y import kappa_y


def field_calc(lambda_, thickness, width, n_medium, n_base, n_enviornment, dL):
    """
    Calculate E-field distribution in a waveguide of medium placed on base with
    surrounding enviornment

    Parameters
    ----------
    lambda_ : float
        Wavelength in vacuum.
    thickness : float
        Thickness of the waveguide.
    width : float
        Width of the waveguide.
    n_medium : float
        Refractive index of the medium.
    n_base : float
        Refractive index of the base.
    n_enviornment : float
        Refractive index of the enviornment.
    dL : float
        Calculation resolution (length).

    Returns
    -------
    Ex : np.array of complex
        E-field distribution.

    """

    k = 2 * np.pi / lambda_

    zeros_x = 2 * np.pi / (thickness * n_medium)
    zeros_y = 2 * np.pi / (width * n_medium)

    kax = fsolve(lambda kappa: kappa_x(kappa, k, n_medium, n_base, n_enviornment, thickness), zeros_x)
    kay = fsolve(lambda kappa: kappa_y(kappa, k, n_medium, n_enviornment, width), zeros_y)

    g3 = np.sqrt((n_medium ** 2 - n_enviornment ** 2) * k ** 2 - kax ** 2)
    g5 = np.sqrt((n_medium ** 2 - n_enviornment ** 2) * k ** 2 - kay ** 2)

    xi = (1 / kax) * np.arctan(- (n_enviornment / n_medium) ** 2 * kax / g3)
    eta = (1 / kay) * np.arctan(- g5 / kay)
    beta = np.sqrt(n_medium**2 * k**2 - kax**2 - kay ** 2)

    f = 0.002654418729345            # sqrt(eps0/mu0)

    # Meshgrid for WG cross section
    X, Y = np.meshgrid(np.arange(-thickness, 0, dL), np.arange(0, width, dL))
    dA = dL ** 2                             # surface element

    Ex = (1j / kax / beta) * (n_medium**2 * k**2 - kax**2) * \
        np.sin(kax * (X + xi)) * np.cos(kay * (Y + eta))
    Hy = 1j * f * n_medium**2 * (k / kax) *\
        np.sin(kax * (X + xi)) * np.cos(kay * (Y + eta))
    Sz = Ex * Hy.conj() / 2

    Anorm = np.sqrt(np.sum(np.sum(Sz)) * dA)**-1

    Ex *= Anorm

    return Ex
