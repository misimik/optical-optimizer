# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 18:24:20 2023

@author: Michal
"""
from numba import njit
import numpy as np
from deff_calc import deff_calc


@njit
def gamma_calc(idler, pumpe, Es, Ei, Ep, T, dA, P, n1, n2, n3, ln):
    # Calculation of gain factor Gamma, refer to "Suhura, Waveguide
    # Nonlinear-Optic Devices" 1st Edition, 2003: Eq. 3.63

    # constants
    eps0 = 8.854187817e-12
    c = 299792458

    # frequencies of idler and pump field
    omega_i = 2 * np.pi * c / idler
    omega_p = 2 * np.pi * c / pumpe

    # calculate effective nonlinear coefficient
    d = deff_calc(n1, n2, n3, ln, T) * 1e-12

    # overlap integral between all electric fields
    A = np.sum(Ei.conj() * Es * Ep.conj())

    # coupling constant
    coup_p = (dA * d * omega_p * eps0 / 2) * A

    gamma = np.abs(coup_p * np.sqrt(P * omega_i / omega_p))

    return gamma
