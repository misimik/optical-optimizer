# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:33:11 2023

@author: Michal Mikolajczyk (michal@mikolajczyk.link)
"""
from numba import njit


@njit
def thermal_expansion(L0, T):
    # Thermal expansion of lithium niobate at temperature T with respect to
    # reference temperature T0 = 19 degreeC
    # parameters taken from J Appl Phys 40, 4637-4641 (1969)

    T0 = 19
    a = 1.53e-5
    b = 5.3e-9

    return (L0 * (1 + a * (T - T0) + b * (T - T0)**2))


# @njit
def reverse_thermal_expansion(L, T):
    # Thermal expansion of lithium niobate at temperature T with respect to
    # reference temperature T0 = 19 degreeC
    # parameters taken from J Appl Phys 40, 4637-4641 (1969)

    T0 = 19
    a = 1.53e-5
    b = 5.3e-9

    return (L / (1 + a * (T - T0) + b * (T - T0)**2))
