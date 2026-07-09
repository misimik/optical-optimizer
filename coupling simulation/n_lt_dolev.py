# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:53:08 2023

@author: Michal
"""
import numpy as np


def n_lt_dolev(lambda_, T):
    # Appl. Phys. B 96, 423-432 (2009)
    # 0.5% MgO doped SLT extraordin?r

    f = (T - 24.5) * (T + 570.82)

    n = np.sqrt(4.5615 + 4.782 * 1e-7*f + (0.08488 + 3.0913 * 1e-8 * f) /
                (lambda_**2 - (0.1927 + 2.7326 * 1e-8 * f) ** 2) +
                (5.5832 + 1.4837 * 1e-5*f) /
                (lambda_**2 - (8.3067 + 1.3647 * 1e-7 * f)**2) -
                0.021696*lambda_**2)

    return n
