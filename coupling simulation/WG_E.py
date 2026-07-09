# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:33:11 2023

@author: Michal Mikolajczyk (michal@mikolajczyk.link)
"""

from n_air_ciddor import n_air_ciddor
from n_lt_dolev import n_lt_dolev
from n_ln_umemura import n_ln_umemura
from field_calc import field_calc

def W_GE(lambda_, d, b, T):
    n_ln = n_ln_umemura(lambda_*1e3, T)
    n_lt = n_lt_dolev(lambda_*1e3, T)
    n_air = n_air_ciddor(lambda_*1e3, T)
    
    E = field_calc( lambda_, d, b, n_ln, n_lt, n_air, dL )
    return E


