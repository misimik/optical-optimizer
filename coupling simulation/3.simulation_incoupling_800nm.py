# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:28:34 2023

@author: Michal Mikolajczyk (michal@mikolajczyk.link)
"""

import numpy as np
import os
import math
from complex_ray_transfer import *
from focus_shift import *
from waveguide_field import waveguide_field
from scipy.optimize import minimize
import matplotlib.pyplot as plt

tab_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
             'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
script_path = os.path.abspath(__file__)


# %% Simulation of colimation of light coming of a fiber

# PM780-HP
# MFD = 5.3 um
# NA = 0.12

#Units in nanometers

w0 = 5.3e-6/2     # Mode Field Radius in SMF-28 fiber at 1310 nm
lambda0 = 798e-9
n = 1           # Refractive index of air
# Complex Beam Parameter at the output of the fiber
q0 = cbp_2(0, lambda0, n, w0)

# %% Build lens models

# C560 lens: f=13.86 mm at 650 nm
# A220 lens: f=11.00 mm at 630 nm

file_folder_path = os.path.dirname(script_path)
C560_filename = "C560_focal_shift.csv"
file_path = os.path.join(file_folder_path, C560_filename)
test_wavelengths = np.linspace(300e-9,2300e-9,1000)
c560_model = build_lens_model(file_path, 450e-9, 1070e-9)
A220_filename = "A220_focal_shift.csv"
file_path = os.path.join(file_folder_path, A220_filename)
a220_model = build_lens_model(file_path, 450e-9, 1070e-9)

# Focal length of C560 at 1300 nm
f_c560_780 = (13.86+extrapolated_focal_shift(lambda0, c560_model))*1e-3
f_a220_780 = (11.00+extrapolated_focal_shift(lambda0, a220_model))*1e-3

# %% Calculate field distribution in the waveguide

# From NTT datasheet assuming thickness is smaller because mode leaks into the
# base
waveguide_width = 13e-6
waveguide_thickness = 12e-6 
dL = 0.1e-6 # Simulation resolution
temperature = 25
E = waveguide_field(lambda0, waveguide_width, waveguide_thickness,
                    temperature, dL )

# %% Function for minimalization to optimize for lens separation

def to_optimize_lens_separation(distance_variables, E, cbp, lambda0, waveguide_width, waveguide_thickness, focal1, focal2 ):
    output_cbp = propagate_two_lenses(cbp, lambda0, focal1, focal2, distance_variables[0], 
                                      distance_variables[1], 
                                      distance_variables[2])
    return np.abs(1 - wg_mode_overlap(E, output_cbp, lambda0, waveguide_width,
                                      waveguide_thickness))


# %% Find best couling starting with different distances between the lenses

test_distances = np.arange(20,120,10)*1e-2 # Distances from 20 to 120 cm
results = np.zeros((test_distances.size, 4))
# Both lenses are A220TM. The one next to the fiber is A220TM-B, and the one
# at the waveguide has custom AR coating.
args = (E, q0, lambda0, waveguide_width, waveguide_thickness, f_a220_780, f_a220_780)


for i, distance in enumerate(test_distances):
    result = minimize(to_optimize_lens_separation, [f_a220_780, distance, f_a220_780], args=args, bounds=[(10e-3, 12e-3), (0.2, 2), (10e-3, 12e-3)], tol=1e-4)
    results[i,:3] = result.x
    results[i,3] = (1-result.fun)
    print(f"Iteration: {i}\tResult: {result.success}")
    


# %% Calculate lenses positions for the factual distance of 75 cm

distance = 0.75
result = minimize(to_optimize_lens_separation, [f_a220_780, distance, f_a220_780], args=args, bounds=[(10e-3, 12e-3), (0.74, 0.76), (10e-3, 12e-3)], tol=1e-4)
# The beam is propagated from the fiber to the waveguide
print(f"Focal length fiber:\t\t{result.x[0]:.6f}\nLens separation:\t\t{result.x[1]:.6f}\nFocal length waveguide:\t{result.x[2]:.6f}")
# Fiber focal: 11.131 mm
# Waveguide focal: 10.934 mm
