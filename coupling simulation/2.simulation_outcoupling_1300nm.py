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

# PM1550-XP
# MFD = 9.3 um
# NA = 0.125

#Units in nanometers

w0 = 9.3e-6/2     # Mode Field Radius in SMF-28 fiber at 1310 nm
lambda0 = 1.3e-6
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
f_c560_1300 = (13.86+extrapolated_focal_shift(lambda0, c560_model))*1e-3
f_a220_1300 = (11.00+extrapolated_focal_shift(lambda0, a220_model))*1e-3

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
    output_cbp = propagate_two_lenses(cbp, lambda0, focal1, focal2, focal1, 
                                      distance_variables[0], 
                                      distance_variables[1])
    return np.abs(1 - wg_mode_overlap(E, output_cbp, lambda0, waveguide_width,
                                      waveguide_thickness))


# %% Find best couling starting with different distances between the lenses

test_distances = np.arange(20,120,10)*1e-2 # Distances from 20 to 120 cm
results = np.zeros((test_distances.size, 3))
args = (E, q0, lambda0, waveguide_width, waveguide_thickness, f_c560_1300, f_a220_1300)


for i, distance in enumerate(test_distances):
    result = minimize(to_optimize_lens_separation, [distance, f_a220_1300], args=args, bounds=[(0.1, 2), (11e-3, 11.5e-3)], tol=1e-4)
    results[i,:2] = result.x
    results[i,2] = (1-result.fun)
    print(f"Iteration: {i}\tResult: {result.success}")
    
    
# %% Estimate tolerances - preparation

delta = np.max(results[:,1])-np.min(results[:,1])
# deltas = np.geomspace(delta*1e-3, 100*delta, 30)
# z3 = np.concatenate([np.mean(results[:,1])-deltas[::-1], np.mean(results[:,1])+deltas])
z3 = np.linspace(np.mean(results[:,1])-30*delta, np.mean(results[:,1])+30*delta, 60)
z2 = np.linspace(20,120,200)*1e-2


Z2, Z3 = np.meshgrid(z2,z3, indexing='ij')
 
W = np.zeros(Z2.shape)

# %% Estimate tolerances - calculation

for i in range(Z2.shape[0]):
    for j in range(Z3.shape[1]):
        W[i,j] = 1-to_optimize_lens_separation((Z2[i,j], Z3[i,j]), *args)
        
# %% Plot

title = "Coupling at 1300 nm for C560\ncollimating and A220TM coupling lenses"
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, ax1 = plt.subplots(figsize=(1000*px, 750*px))
pc = ax1.pcolormesh(Z2, Z3, W)
ax1.set_xlabel("Coupling lenses separation, m", fontsize=14)
ax1.set_ylabel("Waveguide to lens distance, m", fontsize=14)
fig.colorbar(pc, ax=ax1)
ax1.set_title(title, fontsize=18)