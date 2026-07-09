# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:28:34 2023

@author: Michal Mikolajczyk (michal@mikolajczyk.link)
"""

import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import math

script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))

# %%

from complex_ray_transfer import *
from focus_shift import *
from waveguide_field import waveguide_field


tab_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
             'tab:brown', 'tab:pink', 'tab:gray|', 'tab:olive', 'tab:cyan']



# %% Simulation of colimation of light coming of a fiber

# PM1550-XP
# MFD = 13 um
# NA = 0.125

#Units in nanometers

# Mode Field Radius in SMF-28 fiber at 2070 nm. Source is the the Thorlabs
# F028APC-2000 collimator design parameters.
w0 = 1.2e-3/2     
lambda0 = 2070e-9
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
f_c560_2070 = (13.86+extrapolated_focal_shift(lambda0, c560_model))*1e-3
f_a220_2070 = (11.00+extrapolated_focal_shift(lambda0, a220_model))*1e-3

# Focal length at 780 nm
f_a220_780 = (11.00+extrapolated_focal_shift(780e-9, a220_model))*1e-3

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

def to_optimize_lens_separation(variables, E, cbp, lambda0, waveguide_width, waveguide_thickness, focal1 ):
    output_cbp = propagate_two_lenses(cbp, lambda0, variables[3], focal1,
                                      variables[0], variables[1], variables[2])
    return np.abs(1 - wg_mode_overlap(E, output_cbp, lambda0, waveguide_width,
                                      waveguide_thickness))


# %% Measure the mode size in the waveguide

I = np.multiply(E, np.conj(E))
norm_I = np.sum(I)
I = I/norm_I
maximum = np.unravel_index(I.argmax(), I.shape)
x = np.arange(0,waveguide_width,dL)
y = np.arange(0,waveguide_thickness,dL)
Ix = np.abs(I[:,maximum[1]])
Ix = Ix/np.max(Ix)
Iy = np.abs(I[maximum[0],:])
Iy = Iy/np.max(Iy)

mx = np.abs(Ix - 1/np.exp(2))
wx = np.abs(x[mx.argsort()[0]]-x[mx.argsort()[1]])
my = np.abs(Iy - 1/np.exp(2))
wy = np.abs(y[my.argsort()[0]]-y[my.argsort()[1]])
print(f"Mode width x: {wx:.3e}\n"+
      f"Mode width y: {wy:.3e}")


# %% w-based optimization - two lenses


w0 = 1.2e-3/2
w_waveguide = 10.1e-6/2
lambda0 = 2070e-9
n = 1           # Refractive index of air
# Complex Beam Parameter at the saddle point 5.73 mm from the collimator
# The curvature radius of 3 m results in the divergence angle close to the
# specified 0.13 deg. The waist in the saddle is 1.2 mm

q_in = cbp_1(3, lambda0, n, w0)
f_a220_780_optimized = 10.934e-3 

q_test = propagate_two_lenses(q_in, lambda0, 1e4, f_a220_2070, 1.09, 0.25, f_a220_780_optimized)

args = (q_in, lambda0, f_a220_2070, f_a220_780_optimized, w_waveguide)

def optimize_w(variables, cbp, lambda0, focal_2, z3, w_waveguide):
    q_out = propagate_two_lenses(cbp, lambda0, variables[0], focal_2,
                                 variables[1], variables[2],
                                 f_a220_780_optimized)
    return np.abs(w_waveguide-cbp_to_w_alt(q_out, lambda0))*1e6

res = minimize(optimize_w, [0.3, 1, 0.25],
               bounds=((0.3, 0.3), (0.7, 5),(0.23, 0.4)), args=args, tol=1e-4)
    
q_out = propagate_two_lenses(q_in, lambda0, res.x[0], f_a220_2070, res.x[1], res.x[2], f_a220_780_optimized)
print(f"w = {cbp_to_w_alt(q_out, lambda0)}")

# %% w-based optimization - three lenses 

w0 = 1.2e-3/2
w_waveguide = 10.1e-6/2
lambda0 = 2070e-9
n = 1           # Refractive index of air
# Complex Beam Parameter at the saddle point 5.73 mm from the collimator
# The curvature radius of 3 m results in the divergence angle close to the
# specified 0.13 deg. The waist in the saddle is 1.2 mm

q_in = cbp_1(3, lambda0, n, w0)
f_a220_780_optimized = 10.934e-3 
q_test = propagate_two_lenses(q_in, lambda0, 1e4, f_a220_2070, 1.09, 0.25, f_a220_780_optimized)
q_out = propagate_three_lenses(q_in, lambda0, -0.25, 0.25, f_a220_2070, 0.15, 1, 0.25, f_a220_780_optimized)

def optimize_w(variables, cbp, lambda0, focal_1, z4, w_waveguide):
    q_out = propagate_three_lenses(cbp, lambda0, variables[0], variables[1], focal_1, variables[2], variables[3], variables[4], z4)
    return np.abs(w_waveguide-cbp_to_w_alt(q_out, lambda0))*1e6

args = (q_in, lambda0, f_a220_2070, f_a220_780_optimized, w_waveguide)


lens_1_f = (0.153, 0.153)
lens_2_f = (0.5, 0.5)
distance_01 = (0.1, 2)
distance_12 = (0.4, 2)
distance_23 = (0.38, 0.47)

res = minimize(optimize_w, [-0.3, 0.5, 0.15, 1, 0.35],
               bounds=(lens_1_f, lens_2_f, distance_01, distance_12, distance_23), args=args, tol=1e-4)
    
q_out = propagate_three_lenses(q_in, lambda0, res.x[0], res.x[1], f_a220_2070, res.x[2], res.x[3], res.x[4], f_a220_780_optimized)
print(f"w = {cbp_to_w_alt(q_out, lambda0)}")
print(f"focal 1 = {res.x[0]}\n"+\
      f"distance 1 = {res.x[2]}\n"+\
      f"focal 2 = {res.x[1]}\n"+\
      f"distance 2 = {res.x[3]}\n"+\
      f"distance 3 = {res.x[4]}")
    
    
overlap = wg_mode_overlap(E, q_out, lambda0, waveguide_width, waveguide_thickness)
print(f"Overlap = {overlap*100:.2f} %")
    
# %% Verify the w-based optimiziation with E-field overlap

overlap = wg_mode_overlap(E, q_out, lambda0, waveguide_width, waveguide_thickness)
print(f"Overlap = {overlap*100:.2f} %")

# %% Propagation from the fiber to waveguide based on optimized parameters

# Mode of 2070 in the waveguide has 10.1 um of 1/e**2 width.

# plt.close(fig)

q0 = q_in

def propagate(z, cbp, d, f):
    if z < res.x[2]: # To the first lens
        q1 = m_multiply(propagation_M(z), cbp)
        return cbp_to_w_alt(q1, lambda0)
    elif z < res.x[2]+res.x[3]: # From the first to the second lens
        q1 = m_multiply(propagation_M(res.x[2]), cbp)
        q2 = m_multiply(thin_lens_M(res.x[0]), q1)
        q3 = m_multiply(propagation_M(z-res.x[2]), q2)
        return cbp_to_w_alt(q3, lambda0)
    elif z < res.x[2] + res.x[3] + res.x[4]: # After the second lens
        q1 = m_multiply(propagation_M(res.x[2]), cbp)
        q2 = m_multiply(thin_lens_M(res.x[0]), q1)
        q3 = m_multiply(propagation_M(res.x[3]), q2)
        q4 = m_multiply(thin_lens_M(res.x[1]), q3)
        q5 = m_multiply(propagation_M(z-res.x[2]-res.x[3]), q4)
        return cbp_to_w_alt(q5, lambda0)
    elif z < res.x[2] + res.x[3] + res.x[4] + f_a220_780_optimized:
        q1 = m_multiply(propagation_M(res.x[2]), cbp)
        q2 = m_multiply(thin_lens_M(res.x[0]), q1)
        q3 = m_multiply(propagation_M(res.x[3]), q2)
        q4 = m_multiply(thin_lens_M(res.x[1]), q3)
        q5 = m_multiply(propagation_M(res.x[4]), q4)
        q6 = m_multiply(thin_lens_M(f_a220_2070), q5)
        q7 = m_multiply(propagation_M(z-res.x[2]-res.x[3]-res.x[4]), q6)
        return cbp_to_w_alt(q7, lambda0)
    else:
        q1 = m_multiply(propagation_M(res.x[2]), cbp)
        q2 = m_multiply(thin_lens_M(res.x[0]), q1)
        q3 = m_multiply(propagation_M(res.x[3]), q2)
        q4 = m_multiply(thin_lens_M(res.x[1]), q3)
        q5 = m_multiply(propagation_M(res.x[4]), q4)
        q6 = m_multiply(thin_lens_M(f_a220_2070), q5)
        q7 = m_multiply(propagation_M(f_a220_780_optimized), q6)
        return cbp_to_w_alt(q7, lambda0)
    
z = np.linspace(0, 2.5, 20_000)

w = np.zeros(z.shape)
for i, zi in enumerate(z):
    w[i] = propagate(zi, q0, 0.25, 0.25)

title = "Beam half-width while propagating through the system"
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, ax1 = plt.subplots(figsize=(1000*px, 750*px))
ax1.plot(z, w*1e3, '.')
ax1.set_xlabel("Distance along propagation axis, m", fontsize=14)
ax1.set_ylabel("Beam half-width, mm", fontsize=14)
ax1.set_title(title, fontsize=18)
# ax1.legend(loc="lower left")
plt.show()