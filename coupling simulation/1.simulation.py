# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:15:56 2023

@author: Michal Mikolajczyk (michal@mikolajczyk.link)
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import os
from waveguide_field import waveguide_field
from thermal_expansion import thermal_expansion
from scipy.optimize import minimize, curve_fit
from scipy import interpolate

tab_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
             'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
script_path = os.path.abspath(__file__)

# %% Functions definitions

def propagation_M(d):
    """
    Transfer matrix of freespace propagation

    Parameters
    ----------
    d : float
        Propagation distance

    Returns
    -------
    numpy.array
        Transfer matrix.

    """
    return np.array([[1, d], [0, 1]])


def thin_lens_M(f):
    """
    Transfer matrix of a thin lens

    Parameters
    ----------
    f : float
        Focal length.

    Returns
    -------
    numpy.array
        Transfer matrix.

    """
    return np.array([[1, 0], [-1/f, 1]])


def thick_lens_M(n1, n2, R1, R2, t):
    """
    Transfer matrix for thick lens

    Parameters
    ----------
    n2 : float
        Refractive index outisde of the lens
    n2 : float
        Refractive index of glass
    R1 : float
        Radius of the first surface
    R2 : float
        Radius of the second surface
    t : float
        Radius of the lens at the optical axis

    Returns
    -------
    numpy.array
        Transfer matrix.
    """
    M_interface1 = np.array([[1, 0], [(n1-n2)/(R1*n2), n1/n2]])
    M_interface2 = np.array([[1, 0], [(n2-n1)/(R2*n1), n2/n1]])
    # With .dot() use the same order of matrices as on paper
    # Numba is compatible with numpy.dot
    M = M_interface2.dot(propagation_M(t)).dot(M_interface1)
    return M

def cbp_1(R, lambda0, n, w):
    """
    Calculate complex beam parameter, first approach.

    Parameters
    ----------
    R : float
        Wavefront radius of curvature at the position of calculation.
    lambda0 : float
        Wavelength in vacuum.
    n : float
        Refractive index of the medium.
    w : float
        Beam radius (1/e**2).

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return  1/(1/R - 1j*lambda0/(np.pi*n*w**2))


def cbp_2(z, lambda0, n, w0):
    """
    Calculate complex beam parameter, second approach.

    Parameters
    ----------
    z : float
        Distance from the beam waist
    lambda0 : float
        Wavelength.
    n : float
        Rerfractive index of the medium.
    w0 : float
        Beam waist radius.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return z + 1j*np.pi*n*w0**2/lambda0

def cbp_to_w(q, lambda0, n):
    """
    Convert Complex Beam Parameter to waist radius at the location

    Parameters
    ----------
    q : complex
        Complex Beam Parameter
    lambda0 : float
        Wavelength in vacuum.
    n : float
        Refractive index of the medium.

    Returns
    -------
    float
        Beam waist radius.

    """
    return np.emath.sqrt( lambda0 / ( np.pi * n * np.imag( 1/q ) ) )

def cbp_to_w_alt(cbp, lambda_):
    """
    Calculate beam waist radius with the other equation

    Parameters
    ----------
    cbp : complex
        Complex Beam Parameter.
    lambda_: float
        Wavelength.

    Returns
    -------
    wz : float
        Beam waist radius.

    """
    z_r = np.imag(cbp)
    z = np.real(cbp)
    w0 = cbp_to_w0(cbp, lambda0)
    wz = w0*np.sqrt(1+(z/z_r)**2)
    return wz

def cbp_to_w0(cbp, lambda_):
    """
    Calculate beam waist radius in the waist

    Parameters
    ----------
    cbp : complex
        Complex Beam Parameter.
    lambda_: float
        Wavelength.

    Returns
    -------
    w0 : float
        Beam waist radius in the waist.

    """
    z_r = np.imag(cbp)
    w0 = np.sqrt(lambda_*z_r/np.pi)
    return w0

def m_multiply(M,cbp):
    """
    Sigle step multiplication of the complex beam parameter by a transfer
    matrix an normalization

    Parameters
    ----------
    M : numpy.array(2,2)
        Transfer matrix
    cbp : complex
        Complex beam parameter.

    Returns
    -------
    complex
        Complex Beam Parameter.

    """
    return (M[0,0]*cbp+M[0,1])/(M[1,0]*cbp+M[1,1])

    # Gaussian beam field distribution for the light from the fiber to the
    # waveguide
def cbp_to_field_distribution(r, cbp, lambda_):
    """
    Calculate E-field from complex beam parameter and the distance from the
    beam center

    Parameters
    ----------
    r : float
        Radius (distance from beam center).
    cbp : complex
        Complex Beam Parameter.
    lambda_ : float
        Wavelength.

    Returns
    -------
    E : complex
        E-field distribution.

    """
    w_z = cbp_to_w_alt(cbp, lambda_)
    w0 = cbp_to_w0(cbp, lambda_)
    z_r = np.imag(cbp)
    z = np.real(cbp)
    R = 1/np.real(1/cbp)
    phi = math.atan(z/z_r)
    k = 2*np.pi/lambda_
    E = w0/w_z*np.exp(-r**2/w_z**2)*np.exp(-1j*(2*k*(z + r**2/(2*R))-phi ))
    return E
 

def wg_mode_overlap(E, cbp, lambda0, waveguide_x, waveguide_y):
    """
    Calculates an overlap between a waveguide mode E-field and an E-filed of
    a propagated Gaussian beam

    Parameters
    ----------
    E : np.array of complex
        Waveguide mode E-field.
    cbp : complex
        Complex beam parameter at the facet of the waveguide.
    lambda0 : float
        Wavelength in vacuum.
    waveguide_x : float
        Waveguide width.
    waveguide_y : float
        Waveguide thickness.

    Returns
    -------
    overlap : float
        Overlap fraction from 0 to 1.

    """
    
    # Normalize the waveguide mode E field
    I = np.abs(np.multiply(E, np.conj(E)))
    norm_I = np.sum(I)
    I = I/norm_I
    E = E/np.sqrt(norm_I)
    
    # Find the maximum of the waveguide mode
    maximum = np.unravel_index(I.argmax(), I.shape)
      
    # Calculate the Gaussian Beam field
    X, Y = np.meshgrid(np.linspace(0,waveguide_x, I.shape[0]),
                       np.linspace(0, waveguide_y, I.shape[1]),
                       indexing='ij')
    x_center = X[maximum[0], 0]
    y_center = Y[0, maximum[1]]
    E_propagated = cbp_to_field_distribution(np.sqrt((X-x_center)**2+
                                                     (Y-y_center)**2), 
                                             cbp, lambda0)
    
    # Normalize the propagated beam E field
    I_propagated = np.abs( np.multiply( E_propagated,
                                       np.conj( E_propagated ) ) )
    norm_I_propagated = np.sum(I_propagated)
    I_propagated = I_propagated/norm_I_propagated
    E_propagated = E_propagated/np.sqrt(norm_I_propagated)
    
    # Calculate the overlap
    overlap = np.sum(np.abs(np.multiply(E, np.conj(E_propagated))))
    
    return overlap


def propagate_two_lenses(cbp, lambda0, f_1, f_2, z1, z2, z3):
    """
    Function propagating Complex Beam Parameter through two lenses padded at
    the entrance and the output

    Parameters
    ----------
    cbp : complex
        Complex Beam Parameter at the input.
    lambda0 : float
        Wavelength in vacuum.
    f_1 : float
        First lens focal length.
    f_2 : float
        Second lens focal length.
    z1 : float
        Distance from the input to the first lens.
    z2 : float
        Distance from the first lens to the second.
    z3 : float
        Distance from the second lens to the output.

    Returns
    -------
    q5 : complex
        Coplex Beam Parameter at the output.

    """

    # CBP at the first lens
    q1 = m_multiply( propagation_M(z1), cbp )

    # CBP after the first lens
    q2 = m_multiply( thin_lens_M(f_1), q1 )

    # CBP at the second lens
    q3 = m_multiply( propagation_M(z2), q2 )

    # CBP after the second lens
    q4 = m_multiply(thin_lens_M(f_2), q3)

    # CBP at the output
    q5 = m_multiply(propagation_M(z3), q4)
    
    return q5

# %% Create model of the focus shift from the Thorlabs data

def build_lens_model(file_path, lower_linear_limit, upper_linear_limit):
    """
    Build focus shift model from data using interpolation where data is
    available and extrapolation assumin good linear fit at the edges of
    the data

    Parameters
    ----------
    file_path : path
        Path to the focus shift data.
    lower_linear_limit : float
        Wavelength were the focus shift stops being approximated by
        linear/cubic function.
    upper_linear_limit : float
        Wavelength where the focus shift starts againg being approximated by
        linear/cubic function.

    Returns
    -------
    tuple
        Set of parameters sufficient for calculating focus shift in this model.

    """
    with open(file_path, 'r') as file:
        data = np.loadtxt(file, delimiter=';')
        wavelengths = data[:,1]
        focal_shift = data[:,0]
    
        
    
    lower_linear_limit = lower_linear_limit*1e9
    upper_linear_limit = upper_linear_limit*1e9
    # Select linear regions to fit extrapolating functions
    long_linear_wavelengths = wavelengths[wavelengths > upper_linear_limit]
    long_linear_focal_shift = focal_shift[wavelengths > upper_linear_limit]
    
    
    short_linear_wavelengths = wavelengths[wavelengths < lower_linear_limit]
    short_linear_focal_shift = focal_shift[wavelengths < lower_linear_limit]
    
    # Extrapolating functions
    def linear_function(x,a1,a0):
        return a1*x+a0
    
    def cubic_function(x,a2, a1, a0):
        return a2*x**2+a1*x+a0
    
    # Fit the extrapolating functions to the data
    long_popt, long_pcov = curve_fit(linear_function, long_linear_wavelengths,
                                     long_linear_focal_shift)
    
    short_popt, short_pcov = curve_fit(cubic_function, short_linear_wavelengths,
                                       short_linear_focal_shift)
    
    interpolated = interpolate.interp1d(wavelengths, focal_shift)
    model = (lower_linear_limit, upper_linear_limit, long_popt, short_popt,
             interpolated)
    return model


def extrapolated_focal_shift(wavelength, model):
    """
    Define the focal shift funtion for the lens

    Parameters
    ----------
    wavelength : float or np.array[dtype=float]
        Wavelengths for which the shift must be calculated in meters.
    model : tuple
        Set of parameters necessary for calculating the focal shift

    Returns
    -------
    result : float or np.array[dtype=float]
        Focal shift.

    """
    wavelength = wavelength*1e9 # Convert m to nm
    lower_linear_limit, upper_linear_limit, long_popt, short_popt,\
    interpolated = model
        
    scalar = np.isscalar(wavelength)
    if scalar:
        wavelength = np.array([wavelength])
    
    # Create masks for each condition
    mask_interpolate = np.logical_and(lower_linear_limit <= wavelength, wavelength <= upper_linear_limit)
    mask_long = wavelength > upper_linear_limit
    mask_short = wavelength < lower_linear_limit
    
    # Extrapolating functions
    def linear_function(x,a1,a0):
        return a1*x+a0
    
    def cubic_function(x,a2, a1, a0):
        return a2*x**2+a1*x+a0
   
    # Apply conditions
    result = np.empty_like(wavelength)
    result[mask_interpolate] = interpolated(wavelength[mask_interpolate])
    result[mask_long] = linear_function(wavelength[mask_long], *long_popt)
    result[mask_short] = cubic_function(wavelength[mask_short], *short_popt)
    
    if scalar:
        return result.item()
    else:
        return result
    
    
# %% Build lens models

file_folder_path = os.path.dirname(script_path)
C560_filename = "C560_focal_shift.csv"
file_path = os.path.join(file_folder_path, C560_filename)
test_wavelengths = np.linspace(300e-9,2300e-9,1000)
c560_model = build_lens_model(file_path, 450e-9, 1070e-9)
A220_filename = "A220_focal_shift.csv"
file_path = os.path.join(file_folder_path, A220_filename)
a220_model = build_lens_model(file_path, 450e-9, 1070e-9)  

# %% Simulation of colimation of light coming of a fiber

# PM1550-XP
# MFD = 9.3 um
# NA = 0.125

#Units in nanometers


w0 = 9.3e-6/2     # Mode Field Radius in SMF-28 fiber at 1310 nm
lambda0 = 1.3e-6
n = 1           # Refractive index of air
# Focal length of C560 at 1300 nm
f_1300 = (13.86+extrapolated_focal_shift(lambda0, c560_model))*1e-3 

w0_bis = lambda0/np.pi/0.125


# Complex Beam Parameter at the output of the fiber
q0 = cbp_2(0, lambda0, n, w0)
w00 = cbp_to_w_alt(q0, lambda0)

# Verifing the NA
q05 = m_multiply(propagation_M(1), q0)
w05 = cbp_to_w_alt(q05, lambda0)
r = np.linspace(0, 2*0.14, 1000)
intensity = np.abs(cbp_to_field_distribution(r, q05, lambda0))**2
intensity = intensity/intensity.max()
# Find the 1% power radius of the beam
NA = math.sin(math.atan(r[np.abs(intensity-0.01).argmin()]))
print(f"NA: {NA}\t w0: {w00}")


#%%  Plot Gauss field distribution

waveguide_width = 13e-6
waveguide_thickness = 12e-6
dL = 0.1e-6 # Simulation resolution
temperature = 25
E = waveguide_field( lambda0, waveguide_width, waveguide_thickness, temperature, dL )


# %%

Iw = np.multiply(E, np.conj(E))
plt.figure()
plt.imshow(np.abs(Iw).T, origin="lower")
I_overlap = np.abs(np.multiply(E, np.conj(E_propagated)))
plt.figure()
plt.imshow(I_overlap.T, origin="lower")
print(f"Overlap: {np.sum(I_overlap)}")

# %% Function for minimalization to optimize for lens separation

def to_optimize_lens_separation(distance_variables, E, cbp, lambda0, waveguide_width, waveguide_thickness, focal1, focal2 ):
    output_cbp = propagate_two_lenses(cbp, lambda0, focal1, focal2, focal1, 
                                      distance_variables[0], 
                                      distance_variables[1])
    return np.abs(1 - wg_mode_overlap(E, output_cbp, lambda0, waveguide_width,
                                      waveguide_thickness))


# %%    
args = (E, q0, lambda0, waveguide_width, waveguide_thickness, f_1300, 11.21e-3)

res = minimize(to_optimize_lens_separation, [0.75, 11.255e-3], args=args, bounds=[(0.4, 2), (11e-3, 11.5e-3)], tol=1e-4)

# %% Find best couling starting with different distances between the lenses

test_distances = np.arange(20,120,10)*1e-2
results = np.zeros((test_distances.size, 3))

for i, distance in enumerate(test_distances):
    result = minimize(to_optimize_lens_separation, [distance, 11.255e-3], args=args, bounds=[(0.1, 2), (11e-3, 11.5e-3)], tol=1e-4)
    results[i,:2] = result.x
    results[i,2] = (1-result.fun)
    print(f"Iteration: {i}\tResult: {result.success}")
    
    
# %% Estimate tolerances 

delta = np.max(results[:,1])-np.min(results[:,1])
deltas = np.geomspace(delta*1e-3, 100*delta, 30)
z2 = np.linspace(20,120,200)*1e-2
z3 = np.concatenate([np.mean(results[:,1])-deltas[::-1], np.mean(results[:,1])+deltas])

Z2, Z3 = np.meshgrid(z2,z3, indexing='ij')
 
W = np.zeros(Z2.shape)

# %%

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


# %%

from sympy.physics.optics import *

p0 = BeamParameter( lambda0, 0, w=w0 )
fs = FreeSpace(f_1300)
fs2 = FreeSpace(1)
lens = ThinLens(f_1300)
p1 = fs*p0
p1.w.n()
p2 = lens*p1
p3 = fs2*p2



# %% Calculate lens shift models
# and plot the focal shifts with extrapolation for the C560 and A220 lenses


file_folder_path = os.path.dirname(script_path)
C560_filename = "C560_focal_shift.csv"
file_path = os.path.join(file_folder_path, C560_filename)
test_wavelengths = np.linspace(300e-9,2300e-9,1000)
c560_model = build_lens_model(file_path, 450e-9, 1070e-9)


with open(file_path, 'r') as file:
    data = np.loadtxt(file, delimiter=';')
    wavelengths = data[:,1]
    focal_shift = data[:,0]
    


title = "Lens shift with extrapolation"
px = 1/plt.rcParams['figure.dpi']  # pixel in inches
fig, ax1 = plt.subplots(figsize=(1000*px, 750*px))
ax1.plot(wavelengths*1e-9, focal_shift, '.',label="C560 Thorlabs data")
ax1.plot(test_wavelengths,
         extrapolated_focal_shift(test_wavelengths, c560_model),
         '--', label="C560 Extrapolation")    


A220_filename = "A220_focal_shift.csv"
file_path = os.path.join(file_folder_path, A220_filename)
a220_model = build_lens_model(file_path, 450e-9, 1070e-9)   
with open(file_path, 'r') as file:
    data = np.loadtxt(file, delimiter=';')
    wavelengths = data[:,1]
    focal_shift = data[:,0]
    
    
ax1.plot(wavelengths*1e-9, focal_shift, '.', label="A220 Thorlabs data")
ax1.plot(test_wavelengths,
         extrapolated_focal_shift(test_wavelengths, a220_model),
         '--', label="A220 Extrapolation")
ax1.set_title(title, fontsize=18)
ax1.legend(loc="lower right", fontsize=12)
ax1.set_xlabel("Wavelength, m", fontsize=14)
ax1.set_ylabel("Focal shift, mm", fontsize=14)
plt.show()
    
