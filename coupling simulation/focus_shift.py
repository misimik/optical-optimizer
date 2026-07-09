# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:59:23 2023

@author: Michal Mikolajczyk (michal@mikolajczyk.link)
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate

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