# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 18:19:34 2023

@author: Michal Mikolajczyk (michal@mikolajczyk.link)
"""
import numpy as np
import math


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


def flat_interface_M(n_in, n_out):
    """
    Trasfer matrix of refraction at flat interface

    Parameters
    ----------
    n_in : float
        Input material refractive index.
    n_out : float
        Output material refractive index.

    Returns
    -------
    numpy.array
        Transfer matrix.

    """    
    return np.array([[1, 0], [0, n_in/n_out]])


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
    w0 = cbp_to_w0(cbp, lambda_)
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

def propagate_three_lenses(cbp, lambda0, f_1, f_2, f_3, z1, z2, z3, z4):
    """
    Function propagating Complex Beam Parameter through three lenses padded at
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
    f_3 : float
        Third lens focal length.
    z1 : float
        Distance from the input to the first lens.
    z2 : float
        Distance from the first lens to the second.
    z3 : float
        Distance from the second lens to the third.
    z4 : float
        Distance from the third lens to the output.

    Returns
    -------
    q7 : complex
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

    # CBP at the third lens
    q5 = m_multiply(propagation_M(z3), q4)
    
    # CBP after the third lens
    q6 = m_multiply(thin_lens_M(f_3), q5)
    
    # CBP at the output
    q7 = m_multiply(propagation_M(z4), q6)
    
    return q7