"""
Coupling efficiency calculation via mode overlap integrals.

Physics context:
    The power coupled from a free-space Gaussian beam into a waveguide mode
    is given by the normalized overlap integral:
        eta = |∫∫ E_beam * E_wg* dA|^2 / (∫∫ |E_beam|^2 dA * ∫∫ |E_wg|^2 dA)

    This is derived from the continuity of tangential E-field components at
    the waveguide facet. Maximum coupling is achieved when the beam matches
    the mode in both size and wavefront curvature (mode matching).

    For Gaussian beams, eta can exceed 99% when the beam waist size matches
    the mode field diameter and is positioned at the waveguide facet.

References:
    - Hunsperger, "Integrated Optics: Theory and Technology", Springer (2009)
    - Joyce & DeLoach, "Alignment of Gaussian beams", Appl. Opt. 23, 4187 (1984)
"""

import numpy as np
from optical_system import gaussian_field_from_q, q_to_beam_radius


def mode_overlap_integral(E_waveguide, q_beam, lambda0,
                          waveguide_width, waveguide_thickness, dL):
    """
    Calculate the power coupling efficiency between a Gaussian beam and a
    waveguide mode using the overlap integral.

    Both fields are normalized to unit power in the cross-section before
    computing the overlap.

    Parameters
    ----------
    E_waveguide : ndarray, shape (Nx, Ny), complex
        Normalized waveguide mode E-field.
    q_beam : complex
        Complex beam parameter of the incident Gaussian beam at the
        waveguide facet.
    lambda0 : float
        Vacuum wavelength [m].
    waveguide_width : float
        Waveguide width (y-dimension) [m].
    waveguide_thickness : float
        Waveguide thickness (x-dimension) [m].
    dL : float
        Spatial resolution of the waveguide field grid [m].

    Returns
    -------
    eta : float
        Coupling efficiency (0 to 1).
    """
    Nx, Ny = E_waveguide.shape

    # Normalize waveguide mode
    I_wg = np.abs(E_waveguide) ** 2
    norm_wg = np.sqrt(np.sum(I_wg))
    E_wg_norm = E_waveguide / norm_wg

    # Find mode center from intensity peak
    i_max = np.unravel_index(np.argmax(I_wg), E_waveguide.shape)

    # Create spatial grid centered on the mode peak
    x = np.linspace(0, waveguide_width, Nx)
    y = np.linspace(0, waveguide_thickness, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    x_center = X[i_max[0], 0]
    y_center = Y[0, i_max[1]]
    R = np.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2)

    # Compute incident Gaussian beam field at the facet
    E_beam = gaussian_field_from_q(R, q_beam, lambda0, n=1.0)

    # Normalize beam
    I_beam = np.abs(E_beam) ** 2
    norm_beam = np.sqrt(np.sum(I_beam))
    E_beam_norm = E_beam / norm_beam

    # Overlap integral (absolute value accounts for phase differences)
    overlap = np.abs(np.sum(E_beam_norm * np.conj(E_wg_norm)))

    return float(overlap)


def beam_width_at_facet(q_beam, lambda0):
    """
    Convenience: get the 1/e^2 beam radius at the waveguide facet.

    Parameters
    ----------
    q_beam : complex
        Complex beam parameter.
    lambda0 : float
        Vacuum wavelength [m].

    Returns
    -------
    w : float
        Beam radius [m].
    """
    return q_to_beam_radius(q_beam, lambda0)


def estimate_coupling_from_width(w_beam, w_mode_x, w_mode_y):
    """
    Quick estimate of maximum coupling efficiency from beam and mode sizes.

    Physics: For a circular Gaussian beam incident on an elliptical mode,
    the overlap integral reduces to a product of 1D overlaps. For matched
    beams (w_beam = w_mode), the efficiency is:
        eta = 4 * w_beam^2 * w_mode_x * w_mode_y /
              ((w_beam^2 + w_mode_x^2) * (w_beam^2 + w_mode_y^2))

    This is exact when there's no wavefront curvature mismatch.

    Parameters
    ----------
    w_beam : float
        Gaussian beam 1/e^2 radius [m].
    w_mode_x : float
        Waveguide mode 1/e^2 half-width in x [m].
    w_mode_y : float
        Waveguide mode 1/e^2 half-width in y [m].

    Returns
    -------
    eta : float
        Estimated coupling efficiency (0 to 1).
    """
    eta_x = 2.0 * w_beam * w_mode_x / (w_beam ** 2 + w_mode_x ** 2)
    eta_y = 2.0 * w_beam * w_mode_y / (w_beam ** 2 + w_mode_y ** 2)
    return eta_x * eta_y
