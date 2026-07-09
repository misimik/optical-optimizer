"""
Waveguide mode solver for rectangular dielectric waveguides using Marcatili's
method with the effective index approximation.

Physics context:
    The guided modes of a rectangular dielectric waveguide are obtained by
    solving the transcendental eigenvalue equations in the x and y directions
    separately (Marcatili's method). The transverse wave numbers kappa_x and
    kappa_y satisfy:
        tan(kappa_x * d) = n1^2 * kappa_x * (n3^2*g2 + n2^2*g3) /
                           (n3^2*n2^2*kappa_x^2 - n1^4*g2*g3)
    for the x-direction (asymmetric: substrate n2, cladding n3), and
        tan(kappa_y * w) = 2 * kappa_y * g5 / (kappa_y^2 - g5^2)
    for the y-direction (symmetric: cladding n3 on both sides).
    The propagation constant is beta = sqrt(n1^2*k0^2 - kappa_x^2 - kappa_y^2).

    The field distribution for the E^x_{pq} mode (quasi-TE) is:
        E_x = (i/(kappa_x*beta)) * (n1^2*k0^2 - kappa_x^2) *
              sin(kappa_x*(x+xi)) * cos(kappa_y*(y+eta))
    where xi and eta are phase offsets determined by the boundary conditions.

References:
    - Marcatili, "Dielectric Rectangular Waveguide and Directional Coupler
      for Integrated Optics", Bell Syst. Tech. J. 48, 2071 (1969)
    - Suhara, "Waveguide Nonlinear-Optic Devices", Springer (2003)
"""

import numpy as np
from scipy.optimize import fsolve
from numba import njit


# ---------------------------------------------------------------------------
# Refractive index models
# ---------------------------------------------------------------------------

@njit
def n_air_ciddor(lambda_um):
    """
    Refractive index of standard air (Ciddor 1996).

    Physics: Standard air at 15 C, 101.325 kPa, 450 ppm CO2.

    Parameters
    ----------
    lambda_um : float
        Wavelength [um].

    Returns
    -------
    n : float
        Refractive index.
    """
    inv_l2 = lambda_um ** (-2)
    return 1.0 + 0.05792105 / (238.0185 - inv_l2) + 0.00167917 / (57.362 - inv_l2)


@njit
def n_ln_umemura(lambda_um, T=25.0):
    """
    Extraordinary refractive index of 5% MgO-doped congruent LiNbO3.

    Physics: Sellmeier equation with temperature correction. The extraordinary
    index is relevant for quasi-TM modes in x-cut or z-cut waveguides.

    Reference: Umemura et al., Appl. Opt. 53, 25 (2014)

    Parameters
    ----------
    lambda_um : float
        Wavelength [um].
    T : float
        Temperature [C].

    Returns
    -------
    n : float
        Extraordinary refractive index.
    """
    n_20 = np.sqrt(4.54514 + 0.096471 / (lambda_um ** 2 - 0.043763)
                   - 0.021502 * lambda_um ** 2)
    dT = T - 20.0
    Dn = (0.4175 / lambda_um ** 3 - 0.6643 / lambda_um ** 2
          + 0.9036 / lambda_um + 3.5332 - 0.0744 * lambda_um) \
         * (dT + 0.00138 * dT ** 2) * 1e-5
    return n_20 + Dn


@njit
def n_lt_dolev(lambda_um, T=25.0):
    """
    Extraordinary refractive index of 0.5% MgO-doped stoichiometric LiTaO3.

    Physics: SLT is used as the substrate material. Its lower refractive index
    compared to the LiNbO3 film enables waveguiding by total internal reflection.

    Reference: Dolev et al., Appl. Phys. B 96, 423 (2009)

    Parameters
    ----------
    lambda_um : float
        Wavelength [um].
    T : float
        Temperature [C].

    Returns
    -------
    n : float
        Extraordinary refractive index.
    """
    f = (T - 24.5) * (T + 570.82)
    return np.sqrt(
        4.5615 + 4.782e-7 * f
        + (0.08488 + 3.0913e-8 * f) / (lambda_um ** 2 - (0.1927 + 2.7326e-8 * f) ** 2)
        + (5.5832 + 1.4837e-5 * f) / (lambda_um ** 2 - (8.3067 + 1.3647e-7 * f) ** 2)
        - 0.021696 * lambda_um ** 2
    )


@njit
def n_ln_deng(lambda_um, T=25.0):
    """
    Extraordinary refractive index of congruent LiNbO3 (Deng 2006).

    Parameters
    ----------
    lambda_um : float
        Wavelength [um].
    T : float
        Temperature [C].

    Returns
    -------
    n : float
        Extraordinary refractive index.
    """
    f = (T - 24.5) * (T + 570.82)
    a1, a2, a3, a4, a5, a6 = 5.39121, 0.100473, 0.20692, 100.0, 11.34927, 1.544e-2
    b1, b2, b3, b4, b5 = 4.96827e-7, 3.862e-8, -0.89e-8, 2.657e-5, 9.62119e-10
    return np.sqrt(
        a1 + b1 * f + (a2 + b2 * f) / (lambda_um ** 2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (lambda_um ** 2 - a5 ** 2) - (a6 + b5 * f) * lambda_um ** 2
    )


# ---------------------------------------------------------------------------
# Marcatili eigenvalue equations
# ---------------------------------------------------------------------------

@njit
def _kappa_x_eq(kappa, k, n1, n2, n3, d):
    """
    Transcendental equation for kappa_x (asymmetric slab, x-direction).

    Physics: This is the eigenvalue equation for TE modes in an asymmetric
    dielectric slab of thickness d (the waveguide thickness), sandwiched
    between substrate (n2, LiTaO3) and cladding (n3, air).

    Parameters
    ----------
    kappa : float
        Transverse wave number in x-direction [rad/m].
    k : float
        Free-space wave number k0 = 2*pi/lambda0 [rad/m].
    n1 : float
        Core refractive index (LiNbO3 film).
    n2 : float
        Substrate refractive index (LiTaO3).
    n3 : float
        Cladding refractive index (air).
    d : float
        Waveguide thickness [m].

    Returns
    -------
    val : float
        Value of the equation (zero at solution).
    """
    g2 = np.sqrt((n1 ** 2 - n2 ** 2) * k ** 2 - kappa ** 2)
    g3 = np.sqrt((n1 ** 2 - n3 ** 2) * k ** 2 - kappa ** 2)
    tan_kd = np.tan(kappa * d)
    numerator = n1 ** 2 * kappa * (n3 ** 2 * g2 + n2 ** 2 * g3)
    denominator = n3 ** 2 * n2 ** 2 * kappa ** 2 - n1 ** 4 * g2 * g3
    return tan_kd - numerator / denominator


@njit
def _kappa_y_eq(kappa, k, n1, n3, w):
    """
    Transcendental equation for kappa_y (symmetric slab, y-direction).

    Physics: Eigenvalue equation for TM-like modes in a symmetric slab
    of width w. Both sides have the same cladding index (n3).

    Parameters
    ----------
    kappa : float
        Transverse wave number in y-direction [rad/m].
    k : float
        Free-space wave number [rad/m].
    n1 : float
        Core refractive index.
    n3 : float
        Cladding refractive index.
    w : float
        Waveguide width [m].

    Returns
    -------
    val : float
        Value of the equation (zero at solution).
    """
    g5 = np.sqrt((n1 ** 2 - n3 ** 2) * k ** 2 - kappa ** 2)
    tan_kw = np.tan(kappa * w)
    return tan_kw - 2.0 * kappa * g5 / (kappa ** 2 - g5 ** 2)


def solve_kappa(k, n1, n2, n3, d, direction='x'):
    """
    Solve for the transverse wave number kappa of the fundamental mode.

    The initial guess is chosen near cutoff: kappa ~ pi/d for the fundamental
    mode, which lies between 0 and the cutoff value.

    Parameters
    ----------
    k : float
        Free-space wave number [rad/m].
    n1, n2, n3 : float
        Refractive indices (core, substrate, cladding).
    d : float
        Waveguide dimension in the relevant direction [m].
    direction : str
        'x' for asymmetric (thickness), 'y' for symmetric (width).

    Returns
    -------
    kappa : float
        Transverse wave number [rad/m].
    """
    kappa0 = 2.0 * np.pi / (d * n1)

    # The kappa must lie in [0, k*sqrt(n1^2-n2^2)] for x or [0, k*sqrt(n1^2-n3^2)] for y
    if direction == 'x':
        kappa_max = k * np.sqrt(n1 ** 2 - max(n2, n3) ** 2)
        eq = lambda kap: _kappa_x_eq(kap, k, n1, n2, n3, d)
    else:
        kappa_max = k * np.sqrt(n1 ** 2 - n3 ** 2)
        eq = lambda kap: _kappa_y_eq(kap, k, n1, n3, d)

    # Try multiple initial guesses to find the fundamental mode
    for guess in [kappa0, kappa0 * 0.5, kappa0 * 0.3, kappa_max * 0.9]:
        try:
            result, infodict, ier, msg = fsolve(
                eq, guess, full_output=True, xtol=1e-12, maxfev=1000
            )
            if ier == 1 and 0 < result[0] < kappa_max:
                return float(result[0])
        except Exception:
            continue

    raise RuntimeError(
        f"Failed to solve kappa_{direction} for lambda={2*np.pi/k:.3e}m, "
        f"n1={n1:.4f}, n2={n2:.4f}, n3={n3:.4f}, d={d:.3e}m"
    )


# ---------------------------------------------------------------------------
# Field calculation
# ---------------------------------------------------------------------------

def waveguide_mode_field(lambda0, thickness, width, temperature=25.0, dL=0.1e-6):
    """
    Calculate the normalized E-field distribution of the fundamental guided
    mode in a rectangular LiNbO3-on-LiTaO3 waveguide.

    Physics: The mode is the E^x_{11} mode (lowest-order quasi-TE).
    The field is normalized so that the total Poynting flux in the
    cross-section equals 1 W. The spatial grid extends beyond the waveguide
    core into the substrate and cladding to capture evanescent tails.

    Parameters
    ----------
    lambda0 : float
        Vacuum wavelength [m].
    thickness : float
        Waveguide thickness (x-dimension) [m].
    width : float
        Waveguide width (y-dimension) [m].
    temperature : float
        Temperature [C].
    dL : float
        Spatial resolution [m].

    Returns
    -------
    E : ndarray, shape (Nx, Ny), complex
        Normalized electric field distribution.
    """
    lambda_um = lambda0 * 1e6
    n_core = n_ln_umemura(lambda_um, temperature)
    n_substrate = n_lt_dolev(lambda_um, temperature)
    n_clad = n_air_ciddor(lambda_um)

    k = 2.0 * np.pi / lambda0

    kax = solve_kappa(k, n_core, n_substrate, n_clad, thickness, 'x')
    kay = solve_kappa(k, n_core, n_clad, n_clad, width, 'y')

    g3 = np.sqrt((n_core ** 2 - n_clad ** 2) * k ** 2 - kax ** 2)
    g5 = np.sqrt((n_core ** 2 - n_clad ** 2) * k ** 2 - kay ** 2)

    # Phase offsets from boundary conditions
    xi = (1.0 / kax) * np.arctan(-(n_clad / n_core) ** 2 * kax / g3)
    eta = (1.0 / kay) * np.arctan(-g5 / kay)

    # Propagation constant
    beta = np.sqrt(n_core ** 2 * k ** 2 - kax ** 2 - kay ** 2)

    # Spatial grid
    x = np.arange(-thickness, 0 + dL / 2, dL)
    y = np.arange(0, width + dL / 2, dL)
    X, Y = np.meshgrid(x, y, indexing='ij')
    dA = dL ** 2

    # Field components (Marcatili formulation)
    Ex = (1j / (kax * beta)) * (n_core ** 2 * k ** 2 - kax ** 2) \
         * np.sin(kax * (X + xi)) * np.cos(kay * (Y + eta))

    # Impedance of free space: sqrt(mu0/eps0)
    f = np.sqrt(4e-7 * np.pi / 8.854187817e-12)
    Hy = 1j * f * n_core ** 2 * (k / kax) \
         * np.sin(kax * (X + xi)) * np.cos(kay * (Y + eta))

    # Poynting vector (z-component, time-averaged)
    Sz = np.real(Ex * np.conj(Hy)) / 2.0

    # Normalize to unit total power
    total_power = np.sum(Sz) * dA
    Anorm = 1.0 / np.sqrt(total_power)

    Ex *= Anorm

    return Ex


def waveguide_mode_size(E, dL, waveguide_width, waveguide_thickness):
    """
    Estimate the 1/e^2 intensity width and height of the waveguide mode.

    Parameters
    ----------
    E : ndarray, shape (Nx, Ny), complex
        Mode field distribution.
    dL : float
        Spatial resolution [m].
    waveguide_width : float
        Waveguide width [m].
    waveguide_thickness : float
        Waveguide thickness [m].

    Returns
    -------
    w_x : float
        1/e^2 half-width in x (thickness direction) [m].
    w_y : float
        1/e^2 half-width in y (width direction) [m].
    """
    I = np.abs(E) ** 2
    # Normalize
    I = I / np.sum(I)

    # Find the intensity peak (center of the mode)
    max_idx = np.unravel_index(np.argmax(I), I.shape)

    x = np.arange(0, waveguide_width, dL)
    y = np.arange(0, waveguide_thickness, dL)

    Ix = I[:, max_idx[1]] / np.max(I[:, max_idx[1]])
    Iy = I[max_idx[0], :] / np.max(I[max_idx[0], :])

    # Find 1/e^2 crossings
    cross_x = np.where(np.diff(np.sign(Ix - 1.0 / np.exp(2))))[0]
    cross_y = np.where(np.diff(np.sign(Iy - 1.0 / np.exp(2))))[0]

    if len(cross_x) >= 2:
        w_x = np.abs(x[cross_x[-1]] - x[cross_x[0]])
    else:
        w_x = waveguide_width / 2.0

    if len(cross_y) >= 2:
        w_y = np.abs(y[cross_y[-1]] - y[cross_y[0]])
    else:
        w_y = waveguide_thickness / 2.0

    return w_x, w_y
