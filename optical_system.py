"""
Core optical system simulation using ABCD matrix formalism for Gaussian beam
propagation through sequences of optical elements.

Physics context:
    Gaussian beams are the fundamental solution to the paraxial Helmholtz
    equation. Their evolution through an optical system is fully described by
    the complex beam parameter q(z) = z + i*z_R (referenced to the waist),
    where the real part encodes wavefront curvature and the imaginary part
    encodes beam radius. The ABCD law (also called the Kogelnik transformation)
    states that q_out = (A*q_in + B) / (C*q_in + D) for a system with transfer
    matrix [[A,B],[C,D]].

References:
    - Kogelnik & Li, "Laser Beams and Resonators", Appl. Opt. 5, 1550 (1966)
    - Siegman, "Lasers", University Science Books (1986)
"""

import numpy as np


# ---------------------------------------------------------------------------
# ABCD transfer matrices
# ---------------------------------------------------------------------------

def propagation_matrix(d):
    """
    Transfer matrix for free-space propagation over distance d.

    Physics: In the ray-optics picture, a ray at height y with angle theta
    becomes y' = y + d*theta, theta' = theta after distance d.

    Parameters
    ----------
    d : float
        Propagation distance [m].

    Returns
    -------
    M : ndarray, shape (2, 2)
        [[1, d], [0, 1]]
    """
    return np.array([[1.0, d], [0.0, 1.0]])


def thin_lens_matrix(f):
    """
    Transfer matrix for a thin lens of focal length f.

    Physics: A thin lens changes ray angle by -y/f without changing height.
    For Gaussian beams, this adds curvature 1/f to the wavefront.

    Parameters
    ----------
    f : float
        Focal length [m]. Positive for converging, negative for diverging.

    Returns
    -------
    M : ndarray, shape (2, 2)
        [[1, 0], [-1/f, 1]]
    """
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])


def flat_interface_matrix(n_in, n_out):
    """
    Transfer matrix for refraction at a flat dielectric interface.

    Physics: Snell's law in the paraxial approximation gives theta' = (n1/n2)*theta.
    Ray height is unchanged. The ABCD form handles the change in effective
    propagation angle for Gaussian beams.

    Parameters
    ----------
    n_in : float
        Refractive index of input medium.
    n_out : float
        Refractive index of output medium.

    Returns
    -------
    M : ndarray, shape (2, 2)
        [[1, 0], [0, n_in/n_out]]
    """
    return np.array([[1.0, 0.0], [0.0, n_in / n_out]])


def thick_lens_matrix(n_env, n_glass, R1, R2, t):
    """
    Transfer matrix for a thick lens with spherical surfaces.

    Physics: A thick lens has two refracting surfaces separated by glass of
    thickness t. Each surface contributes a refraction matrix and the glass
    contributes a propagation. The sign convention: R > 0 for a surface whose
    center of curvature lies to the right of the vertex.

    Parameters
    ----------
    n_env : float
        Refractive index of surrounding medium (usually air = 1).
    n_glass : float
        Refractive index of lens glass.
    R1 : float
        Radius of curvature of first surface [m].
    R2 : float
        Radius of curvature of second surface [m].
    t : float
        Center thickness of lens [m].

    Returns
    -------
    M : ndarray, shape (2, 2)
    """
    M_r2 = np.array([[1.0, 0.0], [(n_glass - n_env) / (R2 * n_env), n_glass / n_env]])
    M_prop = propagation_matrix(t)
    M_r1 = np.array([[1.0, 0.0], [(n_env - n_glass) / (R1 * n_glass), n_env / n_glass]])
    return M_r2 @ M_prop @ M_r1


# ---------------------------------------------------------------------------
# Complex beam parameter (q-parameter) operations
# ---------------------------------------------------------------------------

def q_from_waist(w0, lambda0, n=1.0, z=0.0):
    """
    Calculate complex beam parameter given waist radius.

    Physics: At distance z from the waist, the complex beam parameter is
    q(z) = z + i*z_R where z_R = pi*n*w0^2/lambda0 is the Rayleigh range.
    At the waist (z=0), q = i*z_R (purely imaginary = planar wavefront).

    Parameters
    ----------
    w0 : float
        Beam waist radius (1/e^2 intensity) [m].
    lambda0 : float
        Vacuum wavelength [m].
    n : float
        Refractive index of medium.
    z : float
        Distance from the waist [m].

    Returns
    -------
    q : complex
        Complex beam parameter.
    """
    z_R = np.pi * n * w0 ** 2 / lambda0
    return z + 1j * z_R


def q_from_curvature(R, w, lambda0, n=1.0):
    """
    Calculate complex beam parameter from wavefront curvature and beam radius.

    Physics: The q-parameter at any position satisfies
    1/q = 1/R - i*lambda0/(pi*n*w^2).

    Parameters
    ----------
    R : float
        Wavefront radius of curvature [m]. inf for planar wavefront.
    w : float
        Beam radius (1/e^2) [m].
    lambda0 : float
        Vacuum wavelength [m].
    n : float
        Refractive index of medium.

    Returns
    -------
    q : complex
        Complex beam parameter.
    """
    return 1.0 / (1.0 / R - 1j * lambda0 / (np.pi * n * w ** 2))


def q_apply_matrix(M, q):
    """
    Apply an ABCD matrix to a complex beam parameter (Kogelnik's ABCD law).

    Physics: q_out = (A*q_in + B) / (C*q_in + D)

    Parameters
    ----------
    M : ndarray, shape (2, 2)
        ABCD transfer matrix.
    q : complex
        Input complex beam parameter.

    Returns
    -------
    q_out : complex
        Transformed complex beam parameter.
    """
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    return (A * q + B) / (C * q + D)


def q_to_beam_radius(q, lambda0, n=1.0):
    """
    Extract beam radius at the position defined by q.

    Physics: The imaginary part of 1/q gives w: 1/q = 1/R - i*lambda0/(pi*n*w^2),
    so w = sqrt(lambda0 / (pi * n * Im(1/q))).

    Parameters
    ----------
    q : complex
        Complex beam parameter.
    lambda0 : float
        Vacuum wavelength [m].
    n : float
        Refractive index of medium.

    Returns
    -------
    w : float
        Beam radius (1/e^2 intensity) [m].
    """
    return np.sqrt(lambda0 / (np.pi * n * abs(np.imag(1.0 / q))))


def q_to_waist_radius(q, lambda0, n=1.0):
    """
    Extract beam waist radius from the q-parameter.

    Physics: w0 = sqrt(lambda0 * z_R / pi), where z_R = Im(q) (since q = z + i*z_R
    when referenced to the waist position).

    Parameters
    ----------
    q : complex
        Complex beam parameter.
    lambda0 : float
        Vacuum wavelength [m].
    n : float
        Refractive index of medium.

    Returns
    -------
    w0 : float
        Beam waist radius [m].
    """
    z_R = np.imag(q)
    return np.sqrt(lambda0 * z_R / (np.pi * n))


def q_to_curvature_radius(q):
    """
    Extract wavefront radius of curvature from the q-parameter.

    Physics: R = 1 / Re(1/q). R = inf means planar wavefront (at the waist).

    Parameters
    ----------
    q : complex
        Complex beam parameter.

    Returns
    -------
    R : float
        Wavefront radius of curvature [m].
    """
    real_part = np.real(1.0 / q)
    if abs(real_part) < 1e-15:
        return np.inf
    return 1.0 / real_part


def q_from_waist_and_distance(w0, z, lambda0, n=1.0):
    """
    Beam radius at distance z from a waist of size w0.

    Physics: w(z) = w0 * sqrt(1 + (z/z_R)^2) where z_R = pi*n*w0^2/lambda0.

    Parameters
    ----------
    w0 : float
        Beam waist radius [m].
    z : float
        Distance from waist [m].
    lambda0 : float
        Vacuum wavelength [m].
    n : float
        Refractive index of medium.

    Returns
    -------
    w : float
        Beam radius at position z [m].
    """
    z_R = np.pi * n * w0 ** 2 / lambda0
    return w0 * np.sqrt(1.0 + (z / z_R) ** 2)


# ---------------------------------------------------------------------------
# Gaussian beam field distribution
# ---------------------------------------------------------------------------

def gaussian_field_amplitude(r, w_z, w0, z, lambda0, n=1.0):
    """
    Complex E-field amplitude of a fundamental Gaussian beam.

    Physics: The TEM00 Gaussian beam field is:
    E(r,z) = (w0/w(z)) * exp(-r^2/w(z)^2) * exp(-i(k*z + k*r^2/(2R) - phi_gouy))
    where phi_gouy = atan(z/z_R) is the Gouy phase shift.

    Parameters
    ----------
    r : ndarray
        Radial distance from beam axis [m].
    w_z : float
        Beam radius at position z [m].
    w0 : float
        Beam waist radius [m].
    z : float
        Distance from waist [m].
    lambda0 : float
        Vacuum wavelength [m].
    n : float
        Refractive index of medium.

    Returns
    -------
    E : ndarray, complex
        Electric field amplitude (unnormalized).
    """
    z_R = np.pi * n * w0 ** 2 / lambda0
    R = z * (1.0 + (z_R / z) ** 2) if abs(z) > 1e-20 else np.inf
    phi = np.arctan(z / z_R)
    k = 2.0 * np.pi * n / lambda0

    amplitude = w0 / w_z * np.exp(-r ** 2 / w_z ** 2)
    if np.isinf(R):
        phase = -1j * (k * z - phi)
    else:
        phase = -1j * (k * (z + r ** 2 / (2.0 * R)) - phi)

    return amplitude * np.exp(phase)


def gaussian_field_from_q(r, q, lambda0, n=1.0):
    """
    Compute E-field from complex beam parameter and radial coordinate.

    Convenience wrapper that extracts w(z), w0, and z from the q-parameter.

    Parameters
    ----------
    r : ndarray
        Radial distance from beam axis [m].
    q : complex
        Complex beam parameter.
    lambda0 : float
        Vacuum wavelength [m].
    n : float
        Refractive index of medium.

    Returns
    -------
    E : ndarray, complex
        Electric field amplitude (unnormalized).
    """
    w_z = q_to_beam_radius(q, lambda0, n)
    w0 = q_to_waist_radius(q, lambda0, n)
    z = np.real(q)
    return gaussian_field_amplitude(r, w_z, w0, z, lambda0, n)


# ---------------------------------------------------------------------------
# Element types and system propagation
# ---------------------------------------------------------------------------

class OpticalElement:
    """
    Base class for any optical element in the system.

    Each element has a position along the optical axis and can produce its
    ABCD matrix. The matrix may depend on wavelength (e.g., achromatic lenses
    have focal shift).
    """

    def __init__(self, position, label=""):
        self.position = position
        self.label = label

    def matrix(self, lambda0):
        """Return the ABCD matrix for this element at given wavelength."""
        raise NotImplementedError


class FreeSpace(OpticalElement):
    """Free-space propagation segment (no optical surface, just distance)."""

    def __init__(self, length, label=""):
        super().__init__(0, label)
        self.length = length

    def matrix(self, lambda0):
        return propagation_matrix(self.length)


class ThinLens(OpticalElement):
    """Thin lens with optional wavelength-dependent focal shift."""

    def __init__(self, focal_length, position=0, focal_shift_fn=None, label=""):
        super().__init__(position, label)
        self.focal_length = focal_length
        self.focal_shift_fn = focal_shift_fn

    def matrix(self, lambda0):
        f = self.focal_length
        if self.focal_shift_fn is not None:
            f += self.focal_shift_fn(lambda0)
        return thin_lens_matrix(f)


class ThickLens(OpticalElement):
    """Thick lens model with two spherical surfaces."""

    def __init__(self, n_glass_fn, R1, R2, t, position=0, label=""):
        super().__init__(position, label)
        self.n_glass_fn = n_glass_fn
        self.R1 = R1
        self.R2 = R2
        self.t = t

    def matrix(self, lambda0):
        n_glass = self.n_glass_fn(lambda0)
        return thick_lens_matrix(1.0, n_glass, self.R1, self.R2, self.t)


class FlatInterface(OpticalElement):
    """Refraction at a flat interface between two media."""

    def __init__(self, n_in_fn, n_out_fn, position=0, label=""):
        super().__init__(position, label)
        self.n_in_fn = n_in_fn
        self.n_out_fn = n_out_fn

    def matrix(self, lambda0):
        return flat_interface_matrix(self.n_in_fn(lambda0), self.n_out_fn(lambda0))


class OpticalSystem:
    """
    A complete optical system: a sequence of elements (propagation + optics)
    that transforms a Gaussian beam from input to output.

    The system is defined as an ordered list of OpticalElement objects.
    Propagation between elements is handled automatically based on their
    positions, but explicit FreeSpace elements can also be inserted.

    Usage:
        system = OpticalSystem()
        system.add(FreeSpace(d1), "collimation")
        system.add(ThinLens(f), "lens_1")
        system.add(FreeSpace(d2), "to_wg")
        q_out = system.propagate(q_in, lambda0)
    """

    def __init__(self, label=""):
        self.elements = []
        self.label = label

    def add(self, element):
        """Append an element to the system."""
        self.elements.append(element)

    def add_propagation(self, d, label=""):
        """Convenience: add a free-space propagation segment."""
        self.add(FreeSpace(d, label))

    def add_thin_lens(self, f, focal_shift_fn=None, label=""):
        """Convenience: add a thin lens."""
        self.add(ThinLens(f, focal_shift_fn=focal_shift_fn, label=label))

    def combined_matrix(self, lambda0):
        """
        Compute the total ABCD matrix by multiplying all element matrices
        in order (from first to last).

        Physics: The matrices multiply from right to left because the beam
        enters the first element first. So M_total = M_N * ... * M_2 * M_1.

        Parameters
        ----------
        lambda0 : float
            Vacuum wavelength [m].

        Returns
        -------
        M : ndarray, shape (2, 2)
            Total transfer matrix.
        """
        if not self.elements:
            return np.eye(2)
        M = self.elements[0].matrix(lambda0)
        for elem in self.elements[1:]:
            M = elem.matrix(lambda0) @ M
        return M

    def propagate(self, q_in, lambda0):
        """
        Propagate a complex beam parameter through the entire system.

        Parameters
        ----------
        q_in : complex
            Input complex beam parameter.
        lambda0 : float
            Vacuum wavelength [m].

        Returns
        -------
        q_out : complex
            Complex beam parameter at system output.
        """
        M = self.combined_matrix(lambda0)
        return q_apply_matrix(M, q_in)

    def trace_beam(self, q_in, lambda0, n_points=1000):
        """
        Trace the beam radius along the optical axis through all elements.

        Useful for generating beam-width-vs-position plots.

        Physics: At each intermediate plane we compute the accumulated ABCD
        matrix and extract w(z) from the q-parameter.

        Parameters
        ----------
        q_in : complex
            Input complex beam parameter.
        lambda0 : float
            Vacuum wavelength [m].
        n_points : int
            Number of sample points along the axis.

        Returns
        -------
        z : ndarray
            Positions along the optical axis [m].
        w : ndarray
            Beam radii at each position [m].
        element_positions : list of dict
            Positions and labels of optical elements for annotation.
        """
        total_length = sum(
            elem.length if isinstance(elem, FreeSpace) else 0.0
            for elem in self.elements
        )
        z = np.linspace(0, total_length, n_points)

        w = np.zeros(n_points)
        q = q_in
        cumulative_z = 0.0
        elem_idx = 0

        for i, zi in enumerate(z):
            if elem_idx < len(self.elements) and zi >= cumulative_z + getattr(
                    self.elements[elem_idx], 'length',
                    self._element_position_estimate(elem_idx)):
                q = q_apply_matrix(self.elements[elem_idx].matrix(lambda0), q)
                cumulative_z += self._element_length(elem_idx)
                elem_idx += 1
            w[i] = q_to_beam_radius(q, lambda0)

        element_positions = self._collect_element_positions()

        return z, w, element_positions

    def _element_length(self, idx):
        """Return the axial extent of element idx."""
        elem = self.elements[idx]
        if isinstance(elem, FreeSpace):
            return elem.length
        return 0.0

    def _element_position_estimate(self, idx):
        """Estimate the position along the axis of element idx (for beam trace)."""
        # Simple: distance from previous FreeSpace elements
        pos = 0.0
        for i in range(idx):
            pos += self._element_length(i)
        return pos

    def _collect_element_positions(self):
        """Collect element positions and labels for plotting."""
        positions = []
        cumulative = 0.0
        for elem in self.elements:
            if isinstance(elem, FreeSpace):
                cumulative += elem.length
            else:
                positions.append({
                    'z': cumulative,
                    'label': elem.label,
                    'type': type(elem).__name__
                })
        return positions


# ---------------------------------------------------------------------------
# Convenience: system built from distances and lenses
# ---------------------------------------------------------------------------

def build_system_from_segments(segments):
    """
    Build an OpticalSystem from a list of segment descriptors.

    Each segment is a dict:
        {'type': 'propagation', 'distance': d, 'label': ...}
        {'type': 'thin_lens', 'focal_length': f, 'focal_shift_fn': fn, 'label': ...}
        {'type': 'flat_interface', 'n_in_fn': fn, 'n_out_fn': fn, 'label': ...}

    Parameters
    ----------
    segments : list of dict
        Ordered list of optical segments.

    Returns
    -------
    system : OpticalSystem
    """
    system = OpticalSystem()
    for seg in segments:
        if seg['type'] == 'propagation':
            system.add_propagation(seg['distance'], seg.get('label', ''))
        elif seg['type'] == 'thin_lens':
            system.add_thin_lens(
                seg['focal_length'],
                focal_shift_fn=seg.get('focal_shift_fn'),
                label=seg.get('label', '')
            )
        elif seg['type'] == 'flat_interface':
            system.add(FlatInterface(
                seg['n_in_fn'], seg['n_out_fn'],
                label=seg.get('label', '')
            ))
        elif seg['type'] == 'thick_lens':
            system.add(ThickLens(
                seg['n_glass_fn'], seg['R1'], seg['R2'], seg['t'],
                label=seg.get('label', '')
            ))
    return system
