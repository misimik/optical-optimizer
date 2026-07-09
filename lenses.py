"""
Lens focal shift models for achromatic doublet and aspheric lenses.

Physics context:
    Refractive lenses exhibit chromatic focal shift: the focal length changes
    with wavelength due to material dispersion. Thorlabs provides measured
    focal shift data for their catalog lenses. We model this using cubic
    spline interpolation within the measured range and linear/cubic
    extrapolation beyond it.

    The focal shift is added to the nominal focal length:
        f(lambda) = f_nominal + delta_f(lambda)

    where the nominal focal length is specified at the design wavelength
    (typically 633 nm for visible lenses).

Usage:
    Provide CSV files with columns: focal_shift_mm, wavelength_nm
"""

import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


class ThorlabsLensModel:
    """
    Focal shift model for a Thorlabs lens based on measured data.

    Uses linear interpolation within the data range and power-law
    extrapolation outside it, fit to the boundary regions of the
    measured data.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with columns: focal_shift_mm, wavelength_nm.
    nominal_focal_length : float
        Nominal focal length at design wavelength [m].
    lower_limit_nm : float
        Below this wavelength, extrapolation is used instead of interpolation.
    upper_limit_nm : float
        Above this wavelength, extrapolation is used.
    """

    def __init__(self, csv_path, nominal_focal_length,
                 lower_limit_nm=450.0, upper_limit_nm=1070.0):
        self.nominal_f = nominal_focal_length

        data = np.loadtxt(csv_path, delimiter=';')
        self.shifts = data[:, 0]          # mm
        self.wavelengths = data[:, 1]     # nm

        self.lower_limit = lower_limit_nm
        self.upper_limit = upper_limit_nm

        # Fit extrapolation for long wavelengths (linear)
        mask_long = self.wavelengths > upper_limit_nm
        if np.sum(mask_long) >= 2:
            popt_long, _ = curve_fit(
                lambda x, a, b: a * x + b,
                self.wavelengths[mask_long],
                self.shifts[mask_long]
            )
            self.long_slope, self.long_intercept = popt_long
        else:
            self.long_slope, self.long_intercept = 0.0, 0.0

        # Fit extrapolation for short wavelengths (quadratic)
        mask_short = self.wavelengths < lower_limit_nm
        if np.sum(mask_short) >= 3:
            popt_short, _ = curve_fit(
                lambda x, a, b, c: a * x ** 2 + b * x + c,
                self.wavelengths[mask_short],
                self.shifts[mask_short]
            )
            self.short_a, self.short_b, self.short_c = popt_short
        else:
            self.short_a, self.short_b, self.short_c = 0.0, 0.0, 0.0

        # Interpolator for the measured range
        self._interp = interp1d(
            self.wavelengths, self.shifts,
            kind='cubic', bounds_error=False, fill_value='extrapolate'
        )

    def focal_shift_mm(self, wavelength_nm):
        """
        Get focal shift at a given wavelength in mm.

        Parameters
        ----------
        wavelength_nm : float or ndarray
            Wavelength [nm].

        Returns
        -------
        shift : float or ndarray
            Focal shift [mm].
        """
        wl = np.asarray(wavelength_nm, dtype=float)
        scalar = wl.ndim == 0
        if scalar:
            wl = wl.reshape([1])

        result = np.empty_like(wl)

        mask_int = (wl >= self.lower_limit) & (wl <= self.upper_limit)
        mask_long = wl > self.upper_limit
        mask_short = wl < self.lower_limit

        result[mask_int] = self._interp(wl[mask_int])
        result[mask_long] = self.long_slope * wl[mask_long] + self.long_intercept
        result[mask_short] = (self.short_a * wl[mask_short] ** 2
                              + self.short_b * wl[mask_short]
                              + self.short_c)

        return float(result[0]) if scalar else result

    def focal_length(self, lambda0):
        """
        Get effective focal length at a given vacuum wavelength.

        Parameters
        ----------
        lambda0 : float
            Vacuum wavelength [m].

        Returns
        -------
        f : float
            Effective focal length [m].
        """
        wl_nm = lambda0 * 1e9
        shift_mm = self.focal_shift_mm(wl_nm)
        return self.nominal_f + shift_mm * 1e-3


def load_thorlens_lens_models(data_dir):
    """
    Load standard lens models for C560TME-C and A220TM lenses.

    Parameters
    ----------
    data_dir : str
        Directory containing the CSV data files.

    Returns
    -------
    models : dict
        Mapping of lens names to ThorlabsLensModel instances.
    """
    c560_path = os.path.join(data_dir, 'C560_focal_shift.csv')
    a220_path = os.path.join(data_dir, 'A220_focal_shift.csv')

    models = {}
    models['C560TME-C'] = ThorlabsLensModel(
        c560_path, nominal_focal_length=13.86e-3
    )
    models['A220TM'] = ThorlabsLensModel(
        a220_path, nominal_focal_length=11.00e-3
    )
    models['A220TM-B'] = ThorlabsLensModel(
        a220_path, nominal_focal_length=11.00e-3
    )

    return models


# Thorlabs standard focal lengths (meters) - available for optimization
THORLABS_STANDARD_FL = np.array([
    2.00e-3, 2.75e-3, 3.10e-3, 4.03e-3, 4.51e-3, 5.00e-3,
    5.50e-3, 5.91e-3, 6.15e-3, 6.24e-3, 7.50e-3, 8.00e-3,
    10.00e-3, 11.00e-3, 12.00e-3, 12.70e-3, 13.86e-3, 15.00e-3,
    15.29e-3, 18.00e-3, 18.40e-3, 19.00e-3, 20.00e-3, 20.10e-3,
    25.00e-3, 25.40e-3, 30.00e-3, 35.00e-3, 40.00e-3, 45.00e-3,
    50.00e-3, 60.00e-3, 75.00e-3, 100.00e-3, 125.00e-3, 150.00e-3,
    175.00e-3, 200.00e-3, 250.00e-3, 300.00e-3, 400.00e-3,
    500.00e-3, 750.00e-3, 1000.00e-3,
])


def nearest_standard_focal_length(f):
    """Return the nearest standard Thorlabs focal length."""
    idx = np.argmin(np.abs(THORLABS_STANDARD_FL - f))
    return THORLABS_STANDARD_FL[idx]
