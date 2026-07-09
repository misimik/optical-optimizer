"""
Optical System Simulator — Gaussian beam propagation and waveguide coupling.

This package provides tools for simulating Gaussian beam propagation through
cascaded optical systems using the ABCD matrix (Kogelnik) formalism, computing
waveguide modes via Marcatili's method, and calculating coupling efficiency
through mode overlap integrals.

Core modules:
    optical_system — ABCD matrices, q-parameter, OpticalSystem class
    waveguide      — Marcatili mode solver, refractive index models
    coupling       — Mode overlap integral and coupling efficiency
    lenses         — Thorlabs lens models with chromatic focal shift

Scripts:
    scripts/final_optimize.py     — Co-optimization of DFG/SPDC setups
    scripts/tolerance_report.py   — Tolerance analysis and beam plots
    scripts/optimize.py           — Earlier optimization (kept for reference)
"""

from optical_system import (
    OpticalSystem, OpticalElement,
    FreeSpace, ThinLens, ThickLens, FlatInterface,
    propagation_matrix, thin_lens_matrix, flat_interface_matrix,
    thick_lens_matrix,
    q_from_waist, q_from_curvature,
    q_apply_matrix, q_to_beam_radius, q_to_waist_radius,
    q_to_curvature_radius, q_from_waist_and_distance,
    gaussian_field_amplitude, gaussian_field_from_q,
    build_system_from_segments,
)

from waveguide import (
    waveguide_mode_field, waveguide_mode_size,
    n_air_ciddor, n_ln_umemura, n_lt_dolev, n_ln_deng,
)

from coupling import (
    mode_overlap_integral, beam_width_at_facet,
    estimate_coupling_from_width,
)

from lenses import (
    ThorlabsLensModel, THORLABS_STANDARD_FL,
    nearest_standard_focal_length, load_thorlens_lens_models,
)

__version__ = "2.0.0"
__all__ = [
    # optical_system
    "OpticalSystem", "OpticalElement", "FreeSpace", "ThinLens",
    "ThickLens", "FlatInterface",
    "propagation_matrix", "thin_lens_matrix", "flat_interface_matrix",
    "thick_lens_matrix",
    "q_from_waist", "q_from_curvature",
    "q_apply_matrix", "q_to_beam_radius", "q_to_waist_radius",
    "q_to_curvature_radius", "q_from_waist_and_distance",
    "gaussian_field_amplitude", "gaussian_field_from_q",
    "build_system_from_segments",
    # waveguide
    "waveguide_mode_field", "waveguide_mode_size",
    "n_air_ciddor", "n_ln_umemura", "n_lt_dolev", "n_ln_deng",
    # coupling
    "mode_overlap_integral", "beam_width_at_facet",
    "estimate_coupling_from_width",
    # lenses
    "ThorlabsLensModel", "THORLABS_STANDARD_FL",
    "nearest_standard_focal_length", "load_thorlens_lens_models",
]
