# Optical System Simulator

Gaussian beam propagation and waveguide coupling optimization using the ABCD matrix (Kogelnik) method.

## Overview

This package simulates the propagation of Gaussian beams through cascaded optical systems — lenses, free-space segments, dielectric interfaces — and computes coupling efficiency into rectangular dielectric waveguides.

**Key features:**

- ABCD matrix formalism for arbitrary sequences of optical elements
- Marcatili's method for rectangular waveguide mode solving (LiNbO3 on LiTaO3)
- Mode overlap integrals for coupling efficiency between Gaussian beams and waveguide modes
- Chromatic focal shift models for Thorlabs catalog lenses (A220TM, C560TME-C)
- Refractive index models (Sellmeier equations with temperature dependence)
- Optimization framework with standard Thorlabs focal length snapping
- Tolerance analysis and beam propagation visualization

## Installation

```bash
pip install numpy scipy matplotlib numba pytest
```

No additional package installation needed — pure Python with standard scientific stack.

## Quick Start

```python
from optical_system import (
    OpticalSystem, q_from_waist, q_to_beam_radius
)
from waveguide import waveguide_mode_field
from coupling import mode_overlap_integral

# Define a simple system: collimating lens + free-space to waveguide
wl = 1.3e-6          # wavelength [m]
w0_fiber = 4.65e-6   # fiber mode field radius [m]
f_lens = 0.011       # lens focal length [m]

system = OpticalSystem()
system.add_propagation(f_lens, "fiber_to_lens")
system.add_thin_lens(f_lens, label="collimator")
system.add_propagation(0.5, "to_waveguide")

# Input beam from fiber
q_in = q_from_waist(w0_fiber, wl)

# Propagate and check beam size at output
q_out = system.propagate(q_in, wl)
w_out = q_to_beam_radius(q_out, wl)
print(f"Beam radius at WG: {w_out*1e6:.1f} um")

# Compute coupling to waveguide mode
E_wg = waveguide_mode_field(wl, 12e-6, 13e-6)
eta = mode_overlap_integral(E_wg, q_out, wl, 13e-6, 12e-6, 0.1e-6)
print(f"Coupling efficiency: {eta*100:.1f}%")
```

## Module Reference

### `optical_system.py` — Beam Propagation Core

| Function / Class | Description |
|---|---|
| `propagation_matrix(d)` | Free-space propagation [[1,d],[0,1]] |
| `thin_lens_matrix(f)` | Thin lens [[1,0],[-1/f,1]] |
| `flat_interface_matrix(n_in, n_out)` | Refraction at flat interface |
| `thick_lens_matrix(n_env, n_glass, R1, R2, t)` | Thick lens |
| `q_from_waist(w0, lambda0, n, z)` | Complex beam parameter from waist |
| `q_from_curvature(R, w, lambda0, n)` | q from curvature and beam size |
| `q_apply_matrix(M, q)` | Apply ABCD matrix to q |
| `q_to_beam_radius(q, lambda0, n)` | Extract beam radius from q |
| `q_to_waist_radius(q, lambda0, n)` | Extract waist radius |
| `q_to_curvature_radius(q)` | Extract wavefront curvature |
| `OpticalSystem` | Composable system class |
| `FreeSpace(d)` | Free-space segment element |
| `ThinLens(f, focal_shift_fn)` | Thin lens with chromatic shift |
| `ThickLens(n_fn, R1, R2, t)` | Thick lens element |
| `FlatInterface(n_in_fn, n_out_fn)` | Interface element |

### `waveguide.py` — Waveguide Mode Solver

| Function | Description |
|---|---|
| `waveguide_mode_field(lambda0, thickness, width, T, dL)` | Solve fundamental mode E-field |
| `waveguide_mode_size(E, dL, width, thickness)` | Extract 1/e^2 mode dimensions |
| `n_ln_umemura(lambda_um, T)` | LiNbO3 5% MgO extraordinary index |
| `n_lt_dolev(lambda_um, T)` | LiTaO3 0.5% MgO extraordinary index |
| `n_air_ciddor(lambda_um)` | Air refractive index |
| `n_ln_deng(lambda_um, T)` | LiNbO3 congruent index (Deng 2006) |

### `coupling.py` — Coupling Efficiency

| Function | Description |
|---|---|
| `mode_overlap_integral(E_wg, q_beam, lambda0, w, h, dL)` | Full overlap integral |
| `estimate_coupling_from_width(w_beam, w_mode_x, w_mode_y)` | Fast estimate from sizes |

### `lenses.py` — Lens Models

| Class / Function | Description |
|---|---|
| `ThorlabsLensModel(csv_path, nominal_fl)` | Chromatic focal shift from data |
| `load_thorlens_lens_models(data_dir)` | Load standard C560/A220 models |
| `nearest_standard_focal_length(f)` | Snap to Thorlabs catalog values |
| `THORLABS_STANDARD_FL` | Array of standard focal lengths [m] |

## Physics Background

### ABCD Matrix Method

Gaussian beams are fully characterized by the complex beam parameter:

```
q(z) = z + i·z_R          where z_R = π·n·w₀²/λ₀
```

Propagation through any paraxial optical system obeys the Kogelnik transformation:

```
q_out = (A·q_in + B) / (C·q_in + D)
```

where [[A,B],[C,D]] is the system's ray transfer matrix (product of individual element matrices).

Beam radius and wavefront curvature are extracted from q:

```
1/q = 1/R − i·λ₀/(π·n·w²)
```

### Waveguide Modes

The fundamental mode of a rectangular dielectric waveguide is found by solving Marcatili's transcendental equations for the transverse wave numbers kappa_x and kappa_y:

```
tan(κₓ·d) = n₁²·κₓ·(n₃²·γ₂ + n₂²·γ₃) / (n₃²·n₂²·κₓ² − n₁⁴·γ₂·γ₃)    [asymmetric, x]
tan(κᵧ·w) = 2·κᵧ·γ₅ / (κᵧ² − γ₅²)                                      [symmetric, y]
```

with γⱼ = √((n₁² − nⱼ²)·k₀² − κ²) and propagation constant β = √(n₁²·k₀² − κₓ² − κᵧ²).

### Coupling Efficiency

Power coupling from Gaussian beam to waveguide mode:

```
η = |∫∫ E_beam · E_wg* dA|² / (∫∫ |E_beam|² dA · ∫∫ |E_wg|² dA)
```

## Optimization Results

The co-optimization of DFG and SPDC setups for a 12×13 µm LiNbO3 waveguide at 2070/1299/798 nm with compact table layout (max 800 mm between elements) achieves:

| Setup | 2070 nm | 1299 nm | 798 nm |
|-------|---------|---------|--------|
| DFG   | 99.27%  | 98.42%  | 99.09% |
| SPDC  | 97.74%  | 90.18%  | 99.39% |

Lenses: f1=250 mm, f2=300 mm (standard Thorlabs catalog).

## Running Tests

```bash
python -m pytest tests/test_optical.py -v
```

26 tests covering ABCD matrix algebra, q-parameter transformations, waveguide mode physics, coupling efficiency bounds, and physical scenarios (collimation, telescope magnification, diffraction-limited focusing).

## Project Structure

```
optical-optimizer-2/
├── __init__.py              # Package init
├── optical_system.py         # ABCD matrices, q-parameter, OpticalSystem
├── waveguide.py              # Marcatili mode solver, refractive indices
├── coupling.py               # Mode overlap integrals
├── lenses.py                 # Thorlabs lens models
├── tests/
│   └── test_optical.py       # 26 unit/integration tests
├── scripts/
│   ├── final_optimize.py     # Co-optimization of DFG/SPDC
│   ├── tolerance_report.py   # Tolerances + beam plots + report
│   └── optimize.py           # Earlier optimization (reference)
├── coupling simulation/      # Original code + lens shift data
└── README.md
```

## References

- Kogelnik & Li, "Laser Beams and Resonators", Appl. Opt. 5, 1550 (1966)
- Siegman, "Lasers", University Science Books (1986)
- Marcatili, "Dielectric Rectangular Waveguide...", Bell Syst. Tech. J. 48, 2071 (1969)
- Suhara, "Waveguide Nonlinear-Optic Devices", Springer (2003)
- Umemura et al., "Sellmeier equation for 5% MgO:LiNbO3", Appl. Opt. 53, 25 (2014)
- Dolev et al., "Refractive index of SLT", Appl. Phys. B 96, 423 (2009)
- Ciddor, "Refractive index of air", Appl. Optics 35, 1566 (1996)

## License

MIT — Michal Mikolajczyk (michal@mikolajczyk.link)
