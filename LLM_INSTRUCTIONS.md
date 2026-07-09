# LLM Instructions — Optical System Simulator

## When to Use This Codebase

You are working with a Python package that simulates Gaussian beam propagation through optical systems and computes coupling efficiency into dielectric waveguides. Use this when the user asks about:

- Designing lens-based optical setups (telescopes, collimators, beam expanders)
- Calculating coupling of free-space beams into fibers or waveguides
- Optimizing lens positions and focal lengths for mode matching
- Analyzing placement tolerances of optical elements
- Simulating beam width evolution through an optical train
- Working with LiNbO3 or LiTaO3 waveguide devices

## System Architecture

**Do NOT modify these core modules** — they are the physics engine:

| File | Purpose |
|------|---------|
| `optical_system.py` | ABCD matrices, complex beam parameter (q), `OpticalSystem` class |
| `waveguide.py` | Marcatili mode solver, Sellmeier refractive index models |
| `coupling.py` | Mode overlap integrals for coupling efficiency |
| `lenses.py` | Thorlabs lens chromatic focal shift models |

**Scripts you CAN modify** — these are the user-facing workflows:

| Script | Purpose |
|--------|---------|
| `scripts/final_optimize.py` | Co-optimization of multi-path optical setups |
| `scripts/tolerance_report.py` | Tolerance sweep + beam plots + markdown report |
| `scripts/optimize.py` | Reference: earlier optimization approach |

## Data Files

Lens focal shift data lives in `coupling simulation/`:
- `C560_focal_shift.csv` — Thorlabs C560TME-C measured shifts
- `A220_focal_shift.csv` — Thorlabs A220TM measured shifts

Format: `focal_shift_mm; wavelength_nm`

## Key API Patterns

### Building an optical system (sequential propagation)

```python
from optical_system import OpticalSystem, q_from_waist

system = OpticalSystem("my_path")
system.add_propagation(0.05, "segment_label")       # free-space [m]
system.add_thin_lens(0.250, label="lens_1")          # focal length [m]
system.add_propagation(0.30)
system.add_thin_lens(0.300, label="lens_2")

q_out = system.propagate(q_in, lambda0)
```

### Computing waveguide mode

```python
from waveguide import waveguide_mode_field, waveguide_mode_size

# 12x13 um LiNbO3 WG at 25 C, 0.1 um resolution
E = waveguide_mode_field(lambda0=1.3e-6, thickness=12e-6, width=13e-6, temperature=25, dL=0.1e-6)
wx, wy = waveguide_mode_size(E, dL=0.1e-6, waveguide_width=13e-6, waveguide_thickness=12e-6)
```

### Computing coupling

```python
from coupling import mode_overlap_integral

eta = mode_overlap_integral(E_wg, q_beam, lambda0, wg_width, wg_thickness, dL)
# Returns 0-1 fraction
```

### Using lens models

```python
from lenses import load_thorlens_lens_models

models = load_thorlens_lens_models("coupling simulation")
f_eff = models['A220TM'].focal_length(2070e-9)   # effective FL at 2070nm
```

## Optimization Pattern

When the user describes an optical setup to optimize:

1. **Define the parameter vector**: Create a numpy array mapping each adjustable distance and focal length to an index.
2. **Write a system builder function**: Takes the parameter vector, returns an `OpticalSystem`.
3. **Write an objective function**: Returns a scalar to minimize. Typical pattern:
   ```python
   def objective(x):
       eta1 = compute_coupling(x, path1_params)
       eta2 = compute_coupling(x, path2_params)
       return -0.5*(eta1 + eta2) + 100*sum(max(0, target - eta)**2)
   ```
4. **Use `scipy.optimize`**: Differential evolution for global search, L-BFGS-B for refinement.
5. **Snap lens focal lengths**: Use `nearest_standard_focal_length()` to enforce Thorlabs catalog values.
6. **Verify with fine grid**: Recompute at dL=0.1µm for final numbers.

## Important Physics Conventions

- **All units in SI**: meters, watts, etc.
- **Beam radius**: 1/e² intensity (not 1/e amplitude)
- **q parameter**: `q = z + i*z_R` where `z_R = pi*n*w0^2/lambda0`
- **Waveguide grid**: Marcatili field is computed on a grid extending beyond the physical boundaries into substrate/cladding
- **Mode normalization**: To unit total power (1 W Poynting flux)
- **Fiber input**: Modeled as Gaussian waist at fiber facet with radius = MFD/2
- **Collimator output**: Modeled with curvature radius and beam radius at the collimator exit plane

## Common Pitfalls

1. **Coupling too low**: Check beam radius at waveguide facet — must approximately match the waveguide mode size (typically 5-7 µm radius for 12x13 µm WG)
2. **Optimizer gets stuck**: The coupling landscape has lots of local minima from phase effects. Always use differential evolution for global search first.
3. **Focal shift matters**: At 2070 nm, the A220TM shifts by ~0.6 mm from its 630 nm design wavelength. Don't use nominal focal lengths at long wavelengths.
4. **Wavefront curvature**: Even with the right beam size, curvature mismatch kills coupling. q must be nearly pure imaginary at the waveguide facet (planar wavefront).
5. **dL resolution**: Optimization at 0.25 µm is fast; final verification should use 0.1 µm for accuracy.

## Default Waveguide Parameters

These are baked into the optimization scripts:

```python
WG_THICKNESS = 12e-6       # m, LiNbO3 film thickness
WG_WIDTH = 13e-6           # m, ridge width
WG_LENGTH = 40e-3          # m, crystal length
TEMPERATURE = 25.0         # C
```

## Standard Fiber Parameters

```python
FIBER_PARAMS = {
    'PM1550-XP': {'w0': 4.65e-6},    # MFD=9.3 um at 1310/1550 nm
    'PM780-XP':  {'w0': 2.65e-6},    # MFD=5.3 um at 780 nm
}
```

## Running Tests

```bash
python -m pytest tests/test_optical.py -v
```

26 tests. Must pass before any PR. Tests verify: ABCD matrix determinants, q-parameter identities, waveguide mode physics, coupling bounds, and physical scenarios (telescope magnification, diffraction-limited focusing).

## Adding a New Lens Model

1. Obtain Thorlabs focal shift data in CSV: `focal_shift_mm; wavelength_nm`
2. Create a `ThorlabsLensModel(csv_path, nominal_focal_length)` instance
3. Add to the `load_thorlens_lens_models()` dictionary
4. Use `model.focal_length(lambda0)` to get effective focal length at any wavelength

## Output Conventions

- Markdown reports in `output/` or `output-2/`
- PNG plots at 150 dpi, 14×6 inch figures
- Beam radius in µm, positions in mm on plots
- Red dashed lines = lenses, solid green line = waveguide facet
