"""
Co-optimization of DFG and SPDC optical setups for waveguide coupling.

This script optimizes the shared and independent parameters of two optical
setups (DFG and SPDC) to maximize coupling efficiency of three wavelengths
(2070 nm, 1299 nm, 798 nm) into a 12x13 um LiNbO3 waveguide.

Optimization strategy:
    1. Pre-compute waveguide modes for all three wavelengths.
    2. Use differential evolution for global search.
    3. Refine with L-BFGS-B local optimization.
    4. Snap lens-1 and lens-2 focal lengths to standard Thorlabs values
       and re-optimize distances.
    5. Report final coupling efficiencies.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from scipy.optimize import differential_evolution, minimize

from optical_system import (
    OpticalSystem, FreeSpace, ThinLens,
    q_from_waist, q_from_curvature, q_to_beam_radius,
    gaussian_field_from_q
)
from waveguide import (
    waveguide_mode_field, waveguide_mode_size,
    n_air_ciddor, n_ln_umemura, n_lt_dolev
)
from coupling import mode_overlap_integral
from lenses import (
    ThorlabsLensModel, THORLABS_STANDARD_FL,
    nearest_standard_focal_length, load_thorlens_lens_models
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WAVELENGTHS = {
    '2070': 2070e-9,
    '1299': 1299e-9,
    '798': 798e-9,
}

# Waveguide dimensions
WG_WIDTH = 13e-6
WG_THICKNESS = 12e-6
WG_LENGTH = 40e-3

# Fiber parameters
FIBER_PARAMS = {
    'PM1550': {'w0': 4.65e-6, 'NA': 0.125},   # MFD = 9.3 um
    'PM780':  {'w0': 2.65e-6, 'NA': 0.12},     # MFD = 5.3 um
}

# Collimator output parameters (F028APC-2000 at 2070 nm)
# From Thorlabs design: waist radius 0.6 mm, slight divergence
COLLIMATOR_OUTPUT = {
    'w': 0.6e-3,        # beam radius at collimator output [m]
    'R': 3.0,           # wavefront curvature radius [m]
}

# Resolution for overlap integral
DL_COARSE = 0.25e-6    # for optimization
DL_FINE = 0.1e-6       # for final verification

# Temperature
TEMPERATURE = 25.0

# ---------------------------------------------------------------------------
# Parameter indices (for the optimization vector x)
# ---------------------------------------------------------------------------
# Shared parameters (indices 0-6):
IDX_L2 = 0       # distance from collimator to lens-1 [m]
IDX_F1 = 1       # lens-1 focal length [m]
IDX_L3 = 2       # distance from lens-1 to lens-2 [m]
IDX_F2 = 3       # lens-2 focal length [m]
IDX_D_L2_L3 = 4  # distance from lens-2 to lens-3 (shared for 2070 nm DFG+SPDC) [m]
IDX_L6 = 5       # distance from lens-4 to lens-3/combiner (shared 798 nm DFG+SPDC) [m]
IDX_L8 = 6       # distance from lens-5 to lens-6 (shared 1299 nm DFG+SPDC) [m]

# DFG-only parameters (indices 7-10):
IDX_L4_DFG = 7       # lens-3 to WG for DFG 2070/798 [m]
IDX_L5_DFG = 8       # fiber-798 to lens-4 for DFG [m]
IDX_L7_DFG = 9       # fiber-1299 to lens-5 for DFG [m]
IDX_L9_DFG = 10      # lens-6 to WG for DFG 1299 [m]

# SPDC-only parameters (indices 11-14):
IDX_L10_SPDC = 11    # lens-3 to WG for SPDC 2070/1299 [m]
IDX_L11_SPDC = 12    # fiber-1299 to lens-5 for SPDC [m]
IDX_L12_SPDC = 13    # fiber-798 to lens-4 for SPDC [m]
IDX_L13_SPDC = 14    # lens-3 to WG for SPDC 798 [m]

N_PARAMS = 15

# ---------------------------------------------------------------------------
# Bounds (meters)
# ---------------------------------------------------------------------------

BOUNDS = [
    (0.05,  1.50),   # l2: collimator to lens-1
    (0.025, 1.00),   # f1: lens-1 focal length (Thorlabs range up to 1000mm)
    (0.05,  1.50),   # l3: lens-1 to lens-2
    (0.025, 1.00),   # f2: lens-2 focal length
    (0.05,  1.00),   # d_l2_l3: lens-2 to lens-3
    (0.05,  1.00),   # l6: lens-4 to lens-3 (798 nm shared)
    (0.05,  1.00),   # l8: lens-5 to lens-6 (1299 nm shared)
    (0.005, 0.050),  # l4 (DFG): lens-3 to WG
    (0.005, 0.030),  # l5 (DFG): fiber-798 to lens-4
    (0.005, 0.030),  # l7 (DFG): fiber-1299 to lens-5
    (0.005, 0.050),  # l9 (DFG): lens-6 to WG
    (0.005, 0.050),  # l10 (SPDC): lens-3 to WG
    (0.005, 0.030),  # l11 (SPDC): fiber-1299 to lens-5
    (0.005, 0.030),  # l12 (SPDC): fiber-798 to lens-4
    (0.005, 0.050),  # l13 (SPDC): lens-3 to WG for 798
]

# ---------------------------------------------------------------------------
# Lens models
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'coupling simulation')
lens_models = load_thorlens_lens_models(DATA_DIR) if os.path.exists(DATA_DIR) else {}

def get_focal_length(lens_name, lambda0):
    """Get effective focal length of a named lens at the given wavelength."""
    if lens_name in lens_models:
        return lens_models[lens_name].focal_length(lambda0)
    # Fallback: return nominal if no model available
    nominal_map = {
        'A220TM': 11.00e-3,
        'A220TM-B': 11.00e-3,
        'C560TME-C': 13.86e-3,
    }
    return nominal_map.get(lens_name, 11.00e-3)


# ---------------------------------------------------------------------------
# Pre-compute waveguide modes
# ---------------------------------------------------------------------------

def compute_waveguide_modes(dL=DL_COARSE):
    """Pre-compute waveguide mode fields for all three wavelengths."""
    modes = {}
    for name, wl in WAVELENGTHS.items():
        print(f"  Computing waveguide mode at {name} nm (dL={dL*1e6:.1f} um)...")
        E = waveguide_mode_field(wl, WG_THICKNESS, WG_WIDTH, TEMPERATURE, dL)
        wx, wy = waveguide_mode_size(E, dL, WG_WIDTH, WG_THICKNESS)
        modes[name] = {
            'E': E,
            'w_x': wx,
            'w_y': wy,
            'dL': dL,
        }
        print(f"    Mode size: {wx*1e6:.2f} x {wy*1e6:.2f} um")
    return modes


# ---------------------------------------------------------------------------
# Optical system builders
# ---------------------------------------------------------------------------

def build_2070_system(x, dist_to_wg):
    """
    Build the 2070 nm optical system (shared telescope + lens-3 + propagation to WG).

    Parameters
    ----------
    x : ndarray
        Optimization parameters.
    dist_to_wg : float
        Distance from lens-3 to WG.

    Returns
    -------
    system : OpticalSystem
    """
    system = OpticalSystem("2070_path")
    f3 = get_focal_length('A220TM', WAVELENGTHS['2070'])

    system.add_propagation(x[IDX_L2], "collimator_to_lens1")
    system.add_thin_lens(x[IDX_F1], label="lens_1")
    system.add_propagation(x[IDX_L3], "lens1_to_lens2")
    system.add_thin_lens(x[IDX_F2], label="lens_2")
    system.add_propagation(x[IDX_D_L2_L3], "lens2_to_lens3")
    system.add_thin_lens(f3, label="lens_3_A220")
    system.add_propagation(dist_to_wg, "lens3_to_wg")

    return system


def build_798_system(x, dist_fiber_to_lens4, dist_to_wg):
    """
    Build the 798 nm optical system (lens-4 + shared lens-3 + propagation to WG).

    Parameters
    ----------
    x : ndarray
        Optimization parameters.
    dist_fiber_to_lens4 : float
        Distance from fiber to lens-4.
    dist_to_wg : float
        Distance from lens-3 to WG.

    Returns
    -------
    system : OpticalSystem
    """
    system = OpticalSystem("798_path")
    f4 = get_focal_length('A220TM-B', WAVELENGTHS['798'])
    f3 = get_focal_length('A220TM', WAVELENGTHS['798'])

    system.add_propagation(dist_fiber_to_lens4, "fiber_to_lens4")
    system.add_thin_lens(f4, label="lens_4_A220B")
    system.add_propagation(x[IDX_L6], "lens4_to_lens3")
    system.add_thin_lens(f3, label="lens_3_A220")
    system.add_propagation(dist_to_wg, "lens3_to_wg")

    return system


def build_1299_system(x, dist_fiber_to_lens5, dist_to_wg):
    """
    Build the 1299 nm optical system (C560 + A220TM + propagation to WG).

    Parameters
    ----------
    x : ndarray
        Optimization parameters.
    dist_fiber_to_lens5 : float
        Distance from fiber to lens-5 (C560).
    dist_to_wg : float
        Distance from lens-6 (A220TM) to WG.

    Returns
    -------
    system : OpticalSystem
    """
    system = OpticalSystem("1299_path")
    f5 = get_focal_length('C560TME-C', WAVELENGTHS['1299'])
    f6 = get_focal_length('A220TM', WAVELENGTHS['1299'])

    system.add_propagation(dist_fiber_to_lens5, "fiber_to_lens5")
    system.add_thin_lens(f5, label="lens_5_C560")
    system.add_propagation(x[IDX_L8], "lens5_to_lens6")
    system.add_thin_lens(f6, label="lens_6_A220")
    system.add_propagation(dist_to_wg, "lens6_to_wg")

    return system


# ---------------------------------------------------------------------------
# Coupling computation for a single path
# ---------------------------------------------------------------------------

def compute_coupling(wg_mode, q_in, system, lambda0):
    """
    Propagate beam through system and compute overlap with waveguide mode.

    Parameters
    ----------
    wg_mode : dict
        Pre-computed waveguide mode data (E, dL).
    q_in : complex
        Input complex beam parameter.
    system : OpticalSystem
        Optical system to propagate through.
    lambda0 : float
        Wavelength [m].

    Returns
    -------
    eta : float
        Coupling efficiency (0 to 1).
    """
    q_out = system.propagate(q_in, lambda0)
    eta = mode_overlap_integral(
        wg_mode['E'], q_out, lambda0,
        WG_WIDTH, WG_THICKNESS, wg_mode['dL']
    )
    return eta


# ---------------------------------------------------------------------------
# Master coupling evaluator
# ---------------------------------------------------------------------------

def evaluate_all_couplings(x, wg_modes):
    """
    Compute coupling efficiencies for all 6 paths (3 wavelengths x 2 setups).

    Parameters
    ----------
    x : ndarray, shape (N_PARAMS,)
        Optimization parameter vector.
    wg_modes : dict
        Pre-computed waveguide modes.

    Returns
    -------
    results : dict
        Nested dict: results[setup][wavelength] = coupling efficiency.
    """
    results = {'DFG': {}, 'SPDC': {}}

    # --- DFG ---
    # 2070 nm
    q_in_2070 = q_from_curvature(
        COLLIMATOR_OUTPUT['R'], COLLIMATOR_OUTPUT['w'],
        WAVELENGTHS['2070']
    )
    sys_2070_dfg = build_2070_system(x, x[IDX_L4_DFG])
    results['DFG']['2070'] = compute_coupling(
        wg_modes['2070'], q_in_2070, sys_2070_dfg, WAVELENGTHS['2070']
    )

    # 798 nm
    q_in_798 = q_from_waist(
        FIBER_PARAMS['PM780']['w0'], WAVELENGTHS['798'], z=0.0
    )
    sys_798_dfg = build_798_system(x, x[IDX_L5_DFG], x[IDX_L4_DFG])
    results['DFG']['798'] = compute_coupling(
        wg_modes['798'], q_in_798, sys_798_dfg, WAVELENGTHS['798']
    )

    # 1299 nm
    q_in_1299 = q_from_waist(
        FIBER_PARAMS['PM1550']['w0'], WAVELENGTHS['1299'], z=0.0
    )
    sys_1299_dfg = build_1299_system(x, x[IDX_L7_DFG], x[IDX_L9_DFG])
    results['DFG']['1299'] = compute_coupling(
        wg_modes['1299'], q_in_1299, sys_1299_dfg, WAVELENGTHS['1299']
    )

    # --- SPDC ---
    # 2070 nm
    sys_2070_spdc = build_2070_system(x, x[IDX_L10_SPDC])
    results['SPDC']['2070'] = compute_coupling(
        wg_modes['2070'], q_in_2070, sys_2070_spdc, WAVELENGTHS['2070']
    )

    # 1299 nm
    sys_1299_spdc = build_1299_system(x, x[IDX_L11_SPDC], x[IDX_L10_SPDC])
    results['SPDC']['1299'] = compute_coupling(
        wg_modes['1299'], q_in_1299, sys_1299_spdc, WAVELENGTHS['1299']
    )

    # 798 nm
    sys_798_spdc = build_798_system(x, x[IDX_L12_SPDC], x[IDX_L13_SPDC])
    results['SPDC']['798'] = compute_coupling(
        wg_modes['798'], q_in_798, sys_798_spdc, WAVELENGTHS['798']
    )

    return results


# ---------------------------------------------------------------------------
# Objective function for optimization
# ---------------------------------------------------------------------------

def objective(x, wg_modes, dfg_target=0.98, spdc_target=0.90):
    """
    Optimization objective: maximize coupling across all paths,
    with penalties for falling below target thresholds.

    obj = -(mean of all 6 etas) + sum(penalties for below-target)

    Where each penalty = weight * (target - eta)^2 for eta < target.
    This creates a smooth landscape that rewards all improvements.

    Parameters
    ----------
    x : ndarray
        Parameter vector.
    wg_modes : dict
        Pre-computed waveguide modes.
    dfg_target : float
        Target coupling for DFG setup.
    spdc_target : float
        Target coupling for SPDC setup.

    Returns
    -------
    value : float
        Objective value to minimize.
    """
    results = evaluate_all_couplings(x, wg_modes)

    etas = []
    penalty = 0.0
    penalty_weight = 100.0

    for wl in ['2070', '1299', '798']:
        eta_dfg = results['DFG'][wl]
        etas.append(eta_dfg)
        if eta_dfg < dfg_target:
            penalty += penalty_weight * (dfg_target - eta_dfg) ** 2

        eta_spdc = results['SPDC'][wl]
        etas.append(eta_spdc)
        if eta_spdc < spdc_target:
            penalty += penalty_weight * (spdc_target - eta_spdc) ** 2

    return -np.mean(etas) + penalty


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def run_optimization(seed=42, maxiter=100, polish=True):
    """
    Run the full optimization pipeline.

    Returns
    -------
    result : OptimizeResult
        Scipy optimization result.
    wg_modes : dict
        Pre-computed waveguide modes (coarse grid).
    """
    print("=" * 60)
    print("Pre-computing waveguide modes...")
    print("=" * 60)
    wg_modes = compute_waveguide_modes(DL_COARSE)

    print("\n" + "=" * 60)
    print("Running global optimization (differential evolution)...")
    print("=" * 60)

    result_de = differential_evolution(
        objective,
        bounds=BOUNDS,
        args=(wg_modes,),
        seed=seed,
        maxiter=maxiter,
        polish=False,
        popsize=50,
        mutation=(0.3, 1.5),
        recombination=0.7,
        disp=True,
        workers=1,
    )

    print(f"\nDE best value: {result_de.fun:.6f}")

    if polish:
        print("\n" + "=" * 60)
        print("Polishing with L-BFGS-B...")
        print("=" * 60)
        result = minimize(
            objective,
            result_de.x,
            args=(wg_modes,),
            method='L-BFGS-B',
            bounds=BOUNDS,
            options={'maxiter': 5000, 'ftol': 1e-12},
        )
        print(f"L-BFGS-B best value: {result.fun:.6f}")
    else:
        result = result_de

    return result, wg_modes


# ---------------------------------------------------------------------------
# Snap focal lengths to standard values
# ---------------------------------------------------------------------------

def snap_and_refine(result, wg_modes):
    """
    Snap lens-1 and lens-2 focal lengths to nearest standard Thorlabs values,
    then re-optimize all distances.

    Parameters
    ----------
    result : OptimizeResult
        Optimization result.
    wg_modes : dict
        Pre-computed waveguide modes.

    Returns
    -------
    x_final : ndarray
        Final parameter vector with standard focal lengths.
    """
    x = result.x.copy()
    x[IDX_F1] = nearest_standard_focal_length(x[IDX_F1])
    x[IDX_F2] = nearest_standard_focal_length(x[IDX_F2])

    print(f"\nSnapped f1 to {x[IDX_F1]*1e3:.2f} mm, f2 to {x[IDX_F2]*1e3:.2f} mm")

    # Fix focal lengths and re-optimize distances
    fixed_params = [IDX_F1, IDX_F2]
    free_params = [i for i in range(N_PARAMS) if i not in fixed_params]

    def objective_distances(x_dist, x_full, wg_modes):
        x_combined = x_full.copy()
        for idx, val in zip(free_params, x_dist):
            x_combined[idx] = val
        return objective(x_combined, wg_modes)

    x0_dist = x[free_params]
    bounds_dist = [BOUNDS[i] for i in free_params]

    print("Re-optimizing distances with fixed focal lengths...")
    result_refined = minimize(
        objective_distances,
        x0_dist,
        args=(x, wg_modes),
        method='L-BFGS-B',
        bounds=bounds_dist,
        options={'maxiter': 5000, 'ftol': 1e-12},
    )

    x_final = x.copy()
    for idx, val in zip(free_params, result_refined.x):
        x_final[idx] = val

    return x_final


# ---------------------------------------------------------------------------
# Final verification with fine grid
# ---------------------------------------------------------------------------

def verify_with_fine_grid(x, result_name=""):
    """
    Re-compute all coupling efficiencies using fine grid resolution.

    Parameters
    ----------
    x : ndarray
        Final parameter vector.
    result_name : str
        Label for output.

    Returns
    -------
    results : dict
        Coupling efficiencies.
    """
    print(f"\n{'='*60}")
    print(f"Final verification: {result_name}")
    print(f"{'='*60}")

    wg_modes_fine = compute_waveguide_modes(DL_FINE)
    results = evaluate_all_couplings(x, wg_modes_fine)

    for setup in ['DFG', 'SPDC']:
        print(f"\n  {setup}:")
        for wl in ['2070', '1299', '798']:
            eta = results[setup][wl]
            bar = '#' * int(eta * 50) + '-' * (50 - int(eta * 50))
            print(f"    {wl} nm: {eta*100:6.2f}%  [{bar}]")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Run optimization
    # Run a shorter global search first
    result, wg_modes = run_optimization(seed=42, maxiter=150, polish=True)

    # The first result is likely near-optimal. For SPDC 798nm, which is
    # independent (only depends on l12 and l13), we can optimize separately.
    # The SPDC 798 path: fiber -> lens-4 -> l6 -> lens-3 -> l13 -> WG
    print("\n" + "=" * 60)
    print("Focused refinement for SPDC 798 nm path...")
    print("=" * 60)

    x = result.x.copy()

    def spdc_798_objective(x13, wg_modes, x_ref, l12_val):
        x_trial = x_ref.copy()
        x_trial[IDX_L13_SPDC] = x13[0]
        results = evaluate_all_couplings(x_trial, wg_modes)
        return -results['SPDC']['798']

    # Optimize l12 and l13 independently for best SPDC 798 coupling
    for l12_guess in np.linspace(0.005, 0.030, 10):
        x_trial = x.copy()
        x_trial[IDX_L12_SPDC] = l12_guess
        res_798 = minimize(
            spdc_798_objective,
            [x[IDX_L13_SPDC]],
            args=(wg_modes, x_trial, l12_guess),
            method='L-BFGS-B',
            bounds=[BOUNDS[IDX_L13_SPDC]],
            options={'maxiter': 1000, 'ftol': 1e-14},
        )
        x_trial[IDX_L13_SPDC] = res_798.x[0]
        results_trial = evaluate_all_couplings(x_trial, wg_modes)
        eta_798 = results_trial['SPDC']['798']
        if eta_798 > 0.85:
            x = x_trial.copy()
            break

    # Re-check all couplings
    results_check = evaluate_all_couplings(x, wg_modes)
    print(f"  After refinement: SPDC 798 = {results_check['SPDC']['798']*100:.2f}%")

    result = type('Result', (), {'x': x, 'fun': objective(x, wg_modes)})()

    # Snap to standard focal lengths
    x_final = snap_and_refine(result, wg_modes)

    # Final verification
    results_final = verify_with_fine_grid(x_final, "Final (standard FL)")

    # Print parameter table
    print(f"\n{'='*60}")
    print("Final parameters (all distances in mm)")
    print(f"{'='*60}")
    param_names = [
        "l2 (collim -> lens1)",
        "f1 (lens-1 FL)",
        "l3 (lens1 -> lens2)",
        "f2 (lens-2 FL)",
        "d_l2_l3 (lens2 -> lens3, shared 2070)",
        "l6 (lens4 -> lens3, shared 798)",
        "l8 (lens5 -> lens6, shared 1299)",
        "l4 (DFG: lens3 -> WG, 2070/798)",
        "l5 (DFG: fiber798 -> lens4)",
        "l7 (DFG: fiber1299 -> lens5)",
        "l9 (DFG: lens6 -> WG, 1299)",
        "l10 (SPDC: lens3 -> WG, 2070/1299)",
        "l11 (SPDC: fiber1299 -> lens5)",
        "l12 (SPDC: fiber798 -> lens4)",
        "l13 (SPDC: lens3 -> WG, 798)",
    ]
    for name, val in zip(param_names, x_final):
        print(f"  {name:40s} = {val*1e3:8.3f} mm")
