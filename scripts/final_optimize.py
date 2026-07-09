"""
Final comprehensive co-optimization of DFG and SPDC setups.

Uses insights from the exploration phase:
- SPDC 798nm path achieves >99% with l12=11.647mm, l13=12.112mm
- DFG paths all achieve >98% with the telescope setup from the first optimization
- The challenge was the initial seed for the DE optimizer

Strategy: Use multi-start local optimization from good initial points.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
from scipy.optimize import minimize, differential_evolution

from optical_system import (
    OpticalSystem, FreeSpace, ThinLens,
    q_from_waist, q_from_curvature, q_to_beam_radius,
)
from waveguide import waveguide_mode_field, waveguide_mode_size
from coupling import mode_overlap_integral
from lenses import load_thorlens_lens_models, nearest_standard_focal_length

# Re-use the parameter indices and other infrastructure from optimize.py
# to keep this script self-contained.

WAVELENGTHS = {'2070': 2070e-9, '1299': 1299e-9, '798': 798e-9}
WG_WIDTH = 13e-6
WG_THICKNESS = 12e-6
TEMPERATURE = 25.0
COLL_W = 0.6e-3
COLL_R = 3.0
FIBER_MFD_HALF = {'1550': 4.65e-6, '780': 2.65e-6}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'coupling simulation')
lens_models = load_thorlens_lens_models(DATA_DIR) if os.path.exists(DATA_DIR) else {}

def get_focal_length(lens_name, lambda0):
    if lens_name in lens_models:
        return lens_models[lens_name].focal_length(lambda0)
    nominal = {'A220TM': 11.00e-3, 'A220TM-B': 11.00e-3, 'C560TME-C': 13.86e-3}
    return nominal.get(lens_name, 11.00e-3)

# Parameter indices
IDX_L2 = 0; IDX_F1 = 1; IDX_L3 = 2; IDX_F2 = 3; IDX_L14 = 4
IDX_L6 = 5; IDX_L8 = 6; IDX_L4_DFG = 7; IDX_L5_DFG = 8; IDX_L7_DFG = 9
IDX_L9_DFG = 10; IDX_L10_SPDC = 11; IDX_L11_SPDC = 12; IDX_L12_SPDC = 13
IDX_L13_SPDC = 14

N_PARAMS = 15

BOUNDS = [
    (0.05, 0.80), (0.025, 1.00), (0.05, 0.80), (0.025, 1.00),
    (0.25, 0.80), (0.50, 0.80), (0.50, 0.80),  # l14>250, l6>500, l8>500, all <=800mm
    (0.005, 0.050), (0.005, 0.050), (0.005, 0.050),  # l4,l5,l7: WG/lens dists
    (0.005, 0.050), (0.005, 0.050), (0.005, 0.050),  # l9,l10,l11
    (0.005, 0.050), (0.005, 0.050),                   # l12,l13
]

DL_OPT = 0.25e-6

def build_and_evaluate(x, wg_modes):
    """Compute all 6 coupling efficiencies."""
    results = {'DFG': {}, 'SPDC': {}}

    q_in_2070 = q_from_curvature(COLL_R, COLL_W, WAVELENGTHS['2070'])
    q_in_798 = q_from_waist(FIBER_MFD_HALF['780'], WAVELENGTHS['798'], z=0.0)
    q_in_1299 = q_from_waist(FIBER_MFD_HALF['1550'], WAVELENGTHS['1299'], z=0.0)

    def propagate_and_overlap(wl_name, system, q_in):
        wl = WAVELENGTHS[wl_name]
        q_out = system.propagate(q_in, wl)
        wm = wg_modes[wl_name]
        return mode_overlap_integral(wm['E'], q_out, wl, WG_WIDTH, WG_THICKNESS, wm['dL'])

    # --- DFG ---
    # 2070 nm
    sys = OpticalSystem()
    f3_2070 = get_focal_length('A220TM', WAVELENGTHS['2070'])
    sys.add_propagation(x[IDX_L2]); sys.add_thin_lens(x[IDX_F1])
    sys.add_propagation(x[IDX_L3]); sys.add_thin_lens(x[IDX_F2])
    sys.add_propagation(x[IDX_L14]); sys.add_thin_lens(f3_2070)
    sys.add_propagation(x[IDX_L4_DFG])
    results['DFG']['2070'] = propagate_and_overlap('2070', sys, q_in_2070)

    # 798 nm
    sys = OpticalSystem()
    f4 = get_focal_length('A220TM-B', WAVELENGTHS['798'])
    f3_798 = get_focal_length('A220TM', WAVELENGTHS['798'])
    sys.add_propagation(x[IDX_L5_DFG]); sys.add_thin_lens(f4)
    sys.add_propagation(x[IDX_L6]); sys.add_thin_lens(f3_798)
    sys.add_propagation(x[IDX_L4_DFG])
    results['DFG']['798'] = propagate_and_overlap('798', sys, q_in_798)

    # 1299 nm
    sys = OpticalSystem()
    f5 = get_focal_length('C560TME-C', WAVELENGTHS['1299'])
    f6 = get_focal_length('A220TM', WAVELENGTHS['1299'])
    sys.add_propagation(x[IDX_L7_DFG]); sys.add_thin_lens(f5)
    sys.add_propagation(x[IDX_L8]); sys.add_thin_lens(f6)
    sys.add_propagation(x[IDX_L9_DFG])
    results['DFG']['1299'] = propagate_and_overlap('1299', sys, q_in_1299)

    # --- SPDC ---
    # 2070 nm
    sys = OpticalSystem()
    sys.add_propagation(x[IDX_L2]); sys.add_thin_lens(x[IDX_F1])
    sys.add_propagation(x[IDX_L3]); sys.add_thin_lens(x[IDX_F2])
    sys.add_propagation(x[IDX_L14]); sys.add_thin_lens(f3_2070)
    sys.add_propagation(x[IDX_L10_SPDC])
    results['SPDC']['2070'] = propagate_and_overlap('2070', sys, q_in_2070)

    # 1299 nm
    sys = OpticalSystem()
    sys.add_propagation(x[IDX_L11_SPDC]); sys.add_thin_lens(f5)
    sys.add_propagation(x[IDX_L8]); sys.add_thin_lens(f6)
    sys.add_propagation(x[IDX_L10_SPDC])
    results['SPDC']['1299'] = propagate_and_overlap('1299', sys, q_in_1299)

    # 798 nm
    sys = OpticalSystem()
    sys.add_propagation(x[IDX_L12_SPDC]); sys.add_thin_lens(f4)
    sys.add_propagation(x[IDX_L6]); sys.add_thin_lens(f3_798)
    sys.add_propagation(x[IDX_L13_SPDC])
    results['SPDC']['798'] = propagate_and_overlap('798', sys, q_in_798)

    return results


def objective(x, wg_modes, dfg_target=0.98, spdc_target=0.90):
    """Mean coupling with squared penalties for below-target."""
    results = build_and_evaluate(x, wg_modes)

    etas = []
    penalty = 0.0
    w = 100.0

    for wl in ['2070', '1299', '798']:
        ed = results['DFG'][wl]; etas.append(ed)
        if ed < dfg_target: penalty += w * (dfg_target - ed) ** 2
        es = results['SPDC'][wl]; etas.append(es)
        if es < spdc_target: penalty += w * (spdc_target - es) ** 2

    return -np.mean(etas) + penalty


if __name__ == '__main__':
    print("Pre-computing waveguide modes (coarse for optimization)...")
    wg_modes = {}
    for name, wl in WAVELENGTHS.items():
        print(f"  {name} nm...")
        E = waveguide_mode_field(wl, WG_THICKNESS, WG_WIDTH, TEMPERATURE, DL_OPT)
        wg_modes[name] = {'E': E, 'dL': DL_OPT}

    # Telescope formula: lenses separated by f1+f2 = 550mm for best collimation.
    # l2 positions the first lens after the collimator; l3 connects lens-1 to lens-2.
    x0 = np.array([
        0.250000,    # l2  - collimator to lens-1 (>=50mm, near f1=250mm for waist re-imaging)
        0.250000,    # f1  = 250mm standard FL
        0.550000,    # l3  - lens-1 to lens-2 (telescope: near f1+f2=550mm)
        0.300000,    # f2  = 300mm standard FL
        0.260000,    # l14 - lens-2 to lens-3 (>250mm)
        0.550000,    # l6  - lens-4 to lens-3 shared 798 (>500mm)
        0.550000,    # l8  - lens-5 to lens-6 shared 1299 (>500mm)
        0.011500,    # l4  (DFG lens3 -> WG)
        0.011500,    # l5  (DFG fiber798 -> lens4)
        0.014000,    # l7  (DFG fiber1299 -> lens5)
        0.011500,    # l9  (DFG lens6 -> WG)
        0.011500,    # l10 (SPDC lens3 -> WG 2070/1299)
        0.014000,    # l11 (SPDC fiber1299 -> lens5)
        0.011500,    # l12 (SPDC fiber798 -> lens4)
        0.011500,    # l13 (SPDC lens3 -> WG 798)
    ])

    print(f"\nInitial objective: {objective(x0, wg_modes):.6f}")

    # Multi-start + differential evolution for SPDC-sensitive parameters
    print("\n" + "=" * 60)
    print("Phase 1: Local multi-start (all free params)...")
    print("=" * 60)

    free = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    bounds_free = [BOUNDS[i] for i in free]

    best_x = x0.copy()
    best_obj = objective(x0, wg_modes)

    for seed_jitter in [0, 1, 2, 3, 4, 5, 6, 7]:
        np.random.seed(seed_jitter * 31)
        x_init = x0.copy()
        for i in free:
            lo, hi = BOUNDS[i]
            x_init[i] = np.clip(x_init[i] + np.random.normal(0, (hi-lo)*0.05), lo, hi)

        def obj_free(xf, x_full):
            xc = x_full.copy()
            for ii, v in zip(free, xf): xc[ii] = v
            return objective(xc, wg_modes)

        res = minimize(
            obj_free, x_init[free], args=(x_init,),
            method='L-BFGS-B', bounds=bounds_free,
            options={'maxiter': 5000, 'ftol': 1e-14},
        )

        xc = x_init.copy()
        for ii, v in zip(free, res.x): xc[ii] = v
        obj_val = objective(xc, wg_modes)

        if obj_val < best_obj:
            best_obj = obj_val
            best_x = xc

        print(f"  Seed {seed_jitter}: obj = {obj_val:.6f}")

    # Phase 2: DE for SPDC-critical parameters (l8, l11, l12, l13) with rest fixed
    print("\n" + "=" * 60)
    print("Phase 2: DE on SPDC-critical params (l8, l10, l11, l12, l13)...")
    print("=" * 60)

    spdc_free = [6, 10, 11, 12, 13]  # l8, l10, l11, l12, l13
    spdc_bounds = [BOUNDS[i] for i in spdc_free]

    def obj_spdc(x_spdc, x_base):
        xc = x_base.copy()
        for ii, v in zip(spdc_free, x_spdc): xc[ii] = v
        return objective(xc, wg_modes)

    de_result = differential_evolution(
        obj_spdc, spdc_bounds,
        args=(best_x,),
        seed=42, maxiter=100, popsize=30,
        mutation=(0.5, 1.5), recombination=0.7,
        disp=True, polish=False,
    )

    x_de = best_x.copy()
    for ii, v in zip(spdc_free, de_result.x): x_de[ii] = v

    # Polish
    def obj_spdc_free(xf):
        return obj_spdc(xf, x_de)

    de_polish = minimize(
        obj_spdc_free, de_result.x,
        method='L-BFGS-B', bounds=spdc_bounds,
        options={'maxiter': 5000, 'ftol': 1e-14},
    )

    for ii, v in zip(spdc_free, de_polish.x): x_de[ii] = v

    obj_de = objective(x_de, wg_modes)
    if obj_de < best_obj:
        best_obj = obj_de
        best_x = x_de

    print(f"  DE + polish obj: {obj_de:.6f}")

    # Final evaluation with fine grid
    print("\n" + "=" * 60)
    print("Final evaluation (fine grid)...")
    print("=" * 60)

    wg_modes_fine = {}
    for name, wl in WAVELENGTHS.items():
        E = waveguide_mode_field(wl, WG_THICKNESS, WG_WIDTH, TEMPERATURE, 0.1e-6)
        wg_modes_fine[name] = {'E': E, 'dL': 0.1e-6}
        wx, wy = waveguide_mode_size(E, 0.1e-6, WG_WIDTH, WG_THICKNESS)
        print(f"  {name} nm mode: {wx*1e6:.2f} x {wy*1e6:.2f} um")

    results = build_and_evaluate(best_x, wg_modes_fine)

    print("\nCoupling efficiencies:")
    for setup in ['DFG', 'SPDC']:
        print(f"\n  {setup}:")
        for wl in ['2070', '1299', '798']:
            eta = results[setup][wl]
            bar = '#' * int(eta * 50) + '-' * (50 - int(eta * 50))
            status = "OK" if ((setup == 'DFG' and eta >= 0.98) or
                              (setup == 'SPDC' and eta >= 0.90)) else "FAIL"
            print(f"    {wl} nm: {eta*100:6.2f}%  [{bar}] {status}")

    # Parameter table
    print(f"\n{'='*60}")
    print("Final parameters (mm)")
    print(f"{'='*60}")
    param_names = [
        "l2  (collim -> lens1)",
        "f1  (lens-1 FL, standard)",
        "l3  (lens1 -> lens2)",
        "f2  (lens-2 FL, standard)",
        "l14 (lens2 -> lens3, shared 2070)",
        "l6  (lens4 -> lens3, shared 798)",
        "l8  (lens5 -> lens6, shared 1299)",
        "l4  (DFG lens3 -> WG)",
        "l5  (DFG fiber798 -> lens4)",
        "l7  (DFG fiber1299 -> lens5)",
        "l9  (DFG lens6 -> WG)",
        "l10 (SPDC lens3 -> WG, 2070/1299)",
        "l11 (SPDC fiber1299 -> lens5)",
        "l12 (SPDC fiber798 -> lens4)",
        "l13 (SPDC lens3 -> WG, 798)",
    ]
    for name, val in zip(param_names, best_x):
        print(f"  {name:35s} = {val*1e3:8.3f}")

    # Element-to-element distances for setup documentation
    print(f"\n{'='*60}")
    print("Element spacing table")
    print(f"{'='*60}")

    # Fixed elements
    f_coll = 5.91e-3  # F028APC-2000 collimator focal length

    dfg_spacing = {
        'DFG 2070': [
            ('Collimator (F028APC-2000)', 0, f_coll, best_x[IDX_L2]),
            ('Lens-1 (f1)', best_x[IDX_F1], best_x[IDX_L2], best_x[IDX_L2] + best_x[IDX_L3]),
            ('Lens-2 (f2)', best_x[IDX_F2], best_x[IDX_L2] + best_x[IDX_L3], best_x[IDX_L2]+best_x[IDX_L3]+best_x[IDX_L14]),
            ('Lens-3 (A220TM)', get_focal_length('A220TM', 2070e-9), None, best_x[IDX_L2]+best_x[IDX_L3]+best_x[IDX_L14]+best_x[IDX_L4_DFG]),
            ('WG facet', None, None, None),
        ]
    }

    print("\n  DFG 2070 nm path:")
    cum_dist = 0
    elements = [
        ("Collimator", 0),
        ("Lens-1", best_x[IDX_L2]),
        ("Lens-2", best_x[IDX_L2] + best_x[IDX_L3]),
        ("Lens-3 (A220TM)", best_x[IDX_L2] + best_x[IDX_L3] + best_x[IDX_L14]),
        ("WG facet", best_x[IDX_L2] + best_x[IDX_L3] + best_x[IDX_L14] + best_x[IDX_L4_DFG]),
    ]
    for i in range(len(elements) - 1):
        name1, pos1 = elements[i]
        name2, pos2 = elements[i+1]
        gap = pos2 - pos1
        print(f"    {name1:20s} -> {name2:20s}: {gap*1e3:8.3f} mm  (at {pos2*1e3:8.1f} mm)")

    print("\n  DFG 798 nm path:")
    elements = [
        ("Fiber (PM780)", 0),
        ("Lens-4 (A220TM-B)", best_x[IDX_L5_DFG]),
        ("Combiner/Dichroic", best_x[IDX_L5_DFG] + best_x[IDX_L6]),
        ("Lens-3 (A220TM)", best_x[IDX_L5_DFG] + best_x[IDX_L6]),
        ("WG facet", best_x[IDX_L5_DFG] + best_x[IDX_L6] + best_x[IDX_L4_DFG]),
    ]
    for i in range(len(elements) - 1):
        name1, pos1 = elements[i]
        name2, pos2 = elements[i+1]
        gap = pos2 - pos1
        print(f"    {name1:20s} -> {name2:20s}: {gap*1e3:8.3f} mm  (at {pos2*1e3:8.1f} mm)")

    print("\n  DFG 1299 nm path:")
    elements = [
        ("Fiber (PM1550)", 0),
        ("Lens-5 (C560TME-C)", best_x[IDX_L7_DFG]),
        ("Lens-6 (A220TM)", best_x[IDX_L7_DFG] + best_x[IDX_L8]),
        ("WG facet", best_x[IDX_L7_DFG] + best_x[IDX_L8] + best_x[IDX_L9_DFG]),
    ]
    for i in range(len(elements) - 1):
        name1, pos1 = elements[i]
        name2, pos2 = elements[i+1]
        gap = pos2 - pos1
        print(f"    {name1:20s} -> {name2:20s}: {gap*1e3:8.3f} mm  (at {pos2*1e3:8.1f} mm)")

    print("\n  SPDC 2070 nm path:")
    elements = [
        ("Collimator", 0),
        ("Lens-1", best_x[IDX_L2]),
        ("Lens-2", best_x[IDX_L2] + best_x[IDX_L3]),
        ("Lens-3 (A220TM)", best_x[IDX_L2] + best_x[IDX_L3] + best_x[IDX_L14]),
        ("WG facet", best_x[IDX_L2] + best_x[IDX_L3] + best_x[IDX_L14] + best_x[IDX_L10_SPDC]),
    ]
    for i in range(len(elements) - 1):
        name1, pos1 = elements[i]
        name2, pos2 = elements[i+1]
        gap = pos2 - pos1
        print(f"    {name1:20s} -> {name2:20s}: {gap*1e3:8.3f} mm  (at {pos2*1e3:8.1f} mm)")

    print("\n  SPDC 1299 nm path:")
    elements = [
        ("Fiber (PM1550)", 0),
        ("Lens-5 (C560TME-C)", best_x[IDX_L11_SPDC]),
        ("Lens-6 (A220TM)", best_x[IDX_L11_SPDC] + best_x[IDX_L8]),
        ("WG facet", best_x[IDX_L11_SPDC] + best_x[IDX_L8] + best_x[IDX_L10_SPDC]),
    ]
    for i in range(len(elements) - 1):
        name1, pos1 = elements[i]
        name2, pos2 = elements[i+1]
        gap = pos2 - pos1
        print(f"    {name1:20s} -> {name2:20s}: {gap*1e3:8.3f} mm  (at {pos2*1e3:8.1f} mm)")

    print("\n  SPDC 798 nm path:")
    elements = [
        ("Fiber (PM780)", 0),
        ("Lens-4 (A220TM-B)", best_x[IDX_L12_SPDC]),
        ("Combiner/Dichroic", best_x[IDX_L12_SPDC] + best_x[IDX_L6]),
        ("Lens-3 (A220TM)", best_x[IDX_L12_SPDC] + best_x[IDX_L6]),
        ("WG facet", best_x[IDX_L12_SPDC] + best_x[IDX_L6] + best_x[IDX_L13_SPDC]),
    ]
    for i in range(len(elements) - 1):
        name1, pos1 = elements[i]
        name2, pos2 = elements[i+1]
        gap = pos2 - pos1
        print(f"    {name1:20s} -> {name2:20s}: {gap*1e3:8.3f} mm  (at {pos2*1e3:8.1f} mm)")
