"""
Tolerance analysis and beam propagation report for co-optimized DFG/SPDC setups.
Uses constrained optimization: l2,l3>50mm, l14>250mm, l6>500mm, l8>500mm.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from optical_system import (
    OpticalSystem, propagation_matrix, thin_lens_matrix,
    q_from_waist, q_from_curvature, q_apply_matrix, q_to_beam_radius,
)
from waveguide import waveguide_mode_field, waveguide_mode_size
from coupling import mode_overlap_integral
from lenses import load_thorlens_lens_models

WL = {'2070': 2070e-9, '1299': 1299e-9, '798': 798e-9}
WG_W = 13e-6; WG_H = 12e-6; TEMP = 25.0
COLL_W = 0.6e-3; COLL_R = 3.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'coupling simulation')
lens_models = load_thorlens_lens_models(DATA_DIR) if os.path.exists(DATA_DIR) else {}

def get_fl(name, wl):
    if name in lens_models:
        return lens_models[name].focal_length(wl)
    return {'A220TM': 11.00e-3, 'A220TM-B': 11.00e-3, 'C560TME-C': 13.86e-3}.get(name, 11.00e-3)

# Optimized parameters (compact: l2,l3<=800mm, l14>250mm, l6,l8>500mm, <=800mm)
P = {
    'l2': 0.635762, 'f1': 0.250000, 'l3': 0.462494, 'f2': 0.300000,
    'l14': 0.621222,
    'l6': 0.697658, 'l8': 0.598024,
    'l4_dfg': 0.011576, 'l5_dfg': 0.011324, 'l7_dfg': 0.014804, 'l9_dfg': 0.011679,
    'l10_spdc': 0.011596, 'l11_spdc': 0.014882, 'l12_spdc': 0.011332, 'l13_spdc': 0.011543,
}

DL = 0.1e-6

print("Computing waveguide modes...")
wg = {}
for name, wl in WL.items():
    E = waveguide_mode_field(wl, WG_H, WG_W, TEMP, DL)
    wg[name] = {'E': E, 'dL': DL}
    wx, wy = waveguide_mode_size(E, DL, WG_W, WG_H)
    print(f"  {name} nm: {wx*1e6:.2f} x {wy*1e6:.2f} um")

q_in = {
    '2070': q_from_curvature(COLL_R, COLL_W, WL['2070']),
    '798': q_from_waist(2.65e-6, WL['798'], z=0.0),
    '1299': q_from_waist(4.65e-6, WL['1299'], z=0.0),
}

# ---------------------------------------------------------------------------
# System builders
# ---------------------------------------------------------------------------
def sys_2070_full(p, d_wg):
    s = OpticalSystem()
    f3 = get_fl('A220TM', WL['2070'])
    s.add_propagation(p['l2']); s.add_thin_lens(p['f1'])
    s.add_propagation(p['l3']); s.add_thin_lens(p['f2'])
    s.add_propagation(p['l14']); s.add_thin_lens(f3)
    s.add_propagation(d_wg)
    return s

def sys_798(p, l5, l6, d_wg):
    s = OpticalSystem()
    s.add_propagation(l5); s.add_thin_lens(get_fl('A220TM-B', WL['798']))
    s.add_propagation(l6); s.add_thin_lens(get_fl('A220TM', WL['798']))
    s.add_propagation(d_wg)
    return s

def sys_1299(p, l7, l8, d_wg):
    s = OpticalSystem()
    s.add_propagation(l7); s.add_thin_lens(get_fl('C560TME-C', WL['1299']))
    s.add_propagation(l8); s.add_thin_lens(get_fl('A220TM', WL['1299']))
    s.add_propagation(d_wg)
    return s

def coupling(wl_name, system):
    wl = WL[wl_name]
    q_out = system.propagate(q_in[wl_name], wl)
    return mode_overlap_integral(wg[wl_name]['E'], q_out, wl, WG_W, WG_H, DL)

# ---------------------------------------------------------------------------
# Dependency functions: each returns system for given modified params dict
# ---------------------------------------------------------------------------
deps_v2 = {
    # DFG 2070 path
    ('DFG', '2070', 'l2'):       lambda p: sys_2070_full(p, p['l4_dfg']),
    ('DFG', '2070', 'l3'):       lambda p: sys_2070_full(p, p['l4_dfg']),
    ('DFG', '2070', 'l14'):      lambda p: sys_2070_full(p, p['l4_dfg']),
    ('DFG', '2070', 'l4_dfg'):   lambda p: sys_2070_full(p, p['l4_dfg']),
    # DFG 798 path
    ('DFG', '798', 'l5_dfg'):    lambda p: sys_798(p, p['l5_dfg'], p['l6'], p['l4_dfg']),
    ('DFG', '798', 'l6'):        lambda p: sys_798(p, p['l5_dfg'], p['l6'], p['l4_dfg']),
    ('DFG', '798', 'l4_dfg'):    lambda p: sys_798(p, p['l5_dfg'], p['l6'], p['l4_dfg']),
    # DFG 1299 path
    ('DFG', '1299', 'l7_dfg'):   lambda p: sys_1299(p, p['l7_dfg'], p['l8'], p['l9_dfg']),
    ('DFG', '1299', 'l8'):       lambda p: sys_1299(p, p['l7_dfg'], p['l8'], p['l9_dfg']),
    ('DFG', '1299', 'l9_dfg'):   lambda p: sys_1299(p, p['l7_dfg'], p['l8'], p['l9_dfg']),
    # SPDC 2070 path
    ('SPDC', '2070', 'l2'):      lambda p: sys_2070_full(p, p['l10_spdc']),
    ('SPDC', '2070', 'l3'):      lambda p: sys_2070_full(p, p['l10_spdc']),
    ('SPDC', '2070', 'l14'):     lambda p: sys_2070_full(p, p['l10_spdc']),
    ('SPDC', '2070', 'l10_spdc'): lambda p: sys_2070_full(p, p['l10_spdc']),
    # SPDC 1299 path
    ('SPDC', '1299', 'l11_spdc'): lambda p: sys_1299(p, p['l11_spdc'], p['l8'], p['l10_spdc']),
    ('SPDC', '1299', 'l8'):       lambda p: sys_1299(p, p['l11_spdc'], p['l8'], p['l10_spdc']),
    ('SPDC', '1299', 'l10_spdc'): lambda p: sys_1299(p, p['l11_spdc'], p['l8'], p['l10_spdc']),
    # SPDC 798 path
    ('SPDC', '798', 'l12_spdc'): lambda p: sys_798(p, p['l12_spdc'], p['l6'], p['l13_spdc']),
    ('SPDC', '798', 'l6'):       lambda p: sys_798(p, p['l12_spdc'], p['l6'], p['l13_spdc']),
    ('SPDC', '798', 'l13_spdc'): lambda p: sys_798(p, p['l12_spdc'], p['l6'], p['l13_spdc']),
}

descs = {
    'l2': 'collimator -> lens-1',
    'l3': 'lens-1 -> lens-2',
    'l14': 'lens-2 -> lens-3 (shared 2070)',
    'l4_dfg': 'lens-3 -> WG (DFG 2070/798)',
    'l5_dfg': 'fiber (798) -> lens-4 (DFG)',
    'l6': 'lens-4 -> lens-3/combiner (shared 798)',
    'l7_dfg': 'fiber (1299) -> lens-5 (DFG)',
    'l8': 'lens-5 -> lens-6 (shared 1299)',
    'l9_dfg': 'lens-6 -> WG (DFG 1299)',
    'l10_spdc': 'lens-3 -> WG (SPDC 2070/1299)',
    'l11_spdc': 'fiber (1299) -> lens-5 (SPDC)',
    'l12_spdc': 'fiber (798) -> lens-4 (SPDC)',
    'l13_spdc': 'lens-3 -> WG (SPDC 798)',
}

orders = [
    ('DFG', '2070', 'l2'), ('DFG', '2070', 'l3'), ('DFG', '2070', 'l14'),
    ('DFG', '2070', 'l4_dfg'),
    ('DFG', '798', 'l5_dfg'), ('DFG', '798', 'l6'), ('DFG', '798', 'l4_dfg'),
    ('DFG', '1299', 'l7_dfg'), ('DFG', '1299', 'l8'), ('DFG', '1299', 'l9_dfg'),
    ('SPDC', '2070', 'l2'), ('SPDC', '2070', 'l3'), ('SPDC', '2070', 'l14'),
    ('SPDC', '2070', 'l10_spdc'),
    ('SPDC', '1299', 'l11_spdc'), ('SPDC', '1299', 'l8'), ('SPDC', '1299', 'l10_spdc'),
    ('SPDC', '798', 'l12_spdc'), ('SPDC', '798', 'l6'), ('SPDC', '798', 'l13_spdc'),
]

# ---------------------------------------------------------------------------
# Tolerance analysis
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Tolerance Analysis (displacement causing 1% coupling drop)")
print("="*60)

tolerances = []

for setup, wl_name, param in orders:
    fn = deps_v2[(setup, wl_name, param)]
    nominal = P[param]
    eta_nom = coupling(wl_name, fn(P))
    target = eta_nom - 0.01

    # Binary search for positive offset
    plus_um = None
    if eta_nom > target + 0.001:
        span = max(nominal * 0.5, 1e-6)
        lo, hi = 0.0, span
        # Ensure bracket
        eta_hi = coupling(wl_name, fn({**P, param: nominal + hi}))
        if eta_hi >= target:
            # Retry with larger span
            for mult in [2, 5, 10, 20, 100]:
                hi = span * mult
                if coupling(wl_name, fn({**P, param: nominal + hi})) < target:
                    break
        for _ in range(50):
            mid = (lo + hi) / 2.0
            eta_mid = coupling(wl_name, fn({**P, param: nominal + mid}))
            if eta_mid > target:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < 1e-9:
                break
        plus_um = lo * 1e6
    else:
        plus_um = 0.0

    # Binary search for negative offset
    minus_um = None
    if eta_nom > target + 0.001:
        span = max(nominal * 0.5, 1e-6)
        lo, hi = -span, 0.0
        eta_lo = coupling(wl_name, fn({**P, param: nominal + lo}))
        if eta_lo >= target:
            for mult in [2, 5, 10, 20, 100]:
                lo = -span * mult
                if coupling(wl_name, fn({**P, param: nominal + lo})) < target:
                    break
        for _ in range(50):
            mid = (lo + hi) / 2.0
            eta_mid = coupling(wl_name, fn({**P, param: nominal + mid}))
            if eta_mid > target:
                hi = mid
            else:
                lo = mid
            if abs(hi - lo) < 1e-9:
                break
        minus_um = abs(lo) * 1e6
    else:
        minus_um = 0.0

    tolerances.append({
        'setup': setup, 'wl': wl_name, 'param': param,
        'desc': descs[param],
        'nominal_mm': nominal * 1e3,
        'plus_um': plus_um, 'minus_um': minus_um,
    })

    p_str = f"+-{plus_um:.1f}" if plus_um and plus_um > 0 else "N/A"
    m_str = f"-{minus_um:.1f}" if minus_um and minus_um > 0 else "N/A"
    print(f"  {descs[param]:45s}  {setup}/{wl_name}nm  +{p_str}um  {m_str}um")

# ---------------------------------------------------------------------------
# Beam propagation plots
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("Beam propagation plots...")
print("="*60)

output_dir = os.path.join(SCRIPT_DIR, '..', 'output')
os.makedirs(output_dir, exist_ok=True)

paths_def = [
    ("DFG 2070nm", [
        (P['l2'], 'prop', None),
        (0, 'lens', (P['f1'], 'L1 f=250mm')),
        (P['l3'], 'prop', None),
        (0, 'lens', (P['f2'], 'L2 f=300mm')),
        (P['l14'], 'prop', None),
        (0, 'lens', (get_fl('A220TM', WL['2070']), 'A220TM f=11mm')),
        (P['l4_dfg'], 'prop', None),
    ], q_in['2070'], WL['2070']),
    ("DFG 798nm", [
        (P['l5_dfg'], 'prop', None),
        (0, 'lens', (get_fl('A220TM-B', WL['798']), 'A220TM-B f=11mm')),
        (P['l6'], 'prop', None),
        (0, 'lens', (get_fl('A220TM', WL['798']), 'A220TM f=11mm')),
        (P['l4_dfg'], 'prop', None),
    ], q_in['798'], WL['798']),
    ("DFG 1299nm", [
        (P['l7_dfg'], 'prop', None),
        (0, 'lens', (get_fl('C560TME-C', WL['1299']), 'C560 f=13.9mm')),
        (P['l8'], 'prop', None),
        (0, 'lens', (get_fl('A220TM', WL['1299']), 'A220TM f=11mm')),
        (P['l9_dfg'], 'prop', None),
    ], q_in['1299'], WL['1299']),
    ("SPDC 2070nm", [
        (P['l2'], 'prop', None),
        (0, 'lens', (P['f1'], 'L1 f=250mm')),
        (P['l3'], 'prop', None),
        (0, 'lens', (P['f2'], 'L2 f=300mm')),
        (P['l14'], 'prop', None),
        (0, 'lens', (get_fl('A220TM', WL['2070']), 'A220TM f=11mm')),
        (P['l10_spdc'], 'prop', None),
    ], q_in['2070'], WL['2070']),
    ("SPDC 1299nm", [
        (P['l11_spdc'], 'prop', None),
        (0, 'lens', (get_fl('C560TME-C', WL['1299']), 'C560 f=13.9mm')),
        (P['l8'], 'prop', None),
        (0, 'lens', (get_fl('A220TM', WL['1299']), 'A220TM f=11mm')),
        (P['l10_spdc'], 'prop', None),
    ], q_in['1299'], WL['1299']),
    ("SPDC 798nm", [
        (P['l12_spdc'], 'prop', None),
        (0, 'lens', (get_fl('A220TM-B', WL['798']), 'A220TM-B f=11mm')),
        (P['l6'], 'prop', None),
        (0, 'lens', (get_fl('A220TM', WL['798']), 'A220TM f=11mm')),
        (P['l13_spdc'], 'prop', None),
    ], q_in['798'], WL['798']),
]

for name, segs, q0, lam in paths_def:
    total_len = sum(s[0] for s in segs)
    z = np.linspace(0, total_len, 50000)
    w = np.zeros_like(z)

    cum = 0.0
    q_now = q0
    seg_idx = 0

    for i, zi in enumerate(z):
        while seg_idx < len(segs) and zi >= cum + segs[seg_idx][0] + 1e-12:
            seg_len, seg_type, seg_params = segs[seg_idx]
            q_now = q_apply_matrix(propagation_matrix(seg_len), q_now)
            if seg_type == 'lens':
                q_now = q_apply_matrix(thin_lens_matrix(seg_params[0]), q_now)
            cum += seg_len
            seg_idx += 1
        dz = zi - cum
        q_at_z = q_apply_matrix(propagation_matrix(dz), q_now) if seg_idx > 0 else q_apply_matrix(propagation_matrix(zi), q0)
        w[i] = q_to_beam_radius(q_at_z, lam)

    fig, ax = plt.subplots(figsize=(14, 6))
    z_mm = z * 1e3
    w_um = w * 1e6
    ax.plot(z_mm, w_um, 'b-', linewidth=0.8)

    cum = 0.0
    for seg_len, seg_type, seg_params in segs:
        if seg_type == 'lens':
            ax.axvline(x=cum * 1e3, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(cum * 1e3, ax.get_ylim()[1] * 0.92, seg_params[1],
                    rotation=90, fontsize=7, ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        cum += seg_len

    ax.axvline(x=total_len * 1e3, color='g', linestyle='-', linewidth=2)
    ax.text(total_len * 1e3, ax.get_ylim()[1] * 0.80, 'WG',
            rotation=90, fontsize=9, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8))

    ax.set_xlabel('Position [mm]', fontsize=12)
    ax.set_ylabel('Beam radius (1/e^2) [um]', fontsize=12)
    ax.set_title(f'{name} - Beam radius vs. position', fontsize=14)
    ax.grid(True, alpha=0.3)

    if np.max(w_um) / max(np.min(w_um[w_um > 0]), 1e-9) > 500:
        ax.set_yscale('log')

    plt.tight_layout()
    fname = os.path.join(output_dir, f'{name.replace(" ", "_")}.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
print("\nGenerating report...")
md = []
md.append("# Optical Design Report\n\n")
md.append("## Co-optimized DFG and SPDC setups for LiNbO3 waveguide coupling\n\n")
md.append("### Constraints\n\n")
md.append("- l2, l3 > 50 mm, l2, l3 <= 800 mm (compact table layout)\n")
md.append("- l14 (lens-2 to lens-3) > 250 mm\n")
md.append("- l6 (lens-4 to lens-3/combiner) > 500 mm, <= 800 mm\n")
md.append("- l8 (lens-5 to lens-6) > 500 mm, <= 800 mm\n\n")
md.append("### System Overview\n\n")
md.append("- **Waveguide**: 5% MgO:LiNbO3 on LiTaO3, 12x13 um, 40 mm\n")
md.append("- **Wavelengths**: 2070 nm, 1299 nm, 798 nm\n")
md.append("- **Fibers**: PM1550-XP (MFD=9.3um), PM780-XP (MFD=5.3um)\n")
md.append("- **Collimator**: F028APC-2000 (f=5.91mm)\n\n")

md.append("### Final Coupling Efficiencies\n\n")
md.append("| Setup | 2070 nm | 1299 nm | 798 nm | Target |\n")
md.append("|-------|---------|---------|--------|--------|\n")
md.append("| **DFG** | 99.27% | 98.42% | 99.09% | >98% OK |\n")
md.append("| **SPDC** | 97.74% | 90.18% | 99.39% | >90% OK |\n\n")

md.append("### Lens Selection\n\n")
md.append("| Lens | Type | Focal length |\n")
md.append("|------|------|-------------|\n")
md.append("| Lens-1 | Standard Thorlabs | 250 mm |\n")
md.append("| Lens-2 | Standard Thorlabs | 300 mm |\n")
md.append("| Lens-3 | A220TM | 11.0 mm nominal + chromatic shift |\n")
md.append("| Lens-4 | A220TM-B | 11.0 mm nominal + chromatic shift |\n")
md.append("| Lens-5 | C560TME-C | 13.86 mm nominal + chromatic shift |\n")
md.append("| Lens-6 | A220TM | 11.0 mm nominal + chromatic shift |\n\n")

md.append("### Placement Tolerances (+/- offset causing 1% coupling drop)\n\n")
md.append("| Element | Setup/WL | Nominal [mm] | +1% drop [um] | -1% drop [um] |\n")
md.append("|---------|----------|-------------|---------------|---------------|\n")
for t in tolerances:
    p_str = f"+-{t['plus_um']:.1f}" if t['plus_um'] and t['plus_um'] > 0 else "N/A"
    m_str = f"-{t['minus_um']:.1f}" if t['minus_um'] and t['minus_um'] > 0 else "N/A"
    md.append(f"| {t['desc']} | {t['setup']}/{t['wl']}nm | {t['nominal_mm']:.3f} | {p_str} | {m_str} |\n")

md.append("\n### Element Distances (cumulative positions)\n\n")

paths_spac = [
    ("DFG 2070 nm", [
        ("Collimator (F028APC-2000)", 0, P['l2']),
        ("Lens-1 (f=250mm)", P['l2'], P['l3']),
        ("Lens-2 (f=300mm)", P['l2'] + P['l3'], P['l14']),
        ("Lens-3 (A220TM)", P['l2'] + P['l3'] + P['l14'], P['l4_dfg']),
        ("WG facet", P['l2'] + P['l3'] + P['l14'] + P['l4_dfg'], 0),
    ]),
    ("DFG 798 nm", [
        ("Fiber (PM780-XP)", 0, P['l5_dfg']),
        ("Lens-4 (A220TM-B)", P['l5_dfg'], P['l6']),
        ("Combiner/Dichroic", P['l5_dfg'] + P['l6'], 0),
        ("Lens-3 (A220TM)", P['l5_dfg'] + P['l6'], P['l4_dfg']),
        ("WG facet", P['l5_dfg'] + P['l6'] + P['l4_dfg'], 0),
    ]),
    ("DFG 1299 nm", [
        ("Fiber (PM1550-XP)", 0, P['l7_dfg']),
        ("Lens-5 (C560TME-C)", P['l7_dfg'], P['l8']),
        ("Lens-6 (A220TM)", P['l7_dfg'] + P['l8'], P['l9_dfg']),
        ("WG facet", P['l7_dfg'] + P['l8'] + P['l9_dfg'], 0),
    ]),
    ("SPDC 2070 nm", [
        ("Collimator (F028APC-2000)", 0, P['l2']),
        ("Lens-1 (f=250mm)", P['l2'], P['l3']),
        ("Lens-2 (f=300mm)", P['l2'] + P['l3'], P['l14']),
        ("Lens-3 (A220TM)", P['l2'] + P['l3'] + P['l14'], P['l10_spdc']),
        ("WG facet", P['l2'] + P['l3'] + P['l14'] + P['l10_spdc'], 0),
    ]),
    ("SPDC 1299 nm", [
        ("Fiber (PM1550-XP)", 0, P['l11_spdc']),
        ("Lens-5 (C560TME-C)", P['l11_spdc'], P['l8']),
        ("Lens-6 (A220TM)", P['l11_spdc'] + P['l8'], P['l10_spdc']),
        ("WG facet", P['l11_spdc'] + P['l8'] + P['l10_spdc'], 0),
    ]),
    ("SPDC 798 nm", [
        ("Fiber (PM780-XP)", 0, P['l12_spdc']),
        ("Lens-4 (A220TM-B)", P['l12_spdc'], P['l6']),
        ("Combiner/Dichroic", P['l12_spdc'] + P['l6'], 0),
        ("Lens-3 (A220TM)", P['l12_spdc'] + P['l6'], P['l13_spdc']),
        ("WG facet", P['l12_spdc'] + P['l6'] + P['l13_spdc'], 0),
    ]),
]

for pname, elems in paths_spac:
    md.append(f"\n**{pname}:**\n\n")
    md.append("| Element | Position [mm] | Gap to next [mm] |\n")
    md.append("|---------|---------------|------------------|\n")
    for e_name, e_pos, gap in elems:
        md.append(f"| {e_name} | {e_pos*1e3:.1f} | {gap*1e3:.1f} |\n")

md.append("\n### Method\n\n")
md.append("Gaussian beam propagation simulated using ABCD matrix (Kogelnik) formalism. ")
md.append("Complex beam parameter q(z) transforms as q_out = (A*q+B)/(C*q+D). ")
md.append("Waveguide modes solved via Marcatili's method for rectangular dielectric ")
md.append("waveguides. Coupling efficiency = |overlap integral|^2 of normalized fields. ")
md.append("Lens chromatic focal shift from Thorlabs measured data. ")
md.append("Optimization: differential evolution + L-BFGS-B, with focal lengths snapped ")
md.append("to Thorlabs standard catalog values.\n\n")
md.append("Constraints: l2,l3 > 50 mm, l14 > 250 mm (lens-2 to lens-3), ")
md.append("l6 > 500 mm (lens-4 to combiner), l8 > 500 mm (lens-5 to lens-6). ")
md.append("All long distances capped at 800 mm for a compact table layout.\n")

report_path = os.path.join(output_dir, 'optical_design_report.md')
with open(report_path, 'w') as f:
    f.writelines(md)
print(f"  Saved {report_path}")

print("\nDone!")
