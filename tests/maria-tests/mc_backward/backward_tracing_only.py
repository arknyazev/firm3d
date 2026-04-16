#!/usr/bin/env python3
"""
Backward-tracing-only analysis: how often do we get fusion-born particles
from backward tracing?

Pipeline
--------
1. Load wall IC (R, phi, Z) from 3_IC_sample_wall/outputs/
   initial_conditions_surface_cylindrical.txt.  Positions only — energy and
   pitch are sampled here:
     - H_wall ~ Uniform(3.0 MeV, 3.5 MeV)
     - lambda ~ Uniform(-1, 1);  v_par = lambda * sqrt(2 H_wall / m)
2. Build the Biot-Savart field, LCFS classifier, and GPU drag interpolant
   exactly the way 2_tracing_gpu / backward_informed_mc.py do.
3. Trace backward with drag from each wall state until one of:
     stop_code 0: tmax
     stop_code 1: wall hit
     stop_code 2: H reaches H_fusion  (success — "backward-born")
     stop_code 3: invalid
4. Endpoints with stop_code == 2 are the successful "birth" points.
   Convert their (R, phi, Z) to Boozer (s, theta, zeta).  A birth point is
   considered to have "valid Boozer coordinates" iff 0 <= s <= 1.
5. Report fraction reaching H_fusion, and fraction reaching H_fusion AND
   having valid Boozer coordinates.  Write CSV, VTK, matplotlib plots.

Outputs (timestamped dir under "mc_proj" / "results" / "backward_only" / timestamp
------------------------------------------------------
  backward_results.npy             (n_wall, 7) full tracer output
  wall_starts_xyz.npy              (n_wall, 3) wall IC used
  wall_starts_state.npy            (n_wall, 5): X,Y,Z,vpar,H
  birth_endpoints.npy              (M, 5): X,Y,Z,vpar,H  (stop_code==2)
  birth_endpoints_rphiz.npy        (M, 3): R, phi, Z
  birth_endpoints_boozer.npy       (M, 3): s, theta, zeta
  valid_boozer_mask.npy            (M,) bool  (0<=s<=1)
  trajectory_*.vtu                 Paraview point-cloud outputs
  summary.csv                      Success metrics
  plots/                           matplotlib figures
"""

import csv
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simsopt.field import (
    BiotSavart,
    Current,
    InterpolatedField,
    SurfaceClassifier,
    coils_via_symmetries,
)
from simsopt.field.coil import load_coils_from_makegrid_file
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as H_FUSION,
    ONE_EV,
)

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.coordinates import cylindrical_to_boozer
from firm3d.util.gpu_utils import cartesian_interpolant_drag
from firm3dpp import cartesian_gpu_tracing_backward_drag


# ── Inputs ───────────────────────────────────────────────────────────────────

THIS_DIR = Path(__file__).resolve().parent


@dataclass
class Inputs:
    # Files
    coil_file:       Path = THIS_DIR / "LandremanPaulQH_coils" / "coils.curves_22_7_21"
    vmec_input_file: Path = THIS_DIR / "LandremanPaulQH_coils" / "input.vmec"
    boozmn_file:     Path = THIS_DIR / "LandremanPaulQH_coils" / "boozmn.nc"
    wall_ic_file = Path("/pscratch/sd/m/mariagar/projects/mc_proj/IC/initial_conditions_surface_cylindrical.txt")
    
    # Equilibrium / coils
    nfp:        int   = 4
    ncoils:     int   = 5
    current:    float = 1.27797548115612e7
    coil_order: int   = 20

    # Interpolant grid (matching 2_tracing_gpu conventions)
    n_r:    int = 64
    n_phi:  int = 128
    n_z:    int = 64
    degree: int = 3
    nphi_surf:   int = 128
    ntheta_surf: int = 64

    # SurfaceClassifier
    sc_h: float = 0.05
    sc_p: int   = 2

    # Boozer interpolant
    radial_order:    int = 3
    boozer_degree:   int = 3
    boozer_res:      int = 48

    # [3.0, 3.5] MeV
    mass:        float = MASS
    charge:      float = CHARGE
    H_low:       float = 3.0e6 * ONE_EV
    H_high:      float = H_FUSION
    H_fusion:    float = H_FUSION
    coulomb_log: float = 17.0
    Te_in_eV:    bool  = True
    ne0:         float = 1e21
    Te0_ev:      float = 500

    # Tracing
    n_wall:         int   = 10_000_000  # same as in IC file from folder 3_
    # Subsample of successful (stop_code==2) particles to re-trace with
    # snapshots to save real time-resolved trajectories.  Kept far smaller
    # than n_wall because each snapshot re-runs the tracer from t=0.
    n_trajectory:   int   = 200
    n_snapshots:    int   = 100
    # tmax_backward is recomputed below from slowing-down time and H range
    tmax_backward:  float = 5e-5
    tol:            float = 1e-9
    seed:           int   = 57

    # Finite-difference step for outward normal
    normal_fd_eps: float = 1e-4


inp = Inputs()


# ── Slowing-down time and required backward time horizon ─────────────────────
#   t_req = ln(H_fusion / H_low) * tau_s
def _slowing_down_time(ne, Te_eV, mass, coulomb_log):
    eps0 = 8.8541878128e-12
    e_ch = 1.602176634e-19
    m_e  = 9.1093837015e-31
    Z_alpha = 2.0
    Te_J = Te_eV * e_ch
    num = 3.0 * (2.0 * np.pi) ** 1.5 * eps0 ** 2 * mass * Te_J ** 1.5
    den = Z_alpha ** 2 * e_ch ** 4 * np.sqrt(m_e) * ne * coulomb_log
    return num / den

_tau_s = _slowing_down_time(inp.ne0, inp.Te0_ev, inp.mass, inp.coulomb_log)
_tmax_required = np.log(inp.H_fusion / inp.H_low) * _tau_s
# Headroom 1.2x so energy stop can trigger before hitting tmax
inp.tmax_backward = float(1.2 * _tmax_required)
print(f"tau_s = {_tau_s:.4e} s, tmax_backward = {inp.tmax_backward:.4e} s "
      f"(= 1.2 * ln(H_fusion/H_low) * tau_s)")


# ── Output directory ─────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = Path("/pscratch/sd/m/mariagar/projects/mc_proj/results/backward_only") / timestamp
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "plots").mkdir(parents=True, exist_ok=True)
print(f"Writing outputs to {out_dir}")


# ── Helpers (reusing from previous scripts) ─────

def wrap_phi(phi, phi_min, phi_max):
    period = phi_max - phi_min
    return (phi - phi_min) % period + phi_min


def summarize_stop_codes(stop_codes):
    uniq, counts = np.unique(stop_codes.astype(int), return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}


def ne_fun(rphiz):
    return np.full(rphiz.shape[0], inp.ne0, dtype=np.float64)


def Te_fun(rphiz):
    return np.full(rphiz.shape[0], inp.Te0_ev, dtype=np.float64)


def write_points_vtu(filename, xyz, point_data=None):
    """Point-cloud .vtu writer matching 2_tracing_gpu/export_points_vtk.py."""
    npts = len(xyz)
    if npts == 0:
        print(f"  (skip {filename.name}: no points)")
        return

    root = ET.Element("VTKFile",
                      type="UnstructuredGrid", version="0.1",
                      byte_order="LittleEndian")
    ugrid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(ugrid, "Piece",
                          NumberOfPoints=str(npts), NumberOfCells=str(npts))

    pts_elem = ET.SubElement(piece, "Points")
    arr = ET.SubElement(pts_elem, "DataArray",
                        type="Float64", NumberOfComponents="3", format="ascii")
    arr.text = " ".join(f"{x:.8e} {y:.8e} {z:.8e}" for x, y, z in xyz)

    cells = ET.SubElement(piece, "Cells")
    for name_, data_, dtype in [
        ("connectivity", range(npts),       "Int32"),
        ("offsets",      range(1, npts + 1), "Int32"),
        ("types",        ["1"] * npts,       "UInt8"),
    ]:
        da = ET.SubElement(cells, "DataArray",
                           type=dtype, Name=name_, format="ascii")
        da.text = " ".join(map(str, data_))

    if point_data:
        pdata = ET.SubElement(piece, "PointData")
        for name, data in point_data.items():
            da = ET.SubElement(pdata, "DataArray",
                               type="Float64", Name=name, format="ascii")
            da.text = " ".join(f"{v:.8e}" for v in data)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(filename), encoding="utf-8", xml_declaration=True)
    print(f"  wrote {filename.name} ({npts} points)")


# Fusion reactivity
def sigmav(T_keV):
    if T_keV > 0:
        return T_keV ** (-2/3) * np.exp(-19.94 * T_keV ** (-1/3))
    return 0.0

def fusion_reactivity(s):
    s = np.asarray(s, dtype=np.float64)
    nD = 1.0 - s**5
    T = 11.5 * (1.0 - s)
    sv = np.array([sigmav(float(t)) for t in T])
    return np.maximum(nD * nD * sv, 0.0)


# ── Build coils, surface, B, classifier, GPU interpolant ─────────────────────
print("\nBuilding coils + field...")
all_coils     = load_coils_from_makegrid_file(str(inp.coil_file), order=inp.coil_order)
base_curves   = [all_coils[i].curve for i in range(inp.ncoils)]
base_currents = [Current(inp.current) for _ in range(inp.ncoils)]

coils  = coils_via_symmetries(base_curves, base_currents, inp.nfp, stellsym=True)
curves = [c.curve for c in coils]
bs     = BiotSavart(coils)

s_input = SurfaceRZFourier.from_vmec_input(
    str(inp.vmec_input_file), range="full torus",
    nphi=inp.nphi_surf, ntheta=inp.ntheta_surf,
)

sc_particle = SurfaceClassifier(s_input, h=inp.sc_h, p=inp.sc_p)

rs    = np.linalg.norm(s_input.gamma()[:, :, 0:2], axis=2)
z_max = np.max(np.abs(s_input.gamma()[:, :, 2]))

rrange   = (np.min(rs), np.max(rs), inp.n_r)
phirange = (0, 2 * np.pi / inp.nfp, inp.n_phi)
zrange   = (0, z_max, inp.n_z)
phi_min, phi_max = phirange[0], phirange[1]

bsh = InterpolatedField(
    bs, inp.degree, rrange, phirange, zrange, True, nfp=inp.nfp, stellsym=True,
)

curves_to_vtk(curves, str(out_dir / "coils_LPQH"), close=True)
s_input.to_vtk(str(out_dir / "surface_LPQH"))

t0 = time.time()
r_range, phi_range, z_range, cell_quad_pts = cartesian_interpolant_drag(
    field=bsh, sc_particle=sc_particle,
    ne_fun=ne_fun, Te_fun=Te_fun,
    nfp=inp.nfp, n_metagrid_pts=inp.n_r,
)
print(f"GPU drag interpolant built in {time.time()-t0:.1f}s")


# ── Step 1: wall states ──────────────────────────────────────────────────────

print("\n--- STEP 1: load wall IC and sample pitch + energy ---")
wall_ic = np.loadtxt(inp.wall_ic_file, comments="#")
R_all   = wall_ic[:, 0]
phi_all = wall_ic[:, 1]
Z_all   = wall_ic[:, 2]
n_avail = len(R_all)
n_wall  = min(inp.n_wall, n_avail)
print(f"  available: {n_avail}, using {n_wall}")

rng = np.random.default_rng(inp.seed)
idx = rng.choice(n_avail, size=n_wall, replace=False)
R_wall, phi_wall, Z_wall = R_all[idx], phi_all[idx], Z_all[idx]

# Drop points outside the LCFS
sd = sc_particle.evaluate_rphiz(
    np.column_stack([R_wall, phi_wall, Z_wall])
).ravel()
inside = sd >= 0
n_outside = int((~inside).sum())
if n_outside:
    print(f"  {n_outside} wall points outside LCFS — dropped")
    R_wall, phi_wall, Z_wall = R_wall[inside], phi_wall[inside], Z_wall[inside]
    n_wall = int(inside.sum())

phi_wall = wrap_phi(phi_wall, phi_min, phi_max)


# ── Pitch sampling: parallel velocity must point into the plasma at the wall ─
# Simplified version: v ≈ v_par * b̂ (drifts dropped). Require
# (v_par b̂) · n̂_out ≤ 0 so the particle moves inward at t=0.
def outward_unit_normal(xyz_wall, eps):
    def sd_xyz(xyz_arr):
        R = np.sqrt(xyz_arr[:, 0] ** 2 + xyz_arr[:, 1] ** 2)
        phi = np.arctan2(xyz_arr[:, 1], xyz_arr[:, 0])
        Z = xyz_arr[:, 2]
        return sc_particle.evaluate_rphiz(np.column_stack([R, phi, Z])).ravel()

    grad_sd = np.zeros_like(xyz_wall)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        grad_sd[:, k] = (sd_xyz(xyz_wall + d) - sd_xyz(xyz_wall - d)) / (2 * eps)

    grad_norm = np.linalg.norm(grad_sd, axis=1, keepdims=True)
    grad_norm[grad_norm == 0.0] = 1.0
    return -grad_sd / grad_norm   # outward normal if signed distance increases inward


_xyz_wall = np.column_stack([
    R_wall * np.cos(phi_wall),
    R_wall * np.sin(phi_wall),
    Z_wall,
])

n_out_hat = outward_unit_normal(_xyz_wall, inp.normal_fd_eps)

bs.set_points(_xyz_wall)
B_xyz = np.asarray(bs.B())
B_mag = np.linalg.norm(B_xyz, axis=1, keepdims=True)
B_mag[B_mag == 0.0] = 1.0
b_hat = B_xyz / B_mag

b_dot_n = np.einsum("ij,ij->i", b_hat, n_out_hat)

H_wall = rng.uniform(inp.H_low, inp.H_high, size=n_wall)
v_total_w = np.sqrt(2.0 * H_wall / inp.mass)

lam_abs = rng.uniform(0.0, 1.0, size=n_wall)
lam_wall = -np.sign(b_dot_n) * lam_abs   # makes (v_par b_hat)·n_out ≤ 0

# fallback where b_hat·n_out is numerically zero
mask_zero = np.isclose(b_dot_n, 0.0)
lam_wall[mask_zero] = rng.uniform(-1.0, 1.0, size=int(np.count_nonzero(mask_zero)))

vtang_w = lam_wall * v_total_w

_vn_par = lam_wall * v_total_w * b_dot_n
print(f"  (v_par b̂)·n̂_out after sampling: max={_vn_par.max():.3e}  "
      f"mean={_vn_par.mean():.3e}  (all ≤ 0)")
print(f"  b̂·n̂_out ≈ 0 fallbacks: {int(mask_zero.sum())}")

# Flatten positions to the [R0,phi0,Z0, R1,phi1,Z1,...] layout the tracer wants
stz_init = np.empty(3 * n_wall, dtype=np.float64)
stz_init[0::3] = R_wall
stz_init[1::3] = phi_wall
stz_init[2::3] = Z_wall

X_w = R_wall * np.cos(phi_wall)
Y_w = R_wall * np.sin(phi_wall)
wall_xyz = np.column_stack([X_w, Y_w, Z_wall])
np.save(out_dir / "wall_starts_xyz.npy", wall_xyz)
np.save(out_dir / "wall_starts_state.npy",
        np.column_stack([X_w, Y_w, Z_wall, vtang_w, H_wall]))


# ── Step 2: backward tracing ─────────────────────────────────────────────────

print("\n--- STEP 2: backward GPU tracing (drag) ---")
speed_ref = float(sqrt(2.0 * inp.H_fusion / inp.mass))

t0 = time.time()
out = cartesian_gpu_tracing_backward_drag(
    cell_quad_pts,
    np.ascontiguousarray(r_range,   dtype=np.float64),
    np.ascontiguousarray(phi_range, dtype=np.float64),
    np.ascontiguousarray(z_range,   dtype=np.float64),
    np.ascontiguousarray(stz_init,  dtype=np.float64),
    float(inp.mass), float(inp.charge), speed_ref,
    np.ascontiguousarray(vtang_w,   dtype=np.float64),
    np.ascontiguousarray(H_wall,    dtype=np.float64),
    float(inp.coulomb_log), bool(inp.Te_in_eV),
    float(inp.tmax_backward), float(inp.tol), int(n_wall),
    float(inp.H_fusion),    # H_stop = fusion birth energy
    True,                    # use_energy_stop
)
bwd = np.asarray(out, dtype=np.float64).reshape(n_wall, 7)
print(f"  tracing done in {time.time()-t0:.2f}s")

np.save(out_dir / "backward_results.npy", bwd)
stop_codes = bwd[:, 6].astype(int)
print(f"  stop codes: {summarize_stop_codes(stop_codes)}")

# Per-category final energy statistics [MeV]
_H_final_all = bwd[:, 5]
_label = {0: "tmax", 1: "wall", 2: "H_fusion", 3: "invalid"}
print("\n  Final-energy stats per stop-code category [MeV]:")
print(f"  {'cat':<10}{'n':>8}{'min':>10}{'max':>10}{'mean':>10}"
      f"{'median':>10}{'std':>10}")
H_stats_rows = []
for code in sorted(np.unique(stop_codes)):
    mask = stop_codes == code
    Hs = _H_final_all[mask] / ONE_EV / 1e6
    if Hs.size == 0:
        continue
    row = dict(
        stop_code=int(code), label=_label.get(int(code), str(code)),
        n=int(Hs.size),
        H_min_MeV=float(Hs.min()), H_max_MeV=float(Hs.max()),
        H_mean_MeV=float(Hs.mean()), H_median_MeV=float(np.median(Hs)),
        H_std_MeV=float(Hs.std()),
    )
    H_stats_rows.append(row)
    print(f"  {row['label']:<10}{row['n']:>8d}"
          f"{row['H_min_MeV']:>10.4f}{row['H_max_MeV']:>10.4f}"
          f"{row['H_mean_MeV']:>10.4f}{row['H_median_MeV']:>10.4f}"
          f"{row['H_std_MeV']:>10.4f}")

with open(out_dir / "final_energy_stats.csv", "w", newline="") as f:
    if H_stats_rows:
        writer = csv.DictWriter(f, fieldnames=list(H_stats_rows[0].keys()))
        writer.writeheader()
        writer.writerows(H_stats_rows)


# ── Step 3: filter success (stop_code == 2) and convert to Boozer ────────────

hit_fusion = stop_codes == 2
M = int(hit_fusion.sum())
frac_fusion = M / n_wall
print(f"\n--- STEP 3: success filtering ---")
print(f"  reached H_fusion (stop_code==2): {M}/{n_wall}  "
      f"({100*frac_fusion:.2f}%)")

if M == 0:
    print("No successes; writing empty summary and exiting.")
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_wall", "n_success", "frac_fusion",
                    "n_success_valid_boozer", "frac_fusion_and_valid"])
        w.writerow([n_wall, 0, 0.0, 0, 0.0])
    raise SystemExit(0)

X_b = bwd[hit_fusion, 1]
Y_b = bwd[hit_fusion, 2]
Z_b = bwd[hit_fusion, 3]
vpar_b = bwd[hit_fusion, 4]
H_b    = bwd[hit_fusion, 5]
t_b    = bwd[hit_fusion, 0]

R_b   = np.sqrt(X_b**2 + Y_b**2)
phi_b = np.arctan2(Y_b, X_b)

birth_xyz   = np.column_stack([X_b, Y_b, Z_b, vpar_b, H_b])
birth_rphiz = np.column_stack([R_b, phi_b, Z_b])
np.save(out_dir / "birth_endpoints.npy", birth_xyz)
np.save(out_dir / "birth_endpoints_rphiz.npy", birth_rphiz)

print("  building Boozer interpolant + converting to Boozer...")
bri = BoozerRadialInterpolant(str(inp.boozmn_file), inp.radial_order, no_K=True)
boozer_field = InterpolatedBoozerField(
    bri, inp.boozer_degree,
    ns_interp=inp.boozer_res,
    ntheta_interp=inp.boozer_res,
    nzeta_interp=inp.boozer_res,
)
# cylindrical_to_boozer raises a RuntimeError as soon as ONE point in the batch
# fails (e.g. it sits numerically outside the Boozer s in [0,1] domain).  With
# 1e6 birth endpoints, the previous per-point fallback was unusably slow.
#
# Strategy:
#   (a) Drop points already known to be outside the LCFS via sc_particle —
#       they cannot have a valid Boozer s and don't need to be inverted.
#   (b) Convert the remainder in chunks; on a chunk failure, recursively split
#       the chunk in half. Only chunks that bottom-out at a single bad point
#       pay the per-point cost. This is O(N + B log C) with tiny B in practice.

boozer_coords = np.full_like(birth_rphiz, np.nan)
n_failed_bz = 0

# (a) Cheap LCFS pre-filter on birth endpoints
sd_birth = sc_particle.evaluate_rphiz(birth_rphiz).ravel()
inside_birth = sd_birth >= 0
n_outside_birth = int((~inside_birth).sum())
if n_outside_birth:
    print(f"  {n_outside_birth}/{M} birth endpoints lie outside LCFS — "
          f"skipping Boozer conversion for these")
inside_idx = np.where(inside_birth)[0]
to_convert = birth_rphiz[inside_idx]

# (b) Chunked conversion with recursive subdivision on failure
def _convert_chunked(field, pts, idx_global, chunk=10_000):
    """Fill boozer_coords[idx_global] in-place; return number of failures."""
    failed = 0
    n = len(pts)
    starts = range(0, n, chunk)
    t_chunk0 = time.time()
    done = 0
    for s in starts:
        e = min(s + chunk, n)
        sub_pts = pts[s:e]
        sub_idx = idx_global[s:e]
        failed += _convert_recurse(field, sub_pts, sub_idx)
        done += (e - s)
        if done % (10 * chunk) == 0 or e == n:
            print(f"    Boozer convert: {done}/{n}  "
                  f"(elapsed {time.time()-t_chunk0:.1f}s, failures so far {failed})")
    return failed

def _convert_recurse(field, pts, idx_global):
    """Try to convert a contiguous block; recurse on failure."""
    if len(pts) == 0:
        return 0
    try:
        out = cylindrical_to_boozer(field, pts)
        boozer_coords[idx_global] = out
        return 0
    except RuntimeError:
        if len(pts) == 1:
            return 1
        mid = len(pts) // 2
        return (_convert_recurse(field, pts[:mid], idx_global[:mid]) +
                _convert_recurse(field, pts[mid:], idx_global[mid:]))

print(f"  converting {len(to_convert)} inside-LCFS endpoints to Boozer...")
t_bz = time.time()
n_failed_bz = _convert_chunked(boozer_field, to_convert, inside_idx)
print(f"  Boozer conversion done in {time.time()-t_bz:.1f}s; "
      f"failures: {n_failed_bz}/{len(to_convert)}")

s_b = boozer_coords[:, 0]
np.save(out_dir / "birth_endpoints_boozer.npy", boozer_coords)

# "Valid" Boozer coordinates == finite s in [0, 1]
valid_bz = np.isfinite(s_b) & (s_b >= 0.0) & (s_b <= 1.0)
M_valid = int(valid_bz.sum())
frac_both = M_valid / n_wall
np.save(out_dir / "valid_boozer_mask.npy", valid_bz)
print(f"  valid Boozer s (0<=s<=1): {M_valid}/{M}")
print(f"  fraction fusion+valid of wall starts: {100*frac_both:.2f}%")


# ── Step 4: summary CSV ──────────────────────────────────────────────────────

with open(out_dir / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["n_wall", n_wall])
    w.writerow(["n_success_stop_code_2", M])
    w.writerow(["frac_fusion", frac_fusion])
    w.writerow(["n_success_valid_boozer", M_valid])
    w.writerow(["frac_fusion_and_valid_boozer", frac_both])
    w.writerow(["H_low_MeV", inp.H_low / ONE_EV / 1e6])
    w.writerow(["H_high_MeV", inp.H_high / ONE_EV / 1e6])
    w.writerow(["H_fusion_MeV", inp.H_fusion / ONE_EV / 1e6])
    w.writerow(["tmax_backward_s", inp.tmax_backward])
    for k, v in summarize_stop_codes(stop_codes).items():
        w.writerow([f"stop_code_{k}_count", v])


# ── Step 5: VTK point-cloud exports ──────────────────────────────────────────

print("\n--- STEP 5: VTK exports ---")
write_points_vtu(out_dir / "wall_starts.vtu", wall_xyz,
                 point_data={"H_init": H_wall, "vpar_init": vtang_w})

birth_xyz_only = np.column_stack([X_b, Y_b, Z_b])
write_points_vtu(out_dir / "birth_endpoints.vtu", birth_xyz_only,
                 point_data={
                     "vpar": vpar_b, "H": H_b, "s_boozer": s_b,
                     "valid_boozer": valid_bz.astype(np.float64),
                     "t_elapsed": t_b,
                 })
write_points_vtu(out_dir / "birth_endpoints_valid.vtu",
                 birth_xyz_only[valid_bz],
                 point_data={"s_boozer": s_b[valid_bz],
                             "vpar": vpar_b[valid_bz]})

# Segments from wall start -> birth endpoint, one per successful particle
wall_xyz_success = wall_xyz[hit_fusion]
seg_pts = np.empty((2 * M, 3), dtype=np.float64)
seg_pts[0::2] = wall_xyz_success
seg_pts[1::2] = birth_xyz_only

root = ET.Element("VTKFile", type="UnstructuredGrid",
                  version="0.1", byte_order="LittleEndian")
ugrid = ET.SubElement(root, "UnstructuredGrid")
piece = ET.SubElement(ugrid, "Piece",
                      NumberOfPoints=str(2 * M), NumberOfCells=str(M))
p = ET.SubElement(piece, "Points")
da = ET.SubElement(p, "DataArray",
                   type="Float64", NumberOfComponents="3", format="ascii")
da.text = " ".join(f"{x:.6e} {y:.6e} {z:.6e}" for x, y, z in seg_pts)
cells = ET.SubElement(piece, "Cells")
conn = ET.SubElement(cells, "DataArray",
                     type="Int32", Name="connectivity", format="ascii")
conn.text = " ".join(str(i) for i in range(2 * M))
off = ET.SubElement(cells, "DataArray",
                    type="Int32", Name="offsets", format="ascii")
off.text = " ".join(str(2 * (i + 1)) for i in range(M))
typ = ET.SubElement(cells, "DataArray",
                    type="UInt8", Name="types", format="ascii")
typ.text = " ".join(["3"] * M)  # 3 = VTK_LINE
cdata = ET.SubElement(piece, "CellData")
for name, data in [("s_boozer", s_b), ("H_final", H_b),
                   ("valid_boozer", valid_bz.astype(np.float64))]:
    da = ET.SubElement(cdata, "DataArray",
                       type="Float64", Name=name, format="ascii")
    da.text = " ".join(f"{v:.6e}" for v in data)
tree = ET.ElementTree(root)
ET.indent(tree, space="  ")
tree.write(str(out_dir / "trajectory_segments.vtu"),
           encoding="utf-8", xml_declaration=True)
print(f"  wrote trajectory_segments.vtu ({M} segments)")


# ── Step 5b: full time-resolved trajectories for a subsample of successes ────
# Re-run the backward tracer for a small subsample of particles that reached
# H_fusion (stop_code == 2), sweeping tmax from ~0 up to inp.tmax_backward.
# Each call returns the end state at that tmax; stacking across calls yields a
# (n_traj, n_snap, ...) trajectory array.  use_energy_stop stays True, so a
# particle that hit H_fusion before a given snapshot's tmax remains frozen at
# its birth endpoint for later snapshots (matches the segments output).

print("\n--- STEP 5b: trajectory snapshots for subsample of successes ---")
n_traj = int(min(inp.n_trajectory, M))
if n_traj == 0:
    print("  no successful particles; skipping trajectory save")
else:
    success_idx = np.flatnonzero(hit_fusion)
    traj_sel = rng.choice(success_idx, size=n_traj, replace=False)

    stz_traj = np.empty(3 * n_traj, dtype=np.float64)
    stz_traj[0::3] = R_wall[traj_sel]
    stz_traj[1::3] = phi_wall[traj_sel]
    stz_traj[2::3] = Z_wall[traj_sel]
    vtang_traj = vtang_w[traj_sel]
    H_traj     = H_wall[traj_sel]

    tmax_snaps = np.linspace(
        inp.tmax_backward / inp.n_snapshots,
        inp.tmax_backward,
        inp.n_snapshots,
    )

    traj_xyz  = np.zeros((n_traj, inp.n_snapshots, 3), dtype=np.float64)
    traj_vpar = np.zeros((n_traj, inp.n_snapshots),    dtype=np.float64)
    traj_H    = np.zeros((n_traj, inp.n_snapshots),    dtype=np.float64)
    traj_time = np.zeros((n_traj, inp.n_snapshots),    dtype=np.float64)
    traj_stop = np.zeros((n_traj, inp.n_snapshots),    dtype=np.int32)

    t0 = time.time()
    for i, tmax_i in enumerate(tmax_snaps):
        out_i = cartesian_gpu_tracing_backward_drag(
            cell_quad_pts,
            np.ascontiguousarray(r_range,   dtype=np.float64),
            np.ascontiguousarray(phi_range, dtype=np.float64),
            np.ascontiguousarray(z_range,   dtype=np.float64),
            np.ascontiguousarray(stz_traj,  dtype=np.float64),
            float(inp.mass), float(inp.charge), speed_ref,
            np.ascontiguousarray(vtang_traj, dtype=np.float64),
            np.ascontiguousarray(H_traj,     dtype=np.float64),
            float(inp.coulomb_log), bool(inp.Te_in_eV),
            float(tmax_i), float(inp.tol), int(n_traj),
            float(inp.H_fusion),
            True,
        )
        arr = np.asarray(out_i, dtype=np.float64).reshape(n_traj, 7)
        traj_time[:, i]    = arr[:, 0]
        traj_xyz[:, i, 0]  = arr[:, 1]
        traj_xyz[:, i, 1]  = arr[:, 2]
        traj_xyz[:, i, 2]  = arr[:, 3]
        traj_vpar[:, i]    = arr[:, 4]
        traj_H[:, i]       = arr[:, 5]
        traj_stop[:, i]    = arr[:, 6].astype(np.int32)
        print(f"  snapshot {i+1}/{inp.n_snapshots}  tmax={tmax_i:.3e}s")
    print(f"  trajectory tracing done in {time.time()-t0:.2f}s")

    np.save(out_dir / "trajectories_xyz.npy",        traj_xyz)
    np.save(out_dir / "trajectories_vpar.npy",       traj_vpar)
    np.save(out_dir / "trajectories_H.npy",          traj_H)
    np.save(out_dir / "trajectories_time.npy",       traj_time)
    np.save(out_dir / "trajectories_stop.npy",       traj_stop)
    np.save(out_dir / "trajectories_tmax.npy",       tmax_snaps)
    np.save(out_dir / "trajectories_sample_idx.npy", traj_sel)
    print(f"  wrote trajectories_*.npy  "
          f"(n_traj={n_traj}, n_snap={inp.n_snapshots})")

    # Paraview polyline export: one VTK_POLY_LINE per particle, prepended with
    # the wall start point (t=0) so each polyline begins at the wall IC and
    # ends at the birth endpoint.
    wall_xyz_traj = wall_xyz[traj_sel]
    pts_per = 1 + inp.n_snapshots
    pts = np.empty((n_traj, pts_per, 3), dtype=np.float64)
    pts[:, 0, :]  = wall_xyz_traj
    pts[:, 1:, :] = traj_xyz

    t_pt    = np.empty((n_traj, pts_per), dtype=np.float64)
    vpar_pt = np.empty((n_traj, pts_per), dtype=np.float64)
    H_pt    = np.empty((n_traj, pts_per), dtype=np.float64)
    t_pt[:, 0]     = 0.0
    t_pt[:, 1:]    = traj_time
    vpar_pt[:, 0]  = vtang_traj
    vpar_pt[:, 1:] = traj_vpar
    H_pt[:, 0]     = H_traj
    H_pt[:, 1:]    = traj_H

    flat_pts  = pts.reshape(-1, 3)
    flat_t    = t_pt.reshape(-1)
    flat_vpar = vpar_pt.reshape(-1)
    flat_H    = H_pt.reshape(-1)
    n_total_pts = flat_pts.shape[0]

    root = ET.Element("VTKFile", type="UnstructuredGrid",
                      version="0.1", byte_order="LittleEndian")
    ugrid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(ugrid, "Piece",
                          NumberOfPoints=str(n_total_pts),
                          NumberOfCells=str(n_traj))

    pts_elem = ET.SubElement(piece, "Points")
    da = ET.SubElement(pts_elem, "DataArray",
                       type="Float64", NumberOfComponents="3", format="ascii")
    da.text = " ".join(f"{x:.6e} {y:.6e} {z:.6e}" for x, y, z in flat_pts)

    cells = ET.SubElement(piece, "Cells")
    conn = ET.SubElement(cells, "DataArray",
                         type="Int32", Name="connectivity", format="ascii")
    conn.text = " ".join(str(i) for i in range(n_total_pts))
    off = ET.SubElement(cells, "DataArray",
                        type="Int32", Name="offsets", format="ascii")
    off.text = " ".join(str(pts_per * (k + 1)) for k in range(n_traj))
    typ = ET.SubElement(cells, "DataArray",
                        type="UInt8", Name="types", format="ascii")
    typ.text = " ".join(["4"] * n_traj)  # 4 = VTK_POLY_LINE

    pdata = ET.SubElement(piece, "PointData")
    for name, data in [("time", flat_t), ("vpar", flat_vpar), ("H", flat_H)]:
        d = ET.SubElement(pdata, "DataArray",
                          type="Float64", Name=name, format="ascii")
        d.text = " ".join(f"{v:.6e}" for v in data)

    cdata = ET.SubElement(piece, "CellData")
    pid = ET.SubElement(cdata, "DataArray",
                        type="Int32", Name="particle_id", format="ascii")
    pid.text = " ".join(str(int(i)) for i in traj_sel)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(out_dir / "trajectories.vtu"),
               encoding="utf-8", xml_declaration=True)
    print(f"  wrote trajectories.vtu ({n_traj} polylines, "
          f"{pts_per} points each)")


# ── Step 6: matplotlib figures ──────────────────────────────────────────────

print("\n--- STEP 6: plots ---")
pdir = out_dir / "plots"

# 1. Wall hits (starts) vs birth endpoints, top-down XY
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(wall_xyz[:, 0], wall_xyz[:, 1], s=1, alpha=0.2,
           label=f"wall starts ({n_wall})", color="grey")
ax.scatter(X_b, Y_b, s=3, alpha=0.5,
           label=f"birth endpoints ({M})", color="C1")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_aspect("equal"); ax.legend()
ax.set_title("Wall starts vs backward-recovered birth endpoints")
fig.tight_layout(); fig.savefig(pdir / "wall_vs_birth_xy.png", dpi=150)
plt.close(fig)

# 2. Birth endpoints in R-Z
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(R_b, Z_b, s=3, alpha=0.5, color="C1",
           label=f"birth endpoints ({M})")
ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
ax.set_aspect("equal"); ax.legend()
ax.set_title("Birth endpoints in R–Z")
fig.tight_layout(); fig.savefig(pdir / "birth_RZ.png", dpi=150)
plt.close(fig)

# 3. s histogram — clip to [0, 1] for the view
fig, ax = plt.subplots(figsize=(7, 4))
s_for_hist = s_b[np.isfinite(s_b)]
bins = np.linspace(0, 1, 41)
ax.hist(np.clip(s_for_hist, 0, 1), bins=bins, density=True, alpha=0.7,
        label=f"backward-born (n={len(s_for_hist)})")
# overlay reactivity p0(s) normalised on [0,1]
s_grid = np.linspace(0, 1, 200)
p0 = fusion_reactivity(s_grid)
p0 = p0 / np.trapz(p0, s_grid)
ax.plot(s_grid, p0, "k--", label="fusion reactivity (ref)")
ax.set_xlabel("s (Boozer flux label)")
ax.set_ylabel("probability density")
ax.set_title("Birth-endpoint s distribution")
ax.set_xlim(0, 1); ax.legend()
fig.tight_layout(); fig.savefig(pdir / "s_histogram.png", dpi=150)
plt.close(fig)

# 4. Reactivity evaluated at each birth-endpoint s
fig, ax = plt.subplots(figsize=(7, 4))
reac_at_births = fusion_reactivity(np.clip(s_for_hist, 0, 1))
ax.hist(reac_at_births, bins=40, alpha=0.7, color="C2")
ax.set_xlabel("fusion reactivity p0(s)  [arb. units]")
ax.set_ylabel("count")
ax.set_title("Reactivity at birth endpoints")
fig.tight_layout(); fig.savefig(pdir / "reactivity_histogram.png", dpi=150)
plt.close(fig)

# 5. Stop-code bar chart
fig, ax = plt.subplots(figsize=(6, 4))
sc_counts = summarize_stop_codes(stop_codes)
labels = {0: "tmax", 1: "wall", 2: "H_fusion", 3: "invalid"}
keys = sorted(sc_counts.keys())
ax.bar([labels.get(k, str(k)) for k in keys],
       [sc_counts[k] for k in keys], color="C0")
ax.set_ylabel("count"); ax.set_title("Backward tracer stop codes")
fig.tight_layout(); fig.savefig(pdir / "stop_codes.png", dpi=150)
plt.close(fig)

print(f"  wrote plots to {pdir}")

print("\nDone.")
print(f"  fraction reaching H_fusion:           {100*frac_fusion:.2f}%")
print(f"  fraction reaching H_fusion AND valid: {100*frac_both:.2f}%")
print(f"Outputs at: {out_dir}")