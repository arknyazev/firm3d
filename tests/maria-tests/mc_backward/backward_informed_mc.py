#!/usr/bin/env python3
"""
Backward-informed importance-sampling Monte Carlo for alpha-particle wall losses.

This script is the full workflow built around the backward-only analysis in
``backward_tracing_only.py``; the backward step here is intentionally identical
to that script (same wall IC, same inward-pitch sampling, same slowing-down
time, same backward tracer, same Boozer conversion, same VTK/plot outputs).
On top of it we add the forward IS estimator and a forward-only baseline.

Algorithm
---------
1. Sample wall states y = (r_w, lambda_w, H_w) ~ r(y):
     * positions from ``initial_conditions_surface_cylindrical.txt``
     * energy  H_w ~ Uniform(H_low, H_fusion)
     * pitch   |lambda| ~ U(0, 1), sign flipped so v_par * b_hat * n_out <= 0
       (particle moves inward at t=0; matches backward_tracing_only.py).
2. Trace backward with deterministic drag (no scattering).
3. Keep endpoints with stop_code == 2 (reached H_fusion).
4. Convert (R, phi, Z) -> Boozer (s, theta, zeta); keep 0 <= s <= 1.
5. For each valid birth endpoint x_j, p0_j = reactivity(s_j).
6. Build a 1-D histogram of successful backward s-values and define the
   per-point score s_score_j by evaluating that histogram at s_j.
7. Construct the support-safe mixture proposal over the discrete cloud
         q_j = (1 - alpha) * (p0_j * s_score_j)/Z_score  +  alpha * p0_j/Z_p0
   with configurable small alpha > 0 so that regions with zero score but
   non-zero p0 still receive positive sampling weight.
8. Draw N forward birth states from q and trace forward with deterministic
   drag only (no scattering).
9. IS estimator:
         Q_hat_IS = (1/N) sum_i A(X_i) * p0_i / q_i,  X_i ~ q
10. Forward-only baseline: sample N birth states from the fusion IC file
    (already distributed as p0) and run the forward tracer with drag.
         Q_hat_FWD = (1/N) sum_i A(X_i),  X_i ~ p0.

Estimator metric definitions (see step 9 / step 10 comments in the code):
    Y_i             = per-sample estimator contribution
    Q_hat           = mean(Y_i)
    Var(Q_hat)      = sample_var(Y_i, ddof=1) / N
    SE              = sqrt(Var(Q_hat))
    cv_estimator    = SE / Q_hat
    cv_single_sample= sqrt(sample_var(Y_i)) / Q_hat
    N_target(c)     = (cv_single_sample / c) ** 2     # c = target cv level

Outputs  (timestamped dir under /pscratch/.../results/backward_is/)
-------------------------------------------------------------------
    backward_results.npy             (n_wall, 7) backward tracer output
    wall_starts_xyz.npy              (n_wall, 3) wall IC used
    wall_starts_state.npy            (n_wall, 5) X, Y, Z, vpar, H
    birth_endpoints.npy              (M, 5) X, Y, Z, vpar, H (stop==2)
    birth_endpoints_rphiz.npy        (M, 3) R, phi, Z
    birth_endpoints_boozer.npy       (M, 3) s, theta, zeta
    valid_boozer_mask.npy            (M,) bool  (0<=s<=1, finite)
    s_score.npy                      (M_valid,) birth-space score
    p0_at_births.npy                 (M_valid,) reactivity at births
    q_weights.npy                    (M_valid,) discrete proposal
    is_sample_idx.npy                (N_is,) indices drawn from q
    forward_is_results.npy           (N_is, 7) IS forward tracer output
    forward_baseline_results.npy     (N_base, 7) forward-only tracer output
    is_weights.npy                   (N_is,) p0/q per IS sample
    metrics_summary.csv              all estimator + IS diagnostics
    trajectories_*.{npy,vtu}         snapshots for backward + forward subsample
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
from firm3dpp import (
    cartesian_gpu_tracing_drag,
    cartesian_gpu_tracing_backward_drag,
)


# ── Inputs ───────────────────────────────────────────────────────────────────

THIS_DIR = Path(__file__).resolve().parent


@dataclass
class Inputs:
    # Files (same conventions as backward_tracing_only.py)
    coil_file:       Path = THIS_DIR / "LandremanPaulQH_coils" / "coils.curves_22_7_21"
    vmec_input_file: Path = THIS_DIR / "LandremanPaulQH_coils" / "input.vmec"
    boozmn_file:     Path = THIS_DIR / "LandremanPaulQH_coils" / "boozmn.nc"
    wall_ic_file:    Path = Path("/pscratch/sd/m/mariagar/projects/mc_proj/IC/"
                                 "initial_conditions_surface_cylindrical.txt")
    # Fusion IC file: produced by 1_sample_fusion_distribution.py.
    # Columns: R, phi, Z, vpar  (positions already distributed as p0(s))
    fusion_ic_file:  Path = Path("/pscratch/sd/m/mariagar/projects/mc_proj/IC/"
                                 "initial_conditions_cylindrical.txt")

    # Equilibrium / coils
    nfp:        int   = 4
    ncoils:     int   = 5
    current:    float = 1.27797548115612e7
    coil_order: int   = 20

    # Interpolant grid (matches backward_tracing_only.py)
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

    # Physics
    mass:        float = MASS
    charge:      float = CHARGE
    H_low:       float = 3.0e6 * ONE_EV   # wall-energy lower bound (same as BW-only)
    H_high:      float = H_FUSION         # wall-energy upper bound
    H_fusion:    float = H_FUSION         # birth energy (stop target)
    coulomb_log: float = 17.0
    Te_in_eV:    bool  = True
    ne0:         float = 1e21             # matches backward_tracing_only.py
    Te0_ev:      float = 100

    # Tracing
    n_wall:        int   = 10_000 
    n_baseline:    int   = 10_000   # forward-only baseline sample count
    N_is:          int   = 10_000   # IS forward sample count
    # tmax_backward is recomputed below from slowing-down time
    tmax_backward: float = 5e-5
    tmax_forward:  float = 1e-2
    tol:           float = 1e-9
    seed:          int   = 57

    # Trajectory snapshots (backward + forward subsamples)
    n_trajectory_backward: int = 200
    n_trajectory_forward:  int = 200
    n_snapshots:           int = 100

    # Finite-difference step for outward normal
    normal_fd_eps: float = 1e-4

    # Score construction
    s_score_nbins: int = 40   # 1-D histogram over Boozer s in [0, 1]

    # Support-safe mixture proposal  q = (1-alpha) * q_tilde + alpha * p0
    alpha_mix: float = 0.05


inp = Inputs()


# ── Slowing-down time and required backward time horizon ─────────────────────

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
inp.tmax_backward = float(1.2 * _tmax_required)
print(f"tau_s = {_tau_s:.4e} s, tmax_backward = {inp.tmax_backward:.4e} s "
      f"(= 1.2 * ln(H_fusion/H_low) * tau_s)")


# ── Output directory ─────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = Path("/pscratch/sd/m/mariagar/projects/mc_proj/results/backward_is") / timestamp
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "plots").mkdir(parents=True, exist_ok=True)
print(f"Writing outputs to {out_dir}")


# ── Helpers ─────────────────────────────────────────────────────────────────

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


def flatten_stz(R, phi, Z):
    n = len(R)
    out = np.empty(3 * n, dtype=np.float64)
    out[0::3] = R
    out[1::3] = phi
    out[2::3] = Z
    return out


def write_points_vtu(filename, xyz, point_data=None):
    npts = len(xyz)
    if npts == 0:
        print(f"  (skip {filename.name}: no points)")
        return
    root = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1",
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
        ("connectivity", range(npts),        "Int32"),
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


def write_polylines_vtu(filename, pts_per_poly, pts, point_data=None,
                        cell_data=None):
    """Write N polylines of pts_per_poly points each as VTK_POLY_LINE."""
    n_poly = pts.shape[0]
    flat = pts.reshape(-1, 3)
    n_total = flat.shape[0]
    root = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1",
                      byte_order="LittleEndian")
    ugrid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(ugrid, "Piece",
                          NumberOfPoints=str(n_total),
                          NumberOfCells=str(n_poly))
    pts_elem = ET.SubElement(piece, "Points")
    da = ET.SubElement(pts_elem, "DataArray",
                       type="Float64", NumberOfComponents="3", format="ascii")
    da.text = " ".join(f"{x:.6e} {y:.6e} {z:.6e}" for x, y, z in flat)
    cells = ET.SubElement(piece, "Cells")
    conn = ET.SubElement(cells, "DataArray",
                         type="Int32", Name="connectivity", format="ascii")
    conn.text = " ".join(str(i) for i in range(n_total))
    off = ET.SubElement(cells, "DataArray",
                        type="Int32", Name="offsets", format="ascii")
    off.text = " ".join(str(pts_per_poly * (k + 1)) for k in range(n_poly))
    typ = ET.SubElement(cells, "DataArray",
                        type="UInt8", Name="types", format="ascii")
    typ.text = " ".join(["4"] * n_poly)  # 4 = VTK_POLY_LINE
    if point_data:
        pdata = ET.SubElement(piece, "PointData")
        for name, data in point_data.items():
            d = ET.SubElement(pdata, "DataArray",
                              type="Float64", Name=name, format="ascii")
            d.text = " ".join(f"{v:.6e}" for v in data.ravel())
    if cell_data:
        cdata = ET.SubElement(piece, "CellData")
        for name, data in cell_data.items():
            d = ET.SubElement(cdata, "DataArray",
                              type="Int32", Name=name, format="ascii")
            d.text = " ".join(str(int(v)) for v in data)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(filename), encoding="utf-8", xml_declaration=True)
    print(f"  wrote {filename.name} ({n_poly} polylines, {pts_per_poly} pts each)")


# ── Fusion reactivity (birth distribution p0 in Boozer s) ────────────────────

def sigmav(T_keV):
    if T_keV > 0:
        return T_keV ** (-2 / 3) * np.exp(-19.94 * T_keV ** (-1 / 3))
    return 0.0


def fusion_reactivity(s):
    """Unnormalised birth density as a function of Boozer s. Non-negative."""
    s = np.asarray(s, dtype=np.float64)
    nD = 1.0 - s ** 5
    T = 11.5 * (1.0 - s)
    sv = np.array([sigmav(float(t)) for t in T])
    return np.maximum(nD * nD * sv, 0.0)


# ── Build coils, surface, B, classifier, GPU drag interpolant ────────────────

print("\nBuilding coils + field...")
all_coils     = load_coils_from_makegrid_file(str(inp.coil_file),
                                              order=inp.coil_order)
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
print(f"GPU drag interpolant built in {time.time() - t0:.1f}s")


# =============================================================================
# STEP 1 — Sample wall states (identical conventions to backward_tracing_only.py)
# =============================================================================

print("\n--- STEP 1: wall IC + pitch/energy sampling ---")
wall_ic = np.loadtxt(inp.wall_ic_file, comments="#")
# The wall-IC file has three columns only: R, phi, Z.
R_all   = wall_ic[:, 0]
phi_all = wall_ic[:, 1]
Z_all   = wall_ic[:, 2]
n_avail = len(R_all)
n_wall  = min(inp.n_wall, n_avail)
print(f"  available: {n_avail}, using {n_wall}")

rng = np.random.default_rng(inp.seed)
idx = rng.choice(n_avail, size=n_wall, replace=False)
R_wall, phi_wall, Z_wall = R_all[idx], phi_all[idx], Z_all[idx]

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


def outward_unit_normal(xyz_wall, eps):
    def sd_xyz(xyz_arr):
        R = np.sqrt(xyz_arr[:, 0] ** 2 + xyz_arr[:, 1] ** 2)
        phi = np.arctan2(xyz_arr[:, 1], xyz_arr[:, 0])
        Z = xyz_arr[:, 2]
        return sc_particle.evaluate_rphiz(np.column_stack([R, phi, Z])).ravel()
    grad_sd = np.zeros_like(xyz_wall)
    for k in range(3):
        d = np.zeros(3); d[k] = eps
        grad_sd[:, k] = (sd_xyz(xyz_wall + d) - sd_xyz(xyz_wall - d)) / (2 * eps)
    grad_norm = np.linalg.norm(grad_sd, axis=1, keepdims=True)
    grad_norm[grad_norm == 0.0] = 1.0
    return -grad_sd / grad_norm


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

lam_abs  = rng.uniform(0.0, 1.0, size=n_wall)
lam_wall = -np.sign(b_dot_n) * lam_abs
mask_zero = np.isclose(b_dot_n, 0.0)
lam_wall[mask_zero] = rng.uniform(-1.0, 1.0, size=int(mask_zero.sum()))

vtang_w = lam_wall * v_total_w
_vn_par = lam_wall * v_total_w * b_dot_n
print(f"  (v_par b_hat)*n_out: max={_vn_par.max():.3e}  mean={_vn_par.mean():.3e}")
print(f"  b_hat*n_out ~ 0 fallbacks: {int(mask_zero.sum())}")

stz_init = flatten_stz(R_wall, phi_wall, Z_wall)

X_w = R_wall * np.cos(phi_wall)
Y_w = R_wall * np.sin(phi_wall)
wall_xyz = np.column_stack([X_w, Y_w, Z_wall])
np.save(out_dir / "wall_starts_xyz.npy", wall_xyz)
np.save(out_dir / "wall_starts_state.npy",
        np.column_stack([X_w, Y_w, Z_wall, vtang_w, H_wall]))


# =============================================================================
# STEP 2 — Backward GPU tracing (deterministic drag only)
# =============================================================================

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
    float(inp.H_fusion), True,
)
bwd = np.asarray(out, dtype=np.float64).reshape(n_wall, 7)
print(f"  tracing done in {time.time() - t0:.2f}s")

np.save(out_dir / "backward_results.npy", bwd)
stop_codes_bwd = bwd[:, 6].astype(int)
print(f"  stop codes: {summarize_stop_codes(stop_codes_bwd)}")


# =============================================================================
# STEP 3 — Keep successful backward traces (stop_code == 2)
# =============================================================================

hit_fusion = stop_codes_bwd == 2
M = int(hit_fusion.sum())
bwd_success_frac = M / n_wall
print(f"\n--- STEP 3: success filtering ---")
print(f"  reached H_fusion (stop_code==2): {M}/{n_wall} "
      f"({100 * bwd_success_frac:.2f}%)")

if M == 0:
    raise SystemExit("No successful backward traces — cannot build proposal.")

X_b = bwd[hit_fusion, 1]
Y_b = bwd[hit_fusion, 2]
Z_b = bwd[hit_fusion, 3]
vpar_b = bwd[hit_fusion, 4]
H_b    = bwd[hit_fusion, 5]
t_b    = bwd[hit_fusion, 0]

R_b   = np.sqrt(X_b ** 2 + Y_b ** 2)
phi_b = np.arctan2(Y_b, X_b)

birth_xyz   = np.column_stack([X_b, Y_b, Z_b, vpar_b, H_b])
birth_rphiz = np.column_stack([R_b, phi_b, Z_b])
np.save(out_dir / "birth_endpoints.npy", birth_xyz)
np.save(out_dir / "birth_endpoints_rphiz.npy", birth_rphiz)


# =============================================================================
# STEP 4 — Convert births to Boozer (s, theta, zeta), keep valid s
# =============================================================================

print("  building Boozer interpolant + converting births to Boozer...")
bri = BoozerRadialInterpolant(str(inp.boozmn_file), inp.radial_order, no_K=True)
boozer_field = InterpolatedBoozerField(
    bri, inp.boozer_degree,
    ns_interp=inp.boozer_res,
    ntheta_interp=inp.boozer_res,
    nzeta_interp=inp.boozer_res,
)

boozer_coords = np.full_like(birth_rphiz, np.nan)
sd_birth = sc_particle.evaluate_rphiz(birth_rphiz).ravel()
inside_birth = sd_birth >= 0
n_outside_birth = int((~inside_birth).sum())
if n_outside_birth:
    print(f"  {n_outside_birth}/{M} birth endpoints outside LCFS — skipped")
inside_idx = np.where(inside_birth)[0]
to_convert = birth_rphiz[inside_idx]


def _convert_recurse(field, pts, idx_global):
    if len(pts) == 0:
        return 0
    try:
        boozer_coords[idx_global] = cylindrical_to_boozer(field, pts)
        return 0
    except RuntimeError:
        if len(pts) == 1:
            return 1
        mid = len(pts) // 2
        return (_convert_recurse(field, pts[:mid], idx_global[:mid])
                + _convert_recurse(field, pts[mid:], idx_global[mid:]))


def _convert_chunked(field, pts, idx_global, chunk=10_000):
    failed = 0
    n = len(pts)
    t_chunk0 = time.time()
    done = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        failed += _convert_recurse(field, pts[s:e], idx_global[s:e])
        done += (e - s)
        if done % (10 * chunk) == 0 or e == n:
            print(f"    Boozer convert: {done}/{n} "
                  f"(elapsed {time.time() - t_chunk0:.1f}s, "
                  f"failures so far {failed})")
    return failed


t_bz = time.time()
n_failed_bz = _convert_chunked(boozer_field, to_convert, inside_idx)
print(f"  Boozer conversion done in {time.time() - t_bz:.1f}s; "
      f"failures: {n_failed_bz}/{len(to_convert)}")

s_b = boozer_coords[:, 0]
np.save(out_dir / "birth_endpoints_boozer.npy", boozer_coords)

valid_bz = np.isfinite(s_b) & (s_b >= 0.0) & (s_b <= 1.0)
M_valid = int(valid_bz.sum())
np.save(out_dir / "valid_boozer_mask.npy", valid_bz)
print(f"  valid Boozer s in [0,1]: {M_valid}/{M}")

if M_valid == 0:
    raise SystemExit("No birth endpoints have valid Boozer s — cannot build p0.")

# Restrict the proposal cloud to valid-s entries from here on.
X_v    = X_b[valid_bz]
Y_v    = Y_b[valid_bz]
Z_v    = Z_b[valid_bz]
R_v    = R_b[valid_bz]
phi_v  = phi_b[valid_bz]
vpar_v = vpar_b[valid_bz]
H_v    = H_b[valid_bz]
s_v    = s_b[valid_bz]


# =============================================================================
# STEP 5 — Physical birth density p0 from reactivity(s) (Boozer s)
# =============================================================================

print("\n--- STEP 5: p0 from reactivity(s) ---")
p0_births = fusion_reactivity(s_v)
np.save(out_dir / "p0_at_births.npy", p0_births)
print(f"  p0 range over births: [{p0_births.min():.3e}, {p0_births.max():.3e}]")
print(f"  p0 > 0 count        : {int((p0_births > 0.0).sum())}/{M_valid}")


# =============================================================================
# STEP 6 — Nonnegative birth-space score s_score(x) from backward samples
#          (1-D histogram in Boozer s; one score per birth point)
# =============================================================================

print("\n--- STEP 6: build 1-D histogram score in Boozer s ---")
s_edges = np.linspace(0.0, 1.0, inp.s_score_nbins + 1)
s_hist, _ = np.histogram(s_v, bins=s_edges)
bin_idx = np.clip(np.searchsorted(s_edges, s_v, side="right") - 1,
                  0, inp.s_score_nbins - 1)
s_score = s_hist[bin_idx].astype(np.float64)
# Keep a normalized copy for plotting; does not change the proposal shape.
s_score_density = s_hist.astype(np.float64)
s_score_density /= np.trapz(
    np.maximum(s_score_density, 0.0),
    0.5 * (s_edges[:-1] + s_edges[1:]),
) if s_hist.sum() > 0 else 1.0

np.save(out_dir / "s_score.npy", s_score)
np.save(out_dir / "s_score_hist.npy", s_hist)
np.save(out_dir / "s_score_edges.npy", s_edges)
print(f"  score range        : [{s_score.min():.3e}, {s_score.max():.3e}]")
print(f"  non-empty bins     : {int((s_hist > 0).sum())}/{inp.s_score_nbins}")


# =============================================================================
# STEP 7 — Support-safe mixture proposal  q = (1-a) * q_tilde + a * p0_disc
# =============================================================================

print("\n--- STEP 7: build support-safe proposal q ---")

# p0_disc_j: discrete p0 restricted to the cloud, normalised to sum=1.
p0_sum = p0_births.sum()
if not (np.isfinite(p0_sum) and p0_sum > 0.0):
    raise RuntimeError("Sum of p0 over cloud is not positive — check reactivity.")
p0_disc = p0_births / p0_sum

# q_tilde_j proportional to p0_j * s_score_j.
q_tilde_unnorm = p0_births * s_score
Z_score = q_tilde_unnorm.sum()
if Z_score > 0.0:
    q_tilde = q_tilde_unnorm / Z_score
else:
    # No overlap between p0 support and score support — fall back to p0_disc.
    print("  WARNING: q_tilde normaliser is zero; defaulting q to p0.")
    q_tilde = p0_disc.copy()

alpha = float(inp.alpha_mix)
if not (0.0 < alpha <= 1.0):
    raise ValueError(f"alpha_mix must satisfy 0 < alpha <= 1, got {alpha}")

q = (1.0 - alpha) * q_tilde + alpha * p0_disc
q_sum = q.sum()
if not (np.isfinite(q_sum) and q_sum > 0.0):
    raise RuntimeError("Proposal q has non-positive sum — aborting.")
q /= q_sum    # numerical cleanup

np.save(out_dir / "q_weights.npy", q)
np.save(out_dir / "q_tilde.npy", q_tilde)
np.save(out_dir / "p0_disc.npy", p0_disc)

print(f"  alpha                : {alpha:.3e}")
print(f"  Z_score (sum p0*s)   : {Z_score:.3e}")
print(f"  q support (q > 0)    : {int((q > 0).sum())}/{M_valid}")
print(f"  q_tilde support      : {int((q_tilde > 0).sum())}/{M_valid}")
print(f"  mixture bound 1/alpha: {1.0 / alpha:.3e}")


# =============================================================================
# STEP 8 — IS forward: sample from q and trace forward with drag only
# =============================================================================

print("\n--- STEP 8: IS forward tracing ---")
N_is = int(inp.N_is)
is_idx = rng.choice(M_valid, size=N_is, p=q, replace=True)
np.save(out_dir / "is_sample_idx.npy", is_idx)

R_is    = R_v[is_idx]
phi_is  = wrap_phi(phi_v[is_idx], phi_min, phi_max)
Z_is    = Z_v[is_idx]
vpar_is = vpar_v[is_idx]
H_is    = np.full(N_is, inp.H_fusion, dtype=np.float64)
stz_is  = flatten_stz(R_is, phi_is, Z_is)

t0 = time.time()
out = cartesian_gpu_tracing_drag(
    cell_quad_pts,
    np.ascontiguousarray(r_range,   dtype=np.float64),
    np.ascontiguousarray(phi_range, dtype=np.float64),
    np.ascontiguousarray(z_range,   dtype=np.float64),
    np.ascontiguousarray(stz_is,    dtype=np.float64),
    float(inp.mass), float(inp.charge), speed_ref,
    np.ascontiguousarray(vpar_is,   dtype=np.float64),
    np.ascontiguousarray(H_is,      dtype=np.float64),
    float(inp.coulomb_log), bool(inp.Te_in_eV),
    float(inp.tmax_forward), float(inp.tol), int(N_is),
    0.0, False,   # no energy stop in forward
)
fwd_is = np.asarray(out, dtype=np.float64).reshape(N_is, 7)
print(f"  IS forward done in {time.time() - t0:.2f}s")
np.save(out_dir / "forward_is_results.npy", fwd_is)

stop_codes_is = fwd_is[:, 6].astype(int)
print(f"  IS forward stop codes: {summarize_stop_codes(stop_codes_is)}")

A_is = (stop_codes_is == 1).astype(np.float64)  # wall-hit indicator
w_is = p0_disc[is_idx] / q[is_idx]               # p0 / q importance weights
Y_is = A_is * w_is                               # per-sample estimator
np.save(out_dir / "is_weights.npy", w_is)


# =============================================================================
# STEP 9 — Forward-only baseline (sample from p0 via fusion IC file)
# =============================================================================

print("\n--- STEP 9: baseline forward tracing ---")
fusion_ic = np.loadtxt(inp.fusion_ic_file, comments="#")
# Columns produced by 1_sample_fusion_distribution.py: R, phi, Z, vpar.
n_fus_avail = len(fusion_ic)
n_base = min(inp.n_baseline, n_fus_avail)
idx_base = rng.choice(n_fus_avail, size=n_base, replace=False)
R_base    = fusion_ic[idx_base, 0]
phi_base  = fusion_ic[idx_base, 1]
Z_base    = fusion_ic[idx_base, 2]
vpar_base = fusion_ic[idx_base, 3].astype(np.float64)
H_base    = np.full(n_base, inp.H_fusion, dtype=np.float64)

sd_base = sc_particle.evaluate_rphiz(
    np.column_stack([R_base, phi_base, Z_base])
).ravel()
inside_base = sd_base >= 0
n_out_base = int((~inside_base).sum())
if n_out_base:
    print(f"  dropping {n_out_base} baseline particles outside LCFS")
    R_base, phi_base, Z_base = R_base[inside_base], phi_base[inside_base], Z_base[inside_base]
    vpar_base = vpar_base[inside_base]
    H_base    = H_base[inside_base]
    n_base = int(inside_base.sum())

phi_base = wrap_phi(phi_base, phi_min, phi_max)
stz_base = flatten_stz(R_base, phi_base, Z_base)

t0 = time.time()
out = cartesian_gpu_tracing_drag(
    cell_quad_pts,
    np.ascontiguousarray(r_range,   dtype=np.float64),
    np.ascontiguousarray(phi_range, dtype=np.float64),
    np.ascontiguousarray(z_range,   dtype=np.float64),
    np.ascontiguousarray(stz_base,  dtype=np.float64),
    float(inp.mass), float(inp.charge), speed_ref,
    np.ascontiguousarray(vpar_base, dtype=np.float64),
    np.ascontiguousarray(H_base,    dtype=np.float64),
    float(inp.coulomb_log), bool(inp.Te_in_eV),
    float(inp.tmax_forward), float(inp.tol), int(n_base),
    0.0, False,
)
fwd_base = np.asarray(out, dtype=np.float64).reshape(n_base, 7)
print(f"  baseline forward done in {time.time() - t0:.2f}s")
np.save(out_dir / "forward_baseline_results.npy", fwd_base)

stop_codes_base = fwd_base[:, 6].astype(int)
print(f"  baseline forward stop codes: {summarize_stop_codes(stop_codes_base)}")

A_base = (stop_codes_base == 1).astype(np.float64)
Y_base = A_base     # Y_i = A_i when X_i ~ p0


# =============================================================================
# STEP 10 — Estimator metrics
#   Y_i = per-sample contribution (A_i for FWD; A_i*p0/q for IS).
#   Q_hat              = mean(Y_i)
#   Var(Q_hat)         = var(Y_i, ddof=1) / N
#   SE                 = sqrt(Var(Q_hat))
#   cv_estimator       = SE / Q_hat
#   cv_single_sample   = sqrt(var(Y_i)) / Q_hat
#   N_target(c)        = (cv_single_sample / c)**2, c in {0.10, 0.05, 0.02}
# =============================================================================

def estimator_metrics(Y, A, method_name, N, cv_targets=(0.10, 0.05, 0.02)):
    """
    Y: per-sample estimator contribution (A_i for FWD, A_i*p0/q for IS).
    A: unweighted wall-hit indicator (used only for the raw hit count).
    """
    Y = np.asarray(Y, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    Q_hat = float(Y.mean())
    var_sample = float(Y.var(ddof=1)) if Y.size > 1 else 0.0
    var_estimator = var_sample / N
    se = float(np.sqrt(var_estimator))
    if Q_hat > 0.0:
        cv_est  = se / Q_hat
        cv_one  = float(np.sqrt(var_sample)) / Q_hat
    else:
        cv_est = cv_one = float("nan")
    out = {
        "method":             method_name,
        "N":                  int(N),
        "N_wall_hits":        int((A > 0).sum()),
        "Q_hat":              Q_hat,
        "sample_variance":    var_sample,
        "estimator_variance": var_estimator,
        "standard_error":     se,
        "cv_estimator":       cv_est,
        "cv_single_sample":   cv_one,
    }
    for c in cv_targets:
        key = f"N_target_cv_{int(round(100 * c)):02d}pct"
        if np.isfinite(cv_one):
            out[key] = float((cv_one / c) ** 2)
        else:
            out[key] = float("nan")
    return out


m_base = estimator_metrics(Y_base, A_base, "FWD", N=n_base)
m_is   = estimator_metrics(Y_is,   A_is,   "IS",  N=N_is)
N_hits_base = m_base["N_wall_hits"]
N_hits_is   = m_is["N_wall_hits"]

# IS-specific diagnostics
w_stats = {
    "w_min":    float(w_is.min()),
    "w_max":    float(w_is.max()),
    "w_mean":   float(w_is.mean()),
    "w_median": float(np.median(w_is)),
    "w_std":    float(w_is.std(ddof=1)) if N_is > 1 else 0.0,
}
ess = float((w_is.sum()) ** 2 / np.sum(w_is ** 2)) if np.any(w_is > 0) else 0.0

vrf = (m_base["estimator_variance"] / m_is["estimator_variance"]
       if m_is["estimator_variance"] > 0 else float("inf"))

efficiency_gain = {}
for key in [k for k in m_base if k.startswith("N_target_cv_")]:
    nb = m_base[key]
    ni = m_is[key]
    if np.isfinite(nb) and np.isfinite(ni) and ni > 0:
        efficiency_gain[f"efficiency_gain_{key[9:]}"] = nb / ni
    else:
        efficiency_gain[f"efficiency_gain_{key[9:]}"] = float("nan")

summary_rows = [m_base, m_is]
is_extra = {
    "method":                          "IS_extras",
    "N":                               N_is,
    "n_wall_backward_attempted":       int(n_wall),
    "n_backward_successes":            int(M),
    "n_birth_valid_boozer":            int(M_valid),
    "backward_success_fraction":       float(M / n_wall),
    "backward_valid_fraction":         float(M_valid / n_wall),
    "alpha_mixture":                   float(alpha),
    "Z_score":                         float(Z_score),
    "sum_p0_disc":                     float(p0_sum),
    **w_stats,
    "effective_sample_size":           ess,
    "variance_reduction_factor":       float(vrf),
    **efficiency_gain,
}
summary_rows.append(is_extra)

# Write per-method CSV
with open(out_dir / "metrics_summary.csv", "w", newline="") as f:
    keys = sorted({k for row in summary_rows for k in row.keys()})
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    for row in summary_rows:
        writer.writerow({k: row.get(k, "") for k in keys})

print("\n--- STEP 10: estimator metrics ---")
for row in summary_rows:
    print(f"  {row['method']}:")
    for k, v in row.items():
        if k == "method":
            continue
        print(f"    {k:32s} = {v}")


# =============================================================================
# STEP 11 — Visualizations
# =============================================================================

print("\n--- STEP 11: plots ---")
pdir = out_dir / "plots"

# --- A. Wall starts vs birth endpoints (XY) ---------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(wall_xyz[:, 0], wall_xyz[:, 1], s=1, alpha=0.2,
           color="grey", label=f"wall starts ({n_wall})")
ax.scatter(X_b, Y_b, s=3, alpha=0.5, color="C1",
           label=f"birth endpoints ({M})")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_aspect("equal"); ax.legend()
ax.set_title("Wall starts vs backward-recovered birth endpoints")
fig.tight_layout(); fig.savefig(pdir / "wall_vs_birth_xy.png", dpi=150)
plt.close(fig)

# --- B. Birth endpoints in R-Z ---------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(R_b, Z_b, s=3, alpha=0.5, color="C1",
           label=f"birth endpoints ({M})")
ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
ax.set_aspect("equal"); ax.legend()
ax.set_title("Birth endpoints in R-Z")
fig.tight_layout(); fig.savefig(pdir / "birth_RZ.png", dpi=150)
plt.close(fig)

# --- C. s_histogram + reactivity -------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 1, 41)
ax.hist(np.clip(s_v, 0, 1), bins=bins, density=True, alpha=0.7,
        label=f"backward-born (n={M_valid})")
s_grid = np.linspace(0, 1, 200)
p0_grid = fusion_reactivity(s_grid)
if np.trapz(p0_grid, s_grid) > 0:
    p0_grid = p0_grid / np.trapz(p0_grid, s_grid)
ax.plot(s_grid, p0_grid, "k--", label="p0(s) = reactivity(s) (normalised)")
ax.set_xlabel("s (Boozer flux label)")
ax.set_ylabel("probability density")
ax.set_title("Backward-recovered births in s vs p0(s)")
ax.set_xlim(0, 1); ax.legend()
fig.tight_layout(); fig.savefig(pdir / "s_histogram.png", dpi=150)
plt.close(fig)

# --- D. Reactivity at birth endpoints --------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(p0_births, bins=40, color="C2", alpha=0.8)
ax.set_xlabel("p0(s) = reactivity(s_j)  [arb. units]")
ax.set_ylabel("count")
ax.set_title("p0 evaluated at backward-recovered births")
fig.tight_layout(); fig.savefig(pdir / "p0_at_births_histogram.png", dpi=150)
plt.close(fig)

# --- E. Stop-code bar chart (backward tracer) ------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
sc_counts = summarize_stop_codes(stop_codes_bwd)
_label = {0: "tmax", 1: "wall", 2: "H_fusion", 3: "invalid"}
keys = sorted(sc_counts.keys())
ax.bar([_label.get(k, str(k)) for k in keys],
       [sc_counts[k] for k in keys], color="C0")
ax.set_ylabel("count")
ax.set_title("Backward tracer stop codes")
fig.tight_layout(); fig.savefig(pdir / "backward_stop_codes.png", dpi=150)
plt.close(fig)

# --- F. Proposal-diagnostic plot: raw score, p0, q in Boozer s ------------
s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

# Continuous curves evaluated on s_grid
raw_score_curve = np.interp(s_grid, s_centers, s_hist.astype(float))
p0_curve = fusion_reactivity(s_grid)
q_curve = raw_score_curve * p0_curve
# Normalise each curve to integrate to 1 for visual comparison.
def _normalise_curve(x, y):
    area = np.trapz(np.maximum(y, 0.0), x)
    return y / area if area > 0 else y
raw_score_norm = _normalise_curve(s_grid, raw_score_curve)
p0_norm        = _normalise_curve(s_grid, p0_curve)
q_norm         = _normalise_curve(s_grid, q_curve)

# Also show the empirical forward-sample q distribution:
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(s_grid, raw_score_norm, label="raw backward score s_score(s) (normalised)",
        color="C0")
ax.plot(s_grid, p0_norm, label="p0(s) (normalised)", color="k", linestyle="--")
ax.plot(s_grid, q_norm, label="final proposal q(s) (normalised)",
        color="C3")
# Forward-drawn empirical q:
s_forward_is = s_v[is_idx]
ax.hist(s_forward_is, bins=s_edges, density=True, alpha=0.3, color="C2",
        label=f"forward birth samples from q (n={N_is})")
ax.set_xlabel("s")
ax.set_ylabel("probability density")
ax.set_xlim(0, 1)
ax.legend()
ax.set_title("Proposal diagnostic: raw score vs p0 vs final q in s")
fig.tight_layout(); fig.savefig(pdir / "proposal_components_s.png", dpi=150)
plt.close(fig)

# --- G. Forward birth samples under p0 vs under q (XY + RZ) ----------------
X_base_arr = R_base * np.cos(phi_base)
Y_base_arr = R_base * np.sin(phi_base)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].scatter(X_base_arr, Y_base_arr, s=2, alpha=0.3, color="C0",
               label=f"baseline ~p0 (N={n_base})")
axs[0].scatter(R_is * np.cos(phi_is), R_is * np.sin(phi_is),
               s=2, alpha=0.3, color="C3", label=f"IS ~q (N={N_is})")
axs[0].set_aspect("equal")
axs[0].set_xlabel("X [m]"); axs[0].set_ylabel("Y [m]")
axs[0].set_title("Forward birth samples (top-down)")
axs[0].legend()
axs[1].scatter(R_base, Z_base, s=2, alpha=0.3, color="C0",
               label=f"baseline ~p0 (N={n_base})")
axs[1].scatter(R_is, Z_is, s=2, alpha=0.3, color="C3",
               label=f"IS ~q (N={N_is})")
axs[1].set_aspect("equal")
axs[1].set_xlabel("R [m]"); axs[1].set_ylabel("Z [m]")
axs[1].set_title("Forward birth samples (R-Z)")
axs[1].legend()
fig.tight_layout(); fig.savefig(pdir / "forward_samples_p0_vs_q.png", dpi=150)
plt.close(fig)

# --- H. Wall-hit outcomes: baseline vs IS ----------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["confined (stop==0)", "wall hit (stop==1)", "other"]
base_counts = [
    int((stop_codes_base == 0).sum()),
    int((stop_codes_base == 1).sum()),
    int(n_base - (stop_codes_base == 0).sum() - (stop_codes_base == 1).sum()),
]
is_counts = [
    int((stop_codes_is == 0).sum()),
    int((stop_codes_is == 1).sum()),
    int(N_is - (stop_codes_is == 0).sum() - (stop_codes_is == 1).sum()),
]
x = np.arange(len(labels)); w = 0.35
ax.bar(x - w / 2, base_counts, width=w, label="baseline (~p0)", color="C0")
ax.bar(x + w / 2, is_counts,   width=w, label="IS (~q)",        color="C3")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("count")
ax.set_title("Forward outcomes: baseline vs IS")
ax.legend()
fig.tight_layout(); fig.savefig(pdir / "wall_hit_outcomes.png", dpi=150)
plt.close(fig)

# --- I. IS weight histogram (log scale) ------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
finite = w_is[np.isfinite(w_is) & (w_is > 0)]
if finite.size:
    ax.hist(finite, bins=np.logspace(np.log10(finite.min()),
                                     np.log10(finite.max() + 1e-300), 50),
            color="C4", alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("IS weight  w = p0 / q")
ax.set_ylabel("count")
ax.set_title(f"IS weights (ESS={ess:.1f} of N={N_is})")
fig.tight_layout(); fig.savefig(pdir / "is_weight_histogram.png", dpi=150)
plt.close(fig)

# --- J. Summary bar-chart of Q_hat, SE, cv ---------------------------------
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
axs[0].bar(["FWD", "IS"], [m_base["Q_hat"], m_is["Q_hat"]],
           yerr=[m_base["standard_error"], m_is["standard_error"]],
           color=["C0", "C3"], capsize=6)
axs[0].set_title("Q_hat ± SE")
axs[1].bar(["FWD", "IS"], [m_base["standard_error"], m_is["standard_error"]],
           color=["C0", "C3"])
axs[1].set_title("Standard error")
axs[2].bar(["FWD", "IS"], [m_base["cv_estimator"], m_is["cv_estimator"]],
           color=["C0", "C3"])
axs[2].set_title("cv_estimator = SE / Q_hat")
fig.tight_layout()
fig.savefig(pdir / "summary_qhat_se_cv.png", dpi=150)
plt.close(fig)

# --- K. N_target comparison (log scale) ------------------------------------
cv_labels = ["10%", "5%", "2%"]
base_N = [m_base[k] for k in ("N_target_cv_10pct", "N_target_cv_05pct",
                               "N_target_cv_02pct")]
is_N = [m_is[k] for k in ("N_target_cv_10pct", "N_target_cv_05pct",
                           "N_target_cv_02pct")]
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(3); w = 0.35
ax.bar(x - w / 2, base_N, width=w, label="FWD baseline", color="C0")
ax.bar(x + w / 2, is_N, width=w, label="IS", color="C3")
ax.set_xticks(x); ax.set_xticklabels(cv_labels)
ax.set_yscale("log")
ax.set_ylabel("Required N (log)")
ax.set_title("Samples required to reach target cv_estimator")
ax.legend()
fig.tight_layout()
fig.savefig(pdir / "summary_N_target.png", dpi=150)
plt.close(fig)

print(f"  wrote plots to {pdir}")


# =============================================================================
# STEP 12 — VTK point-cloud + trajectory snapshots (mirror BW-only script)
# =============================================================================

print("\n--- STEP 12: VTK exports ---")
write_points_vtu(out_dir / "wall_starts.vtu", wall_xyz,
                 point_data={"H_init": H_wall, "vpar_init": vtang_w})

birth_xyz_only = np.column_stack([X_b, Y_b, Z_b])
write_points_vtu(out_dir / "birth_endpoints.vtu", birth_xyz_only,
                 point_data={"vpar": vpar_b, "H": H_b, "s_boozer": s_b,
                             "valid_boozer": valid_bz.astype(np.float64),
                             "t_elapsed": t_b})

forward_is_xyz = np.column_stack([fwd_is[:, 1], fwd_is[:, 2], fwd_is[:, 3]])
write_points_vtu(out_dir / "forward_is_endpoints.vtu", forward_is_xyz,
                 point_data={"vpar": fwd_is[:, 4], "H": fwd_is[:, 5],
                             "stop_code": stop_codes_is.astype(np.float64),
                             "is_weight": w_is,
                             "t_elapsed": fwd_is[:, 0]})

forward_base_xyz = np.column_stack([fwd_base[:, 1], fwd_base[:, 2], fwd_base[:, 3]])
write_points_vtu(out_dir / "forward_baseline_endpoints.vtu", forward_base_xyz,
                 point_data={"vpar": fwd_base[:, 4], "H": fwd_base[:, 5],
                             "stop_code": stop_codes_base.astype(np.float64),
                             "t_elapsed": fwd_base[:, 0]})


# --- Backward trajectory snapshots (subsample of stop_code==2 successes) ---
print("\n--- STEP 12b: backward trajectory snapshots ---")
n_traj_bwd = int(min(inp.n_trajectory_backward, M))
if n_traj_bwd > 0:
    success_idx = np.flatnonzero(hit_fusion)
    traj_sel = rng.choice(success_idx, size=n_traj_bwd, replace=False)

    stz_traj = flatten_stz(R_wall[traj_sel], phi_wall[traj_sel], Z_wall[traj_sel])
    vtang_traj = vtang_w[traj_sel]
    H_traj     = H_wall[traj_sel]

    tmax_snaps = np.linspace(
        inp.tmax_backward / inp.n_snapshots,
        inp.tmax_backward,
        inp.n_snapshots,
    )
    traj_xyz  = np.zeros((n_traj_bwd, inp.n_snapshots, 3), dtype=np.float64)
    traj_vpar = np.zeros((n_traj_bwd, inp.n_snapshots),    dtype=np.float64)
    traj_H    = np.zeros((n_traj_bwd, inp.n_snapshots),    dtype=np.float64)
    traj_time = np.zeros((n_traj_bwd, inp.n_snapshots),    dtype=np.float64)

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
            float(tmax_i), float(inp.tol), int(n_traj_bwd),
            float(inp.H_fusion), True,
        )
        arr = np.asarray(out_i, dtype=np.float64).reshape(n_traj_bwd, 7)
        traj_time[:, i]   = arr[:, 0]
        traj_xyz[:, i, 0] = arr[:, 1]
        traj_xyz[:, i, 1] = arr[:, 2]
        traj_xyz[:, i, 2] = arr[:, 3]
        traj_vpar[:, i]   = arr[:, 4]
        traj_H[:, i]      = arr[:, 5]
    print(f"  backward trajectory tracing done in {time.time() - t0:.2f}s")

    np.save(out_dir / "bwd_trajectories_xyz.npy", traj_xyz)
    np.save(out_dir / "bwd_trajectories_vpar.npy", traj_vpar)
    np.save(out_dir / "bwd_trajectories_H.npy", traj_H)
    np.save(out_dir / "bwd_trajectories_time.npy", traj_time)

    wall_xyz_traj = wall_xyz[traj_sel]
    pts_per = 1 + inp.n_snapshots
    pts = np.empty((n_traj_bwd, pts_per, 3), dtype=np.float64)
    pts[:, 0, :]  = wall_xyz_traj
    pts[:, 1:, :] = traj_xyz
    t_pt    = np.empty((n_traj_bwd, pts_per)); t_pt[:, 0] = 0.0; t_pt[:, 1:] = traj_time
    vpar_pt = np.empty((n_traj_bwd, pts_per)); vpar_pt[:, 0] = vtang_traj; vpar_pt[:, 1:] = traj_vpar
    H_pt    = np.empty((n_traj_bwd, pts_per)); H_pt[:, 0]    = H_traj;    H_pt[:, 1:]    = traj_H
    write_polylines_vtu(out_dir / "bwd_trajectories.vtu",
                        pts_per_poly=pts_per,
                        pts=pts,
                        point_data={"time": t_pt, "vpar": vpar_pt, "H": H_pt},
                        cell_data={"particle_id": traj_sel.astype(np.int32)})


# --- Forward trajectory snapshots (IS subsample) ----------------------------
print("\n--- STEP 12c: forward (IS) trajectory snapshots ---")
n_traj_fwd = int(min(inp.n_trajectory_forward, N_is))
if n_traj_fwd > 0:
    fwd_sel = rng.choice(N_is, size=n_traj_fwd, replace=False)

    stz_fwd_traj = flatten_stz(R_is[fwd_sel], phi_is[fwd_sel], Z_is[fwd_sel])
    vtang_fwd_traj = vpar_is[fwd_sel]
    H_fwd_traj     = H_is[fwd_sel]

    tmax_snaps_fwd = np.linspace(
        inp.tmax_forward / inp.n_snapshots,
        inp.tmax_forward,
        inp.n_snapshots,
    )
    fwd_xyz  = np.zeros((n_traj_fwd, inp.n_snapshots, 3), dtype=np.float64)
    fwd_vpar = np.zeros((n_traj_fwd, inp.n_snapshots),    dtype=np.float64)
    fwd_H    = np.zeros((n_traj_fwd, inp.n_snapshots),    dtype=np.float64)
    fwd_time = np.zeros((n_traj_fwd, inp.n_snapshots),    dtype=np.float64)

    t0 = time.time()
    for i, tmax_i in enumerate(tmax_snaps_fwd):
        out_i = cartesian_gpu_tracing_drag(
            cell_quad_pts,
            np.ascontiguousarray(r_range,   dtype=np.float64),
            np.ascontiguousarray(phi_range, dtype=np.float64),
            np.ascontiguousarray(z_range,   dtype=np.float64),
            np.ascontiguousarray(stz_fwd_traj, dtype=np.float64),
            float(inp.mass), float(inp.charge), speed_ref,
            np.ascontiguousarray(vtang_fwd_traj, dtype=np.float64),
            np.ascontiguousarray(H_fwd_traj,     dtype=np.float64),
            float(inp.coulomb_log), bool(inp.Te_in_eV),
            float(tmax_i), float(inp.tol), int(n_traj_fwd),
            0.0, False,
        )
        arr = np.asarray(out_i, dtype=np.float64).reshape(n_traj_fwd, 7)
        fwd_time[:, i]   = arr[:, 0]
        fwd_xyz[:, i, 0] = arr[:, 1]
        fwd_xyz[:, i, 1] = arr[:, 2]
        fwd_xyz[:, i, 2] = arr[:, 3]
        fwd_vpar[:, i]   = arr[:, 4]
        fwd_H[:, i]      = arr[:, 5]
    print(f"  forward (IS) trajectory tracing done in {time.time() - t0:.2f}s")

    np.save(out_dir / "fwd_is_trajectories_xyz.npy",  fwd_xyz)
    np.save(out_dir / "fwd_is_trajectories_vpar.npy", fwd_vpar)
    np.save(out_dir / "fwd_is_trajectories_H.npy",    fwd_H)
    np.save(out_dir / "fwd_is_trajectories_time.npy", fwd_time)

    birth_xyz_sel = np.column_stack([
        R_is[fwd_sel] * np.cos(phi_is[fwd_sel]),
        R_is[fwd_sel] * np.sin(phi_is[fwd_sel]),
        Z_is[fwd_sel],
    ])
    pts_per = 1 + inp.n_snapshots
    pts = np.empty((n_traj_fwd, pts_per, 3), dtype=np.float64)
    pts[:, 0, :]  = birth_xyz_sel
    pts[:, 1:, :] = fwd_xyz
    t_pt    = np.empty((n_traj_fwd, pts_per)); t_pt[:, 0] = 0.0; t_pt[:, 1:] = fwd_time
    vpar_pt = np.empty((n_traj_fwd, pts_per)); vpar_pt[:, 0] = vtang_fwd_traj; vpar_pt[:, 1:] = fwd_vpar
    H_pt    = np.empty((n_traj_fwd, pts_per)); H_pt[:, 0]    = H_fwd_traj;    H_pt[:, 1:]    = fwd_H
    write_polylines_vtu(out_dir / "fwd_is_trajectories.vtu",
                        pts_per_poly=pts_per,
                        pts=pts,
                        point_data={"time": t_pt, "vpar": vpar_pt, "H": H_pt},
                        cell_data={"is_sample_idx": fwd_sel.astype(np.int32)})


# =============================================================================
# Done
# =============================================================================

print("\nDone.")
print(f"  backward success fraction (stop==2): {100 * bwd_success_frac:.2f}%")
print(f"  Q_hat_FWD = {m_base['Q_hat']:.6e} +/- {m_base['standard_error']:.3e}"
      f"  (N={n_base})")
print(f"  Q_hat_IS  = {m_is['Q_hat']:.6e} +/- {m_is['standard_error']:.3e}"
      f"  (N={N_is})")
print(f"  variance reduction factor (FWD/IS):  {vrf:.3e}")
print(f"  ESS (IS) = {ess:.1f}/{N_is}")
print(f"Outputs at: {out_dir}")
