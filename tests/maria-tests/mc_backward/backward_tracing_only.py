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

Outputs (timestamped dir under outputs_backward_only/)
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
    wall_ic_file:    Path = THIS_DIR / "3_IC_sample_wall" / "outputs" / "initial_conditions_surface_cylindrical.txt"

    # Equilibrium / coils
    nfp:        int   = 4
    ncoils:     int   = 5
    current:    float = 1.27797548115612e7
    coil_order: int   = 20

    # Interpolant grid (must match 2_tracing_gpu conventions)
    n_r:    int = 64
    n_phi:  int = 128
    n_z:    int = 64
    degree: int = 3
    nphi_surf:   int = 128
    ntheta_surf: int = 64

    # SurfaceClassifier
    sc_h: float = 0.05
    sc_p: int   = 2

    # Boozer interpolant (for cylindrical_to_boozer)
    radial_order:    int = 3
    boozer_degree:   int = 3
    boozer_res:      int = 48

    # Physics — energy sampling explicitly [3.0, 3.5] MeV per task spec
    mass:        float = MASS
    charge:      float = CHARGE
    H_low:       float = 3.0e6 * ONE_EV
    H_high:      float = H_FUSION          # 3.5 MeV — D-T alpha birth energy
    H_fusion:    float = H_FUSION
    coulomb_log: float = 17.0
    Te_in_eV:    bool  = True
    ne0:         float = 5e19
    Te0_ev:      float = 5e3

    # Tracing
    n_wall:         int   = 50_000
    tmax_backward:  float = 5e-5
    tol:            float = 1e-9
    seed:           int   = 57


inp = Inputs()

# ── Output directory ─────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = THIS_DIR / "outputs_backward_only" / timestamp
(out_dir / "plots").mkdir(parents=True, exist_ok=True)
print(f"Writing outputs to {out_dir}")


# ── Helpers (reused from backward_informed_mc / export_points_vtk style) ─────

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


# Fusion reactivity (for a reference p_0(s) histogram overlay) —
# identical to 1_IC_sample_1e6_points and backward_informed_mc.
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
# Mirrors 2_tracing_gpu/tracing_gpu.py and backward_informed_mc.py.

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
# Wall IC contains positions only (R, phi, Z) — per 3_IC_sample_wall script.
R_all   = wall_ic[:, 0]
phi_all = wall_ic[:, 1]
Z_all   = wall_ic[:, 2]
n_avail = len(R_all)
n_wall  = min(inp.n_wall, n_avail)
print(f"  available: {n_avail}, using {n_wall}")

rng = np.random.default_rng(inp.seed)
idx = rng.choice(n_avail, size=n_wall, replace=False)
R_wall, phi_wall, Z_wall = R_all[idx], phi_all[idx], Z_all[idx]

# Drop points outside the LCFS (Boozer->cyl roundoff can nudge points out).
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

# Sample uniform pitch and uniform energy per task spec.
lam_wall  = rng.uniform(-1.0, 1.0, size=n_wall)
H_wall    = rng.uniform(inp.H_low, inp.H_high, size=n_wall)
v_total_w = np.sqrt(2.0 * H_wall / inp.mass)
vtang_w   = lam_wall * v_total_w

# Flatten positions to the [R0,phi0,Z0, R1,phi1,Z1,...] layout the tracer wants.
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

# Convert to Boozer via BOOZ_XFORM field.
print("  building Boozer interpolant + converting to Boozer...")
bri = BoozerRadialInterpolant(str(inp.boozmn_file), inp.radial_order, no_K=True)
boozer_field = InterpolatedBoozerField(
    bri, inp.boozer_degree,
    ns_interp=inp.boozer_res,
    ntheta_interp=inp.boozer_res,
    nzeta_interp=inp.boozer_res,
)
boozer_coords = cylindrical_to_boozer(boozer_field, birth_rphiz)
s_b = boozer_coords[:, 0]
np.save(out_dir / "birth_endpoints_boozer.npy", boozer_coords)

# "Valid" Boozer coordinates == lies inside the plasma, s in [0, 1].
valid_bz = (s_b >= 0.0) & (s_b <= 1.0) & np.isfinite(s_b)
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

# Segments from wall start -> birth endpoint, one per successful particle.
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


# ── Step 6: matplotlib figures ──────────────────────────────────────────────

print("\n--- STEP 6: plots ---")
pdir = out_dir / "plots"

# 1. Wall hits (starts) vs birth endpoints, top-down XY.
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

# 2. Birth endpoints in R-Z.
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(R_b, Z_b, s=3, alpha=0.5, color="C1",
           label=f"birth endpoints ({M})")
ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
ax.set_aspect("equal"); ax.legend()
ax.set_title("Birth endpoints in R–Z")
fig.tight_layout(); fig.savefig(pdir / "birth_RZ.png", dpi=150)
plt.close(fig)

# 3. s histogram — clip to [0, 1] for the view.
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

# 4. Reactivity evaluated at each birth-endpoint s.
fig, ax = plt.subplots(figsize=(7, 4))
reac_at_births = fusion_reactivity(np.clip(s_for_hist, 0, 1))
ax.hist(reac_at_births, bins=40, alpha=0.7, color="C2")
ax.set_xlabel("fusion reactivity p0(s)  [arb. units]")
ax.set_ylabel("count")
ax.set_title("Reactivity at birth endpoints")
fig.tight_layout(); fig.savefig(pdir / "reactivity_histogram.png", dpi=150)
plt.close(fig)

# 5. Stop-code bar chart.
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
