"""
GPU guiding-centre tracing of fusion-born alpha particles in the
Landreman-Paul vacuum quasi-helically-symmetric (QH) stellarator equilibrium.

Expected layout (relative to the repo root):
  LandremanPaulQH_coils/
    coils.curves_22_7_21   — MAKEGRID coil file
    input.vmec             — VMEC input (LCFS boundary)
  1_IC_sample_1e6_points/outputs/
    initial_conditions_cylindrical.txt — (R, phi, Z, vpar) per particle
    initial_conditions_boozer.txt      — (s, theta, zeta, vpar) per particle
"""

import os
import time
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

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
    ALPHA_PARTICLE_CHARGE  as CHARGE,
    ALPHA_PARTICLE_MASS    as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)

from firm3d.util.gpu_utils import cartesian_interpolant
from firm3dpp import cartesian_gpu_tracing

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR  = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent
COILS_DIR = REPO_ROOT / "LandremanPaulQH_coils"
IC_DIR    = REPO_ROOT / "1_IC_sample_1e6_points" / "outputs"


# ── Input parameters ──────────────────────────────────────────────────────────

@dataclass
class Inputs:
    # Files
    coil_file:       Path = COILS_DIR / "coils.curves_22_7_21"
    vmec_input_file: Path = COILS_DIR / "input.vmec"
    ic_file_cyl:     Path = IC_DIR / "initial_conditions_cylindrical.txt"
    ic_file_boozer:  Path = IC_DIR / "initial_conditions_boozer.txt"
    nparticles:      int  = 1_000_000   # number of particles to load (set to -1 for all)

    # Equilibrium
    nfp:     int   = 4                   # number of field periods
    ncoils:       int   = 5                   # unique base coil shapes per half field period
    current:      float = 1.27797548115612e7  # coil current [A] — from extcur.curves_22_7_21
    coil_order:   int   = 20                  # Fourier order for coil curve representation

    # Interpolation grid
    n_r:    int = 64   # grid cells in R
    n_phi:  int = 128  # grid cells in φ
    n_z:    int = 64   # grid cells in Z
    degree: int = 3    # spline degree (must be 3 for the GPU CUDA kernel)
    nphi_surf:   int = 128  # surface resolution used for B·n check and VTK output
    ntheta_surf: int = 64

    # SurfaceClassifier (loss criterion)
    sc_h: float = 0.05  # grid spacing [m] — smaller is more accurate near the boundary
    sc_p: int   = 2     # interpolant degree

    # Tracing
    tmax: float = 1e-2  # max integration time [s]
    tol:  float = 1e-9  # ODE solver tolerance


inp = Inputs()

OUT_DIR = str(THIS_DIR / "output") + "/"
os.makedirs(OUT_DIR, exist_ok=True)


# ── 1. Load coils ─────────────────────────────────────────────────────────────
# load_coils_from_makegrid_file returns ncoils × nfp coils, omitting
# stellarator-symmetric images.  coils_via_symmetries reconstructs the full
# set with stellarator symmetry.

all_coils     = load_coils_from_makegrid_file(str(inp.coil_file), order=inp.coil_order)
base_curves   = [all_coils[i].curve for i in range(inp.ncoils)]
base_currents = [Current(inp.current) for _ in range(inp.ncoils)]

coils  = coils_via_symmetries(base_curves, base_currents, inp.nfp, stellsym=True)
curves = [c.curve for c in coils]
bs     = BiotSavart(coils)


# ── 2. Load plasma boundary from VMEC input ──────────────────────────────────
# The VMEC boundary represents first wall in this example

s_input = SurfaceRZFourier.from_vmec_input(
    str(inp.vmec_input_file), range="full torus",
    nphi=inp.nphi_surf, ntheta=inp.ntheta_surf,
)


# ── 3. B·n check ─────────────────────────────────────────────────────────────
# |B·n̂|/|B| should be small on VMEC's LCFS for precise coils.
# A large value means the coil field does not reproduce the target equilibrium.

bs.set_points(s_input.gamma().reshape((-1, 3)))
B   = bs.B().reshape((inp.nphi_surf, inp.ntheta_surf, 3))
BN  = np.sum(B * s_input.unitnormal(), axis=2)
rel = np.abs(BN) / np.linalg.norm(B, axis=2)
print(f"B·n check: mean |B·n|/|B| = {rel.mean():.2e},  max = {rel.max():.2e}")


# ── 4. Save geometry to VTK ───────────────────────────────────────────────────

bs.set_points(s_input.gamma().reshape((-1, 3)))
B_on_surf = bs.B().reshape((inp.nphi_surf, inp.ntheta_surf, 3))
B_N       = np.sum(B_on_surf * s_input.unitnormal(), axis=2)
absB      = np.linalg.norm(B_on_surf, axis=2)

curves_to_vtk(curves, OUT_DIR + "coils_LPQH", close=True)
s_input.to_vtk(OUT_DIR + "surface_LPQH", extra_data={
    "B_N":            B_N[:, :, None],
    "abs_B_N_over_B": (np.abs(B_N) / absB)[:, :, None],
})
print(f"VTK files written to {OUT_DIR}")


# ── 5. Build interpolated field on a cylindrical grid ────────────────────────

sc_particle = SurfaceClassifier(s_input, h=inp.sc_h, p=inp.sc_p)

rs    = np.linalg.norm(s_input.gamma()[:, :, 0:2], axis=2)
z_max = np.max(np.abs(s_input.gamma()[:, :, 2]))

rrange   = (np.min(rs), np.max(rs), inp.n_r)
phirange = (0, 2 * np.pi / inp.nfp, inp.n_phi)
zrange   = (0, z_max, inp.n_z)

bsh = InterpolatedField(
    bs, inp.degree, rrange, phirange, zrange, True, nfp=inp.nfp, stellsym=True
)
print(f"Interpolation grid: {inp.n_r}(R) × {inp.n_phi}(φ) × {inp.n_z}(Z)")
print("  error in B:       ", bsh.estimate_error_B(1000))
print("  error in GradAbsB:", bsh.estimate_error_GradAbsB(1000))


# ── 6. Build the GPU interpolant ──────────────────────────────────────────────
t0 = time.time()
r_range, phi_range, z_range, cell_quad_pts = cartesian_interpolant(
    field=bsh,
    sc_particle=sc_particle,
    nfp=inp.nfp,              # unused by the function, grid is read from bsh
    n_metagrid_pts=inp.n_r,   # unused by the function, grid is read from bsh
)
print(f"GPU interpolant built in {time.time()-t0:.1f}s")


# ── 7. Load initial conditions ────────────────────────────────────────────────
# Cylindrical columns: R_init  phi_init  Z_init  vpar_init
# Boozer columns:      s_init  theta_init  zeta_init  vpar_init
# ic_index tracks each particle's original row in the IC files so Boozer
# coordinates can be recovered after filtering and after tracing.

ic_cyl    = np.loadtxt(inp.ic_file_cyl,    comments="#")
ic_boozer = np.loadtxt(inp.ic_file_boozer, comments="#")
n_in_file = len(ic_cyl)

if inp.nparticles != -1 and inp.nparticles < n_in_file:
    ic_cyl    = ic_cyl[:inp.nparticles]
    ic_boozer = ic_boozer[:inp.nparticles]
elif inp.nparticles > n_in_file:
    print(f"Requested {inp.nparticles} particles but file contains only {n_in_file}; using all.")

ic_index   = np.arange(len(ic_cyl))          # original row index in the IC files
R_init     = ic_cyl[:, 0]
phi_init   = ic_cyl[:, 1]
Z_init     = ic_cyl[:, 2]
vtang      = np.ascontiguousarray(ic_cyl[:, 3], dtype=np.float64)
boozer_init = ic_boozer[:, :3]               # s, theta, zeta  (drop duplicate vpar column)
nparticles = len(R_init)
print(f"Loaded {nparticles} particles from {inp.ic_file_cyl.name}")

# Remove any particles that lie outside the LCFS.  A small fraction can end up
# there due to imprecision in the Boozer→cylindrical coordinate conversion.
sd     = sc_particle.evaluate_rphiz(np.column_stack([R_init, phi_init, Z_init])).ravel()
inside = sd >= 0
n_out  = int(np.sum(~inside))
print(f"\nSigned-distance check: {n_out}/{nparticles} particles outside LCFS "
      f"(sd min={sd.min():.3f}  mean={sd.mean():.3f})")

if n_out > 0:
    np.save(OUT_DIR + "initial_outside_rphiz.npy",
            np.column_stack([R_init[~inside], phi_init[~inside], Z_init[~inside]]))
    R_init      = R_init[inside]
    phi_init    = phi_init[inside]
    Z_init      = Z_init[inside]
    vtang       = vtang[inside]
    boozer_init = boozer_init[inside]
    ic_index    = ic_index[inside]
    nparticles  = int(inside.sum())
    print(f"  Removed {n_out} outside particles; tracing {nparticles}.")

# GPU tracing expects Cartesian positions: [X0, Y0, Z0, X1, Y1, Z1, ...]
stz_init = np.empty(3 * nparticles, dtype=np.float64)
stz_init[0::3] = R_init * np.cos(phi_init)
stz_init[1::3] = R_init * np.sin(phi_init)
stz_init[2::3] = Z_init


# ── 8. GPU particle tracing ───────────────────────────────────────────────────
# Integrates the guiding-centre equations on the GPU until tmax or until a
# particle crosses the LCFS (signed distance < 0).
# Returns [t_final, X, Y, Z, v_par] per particle (X, Y, Z in Cartesian metres).

t0 = time.time()
results = cartesian_gpu_tracing(
    quad_pts=cell_quad_pts,
    srange=np.ascontiguousarray(r_range,   dtype=np.float64),
    trange=np.ascontiguousarray(phi_range, dtype=np.float64),
    zrange=np.ascontiguousarray(z_range,   dtype=np.float64),
    stz_init=np.ascontiguousarray(stz_init, dtype=np.float64),
    m=MASS,
    q=CHARGE,
    vtotal=sqrt(2 * ENERGY / MASS),
    vtang=vtang,
    tmax=inp.tmax,
    tol=inp.tol,
    nparticles=nparticles,
)
print(f"GPU tracing done in {time.time()-t0:.2f}s")


# ── 9. Compute loss fraction ──────────────────────────────────────────────────
# Particles that stopped before 99% of tmax are considered lost to the wall.

results    = np.array(results, dtype=np.float64).reshape(nparticles, 5)
t_final    = results[:, 0]
X_final    = results[:, 1]
Y_final    = results[:, 2]
Z_final    = results[:, 3]
vpar_final = results[:, 4]

R_final   = np.sqrt(X_final**2 + Y_final**2)
phi_final = np.arctan2(Y_final, X_final)

lost_mask = t_final < 0.99 * inp.tmax
print(f"Particles lost: {lost_mask.sum()}/{nparticles} ({100*lost_mask.mean():.1f}%)")


# ── 10. Save results ──────────────────────────────────────────────────────────
# Row i corresponds to ic_index[i] in the input IC files.
# Lost/confined subsets duplicate data in full array, saved for convenience

confined_mask = ~lost_mask

X_init = R_init * np.cos(phi_init)
Y_init = R_init * np.sin(phi_init)
rphiz_init = np.column_stack([R_init, phi_init, Z_init])
xyz_init   = np.column_stack([X_init, Y_init, Z_init])

# Initial conditions of all traced particles
np.save(OUT_DIR + "initial_ic_index.npy",  ic_index)
np.save(OUT_DIR + "initial_xyz.npy",       xyz_init)
np.save(OUT_DIR + "initial_rphiz.npy",     rphiz_init)
np.save(OUT_DIR + "initial_boozer.npy",    boozer_init)
np.save(OUT_DIR + "initial_vtang.npy",     vtang)

# Final state of all traced particles
np.save(OUT_DIR + "final_xyz.npy",         np.column_stack([X_final, Y_final, Z_final]))
np.save(OUT_DIR + "final_rphiz.npy",       np.column_stack([R_final, phi_final, Z_final]))
np.save(OUT_DIR + "final_vpar.npy",        vpar_final)
np.save(OUT_DIR + "final_time.npy",        t_final)

# Lost particles (subset of initial/final arrays above)
np.save(OUT_DIR + "lost_ic_index.npy",       ic_index[lost_mask])
np.save(OUT_DIR + "lost_initial_xyz.npy",    xyz_init[lost_mask])
np.save(OUT_DIR + "lost_initial_rphiz.npy",  rphiz_init[lost_mask])
np.save(OUT_DIR + "lost_initial_boozer.npy", boozer_init[lost_mask])

# Confined particles (subset of initial/final arrays above)
np.save(OUT_DIR + "confined_ic_index.npy",       ic_index[confined_mask])
np.save(OUT_DIR + "confined_initial_xyz.npy",    xyz_init[confined_mask])
np.save(OUT_DIR + "confined_initial_rphiz.npy",  rphiz_init[confined_mask])
np.save(OUT_DIR + "confined_initial_boozer.npy", boozer_init[confined_mask])

print(f"Results saved to {OUT_DIR}")
