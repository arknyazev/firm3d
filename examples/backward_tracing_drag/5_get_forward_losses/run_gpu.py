"""
Sample fusion-born alpha particles, trace them on GPU, save only the lost ones.

Each call generates a fresh batch of `nparticles` ICs in memory (never written
to disk), filters those outside the LCFS, runs GPU guiding-centre tracing, and
writes the lost-particle data to output/ tagged with a unique run_tag.

Rerun (or resubmit the SLURM job) as many times as needed to accumulate losses.
Each 1M-particle run yields ~250 lost particles; 4 GPUs in parallel → ~1000/job.

Usage:
    python run_gpu.py --gpu_id 0 --run_tag 20240402_143022_job12345_gpu0
"""

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.coordinates import boozer_to_cylindrical
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from firm3d.util.constants import ALPHA_PARTICLE_MASS, FUSION_ALPHA_PARTICLE_ENERGY
from firm3d.util.gpu_utils import cartesian_interpolant
from firm3d.util.mpi import comm_world

from firm3dpp import cartesian_gpu_tracing

from simsopt.field import (
    BiotSavart,
    Current,
    InterpolatedField,
    SurfaceClassifier,
    coils_via_symmetries,
)
from simsopt.field.coil import load_coils_from_makegrid_file
from simsopt.geo import SurfaceRZFourier
from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE        as CHARGE,
    ALPHA_PARTICLE_MASS          as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR  = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent
COILS_DIR = REPO_ROOT / "LandremanPaulQH_coils"


# ── Input parameters ───────────────────────────────────────────────────────────

@dataclass
class Inputs:
    # Files
    coil_file:       Path = COILS_DIR / "coils.curves_22_7_21"
    vmec_input_file: Path = COILS_DIR / "input.vmec"
    boozmn_file:     Path = COILS_DIR / "boozmn.nc"

    # IC sampling
    nparticles:    int = 1_000_000
    resolution:    int = 48   # Boozer field interpolation grid points per dimension
    radial_order:  int = 3
    spline_degree: int = 3

    # Equilibrium
    nfp:        int   = 4
    ncoils:     int   = 5
    current:    float = 1.27797548115612e7
    coil_order: int   = 20

    # Interpolation grid
    n_r:    int = 64
    n_phi:  int = 128
    n_z:    int = 64
    degree: int = 3
    nphi_surf:   int = 128
    ntheta_surf: int = 64

    # SurfaceClassifier
    sc_h: float = 0.05
    sc_p: int   = 2

    # Tracing
    tmax: float = 1e-2
    tol:  float = 1e-9


inp = Inputs()

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id",  type=int, default=0,
                    help="GPU index for logging (CUDA_VISIBLE_DEVICES set externally)")
parser.add_argument("--run_tag", type=str, default=None,
                    help="Unique tag for output files; auto-generated if omitted")
args    = parser.parse_args()
gpu_id  = args.gpu_id
run_tag = args.run_tag or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu{gpu_id}"

OUT_DIR = str(THIS_DIR / "output") + "/"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n[GPU {gpu_id}] run_tag = {run_tag}")


# ── 1. Build Boozer field interpolant ──────────────────────────────────────────
print(f"[GPU {gpu_id}] Building Boozer field interpolant from {inp.boozmn_file.name}...")
bri = BoozerRadialInterpolant(
    str(inp.boozmn_file), inp.radial_order, no_K=True, comm=comm_world
)
boozer_field = InterpolatedBoozerField(
    bri,
    inp.spline_degree,
    ns_interp=inp.resolution,
    ntheta_interp=inp.resolution,
    nzeta_interp=inp.resolution,
)


# ── 2. Sample ICs from D-T fusion birth distribution ──────────────────────────
# Reactivity ∝ n_D(s) · n_T(s) · <σv>(T(s))  (same model as 1_IC_sample_1e6_points)

def sigmav(T_keV):
    if T_keV > 0:
        return T_keV ** (-2 / 3) * np.exp(-19.94 * T_keV ** (-1 / 3))
    return 0.0

nD         = lambda s: 1 - s**5
nT         = nD
T_keV_fn   = lambda s: 11.5 * (1 - s)
reactivity = lambda s: nD(s) * nT(s) * sigmav(T_keV_fn(s))

vpar0 = sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)

t0 = time.time()
print(f"[GPU {gpu_id}] Sampling {inp.nparticles:,} ICs...")
points_boozer = initialize_position_profile(
    boozer_field, inp.nparticles, reactivity, comm=comm_world
)
vpar_init  = initialize_velocity_uniform(vpar0, inp.nparticles, comm=comm_world)
points_cyl = boozer_to_cylindrical(boozer_field, points_boozer)
n_sampled  = len(points_cyl)
print(f"[GPU {gpu_id}] IC sampling done in {time.time()-t0:.1f}s")


# ── 3. Load coils + LCFS surface ───────────────────────────────────────────────
all_coils     = load_coils_from_makegrid_file(str(inp.coil_file), order=inp.coil_order)
base_curves   = [all_coils[i].curve for i in range(inp.ncoils)]
base_currents = [Current(inp.current) for _ in range(inp.ncoils)]
coils  = coils_via_symmetries(base_curves, base_currents, inp.nfp, stellsym=True)
bs     = BiotSavart(coils)

s_input = SurfaceRZFourier.from_vmec_input(
    str(inp.vmec_input_file), range="full torus",
    nphi=inp.nphi_surf, ntheta=inp.ntheta_surf,
)


# ── 4. Filter particles outside LCFS ──────────────────────────────────────────
sc_particle = SurfaceClassifier(s_input, h=inp.sc_h, p=inp.sc_p)

R_init   = points_cyl[:, 0]
phi_init = points_cyl[:, 1]
Z_init   = points_cyl[:, 2]

sd        = sc_particle.evaluate_rphiz(np.column_stack([R_init, phi_init, Z_init])).ravel()
inside    = sd >= 0
n_outside = int(np.sum(~inside))
print(f"[GPU {gpu_id}] Outside LCFS: {n_outside}/{n_sampled} "
      f"(sd min={sd.min():.3f}  mean={sd.mean():.3f})")

boozer_init = points_boozer[inside]
R_init      = R_init[inside]
phi_init    = phi_init[inside]
Z_init      = Z_init[inside]
vtang       = np.ascontiguousarray(vpar_init[inside], dtype=np.float64)
n_traced    = int(inside.sum())
print(f"[GPU {gpu_id}] Tracing {n_traced:,} particles...")

# Cartesian for GPU kernel: [X0, Y0, Z0, X1, Y1, Z1, ...]
stz_init = np.empty(3 * n_traced, dtype=np.float64)
stz_init[0::3] = R_init * np.cos(phi_init)
stz_init[1::3] = R_init * np.sin(phi_init)
stz_init[2::3] = Z_init


# ── 5. Build interpolated field + GPU interpolant ─────────────────────────────
rs    = np.linalg.norm(s_input.gamma()[:, :, 0:2], axis=2)
z_max = np.max(np.abs(s_input.gamma()[:, :, 2]))

rrange   = (np.min(rs), np.max(rs), inp.n_r)
phirange = (0, 2 * np.pi / inp.nfp, inp.n_phi)
zrange   = (0, z_max, inp.n_z)

bsh = InterpolatedField(
    bs, inp.degree, rrange, phirange, zrange, True, nfp=inp.nfp, stellsym=True
)

print(f"[GPU {gpu_id}] Grid: {int(rrange[2])}(R) × {int(phirange[2])}(φ) × {int(zrange[2])}(Z)  "
      f"B-error={bsh.estimate_error_B(1000):.2e}")

t0 = time.time()
r_range, phi_range, z_range, cell_quad_pts = cartesian_interpolant(
    field=bsh, sc_particle=sc_particle, nfp=inp.nfp, n_metagrid_pts=inp.n_r,
)
print(f"[GPU {gpu_id}] GPU interpolant built in {time.time()-t0:.1f}s")


# ── 6. GPU tracing ─────────────────────────────────────────────────────────────
t0 = time.time()
results = cartesian_gpu_tracing(
    quad_pts=cell_quad_pts,
    srange=np.ascontiguousarray(r_range,    dtype=np.float64),
    trange=np.ascontiguousarray(phi_range,  dtype=np.float64),
    zrange=np.ascontiguousarray(z_range,    dtype=np.float64),
    stz_init=np.ascontiguousarray(stz_init, dtype=np.float64),
    m=MASS,
    q=CHARGE,
    vtotal=sqrt(2 * ENERGY / MASS),
    vtang=vtang,
    tmax=inp.tmax,
    tol=inp.tol,
    nparticles=n_traced,
)
print(f"[GPU {gpu_id}] GPU tracing done in {time.time()-t0:.2f}s")


# ── 7. Extract lost particles ──────────────────────────────────────────────────
results    = np.array(results, dtype=np.float64).reshape(n_traced, 5)
t_final    = results[:, 0]
X_final    = results[:, 1]
Y_final    = results[:, 2]
Z_final    = results[:, 3]
vpar_final = results[:, 4]

R_final   = np.sqrt(X_final**2 + Y_final**2)
phi_final = np.arctan2(Y_final, X_final)

lost_mask     = t_final < 0.99 * inp.tmax
n_lost        = int(lost_mask.sum())
loss_fraction = n_lost / n_traced

print(f"[GPU {gpu_id}] Lost: {n_lost}/{n_traced} ({100 * loss_fraction:.4f}%)")


# ── 8. Save lost particles only ────────────────────────────────────────────────
# Files are tagged by run_tag so reruns accumulate without overwriting.

tag    = f"_{run_tag}"
X_init = R_init * np.cos(phi_init)
Y_init = R_init * np.sin(phi_init)

np.save(OUT_DIR + f"lost_initial_xyz{tag}.npy",
        np.column_stack([X_init[lost_mask], Y_init[lost_mask], Z_init[lost_mask]]))
np.save(OUT_DIR + f"lost_initial_rphiz{tag}.npy",
        np.column_stack([R_init[lost_mask], phi_init[lost_mask], Z_init[lost_mask]]))
np.save(OUT_DIR + f"lost_initial_boozer{tag}.npy",  boozer_init[lost_mask])
np.save(OUT_DIR + f"lost_initial_vtang{tag}.npy",   vtang[lost_mask])
np.save(OUT_DIR + f"lost_final_xyz{tag}.npy",
        np.column_stack([X_final[lost_mask], Y_final[lost_mask], Z_final[lost_mask]]))
np.save(OUT_DIR + f"lost_final_rphiz{tag}.npy",
        np.column_stack([R_final[lost_mask], phi_final[lost_mask], Z_final[lost_mask]]))
np.save(OUT_DIR + f"lost_final_vpar{tag}.npy",      vpar_final[lost_mask])
np.save(OUT_DIR + f"lost_final_time{tag}.npy",      t_final[lost_mask])

# Run statistics: how many particles passed through each stage
# columns: n_sampled, n_outside_lcfs, n_traced, n_lost, loss_fraction
np.save(OUT_DIR + f"run_stats{tag}.npy",
        np.array([n_sampled, n_outside, n_traced, n_lost, loss_fraction]))

print(f"[GPU {gpu_id}] Saved {n_lost} lost particles → {OUT_DIR}")
print(f"[GPU {gpu_id}] Stats: sampled={n_sampled:,}  outside={n_outside}  "
      f"traced={n_traced:,}  lost={n_lost}  loss_frac={loss_fraction:.4e}")
