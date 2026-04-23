"""
GPU guiding-centre tracing of fusion-born alpha particles through
*perturbed* coil configurations — robustness study.

For each run, coil curves are perturbed by a smooth Gaussian random field
that models manufacturing tolerances.  Two perturbation layers are applied
(matching the approach in simsopt's stage_two_optimization_stochastic.py):

  Systematic error — applied to base curves *before* stellarator symmetry
                     is expanded.  Every symmetric copy of a base coil
                     shares the same error pattern.  Models errors that
                     are correlated across the machine (e.g. a tilted coil
                     form that shifts all copies by the same amount).

  Statistical error — applied *independently* to every coil after symmetry
                      expansion.  Models uncorrelated per-coil assembly
                      scatter (e.g. individual winding tolerances).

Usage
-----
  python 1_trace_perturbed.py --perturbation_id 0    # unperturbed baseline
  python 1_trace_perturbed.py --perturbation_id 3    # perturbed run, seed=3

On Perlmutter, submit the full ensemble with run_perlmutter.sh (array job).

Output
------
All files are tagged by perturbation_id and written to output/:
  loss_summary_NNNN.npy    — [pert_id, nparticles, n_lost, loss_fraction, sigma]
  lost_initial_boozer_NNNN.npy — (s, theta, zeta) of lost particles at birth
  initial_boozer_NNNN.npy  — (s, theta, zeta) of all traced particles at birth
  final_time_NNNN.npy      — integration stop-time per particle
  bn_stats_NNNN.npy        — [mean, max] of |B·n|/|B| on LCFS
"""

import argparse
import os
import time
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np
from numpy.random import PCG64DXSM, Generator

from simsopt.field import (
    BiotSavart,
    Current,
    Coil,
    InterpolatedField,
    SurfaceClassifier,
    coils_via_symmetries,
)
from simsopt.field.coil import load_coils_from_makegrid_file
from simsopt.geo import (
    SurfaceRZFourier,
    GaussianSampler,
    CurvePerturbed,
    PerturbationSample,
)
from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE          as CHARGE,
    ALPHA_PARTICLE_MASS            as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY   as ENERGY,
)

from firm3d.util.gpu_utils import cartesian_interpolant
from firm3dpp import cartesian_gpu_tracing

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR  = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent
COILS_DIR = REPO_ROOT / "LandremanPaulQH_coils"
IC_DIR    = REPO_ROOT / "1_IC_sample_1e6_points" / "outputs"


# ── Input parameters ───────────────────────────────────────────────────────────

@dataclass
class Inputs:
    # Files
    coil_file:       Path = COILS_DIR / "coils.curves_22_7_21"
    vmec_input_file: Path = COILS_DIR / "input.vmec"
    ic_file_cyl:     Path = Path("/pscratch/sd/m/mariagar/projects/mc_proj/IC/initial_conditions_cylindrical.txt")
    ic_file_boozer:  Path = Path("/pscratch/sd/m/mariagar/projects/mc_proj/IC/initial_conditions_boozer.txt")
    nparticles:      int  = 10_000   # particles per ensemble member (~4 min/run)

    # Equilibrium
    nfp:        int   = 4
    ncoils:     int   = 5                   # unique base coil shapes per half field period
    current:    float = 1.27797548115612e7  # coil current [A]
    coil_order: int   = 20

    # Coil perturbation
    # sigma = 1 mm is a realistic tight manufacturing tolerance;
    # try sigma = 3 mm or 5 mm to probe the sensitivity more aggressively.
    sigma:  float = 1e-2   # Gaussian std dev of coil displacement [m]  (1e-3 -> 1mm)
    length: float = 0.5    # spatial correlation length [m] (smoothness)

    # Interpolation grid (same as 2_tracing_gpu)
    n_r:    int = 64
    n_phi:  int = 128
    n_z:    int = 64
    degree: int = 3        # must be 3 for the GPU CUDA kernel
    nphi_surf:   int = 128
    ntheta_surf: int = 64

    # SurfaceClassifier (loss criterion)
    sc_h: float = 0.05
    sc_p: int   = 2

    # Tracing
    tmax: float = 1e-2     # max integration time [s]
    tol:  float = 1e-9


inp = Inputs()

# ── Command-line argument: perturbation ID ─────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Trace alpha particles through a perturbed coil configuration."
)
parser.add_argument(
    "--perturbation_id", type=int, default=0,
    help="0 = exact (baseline) coils; >0 = random seed for Gaussian perturbation",
)
args    = parser.parse_args()
pert_id = 57    #args.perturbation_id

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = str(Path("/pscratch/sd/m/mariagar/projects/mc_proj/results/robustness") / timestamp) + "/"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Writing outputs to {OUT_DIR}")

print(f"\n{'='*62}")
if pert_id == 0:
    print(f"  Perturbation ID: {pert_id}  —  BASELINE (exact coils)")
else:
    print(f"  Perturbation ID: {pert_id}  —  sigma={inp.sigma:.1e} m, "
          f"L={inp.length:.2f} m, seed={pert_id}")
print(f"{'='*62}\n")


# ── 1. Load and (optionally) perturb coils ────────────────────────────────────
# load_coils_from_makegrid_file returns ncoils × nfp coils (without
# stellarator-symmetric images).  coils_via_symmetries reconstructs the full
# set with stellarator symmetry.

all_coils     = load_coils_from_makegrid_file(str(inp.coil_file), order=inp.coil_order)
base_curves   = [all_coils[i].curve for i in range(inp.ncoils)]
base_currents = [Current(inp.current) for _ in range(inp.ncoils)]

if pert_id == 0:
    # ---- Baseline: exact coils, no perturbation --------------------------------
    coils = coils_via_symmetries(base_curves, base_currents, inp.nfp, stellsym=True)
    print("Using exact (unperturbed) coils.")

else:
    # ---- Perturbed coils -------------------------------------------------------
    #
    # GaussianSampler draws a smooth periodic random displacement parameterised
    # by arc length along each coil.  The same sampler is reused for both layers;
    # each call to PerturbationSample draws a fresh independent sample.
    #
    # PCG64DXSM is a high-quality PRNG; seeding by pert_id makes every run
    # fully reproducible while ensuring distinct perturbations per run.

    rg      = Generator(PCG64DXSM(pert_id))
    sampler = GaussianSampler(
        base_curves[0].quadpoints,
        inp.sigma,
        inp.length,
        n_derivs=1,     # also perturbs first derivative → smooth field change
    )

    # Layer 1 — systematic error (same error propagated through stellarator sym.)
    base_curves_pert = [
        CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg))
        for c in base_curves
    ]
    coils_sym = coils_via_symmetries(
        base_curves_pert, base_currents, inp.nfp, stellsym=True
    )

    # Layer 2 — statistical error (independent sample per final coil)
    coils = [
        Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current)
        for c in coils_sym
    ]
    print(f"Gaussian perturbation applied: sigma={inp.sigma:.1e} m, "
          f"L={inp.length:.2f} m, seed={pert_id}")

curves = [c.curve for c in coils]
bs     = BiotSavart(coils)


# ── 2. Load plasma boundary from VMEC input ───────────────────────────────────
# The VMEC LCFS is treated as the first wall.

s_input = SurfaceRZFourier.from_vmec_input(
    str(inp.vmec_input_file), range="full torus",
    nphi=inp.nphi_surf, ntheta=inp.ntheta_surf,
)


# ── 3. B·n check ──────────────────────────────────────────────────────────────
# |B·n̂|/|B| measures how well the (perturbed) coils reproduce the target LCFS.
# Baseline: ~0.01.  Perturbed coils will show larger values — this is one
# diagnostic for how "bad" a particular perturbation realisation is.

bs.set_points(s_input.gamma().reshape((-1, 3)))
B   = bs.B().reshape((inp.nphi_surf, inp.ntheta_surf, 3))
BN  = np.sum(B * s_input.unitnormal(), axis=2)
rel = np.abs(BN) / np.linalg.norm(B, axis=2)
print(f"B·n check:  mean |B·n|/|B| = {rel.mean():.4e},  max = {rel.max():.4e}")

tag = f"_{pert_id:04d}"
np.save(OUT_DIR + f"bn_stats{tag}.npy", np.array([rel.mean(), rel.max()]))


# ── 4. Build interpolated field on a cylindrical grid ─────────────────────────

sc_particle = SurfaceClassifier(s_input, h=inp.sc_h, p=inp.sc_p)

rs    = np.linalg.norm(s_input.gamma()[:, :, 0:2], axis=2)
z_max = np.max(np.abs(s_input.gamma()[:, :, 2]))

rrange   = (np.min(rs), np.max(rs), inp.n_r)
phirange = (0, 2 * np.pi / inp.nfp, inp.n_phi)
zrange   = (0, z_max, inp.n_z)

# stellsym=True note:
#   Layer 1 (systematic) preserves stellarator symmetry — the same error
#   pattern is applied to every base curve before coils_via_symmetries, so
#   all symmetric copies carry the identical displacement.
#   Layer 2 (statistical) *breaks* stellarator symmetry — each coil gets
#   an independent random sample, so the full coil set is no longer symmetric.
#
#   Using stellsym=True here therefore introduces an approximation for
#   perturbed runs: the interpolation grid covers only one half-period and
#   the field is assumed to repeat, which is only true on average across the
#   ensemble.  P small sigma (≤ a few mm) this error is negligible compared
#   to the perturbation itself, and it keeps memory and build-time the same as
#   the baseline.  Set stellsym=False (and double n_phi) for a fully exact
#   treatment of the broken symmetry.
bsh = InterpolatedField(
    bs, inp.degree, rrange, phirange, zrange, True, nfp=inp.nfp, stellsym=True
)
print(f"Interpolation grid: {inp.n_r}(R) × {inp.n_phi}(φ) × {inp.n_z}(Z)")
print("  error in B:       ", bsh.estimate_error_B(1000))
print("  error in GradAbsB:", bsh.estimate_error_GradAbsB(1000))


# ── 5. Build the GPU interpolant ───────────────────────────────────────────────
t0 = time.time()
r_range, phi_range, z_range, cell_quad_pts = cartesian_interpolant(
    field=bsh,
    sc_particle=sc_particle,
    nfp=inp.nfp,
    n_metagrid_pts=inp.n_r,
)
print(f"GPU interpolant built in {time.time()-t0:.1f}s")


# ── 6. Load initial conditions ─────────────────────────────────────────────────
# Use the same first-N particles for every run so differences in loss fraction
# come purely from the coil perturbation, not from IC sampling variance.

ic_cyl    = np.loadtxt(inp.ic_file_cyl,    comments="#")
ic_boozer = np.loadtxt(inp.ic_file_boozer, comments="#")

if inp.nparticles > 0 and inp.nparticles < len(ic_cyl):
    ic_cyl    = ic_cyl[:inp.nparticles]
    ic_boozer = ic_boozer[:inp.nparticles]

ic_index    = np.arange(len(ic_cyl))
R_init      = ic_cyl[:, 0]
phi_init    = ic_cyl[:, 1]
Z_init      = ic_cyl[:, 2]
vtang       = np.ascontiguousarray(ic_cyl[:, 3], dtype=np.float64)
boozer_init = ic_boozer[:, :3]   # s, theta, zeta
nparticles  = len(R_init)
print(f"Loaded {nparticles} particles from {inp.ic_file_cyl.name}")

# Remove any particles that are outside the LCFS (coordinate conversion artefact)
sd     = sc_particle.evaluate_rphiz(np.column_stack([R_init, phi_init, Z_init])).ravel()
inside = sd >= 0
n_out  = int(np.sum(~inside))
print(f"Signed-distance check: {n_out}/{nparticles} particles outside LCFS "
      f"(sd min={sd.min():.3f}  mean={sd.mean():.3f})")

if n_out > 0:
    R_init      = R_init[inside]
    phi_init    = phi_init[inside]
    Z_init      = Z_init[inside]
    vtang       = vtang[inside]
    boozer_init = boozer_init[inside]
    ic_index    = ic_index[inside]
    nparticles  = int(inside.sum())
    print(f"  Removed {n_out} outside particles; tracing {nparticles}.")

# Convert cylindrical → Cartesian for the GPU kernel: [X0, Y0, Z0, X1, ...]
stz_init = np.empty(3 * nparticles, dtype=np.float64)
stz_init[0::3] = R_init * np.cos(phi_init)
stz_init[1::3] = R_init * np.sin(phi_init)
stz_init[2::3] = Z_init


# ── 7. GPU particle tracing ────────────────────────────────────────────────────
# Integrates the guiding-centre equations on the GPU until tmax or until a
# particle crosses the LCFS (signed distance < 0).
# Returns [t_final, X, Y, Z, v_par] per particle.

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
    nparticles=nparticles,
)
print(f"GPU tracing done in {time.time()-t0:.2f}s")


# ── 8. Compute loss fraction ───────────────────────────────────────────────────
# Particles that stopped before 99% of tmax hit the wall.

results       = np.array(results, dtype=np.float64).reshape(nparticles, 5)
t_final       = results[:, 0]
lost_mask     = t_final < 0.99 * inp.tmax
loss_fraction = lost_mask.mean()

print(f"\nPerturbation {pert_id:4d} — Lost: {lost_mask.sum()}/{nparticles} "
      f"({100 * loss_fraction:.3f}%)")


# ── 9. Save results ────────────────────────────────────────────────────────────
# Files are tagged _NNNN so all ensemble members can share the output/ dir.

np.save(OUT_DIR + f"initial_boozer{tag}.npy",    boozer_init)
np.save(OUT_DIR + f"final_time{tag}.npy",        t_final)
np.save(OUT_DIR + f"lost_initial_boozer{tag}.npy", boozer_init[lost_mask])

# Compact one-row summary for the plotting script:
#   [pert_id, nparticles, n_lost, loss_fraction, sigma]
np.save(
    OUT_DIR + f"loss_summary{tag}.npy",
    np.array([
        float(pert_id),
        float(nparticles),
        float(lost_mask.sum()),
        loss_fraction,
        inp.sigma if pert_id > 0 else 0.0,
    ]),
)
print(f"Results saved to {OUT_DIR}")
