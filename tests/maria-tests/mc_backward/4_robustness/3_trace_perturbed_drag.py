"""GPU guiding-centre tracing of fusion-born alpha particles with
DETERMINISTIC DRAG through *perturbed* coil configurations.

Same structure as ``1_trace_perturbed.py`` (coil perturbation model,
stellarator symmetry, loss detection, summary outputs) — but:

  * uses ``cartesian_interpolant_drag`` + ``cartesian_gpu_tracing_drag``
    (matches the forward-tracer path used by the ``mc_comparison/`` scripts),
  * carries the same drag defaults as ``mc_comparison`` (``ne0=1e21``,
    ``Te0_ev=100``, ``coulomb_log=17``),
  * loss detection uses the explicit ``stop_code == 1`` wall hit flag
    (the drag tracer returns 7 cols: ``t, X, Y, Z, vpar, H, stop_code``),
  * accepts ``--sigma`` as a CLI argument (no sed wrapper needed),
  * writes into ``{--out_base}/{sigma}/{timestamp}/`` so that 4 parallel
    processes with different sigmas never collide.

Usage
-----
  python 3_trace_perturbed_drag.py --perturbation_id 57 --sigma 1e-2
  python 3_trace_perturbed_drag.py --perturbation_id 57 --sigma 5e-3 \
      --out_base /pscratch/sd/m/mariagar/projects/mc_proj/results/robustness

On Perlmutter, submit the full sigma ensemble with ``run_sigma_scan.sh``.
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

from firm3d.util.gpu_utils import cartesian_interpolant_drag
from firm3dpp import cartesian_gpu_tracing_drag


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
    nparticles:      int  = 50_000

    # Equilibrium
    nfp:        int   = 4
    ncoils:     int   = 5
    current:    float = 1.27797548115612e7
    coil_order: int   = 20

    # Coil perturbation (sigma is overridden by --sigma on the command line)
    sigma:  float = 1e-2
    length: float = 0.5

    # Interpolation grid
    n_r:    int = 64
    n_phi:  int = 128
    n_z:    int = 64
    degree: int = 3
    nphi_surf:   int = 128
    ntheta_surf: int = 64

    # SurfaceClassifier (loss criterion)
    sc_h: float = 0.05
    sc_p: int   = 2

    # Tracing
    tmax: float = 1e-2
    tol:  float = 1e-9

    # Drag physics (match mc_comparison defaults)
    ne0:         float = 1e21
    Te0_ev:      float = 100.0
    coulomb_log: float = 17.0
    Te_in_eV:    bool  = True


inp = Inputs()


# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Trace fusion alphas with deterministic drag through a "
                "perturbed coil configuration."
)
parser.add_argument("--perturbation_id", type=int, default=57,
                    help="0 = exact coils; >0 = random seed for the "
                         "Gaussian perturbation.  Default 57.")
parser.add_argument("--sigma", type=str, default="1e-2",
                    help="Gaussian std dev of coil displacement [m].  "
                         "Passed as a string so the output path preserves "
                         "the human-readable format (1e-2, 5e-3, ...).")
parser.add_argument("--nparticles", type=int, default=inp.nparticles,
                    help="Number of particles to trace.")
parser.add_argument("--out_base", type=Path,
                    default=Path("/pscratch/sd/m/mariagar/projects/mc_proj/"
                                 "results/robustness"),
                    help="Base output dir; a {sigma}/{timestamp}/ "
                         "subtree is created underneath.")
parser.add_argument("--tmax", type=float, default=inp.tmax)
parser.add_argument("--tol",  type=float, default=inp.tol)
parser.add_argument("--ne0",         type=float, default=inp.ne0)
parser.add_argument("--Te0_ev",      type=float, default=inp.Te0_ev)
parser.add_argument("--coulomb_log", type=float, default=inp.coulomb_log)
args = parser.parse_args()

# Keep the raw sigma string for the output path, parse to float for physics.
sigma_str      = args.sigma
inp.sigma      = float(args.sigma)
inp.nparticles = args.nparticles
inp.tmax       = args.tmax
inp.tol        = args.tol
inp.ne0        = args.ne0
inp.Te0_ev     = args.Te0_ev
inp.coulomb_log = args.coulomb_log

pert_id = args.perturbation_id

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR = str(Path(args.out_base) / sigma_str / timestamp) + "/"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Writing outputs to {OUT_DIR}")

print(f"\n{'='*62}")
if pert_id == 0:
    print(f"  Perturbation ID: {pert_id}  —  BASELINE (exact coils)")
else:
    print(f"  Perturbation ID: {pert_id}  —  sigma={inp.sigma:.1e} m "
          f"({sigma_str}), L={inp.length:.2f} m, seed={pert_id}")
print(f"{'='*62}\n")


# ── 1. Load and (optionally) perturb coils ────────────────────────────────────

all_coils     = load_coils_from_makegrid_file(str(inp.coil_file), order=inp.coil_order)
base_curves   = [all_coils[i].curve for i in range(inp.ncoils)]
base_currents = [Current(inp.current) for _ in range(inp.ncoils)]

if pert_id == 0:
    coils = coils_via_symmetries(base_curves, base_currents, inp.nfp, stellsym=True)
    print("Using exact (unperturbed) coils.")

else:
    rg = Generator(PCG64DXSM(pert_id))
    sampler = GaussianSampler(
        base_curves[0].quadpoints,
        inp.sigma,
        inp.length,
        n_derivs=1,
    )

    # Layer 1 — systematic error
    base_curves_pert = [
        CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg))
        for c in base_curves
    ]
    coils_sym = coils_via_symmetries(
        base_curves_pert, base_currents, inp.nfp, stellsym=True
    )

    # Layer 2 — statistical error
    coils = [
        Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current)
        for c in coils_sym
    ]
    print(f"Gaussian perturbation applied: sigma={inp.sigma:.1e} m, "
          f"L={inp.length:.2f} m, seed={pert_id}")

curves = [c.curve for c in coils]
bs     = BiotSavart(coils)


# ── 2. Load plasma boundary from VMEC input ───────────────────────────────────

s_input = SurfaceRZFourier.from_vmec_input(
    str(inp.vmec_input_file), range="full torus",
    nphi=inp.nphi_surf, ntheta=inp.ntheta_surf,
)


# ── 3. B·n check ──────────────────────────────────────────────────────────────

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

# stellsym=True caveat for perturbed runs — same as in 1_trace_perturbed.py
bsh = InterpolatedField(
    bs, inp.degree, rrange, phirange, zrange, True, nfp=inp.nfp, stellsym=True
)
print(f"Interpolation grid: {inp.n_r}(R) × {inp.n_phi}(φ) × {inp.n_z}(Z)")
print("  error in B:       ", bsh.estimate_error_B(1000))
print("  error in GradAbsB:", bsh.estimate_error_GradAbsB(1000))


# ── 5. Build the GPU DRAG interpolant ──────────────────────────────────────────

def ne_fun(rphiz):
    return np.full(rphiz.shape[0], inp.ne0, dtype=np.float64)

def Te_fun(rphiz):
    return np.full(rphiz.shape[0], inp.Te0_ev, dtype=np.float64)

t0 = time.time()
r_range, phi_range, z_range, cell_quad_pts = cartesian_interpolant_drag(
    field=bsh,
    sc_particle=sc_particle,
    ne_fun=ne_fun,
    Te_fun=Te_fun,
    nfp=inp.nfp,
    n_metagrid_pts=inp.n_r,
)
print(f"GPU drag interpolant built in {time.time()-t0:.1f}s")


# ── 6. Load initial conditions ─────────────────────────────────────────────────

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

stz_init = np.empty(3 * nparticles, dtype=np.float64)
stz_init[0::3] = R_init
stz_init[1::3] = phi_init
stz_init[2::3] = Z_init

H_init = np.full(nparticles, ENERGY, dtype=np.float64)


# ── 7. GPU drag tracing ────────────────────────────────────────────────────────
# Integrates the guiding-centre equations on the GPU with deterministic drag
# until tmax or wall crossing.  Returns [t, X, Y, Z, v_par, H, stop_code]
# per particle (7 cols).  stop_code == 1 iff the particle hit the wall.

speed_ref = float(sqrt(2.0 * ENERGY / MASS))

t0 = time.time()
results = cartesian_gpu_tracing_drag(
    cell_quad_pts,
    np.ascontiguousarray(r_range,    dtype=np.float64),
    np.ascontiguousarray(phi_range,  dtype=np.float64),
    np.ascontiguousarray(z_range,    dtype=np.float64),
    np.ascontiguousarray(stz_init,   dtype=np.float64),
    float(MASS), float(CHARGE), speed_ref,
    np.ascontiguousarray(vtang,      dtype=np.float64),
    np.ascontiguousarray(H_init,     dtype=np.float64),
    float(inp.coulomb_log), bool(inp.Te_in_eV),
    float(inp.tmax), float(inp.tol), int(nparticles),
    0.0, False,   # no energy stop in forward
)
print(f"GPU drag tracing done in {time.time()-t0:.2f}s")


# ── 8. Compute loss fraction ───────────────────────────────────────────────────

results       = np.array(results, dtype=np.float64).reshape(nparticles, 7)
t_final       = results[:, 0]
stop_codes    = results[:, 6].astype(int)
lost_mask     = stop_codes == 1
loss_fraction = lost_mask.mean()

uniq, counts = np.unique(stop_codes, return_counts=True)
print(f"\nStop-code breakdown: {dict(zip(uniq.tolist(), counts.tolist()))}")
print(f"Perturbation {pert_id:4d} — sigma={sigma_str:>6} — "
      f"Lost: {int(lost_mask.sum())}/{nparticles} ({100 * loss_fraction:.3f}%)")


# ── 9. Save results ────────────────────────────────────────────────────────────

np.save(OUT_DIR + f"initial_boozer{tag}.npy",       boozer_init)
np.save(OUT_DIR + f"final_time{tag}.npy",           t_final)
np.save(OUT_DIR + f"stop_codes{tag}.npy",           stop_codes.astype(np.int32))
np.save(OUT_DIR + f"results{tag}.npy",              results)
np.save(OUT_DIR + f"lost_initial_boozer{tag}.npy",  boozer_init[lost_mask])

# Compact one-row summary (same 5 fields as 1_trace_perturbed.py for
# plotting-script compatibility).
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
