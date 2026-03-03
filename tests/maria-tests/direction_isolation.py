import time
import os
from math import sqrt
import numpy as np

from simsopt.configs import get_data
from simsopt.field import InterpolatedField, SurfaceClassifier
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.util import proc0_print
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from simsopt.field.sampling import draw_uniform_on_curve

from firm3d.util.gpu_utils import cartesian_interpolant
from firm3dpp import cartesian_gpu_tracing, cartesian_gpu_tracing_backward



# copied from alex's example (tracing_gpu_ncsx.py)
# ── Parameters ──────────────────────────────────────────────────────────

# Particle parameters:
nparticles = 16 # number of particles to trace
energy = 500 * ONE_EV # kinetic energy of particles [J]
mass = PROTON_MASS  # mass of particles [kg]
charge = ELEMENTARY_CHARGE # charge of particles [C]

tmax_values = np.arange(1e-7, 1e-4 + 1e-7, 1e-7) # trajectory snapshots [s]

seed = 1 # random seed for initial conditions

tol = 1e-9 # ODE solver tolerance

degree = 3 # degree of interpolation for the InterpolatedField
# (for Maria: can this only be 3 for GPU interpolant?)

n = 16 # number of interpolation cells per direction

# Directory for output
out_dir = "./output/"
os.makedirs(out_dir, exist_ok=True)

# ── 1. Magnetic field configuration ─────────────────────────────────────
base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
all_curves = [c.curve for c in bs.coils]

proc0_print("Mean(|B|) on axis =",
            np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
proc0_print("Mean(Axis radius) =",
            np.mean(np.linalg.norm(ma.gamma(), axis=1)))

# ── 2. Save coils to VTK ───────────────────────────────────────────────
curves_to_vtk(all_curves + [ma], out_dir + "coils")
proc0_print(f"Saved coils VTK to {out_dir}coils.vtu")

# ── 3. Build intersection surface and save to VTK ──────────────────────
mpol = 5
ntor = 5
stellsym = True
s = SurfaceRZFourier.from_nphi_ntheta(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
    range="full torus", nphi=64, ntheta=24
)
s.fit_to_curve(ma, 0.20, flip_theta=False)

# Save the surface to VTK
s.to_vtk(out_dir + "surface")
proc0_print(f"Saved surface VTK to {out_dir}surface.vts")

# Build the signed-distance classifier
sc_particle = SurfaceClassifier(s, h=0.1, p=2)

# ── 4. Build the simsopt InterpolatedField ──────────────────────────────
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]

rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2 * np.pi / nfp, n * 2)
zrange = (0, np.max(zs), n // 2)

bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
)
proc0_print("Error in B:      ", bsh.estimate_error_B(1000), flush=True)
proc0_print("Error in GradAbsB:", bsh.estimate_error_GradAbsB(1000), flush=True)

# ── 5. Build the GPU interpolant ────────────────────────────────────────
# cartesian_interpolant evaluates B_cyl, GradAbsB_cyl, and the signed
# distance function on the interpolation grid, then reorders data into
# the cell-local memory layout expected by the CUDA kernel.
t1 = time.time()
r_range, phi_range, z_range, cell_quad_pts = cartesian_interpolant(
    bsh, sc_particle, nfp, n
)
t2 = time.time()
proc0_print(f"GPU interpolant built in {t2-t1:.3f}s", flush=True)

# ── 6. Sample initial conditions on the magnetic axis ───────────────────
speed_total = sqrt(2 * energy / mass)
np.random.seed(seed)

# Draw nparticles points uniformly on the magnetic axis
xyz_init, _ = draw_uniform_on_curve(ma, nparticles, safetyfactor=10)

# Convert XYZ -> cylindrical (R, phi, Z)
R_init = np.sqrt(xyz_init[:, 0]**2 + xyz_init[:, 1]**2)
phi_init = np.arctan2(xyz_init[:, 1], xyz_init[:, 0])
Z_init = xyz_init[:, 2]

# Flatten into [R0, phi0, Z0, R1, phi1, Z1, ...] layout
stz_init = np.empty(3 * nparticles, dtype=np.float64)
stz_init[0::3] = R_init
stz_init[1::3] = phi_init
stz_init[2::3] = Z_init

# Random pitch angles: v_par = u * v_total, u in [-1, 1]
us = np.random.uniform(-1, 1, size=nparticles)
vtang = (us * speed_total).astype(np.float64)

# Save initial conditions
np.save(out_dir + "particles_initial_xyz.npy", xyz_init)
np.save(out_dir + "particles_initial_rphiz.npy",
        np.column_stack([R_init, phi_init, Z_init]))
np.save(out_dir + "particles_initial_vtang.npy", vtang)
proc0_print(f"Saved {nparticles} initial particle positions to {out_dir}")


# TESTS

def run_cartesian(tracer, cell_quad_pts, r_range, phi_range, z_range,
                  stz_init, mass, charge, speed_total, vtang, tmax, tol, nparticles):
    results = tracer(
        cell_quad_pts,
        np.array(r_range, dtype=np.float64),
        np.array(phi_range, dtype=np.float64),
        np.array(z_range, dtype=np.float64),
        np.asarray(stz_init, dtype=np.float64),
        float(mass),
        float(charge),
        float(speed_total),
        np.asarray(vtang, dtype=np.float64),
        float(tmax),
        float(tol),
        int(nparticles),
    )
    return np.asarray(results, dtype=np.float64).reshape(int(nparticles), 5)

def wrap_phi(phi, phi_min, phi_max):
    """
    Wrap phi into [phi_min, phi_max).
    This makes the cylindrical init consistent with the interpolant domain.
    """
    period = phi_max - phi_min
    return (phi - phi_min) % period + phi_min

def xyz_to_stz_flat(xyz, phi_min, phi_max):
    """
    Convert Nx3 XYZ -> flattened cylindrical [R0,phi0,Z0, R1,phi1,Z1, ...]
    with phi wrapped into [phi_min, phi_max).
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    R = np.sqrt(x*x + y*y)
    phi = np.arctan2(y, x)
    phi = wrap_phi(phi, phi_min, phi_max)

    stz = np.empty(3 * xyz.shape[0], dtype=np.float64)
    stz[0::3] = R
    stz[1::3] = phi
    stz[2::3] = z
    return stz

# ----------------------------
# Phase A.0: Make initial phi consistent with phi_range
# ----------------------------
phi_min = float(phi_range[0])
phi_max = float(phi_range[1])

stz_init = np.asarray(stz_init, dtype=np.float64).copy()
stz_init[1::3] = wrap_phi(stz_init[1::3], phi_min, phi_max)

# ----------------------------
# Choose test tmax
# ----------------------------
#test_tmax = float(tmax_values[len(tmax_values)//2])  # mid value
test_tmax = 1e-7
proc0_print(f"\nPhase A tests with tmax={test_tmax:.2e}s", flush=True)

# ----------------------------
# Phase A.1: Direction-flag contamination test
#   Baseline: forward then forward again (no backward call between)
#   Probe:    forward -> backward -> forward, compare forward outputs
# ----------------------------
proc0_print("\n[A1] Direction-flag contamination test", flush=True)

f0 = run_cartesian(cartesian_gpu_tracing,
                   cell_quad_pts, r_range, phi_range, z_range,
                   stz_init, mass, charge, speed_total, vtang,
                   test_tmax, tol, nparticles)

f1 = run_cartesian(cartesian_gpu_tracing,
                   cell_quad_pts, r_range, phi_range, z_range,
                   stz_init, mass, charge, speed_total, vtang,
                   test_tmax, tol, nparticles)

baseline = np.linalg.norm(f1[:, 1:] - f0[:, 1:], axis=1)  # compare xyz+vpar

b  = run_cartesian(cartesian_gpu_tracing_backward,
                   cell_quad_pts, r_range, phi_range, z_range,
                   stz_init, mass, charge, speed_total, vtang,
                   test_tmax, tol, nparticles)

f2 = run_cartesian(cartesian_gpu_tracing,
                   cell_quad_pts, r_range, phi_range, z_range,
                   stz_init, mass, charge, speed_total, vtang,
                   test_tmax, tol, nparticles)

after = np.linalg.norm(f2[:, 1:] - f1[:, 1:], axis=1)

proc0_print(f"median baseline ||f1-f0|| = {np.median(baseline):.3e}", flush=True)
proc0_print(f"median after    ||f2-f1|| = {np.median(after):.3e}", flush=True)

# A simple, scale-aware trigger:
scale = 1.0 + np.median(np.linalg.norm(f1[:, 1:], axis=1))
if np.median(after) > 100 * np.median(baseline) and np.median(after) > 1e-10 * scale:
    proc0_print("WARNING: forward results changed much more after backward call "
                "than the forward/forward baseline. This is consistent with a per-call "
                "state (e.g., dir flag) not being reset.", flush=True)

# ----------------------------
# Phase A.2: Closure test (forward then backward from the forward final state)
#   forward:  (R,phi,Z, vtang) -> (x,y,z,vpar)
#   backward: start from (x,y,z) converted to (R,phi,Z), with vtang := vpar_final
# ----------------------------

proc0_print("\n[A2] Closure test: x0 -> forward -> x1 -> backward -> x0'", flush=True)

# forward output
fwd = f1
t_f = fwd[:, 0]
lost_mask = t_f < 0.99 * test_tmax
keep = ~lost_mask

proc0_print(f"Not-lost particles for closure: {np.sum(keep)}/{nparticles}", flush=True)

if np.sum(keep) > 0:
    n_keep = int(np.sum(keep))

    # start backward from final position of forward run (convert xyz -> (R,phi,Z))
    xyz1 = fwd[keep, 1:4].astype(np.float64)          # (N,3) final xyz
    stz_b_init = xyz_to_stz_flat(xyz1, phi_min, phi_max)                # flattened cylindrical

    # vtang input expects parallel velocity; use vpar from forward output
    vtang_b = fwd[keep, 4].astype(np.float64)         # (N,) final vpar

    bwd_from_fwd = run_cartesian(cartesian_gpu_tracing_backward,
                                 cell_quad_pts, r_range, phi_range, z_range,
                                 stz_b_init, mass, charge, speed_total, vtang_b,
                                 test_tmax, tol, n_keep)

    xyz0 = xyz_init[keep, :].astype(np.float64)       # original xyz
    xyz_back = bwd_from_fwd[:, 1:4]                   # returned xyz

    vpar_back = bwd_from_fwd[:, 4]
    # Compare to original vtang for those particles (sign may depend on convention)
    v0 = vtang[keep].astype(np.float64)

    dv_same = np.abs(vpar_back - v0)
    dv_flip = np.abs(vpar_back + v0)

    proc0_print(f"median |vpar_back - v0|      = {np.median(dv_same):.3e} m/s", flush=True)
    proc0_print(f"median |vpar_back + v0|      = {np.median(dv_flip):.3e} m/s", flush=True)


    err = np.linalg.norm(xyz_back - xyz0, axis=1)
    proc0_print(f"median closure ||x0'-x0|| = {np.median(err):.3e} m", flush=True)
    proc0_print(f"max    closure ||x0'-x0|| = {np.max(err):.3e} m", flush=True)
else:
    proc0_print("All particles were lost in forward run; closure test skipped.", flush=True)