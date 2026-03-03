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


# helper
def run_cartesian(tracer, cell_quad_pts, r_range, phi_range, z_range,
                  stz_init, mass, charge, speed_total, vtang, tmax, tol, nparticles):
    results = tracer(
        cell_quad_pts,
        np.array(r_range, dtype=np.float64),
        np.array(phi_range, dtype=np.float64),
        np.array(z_range, dtype=np.float64),
        stz_init,
        mass,
        charge,
        speed_total,
        vtang,
        tmax,
        tol,
        nparticles
    )
    return np.asarray(results, dtype=np.float64).reshape(nparticles, 5)

# copied from alex's example (tracing_gpu_ncsx.py)
# ── Parameters ──────────────────────────────────────────────────────────

# Particle parameters:
nparticles = 256 # number of particles to trace
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


# direction test: forward -> backward -> forward
test_tmax = float(tmax_values[len(tmax_values)//2])  # mid value
proc0_print(f"\nPhase A direction test with tmax={test_tmax:.2e}s", flush=True)

out_f1 = run_cartesian(cartesian_gpu_tracing,
                       cell_quad_pts, r_range, phi_range, z_range,
                       stz_init, mass, charge, speed_total, vtang,
                       test_tmax, tol, nparticles)

out_b  = run_cartesian(cartesian_gpu_tracing_backward,
                       cell_quad_pts, r_range, phi_range, z_range,
                       stz_init, mass, charge, speed_total, vtang,
                       test_tmax, tol, nparticles)

out_f2 = run_cartesian(cartesian_gpu_tracing,
                       cell_quad_pts, r_range, phi_range, z_range,
                       stz_init, mass, charge, speed_total, vtang,
                       test_tmax, tol, nparticles)

# Compare xyz+vpar only (ignore time column 0)
diff_f  = np.linalg.norm(out_f2[:, 1:] - out_f1[:, 1:], axis=1)
diff_fb = np.linalg.norm(out_f2[:, 1:] - out_b[:, 1:], axis=1)

proc0_print(f"median ||f2-f1|| = {np.median(diff_f):.3e}", flush=True)
proc0_print(f"median ||f2-b || = {np.median(diff_fb):.3e}", flush=True)

if not (np.median(diff_f) < 1e-6 * (1.0 + np.median(np.linalg.norm(out_f1[:,1:], axis=1)))):
    proc0_print("WARNING: forward results changed a lot after backward call. "
                "This often means dir_d was not reset per-call.", flush=True)
    

# closure test: x0 -> forward -> x1 -> backward -> x0' ────────
# Use only particles that were not lost in forward run for the comparison.
proc0_print("\nPhase A closure test (forward then backward)", flush=True)

# forward from initial
fwd = out_f1
t_f = fwd[:, 0]
lost_mask = t_f < 0.99 * test_tmax
keep = ~lost_mask

proc0_print(f"Not-lost particles for closure: {np.sum(keep)}/{nparticles}", flush=True)

if np.sum(keep) > 0:
    # backward start = final xyz, and vpar_final becomes new vtang
    stz_b_init = fwd[keep, 1:4].astype(np.float64).ravel()
    vtang_b = fwd[keep, 4].astype(np.float64)
    n_keep = int(np.sum(keep))

    bwd_from_fwd = run_cartesian(cartesian_gpu_tracing_backward,
                                 cell_quad_pts, r_range, phi_range, z_range,
                                 stz_b_init, mass, charge, speed_total, vtang_b,
                                 test_tmax, tol, n_keep)

    # original xyz for those particles
    # stz_init is cylindrical in your script; but kernel output is xyz.
    # We saved xyz_init earlier; use it:
    xyz0 = xyz_init[keep, :].astype(np.float64)

    xyz_back = bwd_from_fwd[:, 1:4]
    err = np.linalg.norm(xyz_back - xyz0, axis=1)

    proc0_print(f"median closure ||x_back - x0|| = {np.median(err):.3e} m", flush=True)
    proc0_print(f"max    closure ||x_back - x0|| = {np.max(err):.3e} m", flush=True)
else:
    proc0_print("All particles were lost in forward run; closure test skipped.", flush=True)