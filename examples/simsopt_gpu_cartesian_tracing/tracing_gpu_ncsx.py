#!/usr/bin/env python3

"""
This example demonstrates how to use SIMSOPT surfaces and magnetic fields
together with firm3d GPU particle tracing.

It sets up an NCSX configuration, builds an interpolated field and a
surface classifier for the stopping criterion, then traces guiding-center
particles on the GPU using cartesian_gpu_tracing. Initial and final
particle positions are saved to .npy files, and the coils and
intersection surface are exported to VTK for visualization in Paraview.
"""

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
from firm3dpp import cartesian_gpu_tracing

proc0_print("Running tracing_gpu_ncsx.py")
proc0_print("===========================")

# ── Parameters ──────────────────────────────────────────────────────────

# Particle parameters:
nparticles = 1 # number of particles to trace
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

# ── 7. Run GPU particle tracing for different tmax values ────────────────
proc0_print(
    f"\n\tTracing {nparticles} particles on GPU,"
    f"\tfor {len(tmax_values)} different tmax values..."
)

for idx, tmax in enumerate(tmax_values):
    proc0_print(
        f"[{idx+1}/{len(tmax_values)}]",
        f"Tracing with tmax={tmax:.2e}s ...",
        flush=True
    )

    t1 = time.time()
    results = cartesian_gpu_tracing(
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
    t2 = time.time()

    proc0_print(f"  Completed in {t2-t1:.3f}s", flush=True)

    # ── 8. Parse and save results ───────────────────────────────────────────
    # Output is a flat vector of length 5*nparticles:
    # [t_final, x, y, z, v_par] per particle (x, y, z – Cartesian coordinates)
    results = np.array(results).reshape(nparticles, 5)

    t_final = results[:, 0] # [s]
    X_final = results[:, 1] # [m]
    Y_final = results[:, 2] # [m]
    Z_final = results[:, 3] # [m]
    vpar_final = results[:, 4] # [m/s] 

    # Convert XYZ to cylindrical (R, phi, Z)
    R_final = np.sqrt(X_final**2 + Y_final**2)
    phi_final = np.arctan2(Y_final, X_final)
    xyz_final = np.column_stack([X_final, Y_final, Z_final])

    # Save with tmax in filename
    tmax_str = f"{tmax:.2e}".replace(".", "p").replace("-", "m")
    np.save(out_dir + f"particles_final_rphiz_tmax_{tmax_str}.npy",
            np.column_stack([R_final, phi_final, Z_final]))
    np.save(out_dir + f"particles_final_xyz_tmax_{tmax_str}.npy", xyz_final)
    np.save(out_dir + f"particles_final_vpar_tmax_{tmax_str}.npy", vpar_final)
    np.save(out_dir + f"particles_final_time_tmax_{tmax_str}.npy", t_final)

    # Identify lost particles (those that hit the surface before tmax)
    lost_mask = t_final < 0.99 * tmax
    n_lost = np.sum(lost_mask)
    proc0_print(f"Particles lost: {n_lost}/{nparticles}",
                f"({100*n_lost/nparticles:.1f}%)")
    proc0_print(f"Saved to {out_dir}particles_final_*_tmax_{tmax_str}.npy\n")

proc0_print("")
proc0_print("Output files:")
proc0_print(f"  {out_dir}coils.vtu                              - coils + axis (VTK)")
proc0_print(f"  {out_dir}surface.vts                            - intersection surface (VTK)")
proc0_print(f"  {out_dir}particles_initial_xyz.npy              - initial XYZ positions")
proc0_print(f"  {out_dir}particles_initial_rphiz.npy            - initial (R,phi,Z) positions")
proc0_print(f"  {out_dir}particles_initial_vtang.npy            - initial parallel velocities")
proc0_print(f"  {out_dir}particles_final_xyz_tmax_*.npy        - final XYZ positions (per tmax)")
proc0_print(f"  {out_dir}particles_final_rphiz_tmax_*.npy      - final (R,phi,Z) positions (per tmax)")
proc0_print(f"  {out_dir}particles_final_vpar_tmax_*.npy       - final parallel velocities (per tmax)")
proc0_print(f"  {out_dir}particles_final_time_tmax_*.npy       - final times (per tmax)")
proc0_print(f"\n  Note: * = tmax value in scientific notation, e.g., 1p00em04 for 1.00e-04")

proc0_print("")
proc0_print("End of tracing_gpu_ncsx.py")
proc0_print("==========================")

