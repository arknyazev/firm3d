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
nparticles = 1 # number of particles to trace
energy = 500 * ONE_EV # kinetic energy of particles [J]
mass = PROTON_MASS  # mass of particles [kg]
charge = ELEMENTARY_CHARGE # charge of particles [C]

tmax_values = np.arange(1e-7, 1e-5 + 1e-7, 1e-7) # trajectory snapshots [s]
tmax_max = float(np.max(tmax_values))

seed = 1 # random seed for initial conditions

tol_vals = [1e-9, 1e-10, 1e-11] # ODE solver tolerance

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
zrange = zrange  = (np.min(zs), np.max(zs), n)       #(0, np.max(zs), n // 2)

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


# ── Helpers ─────────────────────────────────────────────────────────────

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
    period = phi_max - phi_min
    return (phi - phi_min) % period + phi_min


def xyz_to_stz_flat(xyz, phi_min, phi_max):
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


def tol_tag(tol):
    # stable filesystem-friendly folder name, e.g. 1e-09 -> 1p00em09
    return f"{tol:.2e}".replace(".", "p").replace("-", "m")


def ensure_run_dirs(base_out_dir, tol):
    # ./output/tol_<tag>/forward and ./output/tol_<tag>/backward
    tag = tol_tag(tol)
    tol_dir = os.path.join(base_out_dir, f"tol_{tag}")
    fwd_dir = os.path.join(tol_dir, "forward")
    bwd_dir = os.path.join(tol_dir, "backward")
    os.makedirs(fwd_dir, exist_ok=True)
    os.makedirs(bwd_dir, exist_ok=True)
    return tol_dir, fwd_dir, bwd_dir


def save_results(results, tmax, run_dir, phi_min, phi_max):
    """
    Save particle endpoint arrays for a single run at a single tmax into run_dir.
    """
    t_final = results[:, 0]
    X_final = results[:, 1]
    Y_final = results[:, 2]
    Z_final = results[:, 3]
    vpar_final = results[:, 4]

    R_final = np.sqrt(X_final**2 + Y_final**2)
    phi_final = np.arctan2(Y_final, X_final)
    phi_final = wrap_phi(phi_final, phi_min, phi_max)

    xyz_final = np.column_stack([X_final, Y_final, Z_final])

    tmax_str = f"{tmax:.2e}".replace(".", "p").replace("-", "m")
    np.save(os.path.join(run_dir, f"particles_final_rphiz_tmax_{tmax_str}.npy"),
            np.column_stack([R_final, phi_final, Z_final]))
    np.save(os.path.join(run_dir, f"particles_final_xyz_tmax_{tmax_str}.npy"), xyz_final)
    np.save(os.path.join(run_dir, f"particles_final_vpar_tmax_{tmax_str}.npy"), vpar_final)
    np.save(os.path.join(run_dir, f"particles_final_time_tmax_{tmax_str}.npy"), t_final)

    lost_mask = t_final < 0.99 * tmax
    n_lost = int(np.sum(lost_mask))
    proc0_print(f"Particles lost: {n_lost}/{results.shape[0]} ({100*n_lost/results.shape[0]:.1f}%)", flush=True)


def append_metrics_csv(csv_path, row_dict):
    """
    Append one row (dict) to CSV, creating header if needed.
    """
    import csv
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


# wrap initial phi into interpolant domain
phi_min = float(phi_range[0])
phi_max = float(phi_range[1])

stz_init = np.asarray(stz_init, dtype=np.float64).copy()
stz_init[1::3] = wrap_phi(stz_init[1::3], phi_min, phi_max)

# TEST
for tol in tol_vals:
    proc0_print(f"\n=== tol={tol:.2e} ===", flush=True)

    tol_dir, fwd_dir, bwd_dir = ensure_run_dirs(out_dir, tol)

    np.save(os.path.join(tol_dir, "tmax_values.npy"), np.asarray(tmax_values, dtype=np.float64))
    np.save(os.path.join(tol_dir, "particles_initial_xyz.npy"), xyz_init)
    np.save(os.path.join(tol_dir, "particles_initial_rphiz.npy"), np.column_stack([R_init, wrap_phi(phi_init, phi_min, phi_max), Z_init]))
    np.save(os.path.join(tol_dir, "particles_initial_vtang.npy"), vtang)

    metrics_csv = os.path.join(tol_dir, "metrics.csv")

    # -------------------------
    # (1) FORWARD SWEEP: x0 -> x(tmax)
    # -------------------------
    proc0_print(f"[FWD] sweeping {len(tmax_values)} tmax values from x0", flush=True)

    fwd_at_tmax_max = None  # will hold forward endpoint at tmax_max (needed to start backward sweep)

    for tmax in tmax_values:
        fwd = run_cartesian(
            cartesian_gpu_tracing,
            cell_quad_pts, r_range, phi_range, z_range,
            stz_init, mass, charge, speed_total, vtang,
            float(tmax), float(tol), nparticles
        )
        save_results(fwd, float(tmax), fwd_dir, phi_min, phi_max)

        if abs(float(tmax) - tmax_max) <= 0.0:
            fwd_at_tmax_max = fwd

    if fwd_at_tmax_max is None:
        fwd_at_tmax_max = run_cartesian(
            cartesian_gpu_tracing,
            cell_quad_pts, r_range, phi_range, z_range,
            stz_init, mass, charge, speed_total, vtang,
            float(tmax_max), float(tol), nparticles
        )

    # Determine which particles are still alive at tmax_max
    t_fmax = fwd_at_tmax_max[:, 0]
    keep = ~(t_fmax < 0.99 * tmax_max)
    n_keep = int(np.sum(keep))
    proc0_print(f"[FWD] not-lost at tmax_max={tmax_max:.2e}: {n_keep}/{nparticles}", flush=True)

    if n_keep == 0:
        proc0_print("[BWD] skipped: all particles lost at tmax_max", flush=True)
        append_metrics_csv(metrics_csv, {
            "tol": float(tol),
            "tmax_max": float(tmax_max),
            "nparticles": int(nparticles),
            "n_keep": int(n_keep),
            "median_closure_m": np.nan,
            "max_closure_m": np.nan,
            "median_dv_same": np.nan,
            "median_dv_flip": np.nan,
        })
        continue

    # Build backward initial conditions from forward endpoint at tmax_max
    xyz1 = fwd_at_tmax_max[keep, 1:4].astype(np.float64)
    stz_b_init = xyz_to_stz_flat(xyz1, phi_min, phi_max)
    vtang_b0 = fwd_at_tmax_max[keep, 4].astype(np.float64) ###

    # -------------------------
    # (2) BACKWARD SWEEP: start at x(tmax_max) and integrate backward for each tmax
    # -------------------------
    proc0_print(f"[BWD] sweeping {len(tmax_values)} tmax values from x(tmax_max)", flush=True)

    bwd_at_tmax_max = None

    for tmax in tmax_values:
        bwd = run_cartesian(
            cartesian_gpu_tracing_backward,
            cell_quad_pts, r_range, phi_range, z_range,
            stz_b_init, mass, charge, speed_total, vtang_b0,
            float(tmax), float(tol), n_keep
        )
        save_results(bwd, float(tmax), bwd_dir, phi_min, phi_max)

        if abs(float(tmax) - tmax_max) <= 0.0:
            bwd_at_tmax_max = bwd

    if bwd_at_tmax_max is None:
        bwd_at_tmax_max = run_cartesian(
            cartesian_gpu_tracing_backward,
            cell_quad_pts, r_range, phi_range, z_range,
            stz_b_init, mass, charge, speed_total, vtang_b0,
            float(tmax_max), float(tol), n_keep
        )

    # -------------------------
    # (3) Closure metrics at tmax_max: x0 -> forward(tmax_max) -> backward(tmax_max) -> x0'
    # -------------------------
    xyz0 = xyz_init[keep, :].astype(np.float64)
    xyz_back = bwd_at_tmax_max[:, 1:4]

    err = np.linalg.norm(xyz_back - xyz0, axis=1)
    proc0_print(f"[CLOSURE @ tmax_max] median ||x0'-x0|| = {np.median(err):.3e} m", flush=True)
    proc0_print(f"[CLOSURE @ tmax_max] max    ||x0'-x0|| = {np.max(err):.3e} m", flush=True)

    # Velocity convention diagnostic
    vpar_back = bwd_at_tmax_max[:, 4]
    v0 = vtang[keep].astype(np.float64)
    dv_same = np.abs(vpar_back - v0)
    dv_flip = np.abs(vpar_back + v0)
    proc0_print(f"[VEL CHECK @ tmax_max] median |vpar_back - v0| = {np.median(dv_same):.3e}", flush=True)
    proc0_print(f"[VEL CHECK @ tmax_max] median |vpar_back + v0| = {np.median(dv_flip):.3e}", flush=True)

    append_metrics_csv(metrics_csv, {
        "tol": float(tol),
        "tmax_max": float(tmax_max),
        "nparticles": int(nparticles),
        "n_keep": int(n_keep),
        "median_closure_m": float(np.median(err)),
        "max_closure_m": float(np.max(err)),
        "median_dv_same": float(np.median(dv_same)),
        "median_dv_flip": float(np.median(dv_flip)),
    })

proc0_print("\nDone. Outputs are under ./output/tol_<...>/{forward,backward}/", flush=True)