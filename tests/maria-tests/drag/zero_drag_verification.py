#!/usr/bin/env python3

"""
Verify that when nu_s = 0 and energy stopping is disabled, the new drag
tracers reproduce the existing vacuum Cartesian tracers.
(both forward and backward)
"""

import csv
import os
import time
from pathlib import Path
from datetime import datetime
from math import sqrt

import numpy as np

from simsopt.configs import get_data
from simsopt.field import InterpolatedField, SurfaceClassifier
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.util import proc0_print
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from simsopt.field.sampling import draw_uniform_on_curve

from firm3d.util.gpu_utils import cartesian_interpolant
from firm3dpp import (
    cartesian_gpu_tracing,
    cartesian_gpu_tracing_backward,
    cartesian_gpu_tracing_drag,
    cartesian_gpu_tracing_backward_drag,
)


# =============================================================================
# Parameters
# =============================================================================

nparticles = 4
energy = 500 * ONE_EV
mass = PROTON_MASS
charge = ELEMENTARY_CHARGE

# Keep these modest for first regression/debug pass
tmax_values = np.array([1e-7, 3e-7, 1e-6, 3e-6, 1e-5], dtype=np.float64)
tol_vals = [1e-9, 1e-10]

seed = 1
degree = 3
n = 16

# New drag parameters for regression test
nu_s = 0.0
use_energy_stop = False
H_stop = 3.5e6 * ONE_EV  # arbitrary here, unused because energy stop is off

# Tolerances for comparisons.
atol_time = 1e-14
rtol_time = 1e-11

atol_xyz = 1e-10
rtol_xyz = 1e-9

atol_vpar = 1e-8
rtol_vpar = 1e-8

atol_H = 1e-12
rtol_H = 1e-11


# =============================================================================
# Output directory
# =============================================================================

script_dir = Path(__file__).resolve().parent
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = script_dir / "outputs_drag_zero_regression" / timestamp
out_dir.mkdir(parents=True, exist_ok=True)

proc0_print("Running drag_zero_regression_cartesian.py")
proc0_print("========================================")
proc0_print(f"Saving outputs to: {out_dir}")


# =============================================================================
# Helpers
# =============================================================================

def wrap_phi(phi: np.ndarray, phi_min: float, phi_max: float) -> np.ndarray:
    period = phi_max - phi_min
    return (phi - phi_min) % period + phi_min


def xyz_to_stz_flat(xyz: np.ndarray, phi_min: float, phi_max: float) -> np.ndarray:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    R = np.sqrt(x * x + y * y)
    phi = np.arctan2(y, x)
    phi = wrap_phi(phi, phi_min, phi_max)

    stz = np.empty(3 * xyz.shape[0], dtype=np.float64)
    stz[0::3] = R
    stz[1::3] = phi
    stz[2::3] = z
    return stz


def tol_tag(tol: float) -> str:
    return f"{tol:.2e}".replace(".", "p").replace("-", "m")


def tmax_tag(tmax: float) -> str:
    return f"{tmax:.2e}".replace(".", "p").replace("-", "m")


def ensure_dirs(base_out_dir: Path, tol: float, mode: str):
    tag = tol_tag(tol)
    root = base_out_dir / f"tol_{tag}" / mode
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_csv_row(csv_path: Path, row: dict):
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_cartesian_vacuum(
    tracer,
    cell_quad_pts,
    r_range,
    phi_range,
    z_range,
    stz_init,
    mass,
    charge,
    speed_total,
    vtang,
    tmax,
    tol,
    nparticles,
):
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


def run_cartesian_drag(
    tracer,
    cell_quad_pts,
    r_range,
    phi_range,
    z_range,
    stz_init,
    mass,
    charge,
    speed_total,
    vtang,
    H_init,
    nu_s,
    tmax,
    tol,
    nparticles,
    H_stop,
    use_energy_stop,
):
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
        np.asarray(H_init, dtype=np.float64),
        float(nu_s),
        float(tmax),
        float(tol),
        int(nparticles),
        float(H_stop),
        bool(use_energy_stop),
    )
    return np.asarray(results, dtype=np.float64).reshape(int(nparticles), 7)


def save_compare_bundle(run_dir: Path, prefix: str, vac: np.ndarray, drag: np.ndarray):
    np.save(run_dir / f"{prefix}_vacuum.npy", vac)
    np.save(run_dir / f"{prefix}_drag.npy", drag)


def compare_arrays(name, a, b, atol, rtol):
    ok = np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
    diff = np.abs(a - b)
    max_abs = float(np.max(diff)) if diff.size else 0.0

    denom = np.maximum(np.abs(b), atol)
    rel = diff / denom
    max_rel = float(np.max(rel)) if rel.size else 0.0

    return ok, max_abs, max_rel


def summarize_stop_codes(stop_codes: np.ndarray):
    uniq, counts = np.unique(stop_codes.astype(int), return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}


# =============================================================================
# Build magnetic field and interpolant
# =============================================================================

base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
all_curves = [c.curve for c in bs.coils]

proc0_print(
    "Mean(|B|) on axis =",
    np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)),
)
proc0_print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))

curves_to_vtk(all_curves + [ma], str(out_dir / "coils"))
proc0_print(f"Saved coils VTK to {out_dir / 'coils.vtu'}")

mpol = 5
ntor = 5
stellsym = True
s = SurfaceRZFourier.from_nphi_ntheta(
    mpol=mpol,
    ntor=ntor,
    stellsym=stellsym,
    nfp=nfp,
    range="full torus",
    nphi=64,
    ntheta=24,
)
s.fit_to_curve(ma, 0.20, flip_theta=False)
s.to_vtk(str(out_dir / "surface"))
proc0_print(f"Saved surface VTK to {out_dir / 'surface.vts'}")

sc_particle = SurfaceClassifier(s, h=0.1, p=2)

rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]

rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2 * np.pi / nfp, n * 2)
zrange = (np.min(zs), np.max(zs), n)

bsh = InterpolatedField(
    bs,
    degree,
    rrange,
    phirange,
    zrange,
    True,
    nfp=nfp,
    stellsym=True,
)

proc0_print("Error in B:      ", bsh.estimate_error_B(1000), flush=True)
proc0_print("Error in GradAbsB:", bsh.estimate_error_GradAbsB(1000), flush=True)

t1 = time.time()
r_range, phi_range, z_range, cell_quad_pts = cartesian_interpolant(bsh, sc_particle, nfp, n)
t2 = time.time()
proc0_print(f"GPU interpolant built in {t2 - t1:.3f}s", flush=True)


# =============================================================================
# Initial conditions
# =============================================================================

speed_total = sqrt(2 * energy / mass)
H_total = energy

np.random.seed(seed)
xyz_init, _ = draw_uniform_on_curve(ma, nparticles, safetyfactor=10)

R_init = np.sqrt(xyz_init[:, 0] ** 2 + xyz_init[:, 1] ** 2)
phi_init = np.arctan2(xyz_init[:, 1], xyz_init[:, 0])
Z_init = xyz_init[:, 2]

phi_min = float(phi_range[0])
phi_max = float(phi_range[1])
phi_init = wrap_phi(phi_init, phi_min, phi_max)

stz_init = np.empty(3 * nparticles, dtype=np.float64)
stz_init[0::3] = R_init
stz_init[1::3] = phi_init
stz_init[2::3] = Z_init

us = np.random.uniform(-1, 1, size=nparticles)
vtang = (us * speed_total).astype(np.float64)

# For the drag tracer, initial H is total kinetic energy.
H_init = np.full(nparticles, H_total, dtype=np.float64)

np.save(out_dir / "particles_initial_xyz.npy", xyz_init)
np.save(out_dir / "particles_initial_rphiz.npy", np.column_stack([R_init, phi_init, Z_init]))
np.save(out_dir / "particles_initial_vtang.npy", vtang)
np.save(out_dir / "particles_initial_H.npy", H_init)

proc0_print(f"Saved {nparticles} initial particle positions to {out_dir}")


# =============================================================================
# Main regression test
# =============================================================================

summary_csv = out_dir / "summary.csv"

n_cases = 0
n_cases_passed = 0

for tol in tol_vals:
    proc0_print(f"\n=== tol = {tol:.2e} ===", flush=True)

    fwd_dir = ensure_dirs(out_dir, tol, "forward")
    bwd_dir = ensure_dirs(out_dir, tol, "backward")

    # -------------------------------------------------------------------------
    # Forward comparison: vacuum vs drag(nu_s=0)
    # -------------------------------------------------------------------------
    proc0_print("[FWD] comparing old vacuum tracer vs new drag tracer (nu_s=0)", flush=True)

    fwd_vac_at_tmax_max = None
    fwd_drag_at_tmax_max = None

    for tmax in tmax_values:
        tag = tmax_tag(float(tmax))

        vac = run_cartesian_vacuum(
            cartesian_gpu_tracing,
            cell_quad_pts,
            r_range,
            phi_range,
            z_range,
            stz_init,
            mass,
            charge,
            speed_total,
            vtang,
            float(tmax),
            float(tol),
            nparticles,
        )

        drag = run_cartesian_drag(
            cartesian_gpu_tracing_drag,
            cell_quad_pts,
            r_range,
            phi_range,
            z_range,
            stz_init,
            mass,
            charge,
            speed_total,
            vtang,
            H_init,
            nu_s,
            float(tmax),
            float(tol),
            nparticles,
            H_stop,
            use_energy_stop,
        )

        save_compare_bundle(fwd_dir, f"tmax_{tag}", vac, drag)

        t_vac = vac[:, 0]
        xyz_vac = vac[:, 1:4]
        vpar_vac = vac[:, 4]

        t_drag = drag[:, 0]
        xyz_drag = drag[:, 1:4]
        vpar_drag = drag[:, 4]
        H_drag = drag[:, 5]
        stop_drag = drag[:, 6].astype(int)

        ok_t, max_abs_t, max_rel_t = compare_arrays("t_final", t_drag, t_vac, atol_time, rtol_time)
        ok_xyz, max_abs_xyz, max_rel_xyz = compare_arrays("xyz", xyz_drag, xyz_vac, atol_xyz, rtol_xyz)
        ok_v, max_abs_v, max_rel_v = compare_arrays("vpar", vpar_drag, vpar_vac, atol_vpar, rtol_vpar)
        ok_H, max_abs_H, max_rel_H = compare_arrays("H", H_drag, H_init, atol_H, rtol_H)

        no_energy_stops = np.all(stop_drag != 2)

        row = {
            "mode": "forward",
            "tol": float(tol),
            "tmax": float(tmax),
            "ok_t": bool(ok_t),
            "ok_xyz": bool(ok_xyz),
            "ok_vpar": bool(ok_v),
            "ok_H_const": bool(ok_H),
            "no_energy_stops": bool(no_energy_stops),
            "max_abs_t": max_abs_t,
            "max_rel_t": max_rel_t,
            "max_abs_xyz": max_abs_xyz,
            "max_rel_xyz": max_rel_xyz,
            "max_abs_vpar": max_abs_v,
            "max_rel_vpar": max_rel_v,
            "max_abs_H": max_abs_H,
            "max_rel_H": max_rel_H,
            "stop_codes_drag": summarize_stop_codes(stop_drag),
        }
        save_csv_row(summary_csv, row)

        assert ok_t, f"[FWD, tol={tol:.2e}, tmax={tmax:.2e}] t_final mismatch"
        assert ok_xyz, f"[FWD, tol={tol:.2e}, tmax={tmax:.2e}] xyz mismatch"
        assert ok_v, f"[FWD, tol={tol:.2e}, tmax={tmax:.2e}] vpar mismatch"
        assert ok_H, f"[FWD, tol={tol:.2e}, tmax={tmax:.2e}] H is not constant when nu_s=0"
        assert no_energy_stops, f"[FWD, tol={tol:.2e}, tmax={tmax:.2e}] unexpected energy stop with use_energy_stop=False"

        n_cases += 1
        n_cases_passed += 1

        proc0_print(
            f"  tmax={tmax:.2e} : PASS | "
            f"max|Δt|={max_abs_t:.3e}, "
            f"max|Δxyz|={max_abs_xyz:.3e}, "
            f"max|Δvpar|={max_abs_v:.3e}, "
            f"max|ΔH|={max_abs_H:.3e}",
            flush=True,
        )

        if np.isclose(tmax, np.max(tmax_values)):
            fwd_vac_at_tmax_max = vac
            fwd_drag_at_tmax_max = drag

    assert fwd_vac_at_tmax_max is not None
    assert fwd_drag_at_tmax_max is not None

    # Keep only particles not lost in the vacuum forward run at tmax_max.
    tmax_max = float(np.max(tmax_values))
    keep = ~(fwd_vac_at_tmax_max[:, 0] < 0.99 * tmax_max)
    n_keep = int(np.sum(keep))
    proc0_print(f"[FWD] not-lost at tmax_max={tmax_max:.2e}: {n_keep}/{nparticles}", flush=True)

    if n_keep == 0:
        proc0_print("[BWD] skipped: all particles lost in forward vacuum run at tmax_max", flush=True)
        continue

    # -------------------------------------------------------------------------
    # Backward comparison: old backward vacuum vs new backward drag(nu_s=0)
    # Start from the same forward vacuum endpoint.
    # -------------------------------------------------------------------------
    xyz1 = fwd_vac_at_tmax_max[keep, 1:4].astype(np.float64)
    stz_b_init = xyz_to_stz_flat(xyz1, phi_min, phi_max)
    vtang_b0 = fwd_vac_at_tmax_max[keep, 4].astype(np.float64)
    H_b0 = np.full(n_keep, H_total, dtype=np.float64)

    proc0_print("[BWD] comparing old backward tracer vs new backward drag tracer (nu_s=0)", flush=True)

    bwd_vac_at_tmax_max = None
    bwd_drag_at_tmax_max = None

    for tmax in tmax_values:
        tag = tmax_tag(float(tmax))

        vac = run_cartesian_vacuum(
            cartesian_gpu_tracing_backward,
            cell_quad_pts,
            r_range,
            phi_range,
            z_range,
            stz_b_init,
            mass,
            charge,
            speed_total,
            vtang_b0,
            float(tmax),
            float(tol),
            n_keep,
        )

        drag = run_cartesian_drag(
            cartesian_gpu_tracing_backward_drag,
            cell_quad_pts,
            r_range,
            phi_range,
            z_range,
            stz_b_init,
            mass,
            charge,
            speed_total,
            vtang_b0,
            H_b0,
            nu_s,
            float(tmax),
            float(tol),
            n_keep,
            H_stop,
            use_energy_stop,
        )

        save_compare_bundle(bwd_dir, f"tmax_{tag}", vac, drag)

        t_vac = vac[:, 0]
        xyz_vac = vac[:, 1:4]
        vpar_vac = vac[:, 4]

        t_drag = drag[:, 0]
        xyz_drag = drag[:, 1:4]
        vpar_drag = drag[:, 4]
        H_drag = drag[:, 5]
        stop_drag = drag[:, 6].astype(int)

        ok_t, max_abs_t, max_rel_t = compare_arrays("t_final", t_drag, t_vac, atol_time, rtol_time)
        ok_xyz, max_abs_xyz, max_rel_xyz = compare_arrays("xyz", xyz_drag, xyz_vac, atol_xyz, rtol_xyz)
        ok_v, max_abs_v, max_rel_v = compare_arrays("vpar", vpar_drag, vpar_vac, atol_vpar, rtol_vpar)
        ok_H, max_abs_H, max_rel_H = compare_arrays("H", H_drag, H_b0, atol_H, rtol_H)

        no_energy_stops = np.all(stop_drag != 2)

        row = {
            "mode": "backward",
            "tol": float(tol),
            "tmax": float(tmax),
            "ok_t": bool(ok_t),
            "ok_xyz": bool(ok_xyz),
            "ok_vpar": bool(ok_v),
            "ok_H_const": bool(ok_H),
            "no_energy_stops": bool(no_energy_stops),
            "max_abs_t": max_abs_t,
            "max_rel_t": max_rel_t,
            "max_abs_xyz": max_abs_xyz,
            "max_rel_xyz": max_rel_xyz,
            "max_abs_vpar": max_abs_v,
            "max_rel_vpar": max_rel_v,
            "max_abs_H": max_abs_H,
            "max_rel_H": max_rel_H,
            "stop_codes_drag": summarize_stop_codes(stop_drag),
        }
        save_csv_row(summary_csv, row)

        assert ok_t, f"[BWD, tol={tol:.2e}, tmax={tmax:.2e}] t_final mismatch"
        assert ok_xyz, f"[BWD, tol={tol:.2e}, tmax={tmax:.2e}] xyz mismatch"
        assert ok_v, f"[BWD, tol={tol:.2e}, tmax={tmax:.2e}] vpar mismatch"
        assert ok_H, f"[BWD, tol={tol:.2e}, tmax={tmax:.2e}] H is not constant when nu_s=0"
        assert no_energy_stops, f"[BWD, tol={tol:.2e}, tmax={tmax:.2e}] unexpected energy stop with use_energy_stop=False"

        n_cases += 1
        n_cases_passed += 1

        proc0_print(
            f"  tmax={tmax:.2e} : PASS | "
            f"max|Δt|={max_abs_t:.3e}, "
            f"max|Δxyz|={max_abs_xyz:.3e}, "
            f"max|Δvpar|={max_abs_v:.3e}, "
            f"max|ΔH|={max_abs_H:.3e}",
            flush=True,
        )

        if np.isclose(tmax, np.max(tmax_values)):
            bwd_vac_at_tmax_max = vac
            bwd_drag_at_tmax_max = drag

    # Optional closure comparison at tmax_max for sanity.
    if bwd_vac_at_tmax_max is not None and bwd_drag_at_tmax_max is not None:
        xyz0 = xyz_init[keep, :].astype(np.float64)

        xyz_back_vac = bwd_vac_at_tmax_max[:, 1:4]
        xyz_back_drag = bwd_drag_at_tmax_max[:, 1:4]

        err_vac = np.linalg.norm(xyz_back_vac - xyz0, axis=1)
        err_drag = np.linalg.norm(xyz_back_drag - xyz0, axis=1)

        proc0_print(
            f"[CLOSURE @ tmax_max, vacuum] median ||x0'-x0|| = {np.median(err_vac):.3e} m",
            flush=True,
        )
        proc0_print(
            f"[CLOSURE @ tmax_max, drag0 ] median ||x0'-x0|| = {np.median(err_drag):.3e} m",
            flush=True,
        )


proc0_print("")
proc0_print(f"Passed {n_cases_passed}/{n_cases} regression comparisons.")
proc0_print(f"Summary CSV: {summary_csv}")
proc0_print("Done.")