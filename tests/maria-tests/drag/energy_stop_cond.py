#!/usr/bin/env python3

"""
Test drag tracing energy evolution and backward H-stop behavior.

This script does two things:

1) Forward/backward consistency check for H evolution:
   dH/dt = -nu_s H

   With the current implementation:
   - forward tracing should decrease H
   - backward tracing should increase H

   Since the tracer returns elapsed time t >= 0 in both cases, we expect:
   - forward:  ln(H_final) - ln(H0) ~= -nu_s * t_final
              => ln(H_final) + nu_s * t_final ~= ln(H0)
   - backward: ln(H_final) - ln(H0) ~= +nu_s * t_final
              => ln(H_final) - nu_s * t_final ~= ln(H0)

2) Backward tracing with energy stopping enabled:
   stop when H >= H_stop
"""

import csv
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
    cartesian_gpu_tracing_drag,
    cartesian_gpu_tracing_backward_drag,
)


# =============================================================================
# Parameters
# =============================================================================

nparticles = 8
mass = PROTON_MASS
charge = ELEMENTARY_CHARGE

# Initial particle energy
energy0_ev = 0.5e6                # 0.5 MeV
energy0 = energy0_ev * ONE_EV     # joules

# Backward stopping threshold
H_stop_ev = 1.0e6                 # lowered this to reach it
H_stop = H_stop_ev * ONE_EV       # joules

# drag coefficient used in the H law: dH/dt = -nu_s H
nu_s = 4.0e5                      # 1/s, choose something strong enough to see effect
Q0 = 0.0                          # unused here; keep source integral turned off for clarity

# tracing / numerical params
tol = 1e-10
seed = 1
degree = 3
n = 16

# choose tmax so backward run has time to reach H_stop
# needed elapsed time is ln(H_stop/H0)/nu_s
required_backward_time = np.log(H_stop / energy0) / nu_s
tmax_forward = 2.0e-6
tmax_backward = max(1.2 * required_backward_time, 2.0e-6)

# loose-ish numerical tolerances for invariant checks
abs_tol_log = 5e-7
rel_tol_log = 5e-6
abs_tol_H = 5e-7 * energy0
rel_tol_H = 5e-6


# =============================================================================
# Output directory
# =============================================================================

script_dir = Path(__file__).resolve().parent
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = script_dir / "outputs_drag_H_stop_test" / timestamp
out_dir.mkdir(parents=True, exist_ok=True)

proc0_print("Running drag_H_stop_test.py")
proc0_print("========================================")
proc0_print(f"Saving outputs to: {out_dir}")
proc0_print(f"Initial energy: {energy0_ev/1e6:.3f} MeV = {energy0:.6e} J")
proc0_print(f"H_stop:         {H_stop_ev/1e6:.3f} MeV = {H_stop:.6e} J")
proc0_print(f"nu_s:           {nu_s:.6e} 1/s")
proc0_print(f"required backward elapsed time to hit H_stop ≈ {required_backward_time:.6e} s")
proc0_print(f"tmax_forward:   {tmax_forward:.6e} s")
proc0_print(f"tmax_backward:  {tmax_backward:.6e} s")


# =============================================================================
# Helpers
# =============================================================================

def wrap_phi(phi: np.ndarray, phi_min: float, phi_max: float) -> np.ndarray:
    period = phi_max - phi_min
    return (phi - phi_min) % period + phi_min


def summarize_stop_codes(stop_codes: np.ndarray):
    uniq, counts = np.unique(stop_codes.astype(int), return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}


def save_csv(path, rows):
    if not rows:
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    Q0,
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
        float(Q0),
        float(tmax),
        float(tol),
        int(nparticles),
        float(H_stop),
        bool(use_energy_stop),
    )
    return np.asarray(results, dtype=np.float64).reshape(int(nparticles), 8)
    # columns: [t, x, y, z, vpar, H, I_Q, stop_reason]


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

speed_total = sqrt(2 * energy0 / mass)

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

# Ensure initial H is physically valid: H >= (1/2) m vpar^2
Kpar0 = 0.5 * mass * vtang**2
assert np.all(energy0 >= Kpar0), "Initial H is smaller than parallel kinetic energy for some particles."

H_init = np.full(nparticles, energy0, dtype=np.float64)

np.save(out_dir / "particles_initial_xyz.npy", xyz_init)
np.save(out_dir / "particles_initial_rphiz.npy", np.column_stack([R_init, phi_init, Z_init]))
np.save(out_dir / "particles_initial_vtang.npy", vtang)
np.save(out_dir / "particles_initial_H.npy", H_init)

proc0_print(f"Saved {nparticles} initial particle positions to {out_dir}")


# =============================================================================
# Part 1: Check H law forward / backward without energy stopping
# =============================================================================

proc0_print("")
proc0_print("Part 1: checking H evolution law without energy stopping")
proc0_print("--------------------------------------------------------")

fwd = run_cartesian_drag(
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
    Q0,
    tmax_forward,
    tol,
    nparticles,
    H_stop=0.0,
    use_energy_stop=False,
)

bwd = run_cartesian_drag(
    cartesian_gpu_tracing_backward_drag,
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
    Q0,
    tmax_forward,
    tol,
    nparticles,
    H_stop=0.0,
    use_energy_stop=False,
)

np.save(out_dir / "forward_no_stop.npy", fwd)
np.save(out_dir / "backward_no_stop.npy", bwd)

# unpack
t_f = fwd[:, 0]
H_f = fwd[:, 5]
stop_f = fwd[:, 7].astype(int)

t_b = bwd[:, 0]
H_b = bwd[:, 5]
stop_b = bwd[:, 7].astype(int)

# monotonic direction checks
forward_should_decrease = np.all(H_f < H_init)
backward_should_increase = np.all(H_b > H_init)

# invariant checks
logH0 = np.log(H_init)
inv_f = np.log(H_f) + nu_s * t_f
inv_b = np.log(H_b) - nu_s * t_b

res_f = inv_f - logH0
res_b = inv_b - logH0

ok_inv_f = np.allclose(inv_f, logH0, atol=abs_tol_log, rtol=rel_tol_log)
ok_inv_b = np.allclose(inv_b, logH0, atol=abs_tol_log, rtol=rel_tol_log)

# direct H-law checks
H_pred_f = H_init * np.exp(-nu_s * t_f)
H_pred_b = H_init * np.exp(+nu_s * t_b)

ok_H_f = np.allclose(H_f, H_pred_f, atol=abs_tol_H, rtol=rel_tol_H)
ok_H_b = np.allclose(H_b, H_pred_b, atol=abs_tol_H, rtol=rel_tol_H)

proc0_print(f"Forward stop codes:  {summarize_stop_codes(stop_f)}")
proc0_print(f"Backward stop codes: {summarize_stop_codes(stop_b)}")
proc0_print(f"Forward H decreases for all particles:  {forward_should_decrease}")
proc0_print(f"Backward H increases for all particles: {backward_should_increase}")
proc0_print(f"Forward invariant ok:  {ok_inv_f} | max|res|={np.max(np.abs(res_f)):.3e}")
proc0_print(f"Backward invariant ok: {ok_inv_b} | max|res|={np.max(np.abs(res_b)):.3e}")
proc0_print(f"Forward H law ok:      {ok_H_f} | max rel err={np.max(np.abs((H_f - H_pred_f)/H_pred_f)):.3e}")
proc0_print(f"Backward H law ok:     {ok_H_b} | max rel err={np.max(np.abs((H_b - H_pred_b)/H_pred_b)):.3e}")

rows_part1 = []
for i in range(nparticles):
    rows_part1.append({
        "particle": i,
        "mode": "forward",
        "t_final": float(t_f[i]),
        "H_initial_J": float(H_init[i]),
        "H_final_J": float(H_f[i]),
        "lnH0": float(logH0[i]),
        "lnHf": float(np.log(H_f[i])),
        "lnHf_plus_nu_t": float(inv_f[i]),
        "residual": float(res_f[i]),
        "stop_reason": int(stop_f[i]),
    })
for i in range(nparticles):
    rows_part1.append({
        "particle": i,
        "mode": "backward",
        "t_final": float(t_b[i]),
        "H_initial_J": float(H_init[i]),
        "H_final_J": float(H_b[i]),
        "lnH0": float(logH0[i]),
        "lnHf": float(np.log(H_b[i])),
        "lnHf_minus_nu_t": float(inv_b[i]),
        "residual": float(res_b[i]),
        "stop_reason": int(stop_b[i]),
    })

save_csv(out_dir / "part1_H_law.csv", rows_part1)

assert forward_should_decrease, "Forward run did not decrease H for all particles."
assert backward_should_increase, "Backward run did not increase H for all particles."
assert ok_inv_f, "Forward invariant ln(H) + nu_s t ~= const failed."
assert ok_inv_b, "Backward invariant ln(H) - nu_s t ~= const failed."
assert ok_H_f, "Forward H(t)=H0 exp(-nu_s t) check failed."
assert ok_H_b, "Backward H(t)=H0 exp(+nu_s t) check failed."


# =============================================================================
# Part 2: Backward tracing with H_stop
# =============================================================================

proc0_print("")
proc0_print(f"Part 2: backward tracing with H-stop at {H_stop_ev} eV")
proc0_print("-----------------------------------------------")

bwd_stop = run_cartesian_drag(
    cartesian_gpu_tracing_backward_drag,
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
    Q0,
    tmax_backward,
    tol,
    nparticles,
    H_stop=H_stop,
    use_energy_stop=True,
)

H_final = bwd_stop[:, 5]
stop_codes = bwd_stop[:, 7].astype(int)

print("max final H [MeV] =", np.max(H_final) / ONE_EV / 1e6)
print("target H_stop [MeV] =", H_stop / ONE_EV / 1e6)
print("num with H_final >= H_stop:", np.sum(H_final >= H_stop))
print("num stop_code == 2:", np.sum(stop_codes == 2))

np.save(out_dir / "backward_with_H_stop.npy", bwd_stop)

t_s = bwd_stop[:, 0]
xyz_s = bwd_stop[:, 1:4]
vpar_s = bwd_stop[:, 4]
H_s = bwd_stop[:, 5]
IQ_s = bwd_stop[:, 6]
stop_s = bwd_stop[:, 7].astype(int)

proc0_print(f"Backward-with-stop stop codes: {summarize_stop_codes(stop_s)}")

hit_energy = (stop_s == 2)
n_hit_energy = int(np.sum(hit_energy))
proc0_print(f"Particles hitting energy stop: {n_hit_energy}/{nparticles}")

# consistency checks for those that hit stop
if n_hit_energy > 0:
    # Since stop is checked after an accepted step, final H may be slightly above threshold.
    at_or_above = np.all(H_s[hit_energy] >= H_stop)
    proc0_print(f"All energy-stopped particles satisfy H_final >= H_stop: {at_or_above}")
    assert at_or_above, "Some particles reported energy-stop but final H < H_stop."

# particles that did not hit stop should have reached tmax or wall
non_energy = ~hit_energy
if np.any(non_energy):
    proc0_print(
        f"Non-energy-stop reasons among remaining particles: "
        f"{summarize_stop_codes(stop_s[non_energy])}"
    )

# expected elapsed time to threshold from exact law
t_hit_exact = np.log(H_stop / H_init) / nu_s
if n_hit_energy > 0:
    proc0_print(
        f"Exact threshold time (same for all particles here) ≈ {t_hit_exact[0]:.6e} s"
    )
    proc0_print(
        f"Median observed elapsed time among energy-stopped particles = "
        f"{np.median(t_s[hit_energy]):.6e} s"
    )

rows_part2 = []
for i in range(nparticles):
    rows_part2.append({
        "particle": i,
        "t_final": float(t_s[i]),
        "x_final": float(xyz_s[i, 0]),
        "y_final": float(xyz_s[i, 1]),
        "z_final": float(xyz_s[i, 2]),
        "vpar_final": float(vpar_s[i]),
        "H_initial_J": float(H_init[i]),
        "H_final_J": float(H_s[i]),
        "H_final_MeV": float(H_s[i] / ONE_EV / 1e6),
        "H_stop_J": float(H_stop),
        "H_stop_MeV": float(H_stop / ONE_EV / 1e6),
        "IQ_final": float(IQ_s[i]),
        "stop_reason": int(stop_s[i]),
        "hit_energy_stop": bool(stop_s[i] == 2),
        "exact_threshold_time_s": float(t_hit_exact[i]),
        "lnH_minus_nu_t": float(np.log(H_s[i]) - nu_s * t_s[i]),
        "residual_to_lnH0": float((np.log(H_s[i]) - nu_s * t_s[i]) - np.log(H_init[i])),
    })

save_csv(out_dir / "part2_backward_H_stop.csv", rows_part2)

proc0_print("")
proc0_print("Done.")
proc0_print(f"Saved Part 1 CSV: {out_dir / 'part1_H_law.csv'}")
proc0_print(f"Saved Part 2 CSV: {out_dir / 'part2_backward_H_stop.csv'}")
proc0_print(f"Saved arrays under: {out_dir}")