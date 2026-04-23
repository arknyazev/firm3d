#!/usr/bin/env python3
"""Backward-informed importance sampling on the valid fusion birth pool,
with perturbed coils.

Stage A — backward pilot
------------------------
Same workflow as ``mc_backward/backward_tracing_only.py`` but on the
requested perturbed-coil field:
  1. Load wall IC (R, phi, Z).
  2. Sample H_wall ~ U(H_low, H_fusion); sample |lambda| ~ U(0,1) with sign
     flipped so (v_par b_hat) * n_out <= 0 at t=0.
  3. Backward GPU trace with deterministic drag; use_energy_stop=True.
  4. Keep endpoints with stop_code == 2 (reached H_fusion).
  5. Convert (R, phi, Z) -> Boozer; keep 0 <= s <= 1.

Stage B — build score + proposal on fusion pool
-----------------------------------------------
  6. Build a 1-D histogram of successful backward endpoints in Boozer s
     (uniform bins on [0,1]).  Counts = s_hist.
  7. For each fusion pool marker in bin b_i, score_i = s_hist[b_i].
  8. q_tilde_i = (1 - alpha_mix) * score_i + alpha_mix.   (alpha_mix > 0
     guarantees strictly positive proposal mass on every pool marker —
     independent of what the backward pilot did or didn't find.)
  9. q_i = q_tilde_i / sum_j q_tilde_j.

IMPORTANT: the target is the empirical uniform on the valid fusion pool —
the backward cloud is only pilot information used to shape the proposal.
NO reactivity weighting is applied on top of fusion_ic_file.

Estimator
---------
    p_target_i = 1 / N_pool
    w_i        = p_target_i / q_i
    Y_i        = A(X_i) * w_i
    Q_hat      = (1/N) * sum_k Y_k,  X_k ~ q
"""
import argparse
import time
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np

from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as H_FUSION,
    ONE_EV,
)
from firm3dpp import (
    cartesian_gpu_tracing_backward_drag,
    cartesian_gpu_tracing_drag,
)

from perturbed_field_utils import (
    FieldConfig, build_perturbed_field, flatten_stz, wrap_phi,
)
from birth_pool_utils import (
    build_boozer_interpolant, convert_successes_to_boozer,
    ensure_valid_pool, load_fusion_pool,
)
from estimator_utils import (
    estimator_metrics, is_weight_diagnostics, summarize_stop_codes,
    write_metrics_csv,
)
from plot_utils import plot_s_hist, plot_weight_hist, plot_xy_rz
from vtk_utils import (
    trace_snapshots, write_coils_and_surface_vtk, write_points_vtu,
    write_trajectory_polylines,
)


THIS_DIR = Path(__file__).resolve().parent
COILS_DIR = THIS_DIR.parent / "mc_backward" / "LandremanPaulQH_coils"


def parse_args():
    p = argparse.ArgumentParser(
        description="Backward-informed IS wall-hit estimator on perturbed "
                    "coils.")
    p.add_argument("--perturbation_id", type=int, default=57,
                   help="Perturbation seed (0 = baseline). Default 57.")
    p.add_argument("--n_samples", type=int, default=10_000,
                   help="Number of forward samples drawn from q.")
    p.add_argument("--n_pool", type=int, default=50_000,
                   help="Max rows read from fusion_ic_file (first-N).")
    p.add_argument("--n_pilot", type=int, default=100_000,
                   help="Number of backward wall starts (pilot sample).")
    p.add_argument("--s_score_nbins", type=int, default=40,
                   help="Equal-width bins on [0,1] in Boozer s for score.")
    p.add_argument("--alpha_mix", type=float, default=0.05,
                   help="Additive floor on per-marker score to guarantee "
                        "positive proposal mass (strictly > 0).")
    p.add_argument("--seed", type=int, default=57,
                   help="RNG seed used for pilot pitch/energy and proposal "
                        "sampling.")
    p.add_argument("--coil_file", type=Path,
                   default=COILS_DIR / "coils.curves_22_7_21")
    p.add_argument("--vmec_input_file", type=Path,
                   default=COILS_DIR / "input.vmec")
    p.add_argument("--boozmn_file", type=Path,
                   default=COILS_DIR / "boozmn.nc")
    p.add_argument("--fusion_ic_file", type=Path,
                   default=Path("/pscratch/sd/m/mariagar/projects/mc_proj/IC/"
                                "initial_conditions_cylindrical.txt"))
    p.add_argument("--fusion_boozer_file", type=Path,
                   default=Path("/pscratch/sd/m/mariagar/projects/mc_proj/IC/"
                                "initial_conditions_boozer.txt"))
    p.add_argument("--wall_ic_file", type=Path,
                   default=Path("/pscratch/sd/m/mariagar/projects/mc_proj/IC/"
                                "initial_conditions_surface_cylindrical.txt"))
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--H_low_MeV", type=float, default=3.0,
                   help="Lower bound on wall energy for backward pilot.")
    p.add_argument("--tmax_forward", type=float, default=1e-2)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--ne0", type=float, default=1e21)
    p.add_argument("--Te0_ev", type=float, default=100.0)
    p.add_argument("--coulomb_log", type=float, default=17.0)
    p.add_argument("--normal_fd_eps", type=float, default=1e-4)
    p.add_argument("--save_trajectories", action="store_true",
                   help="Re-trace forward + backward subsamples with "
                        "snapshots to build Paraview polylines.")
    p.add_argument("--n_trajectory", type=int, default=200,
                   help="Subsample size for trajectory polylines (applied "
                        "to both forward and backward).")
    p.add_argument("--n_snapshots", type=int, default=100,
                   help="Number of tmax snapshots per trajectory.")
    return p.parse_args()


# ── Slowing-down time for backward tmax sizing ──────────────────────────────
def _slowing_down_time(ne, Te_eV, mass, coulomb_log):
    eps0 = 8.8541878128e-12
    e_ch = 1.602176634e-19
    m_e  = 9.1093837015e-31
    Z_alpha = 2.0
    Te_J = Te_eV * e_ch
    num = 3.0 * (2.0 * np.pi) ** 1.5 * eps0 ** 2 * mass * Te_J ** 1.5
    den = Z_alpha ** 2 * e_ch ** 4 * np.sqrt(m_e) * ne * coulomb_log
    return num / den


# ── Outward unit normal via finite-difference of signed distance ────────────
def _outward_unit_normal(xyz_wall, sc_particle, eps):
    def sd_xyz(xyz):
        R = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        Z = xyz[:, 2]
        return sc_particle.evaluate_rphiz(
            np.column_stack([R, phi, Z])
        ).ravel()

    grad_sd = np.zeros_like(xyz_wall)
    for k in range(3):
        d = np.zeros(3); d[k] = eps
        grad_sd[:, k] = (sd_xyz(xyz_wall + d) - sd_xyz(xyz_wall - d)) / (2 * eps)
    grad_norm = np.linalg.norm(grad_sd, axis=1, keepdims=True)
    grad_norm[grad_norm == 0.0] = 1.0
    return -grad_sd / grad_norm


def run_backward_pilot(args, field, rng):
    """Run the backward pilot (wall IC + pitch/energy sampling + backward GPU
    trace + Boozer of successes) and return (s_success, diagnostics)."""
    print("\n--- Stage A: backward pilot ---")
    H_low  = args.H_low_MeV * 1e6 * ONE_EV
    H_high = H_FUSION

    tau_s = _slowing_down_time(args.ne0, args.Te0_ev, MASS, args.coulomb_log)
    tmax_backward = float(1.2 * np.log(H_high / H_low) * tau_s)
    print(f"  tau_s={tau_s:.4e} s, tmax_backward={tmax_backward:.4e} s")

    wall_ic = np.loadtxt(str(args.wall_ic_file), comments="#")
    R_all   = wall_ic[:, 0]
    phi_all = wall_ic[:, 1]
    Z_all   = wall_ic[:, 2]
    n_avail = len(R_all)
    n_pilot = int(min(args.n_pilot, n_avail))
    print(f"  wall IC available: {n_avail}, pilot size: {n_pilot}")

    idx = rng.choice(n_avail, size=n_pilot, replace=False)
    R_wall, phi_wall, Z_wall = R_all[idx], phi_all[idx], Z_all[idx]

    # Drop wall IC points outside LCFS
    sd = field["sc_particle"].evaluate_rphiz(
        np.column_stack([R_wall, phi_wall, Z_wall])
    ).ravel()
    inside = sd >= 0
    n_out = int((~inside).sum())
    if n_out:
        print(f"  {n_out} wall points outside LCFS — dropped")
        R_wall, phi_wall, Z_wall = R_wall[inside], phi_wall[inside], Z_wall[inside]
        n_pilot = int(inside.sum())
    phi_wall = wrap_phi(phi_wall, field["phi_min"], field["phi_max"])

    xyz_wall = np.column_stack([
        R_wall * np.cos(phi_wall),
        R_wall * np.sin(phi_wall),
        Z_wall,
    ])
    n_out_hat = _outward_unit_normal(xyz_wall, field["sc_particle"],
                                     args.normal_fd_eps)

    field["bs"].set_points(xyz_wall)
    B_xyz = np.asarray(field["bs"].B())
    B_mag = np.linalg.norm(B_xyz, axis=1, keepdims=True)
    B_mag[B_mag == 0.0] = 1.0
    b_hat = B_xyz / B_mag
    b_dot_n = np.einsum("ij,ij->i", b_hat, n_out_hat)

    H_wall = rng.uniform(H_low, H_high, size=n_pilot)
    v_total = np.sqrt(2.0 * H_wall / MASS)

    lam_abs  = rng.uniform(0.0, 1.0, size=n_pilot)
    lam_wall = -np.sign(b_dot_n) * lam_abs
    mask_zero = np.isclose(b_dot_n, 0.0)
    lam_wall[mask_zero] = rng.uniform(-1.0, 1.0,
                                      size=int(mask_zero.sum()))
    vtang_w = lam_wall * v_total

    stz_init = flatten_stz(R_wall, phi_wall, Z_wall)

    print("  backward GPU tracing...")
    speed_ref = float(sqrt(2.0 * H_FUSION / MASS))
    t0 = time.time()
    out = cartesian_gpu_tracing_backward_drag(
        field["cell_quad_pts"],
        np.ascontiguousarray(field["r_range"],   dtype=np.float64),
        np.ascontiguousarray(field["phi_range"], dtype=np.float64),
        np.ascontiguousarray(field["z_range"],   dtype=np.float64),
        np.ascontiguousarray(stz_init, dtype=np.float64),
        float(MASS), float(CHARGE), speed_ref,
        np.ascontiguousarray(vtang_w, dtype=np.float64),
        np.ascontiguousarray(H_wall,  dtype=np.float64),
        float(args.coulomb_log), True,
        float(tmax_backward), float(args.tol), int(n_pilot),
        float(H_FUSION), True,
    )
    bwd = np.asarray(out, dtype=np.float64).reshape(n_pilot, 7)
    print(f"  backward tracing done in {time.time() - t0:.2f}s")

    stop_codes_bwd = bwd[:, 6].astype(int)
    sc_bwd_counts = summarize_stop_codes(stop_codes_bwd)
    print(f"  backward stop codes: {sc_bwd_counts}")

    hit_fusion = stop_codes_bwd == 2
    M = int(hit_fusion.sum())
    print(f"  successes (stop_code==2): {M}/{n_pilot} "
          f"({100 * M / max(n_pilot, 1):.2f}%)")

    if M == 0:
        raise SystemExit("No backward successes in pilot; cannot build "
                         "backward-informed proposal.")

    X_b = bwd[hit_fusion, 1]
    Y_b = bwd[hit_fusion, 2]
    Z_b = bwd[hit_fusion, 3]
    R_b = np.sqrt(X_b ** 2 + Y_b ** 2)
    phi_b = np.arctan2(Y_b, X_b)
    birth_rphiz = np.column_stack([R_b, phi_b, Z_b])

    boozer_field = build_boozer_interpolant(args.boozmn_file)
    s_all, theta_all, zeta_all, valid, succ_diag = convert_successes_to_boozer(
        birth_rphiz, field["sc_particle"], boozer_field,
    )
    M_valid = int(valid.sum())
    s_success = s_all[valid]
    print(f"  backward valid Boozer s: {M_valid}/{M}")

    diag = {
        "n_pilot":                      int(n_pilot),
        "n_backward_success":           int(M),
        "n_backward_success_valid":     int(M_valid),
        "tmax_backward_s":              float(tmax_backward),
        "H_low_MeV":                    float(args.H_low_MeV),
    }
    for code, count in sc_bwd_counts.items():
        diag[f"bwd_stop_code_{code}_count"] = int(count)

    # Wall IC state (full pilot) and success-subset state, kept so the
    # trajectory polylines can start at the wall and end at the birth
    # endpoint without re-running the pilot.
    wall_xyz = np.column_stack([
        R_wall * np.cos(phi_wall), R_wall * np.sin(phi_wall), Z_wall,
    ])

    pilot = {
        "bwd":             bwd,
        "hit_fusion":      hit_fusion,
        "R_b":             R_b,
        "phi_b":           phi_b,
        "Z_b":             Z_b,
        "vpar_b":          bwd[hit_fusion, 4],
        "H_b":             bwd[hit_fusion, 5],
        "t_b":             bwd[hit_fusion, 0],
        "s_success":       s_success,
        "birth_rphiz":     birth_rphiz,
        "birth_s_all":     s_all,
        "birth_valid":     valid,
        "boozer_field":    boozer_field,
        # Wall IC (all pilot starts)
        "wall_R":          R_wall,
        "wall_phi":        phi_wall,
        "wall_Z":          Z_wall,
        "wall_xyz":        wall_xyz,
        "wall_H":          H_wall,
        "wall_vpar":       vtang_w,
        "tmax_backward":   float(tmax_backward),
    }
    return pilot, diag


def main():
    args = parse_args()

    out_dir = args.out_dir or (
        Path("/pscratch/sd/m/mariagar/projects/mc_proj/results/mc_comparison")
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        / "backward_informed_is"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {out_dir}")

    alpha = float(args.alpha_mix)
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha_mix must satisfy 0 < alpha <= 1, got {alpha}")

    # Field ------------------------------------------------------------------
    cfg = FieldConfig(coil_file=args.coil_file,
                      vmec_input_file=args.vmec_input_file)

    def ne_fun(rphiz): return np.full(rphiz.shape[0], args.ne0,
                                      dtype=np.float64)
    def Te_fun(rphiz): return np.full(rphiz.shape[0], args.Te0_ev,
                                      dtype=np.float64)

    field = build_perturbed_field(cfg, args.perturbation_id, ne_fun, Te_fun)
    np.save(out_dir / "bn_stats.npy", field["bn_stats"])

    # Paraview-friendly exports of the field geometry
    write_coils_and_surface_vtk(out_dir, field["curves"], field["s_input"])

    rng = np.random.default_rng(args.seed)

    # Stage A — backward pilot --------------------------------------------
    pilot, pilot_diag = run_backward_pilot(args, field, rng)

    np.save(out_dir / "backward_results.npy",       pilot["bwd"])
    np.save(out_dir / "backward_birth_rphiz.npy",   pilot["birth_rphiz"])
    np.save(out_dir / "backward_birth_s.npy",       pilot["birth_s_all"])
    np.save(out_dir / "backward_birth_valid.npy",   pilot["birth_valid"])
    np.save(out_dir / "backward_s_success.npy",     pilot["s_success"])

    # Pool ----------------------------------------------------------------
    print("\n--- Loading fusion birth pool ---")
    raw_pool = load_fusion_pool(args.fusion_ic_file, args.n_pool,
                                fusion_boozer_file=args.fusion_boozer_file)
    pool, s_pool, theta_pool, zeta_pool, pool_diag = ensure_valid_pool(
        raw_pool, field["sc_particle"], pilot["boozer_field"],
    )
    N_pool = len(pool["R"])
    if N_pool == 0:
        raise SystemExit("No valid markers in fusion pool; cannot run.")

    np.save(out_dir / "pool_R.npy",     pool["R"])
    np.save(out_dir / "pool_phi.npy",   pool["phi"])
    np.save(out_dir / "pool_Z.npy",     pool["Z"])
    np.save(out_dir / "pool_vpar.npy",  pool["vpar"])
    np.save(out_dir / "pool_s.npy",     s_pool)
    np.save(out_dir / "pool_theta.npy", theta_pool)
    np.save(out_dir / "pool_zeta.npy",  zeta_pool)

    # Stage B — backward-histogram score on pool ---------------------------
    print("\n--- Stage B: score + proposal on pool ---")
    s_edges = np.linspace(0.0, 1.0, args.s_score_nbins + 1)
    s_hist, _ = np.histogram(pilot["s_success"], bins=s_edges)

    pool_bin_idx = np.clip(
        np.searchsorted(s_edges, s_pool, side="right") - 1,
        0, args.s_score_nbins - 1,
    )
    score = s_hist[pool_bin_idx].astype(np.float64)

    # Proposal on the pool:  q_tilde_i = (1 - alpha_mix) * score_i + alpha_mix
    q_tilde = (1.0 - alpha) * score + alpha
    q_sum = float(q_tilde.sum())
    if not (np.isfinite(q_sum) and q_sum > 0):
        raise RuntimeError("q_tilde has non-positive / non-finite sum.")
    q = q_tilde / q_sum

    # Target + IS weight per pool marker
    p_target = 1.0 / float(N_pool)
    w_per_pool_marker = p_target / q

    np.save(out_dir / "s_edges.npy",           s_edges)
    np.save(out_dir / "backward_s_hist.npy",   s_hist)
    np.save(out_dir / "score_on_pool.npy",     score)
    np.save(out_dir / "q_weights.npy",         q)

    print(f"  alpha_mix            : {alpha:.3e}")
    print(f"  s_score_nbins        : {args.s_score_nbins}")
    print(f"  bwd non-empty bins   : {int((s_hist > 0).sum())}/"
          f"{args.s_score_nbins}")
    print(f"  q support            : {int((q > 0).sum())}/{N_pool} "
          f"(should equal N_pool when alpha>0)")
    print(f"  q min / max          : {q.min():.3e} / {q.max():.3e}")

    # Sample from q -------------------------------------------------------
    N = int(args.n_samples)
    sample_idx = rng.choice(N_pool, size=N, replace=True, p=q)
    np.save(out_dir / "sample_idx.npy", sample_idx)

    R_s    = pool["R"][sample_idx]
    phi_s  = wrap_phi(pool["phi"][sample_idx],
                      field["phi_min"], field["phi_max"])
    Z_s    = pool["Z"][sample_idx]
    vpar_s = pool["vpar"][sample_idx]
    H_s    = np.full(N, H_FUSION, dtype=np.float64)
    stz    = flatten_stz(R_s, phi_s, Z_s)
    w_draw = w_per_pool_marker[sample_idx]
    np.save(out_dir / "is_weights.npy", w_draw)

    # Forward trace --------------------------------------------------------
    print("\n--- Forward GPU tracing (drag) ---")
    speed_ref = float(sqrt(2.0 * H_FUSION / MASS))
    t0 = time.time()
    out = cartesian_gpu_tracing_drag(
        field["cell_quad_pts"],
        np.ascontiguousarray(field["r_range"],   dtype=np.float64),
        np.ascontiguousarray(field["phi_range"], dtype=np.float64),
        np.ascontiguousarray(field["z_range"],   dtype=np.float64),
        np.ascontiguousarray(stz, dtype=np.float64),
        float(MASS), float(CHARGE), speed_ref,
        np.ascontiguousarray(vpar_s, dtype=np.float64),
        np.ascontiguousarray(H_s,    dtype=np.float64),
        float(args.coulomb_log), True,
        float(args.tmax_forward), float(args.tol), int(N),
        0.0, False,
    )
    fwd = np.asarray(out, dtype=np.float64).reshape(N, 7)
    print(f"  tracing done in {time.time() - t0:.2f}s")
    np.save(out_dir / "forward_results.npy", fwd)

    stop_codes = fwd[:, 6].astype(int)
    sc_counts = summarize_stop_codes(stop_codes)
    print(f"  stop codes: {sc_counts}")

    # Estimator ------------------------------------------------------------
    A = (stop_codes == 1).astype(np.float64)
    Y = A * w_draw

    metrics = estimator_metrics(Y, "BACKWARD_IS", N=N)
    w_diag = is_weight_diagnostics(w_draw)
    metrics.update(w_diag)
    metrics.update({
        "N_wall_hits":           int(A.sum()),
        "N_pool":                int(N_pool),
        "perturbation_id":       int(args.perturbation_id),
        "s_score_nbins":         int(args.s_score_nbins),
        "alpha_mix":             alpha,
        "bn_mean":               float(field["bn_stats"][0]),
        "bn_max":                float(field["bn_stats"][1]),
        "seed":                  int(args.seed),
        "pool_n_input":          pool_diag["n_input"],
        "pool_n_outside_LCFS":   pool_diag["n_outside"],
        "pool_n_boozer_failed":  pool_diag["n_bz_failed"],
        "pool_n_boozer_invalid": pool_diag["n_bz_invalid"],
    })
    metrics.update(pilot_diag)
    for code, count in sc_counts.items():
        metrics[f"stop_code_{code}_count"] = count

    write_metrics_csv(out_dir / "metrics_summary.csv", [metrics])

    with open(out_dir / "run_config.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k} = {v}\n")

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k:32s} = {v}")

    # VTK point clouds ----------------------------------------------------
    print("\n--- VTK exports ---")

    # Wall IC used by the backward pilot
    write_points_vtu(out_dir / "wall_starts.vtu", pilot["wall_xyz"],
                     point_data={"H_init":    pilot["wall_H"],
                                 "vpar_init": pilot["wall_vpar"]})

    # Backward-success endpoints (labelled with Boozer validity)
    X_be = pilot["R_b"] * np.cos(pilot["phi_b"])
    Y_be = pilot["R_b"] * np.sin(pilot["phi_b"])
    birth_xyz = np.column_stack([X_be, Y_be, pilot["Z_b"]])
    write_points_vtu(out_dir / "backward_endpoints.vtu", birth_xyz,
                     point_data={"vpar":         pilot["vpar_b"],
                                 "H":            pilot["H_b"],
                                 "s_boozer":     pilot["birth_s_all"],
                                 "valid_boozer": pilot["birth_valid"]
                                                     .astype(np.float64),
                                 "t_elapsed":    pilot["t_b"]})

    # Sampled births (drawn from q) and forward endpoints
    X_s = R_s * np.cos(phi_s); Y_s = R_s * np.sin(phi_s)
    sampled_xyz = np.column_stack([X_s, Y_s, Z_s])
    write_points_vtu(out_dir / "sampled_births.vtu", sampled_xyz,
                     point_data={"vpar_init": vpar_s,
                                 "H_init":    H_s,
                                 "s_boozer":  s_pool[sample_idx],
                                 "is_weight": w_draw})

    fwd_xyz = np.column_stack([fwd[:, 1], fwd[:, 2], fwd[:, 3]])
    write_points_vtu(out_dir / "forward_endpoints.vtu", fwd_xyz,
                     point_data={"vpar":       fwd[:, 4],
                                 "H":          fwd[:, 5],
                                 "stop_code":  stop_codes.astype(np.float64),
                                 "t_elapsed":  fwd[:, 0],
                                 "wall_hit":   A,
                                 "is_weight":  w_draw})

    # Optional trajectory polylines ---------------------------------------
    if args.save_trajectories:
        # Backward trajectories: subsample of successful (stop_code==2)
        print("\n--- Backward trajectory snapshots ---")
        success_idx = np.flatnonzero(pilot["hit_fusion"])
        n_traj_bwd = int(min(args.n_trajectory, success_idx.size))
        if n_traj_bwd > 0:
            bwd_sel = rng.choice(success_idx, size=n_traj_bwd, replace=False)
            snap_xyz, snap_vpar, snap_H, snap_time, _, _ = trace_snapshots(
                tracer=cartesian_gpu_tracing_backward_drag, field=field,
                R_init=pilot["wall_R"][bwd_sel],
                phi_init=pilot["wall_phi"][bwd_sel],
                Z_init=pilot["wall_Z"][bwd_sel],
                vpar_init=pilot["wall_vpar"][bwd_sel],
                H_init=pilot["wall_H"][bwd_sel],
                mass=MASS, charge=CHARGE, speed_ref=speed_ref,
                coulomb_log=args.coulomb_log, Te_in_eV=True,
                tmax=pilot["tmax_backward"], tol=args.tol,
                n_snapshots=int(args.n_snapshots),
                H_stop=H_FUSION, use_energy_stop=True,
                label="backward snapshots",
            )
            np.save(out_dir / "bwd_trajectories_xyz.npy",  snap_xyz)
            np.save(out_dir / "bwd_trajectories_vpar.npy", snap_vpar)
            np.save(out_dir / "bwd_trajectories_H.npy",    snap_H)
            np.save(out_dir / "bwd_trajectories_time.npy", snap_time)
            np.save(out_dir / "bwd_trajectories_idx.npy",  bwd_sel)

            write_trajectory_polylines(
                out_dir / "bwd_trajectories.vtu",
                initial_xyz=pilot["wall_xyz"][bwd_sel],
                snap_xyz=snap_xyz, snap_time=snap_time,
                snap_vpar=snap_vpar, snap_H=snap_H,
                initial_vpar=pilot["wall_vpar"][bwd_sel],
                initial_H=pilot["wall_H"][bwd_sel],
                particle_ids=bwd_sel,
            )

        # Forward trajectories: subsample of the IS forward draws
        print("\n--- Forward trajectory snapshots ---")
        n_traj_fwd = int(min(args.n_trajectory, N))
        if n_traj_fwd > 0:
            fwd_sel = rng.choice(N, size=n_traj_fwd, replace=False)
            snap_xyz, snap_vpar, snap_H, snap_time, _, _ = trace_snapshots(
                tracer=cartesian_gpu_tracing_drag, field=field,
                R_init=R_s[fwd_sel], phi_init=phi_s[fwd_sel],
                Z_init=Z_s[fwd_sel],
                vpar_init=vpar_s[fwd_sel], H_init=H_s[fwd_sel],
                mass=MASS, charge=CHARGE, speed_ref=speed_ref,
                coulomb_log=args.coulomb_log, Te_in_eV=True,
                tmax=args.tmax_forward, tol=args.tol,
                n_snapshots=int(args.n_snapshots),
                H_stop=0.0, use_energy_stop=False,
                label="forward snapshots",
            )
            np.save(out_dir / "fwd_trajectories_xyz.npy",  snap_xyz)
            np.save(out_dir / "fwd_trajectories_vpar.npy", snap_vpar)
            np.save(out_dir / "fwd_trajectories_H.npy",    snap_H)
            np.save(out_dir / "fwd_trajectories_time.npy", snap_time)
            np.save(out_dir / "fwd_trajectories_idx.npy",  fwd_sel)

            write_trajectory_polylines(
                out_dir / "fwd_trajectories.vtu",
                initial_xyz=sampled_xyz[fwd_sel],
                snap_xyz=snap_xyz, snap_time=snap_time,
                snap_vpar=snap_vpar, snap_H=snap_H,
                initial_vpar=vpar_s[fwd_sel],
                initial_H=H_s[fwd_sel],
                particle_ids=fwd_sel,
            )

    # Plots ---------------------------------------------------------------
    pdir = out_dir / "plots"
    plot_xy_rz(pdir / "sampled_XY_RZ.png", R_s, phi_s, Z_s,
               f"backward-IS samples (N={N})", color="C3")

    plot_s_hist(pdir / "pool_s_hist.png", s_pool,
                "Valid fusion pool — s distribution")
    plot_s_hist(pdir / "backward_success_s_hist.png", pilot["s_success"],
                "Backward-success s distribution (pilot)")
    plot_s_hist(pdir / "sampled_s_hist.png", s_pool[sample_idx],
                "backward-IS samples — s distribution")
    plot_weight_hist(pdir / "weight_hist.png", w_draw,
                     "backward-IS weights",
                     ess=w_diag["effective_sample_size"])

    # Plot score on pool (grouped by bin for shape)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    ax.bar(centers, s_hist, width=(centers[1] - centers[0]) * 0.95,
           alpha=0.7, color="C0", label="backward s_hist (counts)")
    ax.set_xlabel("s (Boozer)")
    ax.set_ylabel("backward successes per bin")
    ax.set_title("Backward pilot s-histogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pdir / "backward_s_hist.png", dpi=150)
    plt.close(fig)

    print(f"\nDone. Outputs at {out_dir}")


if __name__ == "__main__":
    main()
