#!/usr/bin/env python3
"""Uniform-s importance sampling on the valid fusion birth pool, with
perturbed coils.  Non-backward-informed control for backward_informed_is.

Target
------
    Q = (1 / N_pool) * sum_{i=1..N_pool} A(x_i)

where ``{x_i}`` is the same valid fusion birth pool the other two methods
use.  NO reactivity weighting is applied.

Proposal
--------
Discrete proposal on the pool that is approximately uniform in Boozer s:

    1. Partition [0, 1] into ``s_score_nbins`` equal bins.
    2. Count fusion pool markers in each bin.  For a marker in bin b with
       occupancy n_bin(b) >= 1, set q_tilde_i = 1 / n_bin(i).
    3. Normalise q_i = q_tilde_i / sum_j q_tilde_j.

By construction q_i > 0 for every pool marker.  Every bin that contains any
pool marker receives equal TOTAL probability mass, so sampling from q is
uniform in s across occupied bins.

Estimator
---------
    p_target_i = 1 / N_pool
    w_i        = p_target_i / q_i
    Y_i        = A(X_i) * w_i
    Q_hat      = (1/N) * sum_k Y_k,   X_k ~ q
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
)
from firm3dpp import cartesian_gpu_tracing_drag

from perturbed_field_utils import (
    FieldConfig, build_perturbed_field, flatten_stz, wrap_phi,
)
from birth_pool_utils import (
    build_boozer_interpolant, ensure_valid_pool, load_fusion_pool,
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
        description="Uniform-s IS wall-hit estimator on perturbed coils.")
    p.add_argument("--perturbation_id", type=int, default=57,
                   help="Perturbation seed (0 = baseline). Default 57.")
    p.add_argument("--n_samples", type=int, default=10_000,
                   help="Number of forward samples drawn from q.")
    p.add_argument("--n_pool", type=int, default=50_000,
                   help="Max rows read from fusion_ic_file (first-N).")
    p.add_argument("--s_score_nbins", type=int, default=40,
                   help="Equal-width bins on [0,1] in Boozer s for q.")
    p.add_argument("--seed", type=int, default=57,
                   help="RNG seed used for proposal sampling.")
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
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--tmax_forward", type=float, default=1e-2)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--ne0", type=float, default=1e21)
    p.add_argument("--Te0_ev", type=float, default=100.0)
    p.add_argument("--coulomb_log", type=float, default=17.0)
    p.add_argument("--save_trajectories", action="store_true",
                   help="Re-trace a subsample with snapshots to build "
                        "Paraview polylines.")
    p.add_argument("--n_trajectory", type=int, default=200,
                   help="Subsample size for trajectory polylines.")
    p.add_argument("--n_snapshots", type=int, default=100,
                   help="Number of tmax snapshots per trajectory.")
    p.add_argument("--tmax_forward_trajectory", type=float, default=2e-6,
                   help="tmax used ONLY for forward trajectory snapshots. "
                        "Must be << tmax_forward so snapshot step dt is "
                        "comparable to the alpha gyro-period (~6e-8 s) and "
                        "polylines look like resolved orbits instead of "
                        "random jumps.  Does not affect the estimator.")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = args.out_dir or (
        Path("/pscratch/sd/m/mariagar/projects/mc_proj/results/mc_comparison")
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        / "uniform_s_is"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {out_dir}")

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

    # Pool -------------------------------------------------------------------
    print("\n--- Loading fusion birth pool ---")
    raw_pool = load_fusion_pool(args.fusion_ic_file, args.n_pool,
                                fusion_boozer_file=args.fusion_boozer_file)
    boozer_field = build_boozer_interpolant(args.boozmn_file)
    pool, s_pool, theta_pool, zeta_pool, pool_diag = ensure_valid_pool(
        raw_pool, field["sc_particle"], boozer_field,
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

    # Proposal q (uniform in Boozer s, discrete on pool) --------------------
    print("\n--- Building uniform-s proposal q ---")
    s_edges = np.linspace(0.0, 1.0, args.s_score_nbins + 1)
    hist, _ = np.histogram(s_pool, bins=s_edges)
    bin_idx = np.clip(
        np.searchsorted(s_edges, s_pool, side="right") - 1,
        0, args.s_score_nbins - 1,
    )
    n_bin_at_marker = hist[bin_idx]
    if np.any(n_bin_at_marker == 0):
        # Every pool marker sits in the bin corresponding to its own s, so
        # the count there is >= 1 by construction.  Guard anyway.
        raise RuntimeError(
            "Encountered a pool marker in an empty bin — this is impossible "
            "unless s_pool contains NaNs that slipped past ensure_valid_pool."
        )

    q_tilde = 1.0 / n_bin_at_marker.astype(np.float64)
    q_sum = float(q_tilde.sum())
    if not (np.isfinite(q_sum) and q_sum > 0):
        raise RuntimeError("q_tilde has non-positive / non-finite sum.")
    q = q_tilde / q_sum

    # Target: uniform over pool; IS weight w = (1/N_pool) / q
    p_target = 1.0 / float(N_pool)
    w_per_pool_marker = p_target / q  # per-pool weight; subset below

    np.save(out_dir / "q_weights.npy", q)
    np.save(out_dir / "s_edges.npy", s_edges)
    np.save(out_dir / "s_bin_counts.npy", hist)
    print(f"  s_score_nbins        : {args.s_score_nbins}")
    print(f"  non-empty bins       : {int((hist > 0).sum())}/"
          f"{args.s_score_nbins}")
    print(f"  q support            : {int((q > 0).sum())}/{N_pool}")
    print(f"  q min / max          : {q.min():.3e} / {q.max():.3e}")

    # Sample from q ---------------------------------------------------------
    rng = np.random.default_rng(args.seed)
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

    metrics = estimator_metrics(Y, "UNIF_S_IS", N=N)
    w_diag = is_weight_diagnostics(w_draw)
    metrics.update(w_diag)
    metrics.update({
        "N_wall_hits":           int(A.sum()),
        "N_pool":                int(N_pool),
        "perturbation_id":       int(args.perturbation_id),
        "s_score_nbins":         int(args.s_score_nbins),
        "bn_mean":               float(field["bn_stats"][0]),
        "bn_max":                float(field["bn_stats"][1]),
        "seed":                  int(args.seed),
        "pool_n_input":          pool_diag["n_input"],
        "pool_n_outside_LCFS":   pool_diag["n_outside"],
        "pool_n_boozer_failed":  pool_diag["n_bz_failed"],
        "pool_n_boozer_invalid": pool_diag["n_bz_invalid"],
    })
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
        print("\n--- Forward trajectory snapshots ---")
        n_traj = int(min(args.n_trajectory, N))
        if n_traj > 0:
            traj_sel = rng.choice(N, size=n_traj, replace=False)
            snap_xyz, snap_vpar, snap_H, snap_time, _, _ = trace_snapshots(
                tracer=cartesian_gpu_tracing_drag, field=field,
                R_init=R_s[traj_sel], phi_init=phi_s[traj_sel],
                Z_init=Z_s[traj_sel],
                vpar_init=vpar_s[traj_sel], H_init=H_s[traj_sel],
                mass=MASS, charge=CHARGE, speed_ref=speed_ref,
                coulomb_log=args.coulomb_log, Te_in_eV=True,
                tmax=args.tmax_forward_trajectory, tol=args.tol,
                n_snapshots=int(args.n_snapshots),
                H_stop=0.0, use_energy_stop=False,
                label="forward snapshots",
            )
            np.save(out_dir / "fwd_trajectories_xyz.npy",  snap_xyz)
            np.save(out_dir / "fwd_trajectories_vpar.npy", snap_vpar)
            np.save(out_dir / "fwd_trajectories_H.npy",    snap_H)
            np.save(out_dir / "fwd_trajectories_time.npy", snap_time)
            np.save(out_dir / "fwd_trajectories_idx.npy",  traj_sel)

            initial_xyz = sampled_xyz[traj_sel]
            write_trajectory_polylines(
                out_dir / "fwd_trajectories.vtu",
                initial_xyz=initial_xyz,
                snap_xyz=snap_xyz, snap_time=snap_time,
                snap_vpar=snap_vpar, snap_H=snap_H,
                initial_vpar=vpar_s[traj_sel],
                initial_H=H_s[traj_sel],
                particle_ids=traj_sel,
            )

    # Plots ---------------------------------------------------------------
    pdir = out_dir / "plots"
    plot_xy_rz(pdir / "sampled_XY_RZ.png", R_s, phi_s, Z_s,
               f"uniform-s IS samples (N={N})", color="C2")
    plot_s_hist(pdir / "pool_s_hist.png", s_pool,
                "Valid fusion pool — s distribution")
    plot_s_hist(pdir / "sampled_s_hist.png", s_pool[sample_idx],
                "uniform-s IS samples — s distribution (should be ~flat)")
    plot_weight_hist(pdir / "weight_hist.png", w_draw,
                     "uniform-s IS weights",
                     ess=w_diag["effective_sample_size"])

    print(f"\nDone. Outputs at {out_dir}")


if __name__ == "__main__":
    main()
