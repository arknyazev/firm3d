#!/usr/bin/env python3
"""Forward Monte Carlo on the valid fusion birth pool, with perturbed coils.

Target
------
    Q = (1 / N_pool) * sum_{i=1..N_pool} A(x_i)

where ``{x_i}`` is the empirical valid fusion birth pool from
``fusion_ic_file`` after the LCFS and Boozer-validity filters (same filters
that methods 2 and 3 apply, so all three methods share an identical pool and
target).  ``A`` is the forward-tracing wall-hit indicator under the requested
perturbed-coil field.

Estimator
---------
    Q_hat_FWD = (1/N) * sum_k A(X_k),  X_k ~ Uniform over the valid pool
    Y_k       = A(X_k)  (per-sample contribution; no reactivity weighting)

The fusion IC file is already distributed according to the physical birth
law, so sampling it uniformly IS the target measure — no weighting is
needed.
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
    estimator_metrics, summarize_stop_codes, write_metrics_csv,
)
from plot_utils import plot_s_hist, plot_xy_rz
from vtk_utils import (
    trace_snapshots, write_coils_and_surface_vtk, write_points_vtu,
    write_trajectory_polylines,
)


THIS_DIR = Path(__file__).resolve().parent
COILS_DIR = THIS_DIR.parent / "mc_backward" / "LandremanPaulQH_coils"


def parse_args():
    p = argparse.ArgumentParser(
        description="Forward MC wall-hit estimator on perturbed coils.")
    p.add_argument("--perturbation_id", type=int, default=57,
                   help="Perturbation seed (0 = baseline). Default 57.")
    p.add_argument("--n_samples", type=int, default=10_000,
                   help="Number of forward samples drawn uniformly from pool.")
    p.add_argument("--n_pool", type=int, default=50_000,
                   help="Max rows read from fusion_ic_file (first-N).")
    p.add_argument("--seed", type=int, default=57,
                   help="RNG seed used for pool sampling.")
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
                                "initial_conditions_boozer.txt"),
                   help="Pre-computed Boozer (s,theta,zeta) aligned with "
                        "fusion_ic_file; used when present.")
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
        / "forward_mc"
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

    # Sample uniformly from pool --------------------------------------------
    rng = np.random.default_rng(args.seed)
    N = int(args.n_samples)
    sample_idx = rng.integers(0, N_pool, size=N)
    np.save(out_dir / "sample_idx.npy", sample_idx)

    R_s    = pool["R"][sample_idx]
    phi_s  = wrap_phi(pool["phi"][sample_idx],
                      field["phi_min"], field["phi_max"])
    Z_s    = pool["Z"][sample_idx]
    vpar_s = pool["vpar"][sample_idx]
    H_s    = np.full(N, H_FUSION, dtype=np.float64)
    stz    = flatten_stz(R_s, phi_s, Z_s)

    # Forward trace (drag only, no energy stop) -----------------------------
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

    # Estimator -------------------------------------------------------------
    A = (stop_codes == 1).astype(np.float64)
    Y = A  # forward MC: Y = A

    metrics = estimator_metrics(Y, "FWD", N=N)
    metrics.update({
        "N_wall_hits":           int(A.sum()),
        "N_pool":                int(N_pool),
        "perturbation_id":       int(args.perturbation_id),
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

    # VTK point clouds ------------------------------------------------------
    print("\n--- VTK exports ---")
    X_s = R_s * np.cos(phi_s); Y_s = R_s * np.sin(phi_s)
    sampled_xyz = np.column_stack([X_s, Y_s, Z_s])
    write_points_vtu(out_dir / "sampled_births.vtu", sampled_xyz,
                     point_data={"vpar_init": vpar_s,
                                 "H_init":    H_s,
                                 "s_boozer":  s_pool[sample_idx]})

    fwd_xyz = np.column_stack([fwd[:, 1], fwd[:, 2], fwd[:, 3]])
    write_points_vtu(out_dir / "forward_endpoints.vtu", fwd_xyz,
                     point_data={"vpar":       fwd[:, 4],
                                 "H":          fwd[:, 5],
                                 "stop_code":  stop_codes.astype(np.float64),
                                 "t_elapsed":  fwd[:, 0],
                                 "wall_hit":   A})

    # Optional trajectory polylines ----------------------------------------
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

    # Plots -----------------------------------------------------------------
    pdir = out_dir / "plots"
    plot_xy_rz(pdir / "pool_XY_RZ.png", pool["R"], pool["phi"], pool["Z"],
               f"Valid fusion pool (N={N_pool})", color="grey")
    plot_xy_rz(pdir / "sampled_XY_RZ.png", R_s, phi_s, Z_s,
               f"FWD samples (N={N})", color="C0")
    plot_s_hist(pdir / "pool_s_hist.png", s_pool,
                "Valid fusion pool — s distribution")
    plot_s_hist(pdir / "sampled_s_hist.png", s_pool[sample_idx],
                "FWD samples — s distribution")

    print(f"\nDone. Outputs at {out_dir}")


if __name__ == "__main__":
    main()
