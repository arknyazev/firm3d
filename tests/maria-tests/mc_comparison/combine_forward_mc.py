#!/usr/bin/env python3
"""Combine per-shard ``forward_mc_perturbed.py`` runs into a single
"gold standard" FWD estimate.

Expects a directory layout like::

    <run_dir>/
        shard_0/   forward_results.npy, metrics_summary.csv, ...
        shard_1/   ...
        shard_2/   ...
        shard_3/   ...

where each ``shard_k`` was produced with the same ``--perturbation_id`` and
``--n_pool`` but a different ``--seed`` (so the sampled pool indices are IID
across shards).  Because the FWD estimator is an IID Bernoulli average, the
combined estimator is just

    Y_all = concat([ Y_k  for k in shards ])
    Q_hat = mean(Y_all)
    SE    = sqrt( var(Y_all, ddof=1) / N_total )

No per-shard weighting needed.

Writes into ``<run_dir>/<out_name>/`` (default ``combined``):
    Y_all.npy             (N_total,) per-sample contributions
    metrics_combined.csv  row with combined Q_hat, SE, CVs, N_target, ...
    shard_qhats.csv       per-shard Q_hat, SE, N, n_hits

Also prints per-shard z-scores as a consistency check — non-coherent shards
(|z| > 3) are a sign of a seed collision or a bugged shard.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Combine per-shard FWD outputs into a gold estimate.")
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Parent directory containing shard_* subdirs.")
    p.add_argument("--out_name", type=str, default="combined",
                   help="Subdir of run_dir for combined output "
                        "(default: combined).")
    p.add_argument("--cv_targets", type=float, nargs="+",
                   default=[0.10, 0.05, 0.02, 0.01],
                   help="cv_estimator targets for N_target columns.")
    return p.parse_args()


def find_shards(run_dir):
    shards = sorted(p for p in run_dir.glob("shard_*") if p.is_dir())
    if not shards:
        raise SystemExit(f"No shard_* subdirs under {run_dir}")
    return shards


def load_shard(shard_dir):
    fr = np.load(shard_dir / "forward_results.npy")
    stop = fr[:, 6].astype(int)
    A = (stop == 1).astype(np.float64)
    return A


def per_sample_metrics(Y, cv_targets):
    """Return a dict matching the estimator_utils layout used by the
    per-method scripts, so the combined CSV is directly comparable."""
    Y = np.asarray(Y, dtype=np.float64)
    N = int(Y.size)
    Q = float(Y.mean()) if N else float("nan")
    var_s = float(Y.var(ddof=1)) if N > 1 else 0.0
    var_est = var_s / N if N else float("nan")
    se = float(np.sqrt(var_est)) if N else float("nan")
    cv_est = se / Q if Q > 0 else float("nan")
    cv_one = float(np.sqrt(var_s)) / Q if Q > 0 else float("nan")

    out = {
        "method":             "FWD_GOLD",
        "N":                  N,
        "Q_hat":              Q,
        "sample_variance":    var_s,
        "estimator_variance": var_est,
        "standard_error":     se,
        "cv_estimator":       cv_est,
        "cv_single_sample":   cv_one,
        "N_wall_hits":        int(Y.sum()) if N else 0,
    }
    for c in cv_targets:
        key = f"N_target_cv_{int(round(100 * c)):02d}pct"
        out[key] = (float((cv_one / c) ** 2)
                    if np.isfinite(cv_one) else float("nan"))
    return out


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    shards = find_shards(run_dir)
    print(f"Found {len(shards)} shard(s) under {run_dir}")

    Ys = []
    per_shard = []
    for d in shards:
        try:
            A = load_shard(d)
        except FileNotFoundError:
            print(f"  WARNING: {d.name} missing forward_results.npy — skipping")
            continue
        Ys.append(A)
        N = len(A)
        Q = float(A.mean())
        var_s = float(A.var(ddof=1)) if N > 1 else 0.0
        SE = float(np.sqrt(var_s / N)) if N else 0.0
        per_shard.append({
            "shard":       d.name,
            "N":           N,
            "n_hits":      int(A.sum()),
            "Q_hat":       Q,
            "standard_error": SE,
        })
        print(f"  {d.name}: N={N:d}, hits={int(A.sum()):d}, "
              f"Q={Q:.6e}, SE={SE:.3e}")

    if not Ys:
        print("No usable shards. Exiting.")
        sys.exit(1)

    Y_all = np.concatenate(Ys)
    combined = per_sample_metrics(Y_all, args.cv_targets)
    combined["N_shards"] = len(Ys)

    # Pairwise shard z-scores, for a consistency check.
    print("\nPer-shard |z| vs combined Q_hat (>2 is unusual, >3 is suspicious):")
    for row in per_shard:
        denom = row["standard_error"]
        if denom > 0:
            z = abs(row["Q_hat"] - combined["Q_hat"]) / denom
            row["z_vs_combined"] = z
            flag = "" if z < 2 else ("  <-- check" if z >= 3 else "")
            print(f"  {row['shard']:<10}  |z|={z:.2f}{flag}")
        else:
            row["z_vs_combined"] = float("nan")

    print("\nPairwise |z| between shards:")
    for i in range(len(per_shard)):
        for j in range(i + 1, len(per_shard)):
            a = per_shard[i]; b = per_shard[j]
            denom = np.sqrt(a["standard_error"] ** 2
                            + b["standard_error"] ** 2)
            if denom > 0:
                z = abs(a["Q_hat"] - b["Q_hat"]) / denom
                flag = "" if z < 2 else ("  <-- check" if z >= 3 else "")
                print(f"  {a['shard']} vs {b['shard']}: |z|={z:.2f}{flag}")

    # Write output
    out_dir = run_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "Y_all.npy", Y_all)

    with open(out_dir / "metrics_combined.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined.keys()))
        writer.writeheader()
        writer.writerow(combined)

    with open(out_dir / "shard_qhats.csv", "w", newline="") as f:
        keys = list(per_shard[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in per_shard:
            writer.writerow(r)

    print("\nCombined estimator:")
    print(f"  N_total       = {combined['N']}")
    print(f"  N_wall_hits   = {combined['N_wall_hits']}")
    print(f"  Q_hat         = {combined['Q_hat']:.6e}")
    print(f"  standard_error= {combined['standard_error']:.3e}")
    print(f"  cv_estimator  = {combined['cv_estimator']:.4f}")
    print(f"  wrote         : {out_dir}")


if __name__ == "__main__":
    main()
