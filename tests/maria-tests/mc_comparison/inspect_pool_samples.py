#!/usr/bin/env python3
"""Local helper for picking pool markers to visualize as trajectories.

Reads the outputs of a completed ``forward_mc_perturbed.py`` run — the
per-sample tracer results, the sample→pool index mapping, and the pool's
cylindrical + Boozer coordinates — and prints tables of lost / confined
particles with their (s, R, Z) values so you can hand-pick indices that
span interesting regions.

Also prints a suggested mix: N_lost + N_confined particles chosen to span
the Boozer-s range.

No GPU, no simsopt, no firm3d — runs on a laptop.  Just numpy + (optional)
matplotlib.

Usage
-----
    python inspect_pool_samples.py \
        --dir ~/Desktop/mc_plasma/results/mc_comparison/<run>/forward_mc/

    python inspect_pool_samples.py --dir <...> --plot

    python inspect_pool_samples.py --dir <...> --n_lost 10 --n_confined 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, required=True,
                   help="Path to a forward_mc/ run dir containing "
                        "forward_results.npy, sample_idx.npy, pool_*.npy")
    p.add_argument("--n_lost", type=int, default=5,
                   help="Suggested # of lost particles spanning s "
                        "(default: 5).")
    p.add_argument("--n_confined", type=int, default=5,
                   help="Suggested # of confined particles spanning s "
                        "(default: 5).")
    p.add_argument("--show_all", action="store_true",
                   help="Print the full per-particle table (can be long).")
    p.add_argument("--plot", action="store_true",
                   help="Also open a quick matplotlib figure.")
    return p.parse_args()


def span_quantiles(n_total, n_pick):
    """Return n_pick evenly-spaced integer indices in [0, n_total).
    Safe for n_total <= n_pick (returns all indices)."""
    if n_total == 0:
        return np.empty(0, dtype=int)
    if n_total <= n_pick:
        return np.arange(n_total)
    return np.linspace(0, n_total - 1, n_pick).round().astype(int)


def main():
    args = parse_args()
    d = args.dir.expanduser().resolve()
    if not d.exists():
        print(f"ERROR: dir does not exist: {d}", file=sys.stderr)
        sys.exit(1)

    # --- Load ---
    required = ["forward_results.npy", "sample_idx.npy",
                "pool_R.npy", "pool_phi.npy", "pool_Z.npy", "pool_s.npy"]
    missing = [f for f in required if not (d / f).exists()]
    if missing:
        print(f"ERROR: missing files in {d}: {missing}", file=sys.stderr)
        sys.exit(1)

    fr       = np.load(d / "forward_results.npy")
    sidx     = np.load(d / "sample_idx.npy")
    pool_R   = np.load(d / "pool_R.npy")
    pool_phi = np.load(d / "pool_phi.npy")
    pool_Z   = np.load(d / "pool_Z.npy")
    pool_s   = np.load(d / "pool_s.npy")

    vpar_pool = (np.load(d / "pool_vpar.npy") if (d / "pool_vpar.npy").exists()
                 else None)

    stop = fr[:, 6].astype(int)
    N = len(fr)
    N_hits = int((stop == 1).sum())
    N_conf = int((stop == 0).sum())

    print(f"Run dir         : {d}")
    print(f"N (samples)     : {N}")
    print(f"wall hits (A=1) : {N_hits}  ({100 * N_hits / max(N, 1):.3f}%)")
    print(f"confined (stop=0): {N_conf}  ({100 * N_conf / max(N, 1):.3f}%)")
    other = N - N_hits - N_conf
    if other:
        uniq, cnt = np.unique(stop[(stop != 0) & (stop != 1)],
                              return_counts=True)
        print(f"other stop codes: {dict(zip(uniq.tolist(), cnt.tolist()))}")
    print(f"pool size       : {len(pool_R)}")
    print(f"pool s range    : [{pool_s.min():.3f}, {pool_s.max():.3f}]")

    # --- Map sample_k -> pool index, per outcome, deduplicated ---
    def per_outcome(mask, label):
        ks = np.where(mask)[0]
        pool_idx = np.unique(sidx[ks])
        return {
            "label":    label,
            "n_samples": int(mask.sum()),
            "pool_idx": pool_idx,
            "s":        pool_s[pool_idx],
            "R":        pool_R[pool_idx],
            "phi":      pool_phi[pool_idx],
            "Z":        pool_Z[pool_idx],
        }

    lost  = per_outcome(stop == 1, "lost")
    conf  = per_outcome(stop == 0, "confined")

    print(f"\nunique lost pool markers     : {len(lost['pool_idx'])}")
    print(f"unique confined pool markers : {len(conf['pool_idx'])}")

    # --- Tables ---
    def print_table(group):
        idx = group["pool_idx"]
        ord_ = np.argsort(group["s"])
        print(f"\n--- {group['label'].upper()} ({len(idx)} unique pool markers; "
              f"showing {'all' if args.show_all else 'up to 30'}) ---")
        hdr = f"{'pool_idx':>10}  {'s':>7}  {'R':>7}  {'phi':>8}  {'Z':>7}"
        if vpar_pool is not None:
            hdr += f"  {'vpar':>11}"
        print(hdr)
        rows = ord_ if args.show_all else ord_[
            span_quantiles(len(ord_), 30)
        ]
        for i in rows:
            p = idx[i]
            line = (f"{p:>10d}  "
                    f"{group['s'][i]:>7.3f}  "
                    f"{group['R'][i]:>7.3f}  "
                    f"{group['phi'][i]:>8.3f}  "
                    f"{group['Z'][i]:>7.3f}")
            if vpar_pool is not None:
                line += f"  {vpar_pool[p]:>11.3e}"
            print(line)

    print_table(lost)
    print_table(conf)

    # --- Suggested picks ---
    def suggest(group, n):
        ord_ = np.argsort(group["s"])
        picks = ord_[span_quantiles(len(ord_), n)]
        return [int(group["pool_idx"][i]) for i in picks]

    lost_picks = suggest(lost, args.n_lost)
    conf_picks = suggest(conf, args.n_confined)
    combined_picks = lost_picks + conf_picks

    print(f"\n--- Suggested {args.n_lost} LOST pool indices "
          f"(spanning s) ---")
    for p in lost_picks:
        print(f"  {p:>10d}   s={pool_s[p]:.3f}  R={pool_R[p]:.3f}  "
              f"Z={pool_Z[p]:.3f}")

    print(f"\n--- Suggested {args.n_confined} CONFINED pool indices "
          f"(spanning s) ---")
    for p in conf_picks:
        print(f"  {p:>10d}   s={pool_s[p]:.3f}  R={pool_R[p]:.3f}  "
              f"Z={pool_Z[p]:.3f}")

    print("\nAll suggested picks (paste into --viz_indices for "
          "trajectory_viz.py):")
    print("  " + ",".join(str(p) for p in combined_picks))

    # --- Optional quick plot ---
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("(matplotlib not installed — skipping plot)")
            return

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        axs[0].hist(pool_s, bins=40, alpha=0.4, color="grey",
                    label=f"pool (N={len(pool_s)})")
        axs[0].hist(lost["s"], bins=40, alpha=0.6, color="C3",
                    label=f"lost unique (n={len(lost['pool_idx'])})")
        axs[0].hist(conf["s"], bins=40, alpha=0.4, color="C0",
                    label=f"confined unique (n={len(conf['pool_idx'])})")
        axs[0].set_xlabel("Boozer s")
        axs[0].set_ylabel("count")
        axs[0].set_title("s distribution by outcome")
        axs[0].legend()

        axs[1].scatter(conf["R"], conf["Z"], s=3, alpha=0.2, color="C0",
                       label="confined")
        axs[1].scatter(lost["R"], lost["Z"], s=6, alpha=0.7, color="C3",
                       label="lost")
        axs[1].scatter([pool_R[p] for p in lost_picks],
                       [pool_Z[p] for p in lost_picks],
                       s=80, facecolors="none", edgecolors="black",
                       linewidths=1.5, label="picks (lost)")
        axs[1].scatter([pool_R[p] for p in conf_picks],
                       [pool_Z[p] for p in conf_picks],
                       s=80, facecolors="none", edgecolors="black",
                       linewidths=1.5, linestyle="--",
                       label="picks (confined)")
        axs[1].set_aspect("equal")
        axs[1].set_xlabel("R [m]"); axs[1].set_ylabel("Z [m]")
        axs[1].set_title("RZ — outcomes + picks")
        axs[1].legend(fontsize=8)

        X_conf = conf["R"] * np.cos(conf["phi"])
        Y_conf = conf["R"] * np.sin(conf["phi"])
        X_lost = lost["R"] * np.cos(lost["phi"])
        Y_lost = lost["R"] * np.sin(lost["phi"])
        axs[2].scatter(X_conf, Y_conf, s=3, alpha=0.2, color="C0",
                       label="confined")
        axs[2].scatter(X_lost, Y_lost, s=6, alpha=0.7, color="C3",
                       label="lost")
        axs[2].set_aspect("equal")
        axs[2].set_xlabel("X [m]"); axs[2].set_ylabel("Y [m]")
        axs[2].set_title("XY — outcomes")
        axs[2].legend(fontsize=8)

        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
