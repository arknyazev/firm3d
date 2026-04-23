#!/usr/bin/env python3
"""Post-processing comparison plots for the three MC workflows.

Reads the timestamped run directory produced by
``run_three_methods_per_perturbation.sh`` (layout::

    <run_dir>/
        forward_mc/           metrics_summary.csv, forward_results.npy, ...
        uniform_s_is/         ...
        backward_informed_is/ ...

) and writes a set of comparison plots into ``<run_dir>/<out_subdir>/``.

No Python package dependencies outside numpy + matplotlib.  No coupling to
the estimator scripts — this is pure I/O + plotting.  Optionally also
reads a gold FWD directory (produced by ``run_forward_mc_gold.sh`` +
``combine_forward_mc.py``) and overlays it where relevant.

Plots produced
--------------
    qhat_bars.png           Q_hat ± 2 SE, one bar per method
    running_qhat.png        running Q_hat(1:k) convergence per method
    ntarget_vs_cv.png       samples required to reach target CV (log-log)
    sampled_births.png      (XY, RZ) sampled-birth distribution per method
    weight_hists.png        IS weight distributions (UNIF_S_IS + BACKWARD_IS)
    wall_hit_locations.png  (XY, RZ) where the wall-hit samples hit
    bootstrap.png           bootstrap distribution of Q_hat per method
    zscore_matrix.png       pairwise |z|-score between methods
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_DIRS = {
    "FWD":         "forward_mc",
    "UNIF_S_IS":   "uniform_s_is",
    "BACKWARD_IS": "backward_informed_is",
}
METHOD_COLORS = {
    "FWD":         "C0",
    "UNIF_S_IS":   "C2",
    "BACKWARD_IS": "C3",
    "FWD_GOLD":    "black",
}


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Build comparison plots across the three MC methods.")
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Parent directory containing forward_mc/, "
                        "uniform_s_is/, backward_informed_is/ subdirs.")
    p.add_argument("--gold_dir", type=Path, default=None,
                   help="Optional directory containing a combined gold FWD "
                        "estimate (written by combine_forward_mc.py).")
    p.add_argument("--out_subdir", type=str, default="comparison",
                   help="Subdirectory of --run_dir for plot output.")
    p.add_argument("--bootstrap_n", type=int, default=1000,
                   help="Number of bootstrap resamples per method.")
    p.add_argument("--thin", type=int, default=100,
                   help="Thinning factor for the running-Q_hat plot "
                        "(plot one in every `thin` samples).")
    return p.parse_args()


# ── I/O helpers ─────────────────────────────────────────────────────────────

def _auto_cast(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        pass
    return v


def read_metrics_csv(path):
    if not path.exists():
        return {}
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [{k: _auto_cast(v) for k, v in row.items()} for row in reader]
    return rows[0] if rows else {}


def _maybe_load(path):
    p = Path(path)
    return np.load(p) if p.exists() else None


def load_method(run_dir, name):
    """Load everything we need for one method.  Missing files are OK —
    downstream plots that need them will skip themselves cleanly."""
    d = run_dir / METHOD_DIRS[name]
    if not d.exists():
        return None
    data = {
        "name":            name,
        "dir":             d,
        "metrics":         read_metrics_csv(d / "metrics_summary.csv"),
        "forward_results": _maybe_load(d / "forward_results.npy"),
        "sample_idx":      _maybe_load(d / "sample_idx.npy"),
        "pool_R":          _maybe_load(d / "pool_R.npy"),
        "pool_phi":        _maybe_load(d / "pool_phi.npy"),
        "pool_Z":          _maybe_load(d / "pool_Z.npy"),
        "pool_s":          _maybe_load(d / "pool_s.npy"),
        "is_weights":      _maybe_load(d / "is_weights.npy"),
    }
    return data


def load_gold(gold_dir):
    if gold_dir is None or not gold_dir.exists():
        return None
    return {
        "name":    "FWD_GOLD",
        "dir":     gold_dir,
        "metrics": read_metrics_csv(gold_dir / "metrics_combined.csv"),
        "Y":       _maybe_load(gold_dir / "Y_all.npy"),
    }


def per_sample_Y(data):
    """Per-sample estimator contribution.  FWD: Y=A.  IS: Y=A*w."""
    fr = data.get("forward_results")
    if fr is None:
        return None
    stop = fr[:, 6].astype(int)
    A = (stop == 1).astype(np.float64)
    w = data.get("is_weights")
    return A * w if w is not None else A


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_qhat_bars(methods, out_path, gold=None):
    names, qs, ses = [], [], []
    for m in methods:
        q = m["metrics"].get("Q_hat")
        s = m["metrics"].get("standard_error")
        if q is None or s is None:
            continue
        names.append(m["name"])
        qs.append(q)
        ses.append(s)
    if not names:
        print("  qhat_bars: no data"); return

    xs = np.arange(len(names))
    colors = [METHOD_COLORS.get(n, "grey") for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(xs, qs, yerr=[2 * s for s in ses], color=colors, capsize=6,
           edgecolor="black")
    for x, q, s in zip(xs, qs, ses):
        ax.text(x, q + 2 * s, f"{q:.3e}±{s:.1e}",
                ha="center", va="bottom", fontsize=9)

    if gold and gold["metrics"].get("Q_hat") is not None:
        gq = gold["metrics"]["Q_hat"]
        gs = gold["metrics"].get("standard_error", 0.0) or 0.0
        ax.axhline(gq, color="black", linestyle="--", alpha=0.7,
                   label=f"gold FWD = {gq:.3e} ± {gs:.1e}")
        ax.axhspan(gq - 2 * gs, gq + 2 * gs, color="black", alpha=0.08)
        ax.legend()

    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("Q_hat")
    ax.set_title("Wall-hit probability estimates  (error bars = 2 × SE)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_running_qhat(methods, out_path, thin=100, gold=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for m in methods:
        Y = per_sample_Y(m)
        if Y is None or len(Y) == 0:
            continue
        n = len(Y)
        k = np.arange(1, n + 1)
        csum = np.cumsum(Y)
        csq = np.cumsum(Y * Y)
        mean = csum / k
        # unbiased sample variance of the first k Ys
        var = np.where(k > 1, (csq / k - mean ** 2) * k / np.maximum(k - 1, 1), 0.0)
        se = np.sqrt(np.maximum(var / k, 0.0))
        step = max(thin, 1)
        idx = np.arange(step - 1, n, step)
        color = METHOD_COLORS.get(m["name"], "grey")
        ax.plot(k[idx], mean[idx], label=m["name"], color=color, lw=1.5)
        ax.fill_between(k[idx], mean[idx] - 2 * se[idx], mean[idx] + 2 * se[idx],
                        alpha=0.15, color=color)
        plotted = True

    if gold and gold["metrics"].get("Q_hat") is not None:
        gq = gold["metrics"]["Q_hat"]
        ax.axhline(gq, color="black", linestyle="--", alpha=0.6,
                   label="gold FWD Q_hat")

    if not plotted:
        print("  running_qhat: no data"); plt.close(fig); return

    ax.set_xlabel("sample index k")
    ax.set_ylabel("running Q_hat(1:k)  ± 2 SE")
    ax.set_title("Convergence of the running estimator")
    ax.set_xscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_ntarget_vs_cv(methods, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    cvs = [0.10, 0.05, 0.02]
    keys = [f"N_target_cv_{int(100 * c):02d}pct" for c in cvs]
    plotted = False
    for m in methods:
        ns = [m["metrics"].get(k) for k in keys]
        if any(n is None for n in ns):
            continue
        color = METHOD_COLORS.get(m["name"], "grey")
        ax.plot(cvs, ns, "o-", label=m["name"], color=color, lw=1.5)
        plotted = True
    if not plotted:
        print("  ntarget_vs_cv: no data"); plt.close(fig); return
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("target cv_estimator")
    ax.set_ylabel("N_samples required")
    ax.set_title("Samples required to reach a target CV")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_sampled_births(methods, out_path):
    valid = [m for m in methods
             if m["sample_idx"] is not None and m["pool_R"] is not None]
    if not valid:
        print("  sampled_births: no data"); return
    ncol = len(valid)
    fig, axs = plt.subplots(2, ncol, figsize=(4.5 * ncol, 9),
                            squeeze=False)

    for col, m in enumerate(valid):
        sidx = m["sample_idx"]
        R_pool   = m["pool_R"]
        phi_pool = m["pool_phi"]
        Z_pool   = m["pool_Z"]

        R = R_pool[sidx]
        phi = phi_pool[sidx]
        Z = Z_pool[sidx]
        X = R * np.cos(phi); Y = R * np.sin(phi)
        color = METHOD_COLORS.get(m["name"], "grey")

        # XY
        ax_xy = axs[0, col]
        ax_xy.scatter(R_pool * np.cos(phi_pool), R_pool * np.sin(phi_pool),
                      s=0.5, alpha=0.03, color="grey")
        ax_xy.scatter(X, Y, s=1, alpha=0.25, color=color)
        ax_xy.set_aspect("equal")
        ax_xy.set_xlabel("X [m]"); ax_xy.set_ylabel("Y [m]")
        ax_xy.set_title(f"{m['name']} — XY  (N={len(sidx)})")

        # RZ
        ax_rz = axs[1, col]
        ax_rz.scatter(R_pool, Z_pool, s=0.5, alpha=0.03, color="grey")
        ax_rz.scatter(R, Z, s=1, alpha=0.25, color=color)
        ax_rz.set_aspect("equal")
        ax_rz.set_xlabel("R [m]"); ax_rz.set_ylabel("Z [m]")
        ax_rz.set_title(f"{m['name']} — RZ")

    fig.suptitle("Sampled birth positions (pool in grey)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_weight_hists(methods, out_path):
    is_methods = [m for m in methods if m.get("is_weights") is not None]
    if not is_methods:
        print("  weight_hists: no IS methods"); return

    all_w = np.concatenate([m["is_weights"] for m in is_methods])
    w_pos = all_w[(all_w > 0) & np.isfinite(all_w)]
    if w_pos.size == 0:
        print("  weight_hists: no positive weights"); return

    bins = np.logspace(np.log10(w_pos.min()), np.log10(w_pos.max() + 1e-300), 50)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in is_methods:
        w = m["is_weights"]
        wp = w[(w > 0) & np.isfinite(w)]
        ess = (float(wp.sum() ** 2 / np.sum(wp ** 2)) if wp.sum() > 0 else 0.0)
        ax.hist(wp, bins=bins, alpha=0.6,
                color=METHOD_COLORS.get(m["name"], "grey"),
                label=f"{m['name']} (N={len(w)}, ESS={ess:.0f})",
                edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("IS weight w = (1/N_pool) / q")
    ax.set_ylabel("count")
    ax.set_title("IS weight distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_wall_hit_locations(methods, out_path):
    valid = [m for m in methods if m["forward_results"] is not None]
    if not valid:
        print("  wall_hit_locations: no data"); return
    ncol = len(valid)
    fig, axs = plt.subplots(2, ncol, figsize=(4.5 * ncol, 9),
                            squeeze=False)
    for col, m in enumerate(valid):
        fr = m["forward_results"]
        stop = fr[:, 6].astype(int)
        hit = stop == 1
        color = METHOD_COLORS.get(m["name"], "grey")
        if hit.sum() == 0:
            for row in (0, 1):
                axs[row, col].text(0.5, 0.5, "no wall hits", ha="center",
                                   transform=axs[row, col].transAxes)
                axs[row, col].set_title(f"{m['name']}")
            continue
        X = fr[hit, 1]; Y = fr[hit, 2]; Z = fr[hit, 3]
        R = np.sqrt(X * X + Y * Y)
        axs[0, col].scatter(X, Y, s=2, alpha=0.4, color=color)
        axs[0, col].set_aspect("equal")
        axs[0, col].set_xlabel("X [m]"); axs[0, col].set_ylabel("Y [m]")
        axs[0, col].set_title(f"{m['name']} — wall hits XY  (n={int(hit.sum())})")
        axs[1, col].scatter(R, Z, s=2, alpha=0.4, color=color)
        axs[1, col].set_aspect("equal")
        axs[1, col].set_xlabel("R [m]"); axs[1, col].set_ylabel("Z [m]")
        axs[1, col].set_title(f"{m['name']} — wall hits RZ")

    fig.suptitle("Where the wall-hit samples landed", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_bootstrap(methods, out_path, n_bootstrap=1000, gold=None):
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for m in methods:
        Y = per_sample_Y(m)
        if Y is None or len(Y) == 0:
            continue
        n = len(Y)
        boots = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            boots[b] = Y[rng.integers(0, n, size=n)].mean()
        color = METHOD_COLORS.get(m["name"], "grey")
        ax.hist(boots, bins=50, alpha=0.55, color=color,
                label=f"{m['name']} (mean={boots.mean():.3e})",
                density=True)
        plotted = True

    if gold and gold.get("Y") is not None:
        Y = gold["Y"]
        n = len(Y)
        boots = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            boots[b] = Y[rng.integers(0, n, size=n)].mean()
        ax.hist(boots, bins=50, alpha=0.55, color="black",
                label=f"FWD_GOLD (mean={boots.mean():.3e})", density=True)
        plotted = True

    if not plotted:
        print("  bootstrap: no data"); plt.close(fig); return

    ax.set_xlabel("bootstrap Q_hat")
    ax.set_ylabel("density")
    ax.set_title(f"Bootstrap distribution of Q_hat  (B={n_bootstrap})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_zscore_matrix(methods, out_path, gold=None):
    labelled = list(methods) + ([gold] if gold is not None else [])
    names = [m["name"] for m in labelled
             if m["metrics"].get("Q_hat") is not None
             and m["metrics"].get("standard_error") is not None]
    if len(names) < 2:
        print("  zscore_matrix: need >=2 methods"); return

    valid = [m for m in labelled if m["name"] in names]
    n = len(valid)
    Z = np.full((n, n), np.nan)
    for i, a in enumerate(valid):
        for j, b in enumerate(valid):
            if i == j:
                continue
            qa = a["metrics"]["Q_hat"]
            qb = b["metrics"]["Q_hat"]
            sa = a["metrics"]["standard_error"]
            sb = b["metrics"]["standard_error"]
            denom = np.sqrt(sa * sa + sb * sb)
            Z[i, j] = abs(qa - qb) / denom if denom > 0 else np.nan

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(Z, cmap="RdYlGn_r", vmin=0, vmax=4)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="black")
            elif np.isnan(Z[i, j]):
                ax.text(j, i, "?", ha="center", va="center", color="black")
            else:
                ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                        color="white" if Z[i, j] > 2 else "black")
    fig.colorbar(im, ax=ax, label="|z|")
    ax.set_title("Pairwise |z|-score between methods\n"
                 "(|z| > 3 = genuine disagreement)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


# ── Driver ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"ERROR: run_dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = []
    for name in METHOD_DIRS:
        m = load_method(run_dir, name)
        if m is None:
            print(f"  skipping {name}: dir not found")
            continue
        q = m["metrics"].get("Q_hat", "?")
        s = m["metrics"].get("standard_error", "?")
        n = m["metrics"].get("N", "?")
        print(f"  loaded {name}: N={n}, Q_hat={q}, SE={s}")
        methods.append(m)

    if not methods:
        print("No methods loaded. Exiting.")
        sys.exit(1)

    gold = load_gold(args.gold_dir)
    if gold is not None:
        print(f"  loaded FWD_GOLD from {args.gold_dir}: "
              f"Q_hat={gold['metrics'].get('Q_hat')}, "
              f"SE={gold['metrics'].get('standard_error')}")

    print(f"\nWriting plots to: {out_dir}")
    plot_qhat_bars(         methods, out_dir / "qhat_bars.png", gold=gold)
    plot_running_qhat(      methods, out_dir / "running_qhat.png",
                            thin=args.thin, gold=gold)
    plot_ntarget_vs_cv(     methods, out_dir / "ntarget_vs_cv.png")
    plot_sampled_births(    methods, out_dir / "sampled_births.png")
    plot_weight_hists(      methods, out_dir / "weight_hists.png")
    plot_wall_hit_locations(methods, out_dir / "wall_hit_locations.png")
    plot_bootstrap(         methods, out_dir / "bootstrap.png",
                            n_bootstrap=args.bootstrap_n, gold=gold)
    plot_zscore_matrix(     methods, out_dir / "zscore_matrix.png", gold=gold)
    print("Done.")


if __name__ == "__main__":
    main()
