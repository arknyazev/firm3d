#!/usr/bin/env python3
"""Post-process a backward-informed slowdown-time sweep.

Expected layout:

    <run_dir>/
        tau_x1/
        tau_x2/
        tau_x5/
        logs/

The script discovers tau_x* directories automatically.

Outputs under <run_dir>/<out_subdir>/:
    tau_sweep_summary.csv
    qhat_vs_tau.png
    efficiency_vs_tau.png
    backward_health_vs_tau.png
    proposal_health_vs_tau.png
    weight_hists.png
    backward_score_hist_overlay.png
    sampled_s_hist_overlay.png
    loss_time_hist_overlay.png
    running_qhat.png
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TAU_RE = re.compile(r"^tau_x(?P<factor>[0-9]+(?:\.[0-9]+)?)$")


def parse_args():
    p = argparse.ArgumentParser(
        description="Build comparison plots across tau_x* slowdown-sweep runs."
    )
    p.add_argument("--run_dir", type=Path, required=True,
                   help="Parent directory containing tau_x* subdirectories.")
    p.add_argument("--out_subdir", type=str, default="tau_sweep_comparison",
                   help="Subdirectory of --run_dir for plot output.")
    p.add_argument("--thin", type=int, default=100,
                   help="Thinning factor for running-Q_hat plot.")
    p.add_argument("--hist_bins", type=int, default=50,
                   help="Default bin count for overlay histograms.")
    return p.parse_args()


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


def read_run_config(path):
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            if " = " not in line:
                continue
            key, value = line.rstrip("\n").split(" = ", 1)
            out[key] = _auto_cast(value)
    return out


def maybe_load(path):
    p = Path(path)
    return np.load(p) if p.exists() else None


def discover_runs(run_dir):
    complete = []
    incomplete = []
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        m = TAU_RE.match(d.name)
        if not m:
            continue
        factor = float(m.group("factor"))
        metrics_path = d / "metrics_summary.csv"
        if not metrics_path.exists():
            incomplete.append((factor, d))
            continue
        metrics = read_metrics_csv(metrics_path)
        if not metrics:
            incomplete.append((factor, d))
            continue
        complete.append({
            "factor": factor,
            "label": d.name,
            "dir": d,
            "metrics": metrics,
            "config": read_run_config(d / "run_config.txt"),
        })
    complete.sort(key=lambda r: (r["factor"], r["label"]))
    incomplete.sort(key=lambda r: (r[0], r[1].name))
    return complete, incomplete


def colors_for(n):
    cmap = plt.get_cmap("viridis")
    if n <= 1:
        return [cmap(0.45)]
    return [cmap(i / (n - 1)) for i in range(n)]


def factors_and_labels(runs):
    x = np.array([r["factor"] for r in runs], dtype=float)
    labels = [r["label"] for r in runs]
    return x, labels


def metric_array(runs, key, default=np.nan):
    vals = []
    for r in runs:
        v = r["metrics"].get(key, default)
        vals.append(float(v) if v is not None else default)
    return np.asarray(vals, dtype=float)


def maybe_log_x(ax, x):
    finite = x[np.isfinite(x) & (x > 0)]
    if finite.size >= 2 and finite.max() / finite.min() > 1.5:
        ax.set_xscale("log")


def write_summary_csv(runs, out_path):
    keys = [
        "label", "factor", "ne0", "Te0_ev", "coulomb_log",
        "backward_tmax_factor", "tmax_backward_s",
        "N", "Q_hat", "standard_error", "cv_estimator",
        "effective_sample_size", "ess_fraction", "N_wall_hits",
        "n_pilot", "n_backward_success", "backward_success_fraction",
        "n_backward_success_valid", "valid_success_fraction",
        "bwd_non_empty_bins", "s_score_nbins", "non_empty_bin_fraction",
        "pool_markers_nonzero_score", "frac_pool_nonzero_score",
        "w_min", "w_max", "w_mean", "w_std",
    ]
    rows = []
    for r in runs:
        m = r["metrics"]
        c = r["config"]
        n = m.get("N")
        ess = m.get("effective_sample_size")
        n_pilot = m.get("n_pilot")
        n_succ = m.get("n_backward_success")
        n_valid = m.get("n_backward_success_valid")
        n_bins = m.get("s_score_nbins")
        non_empty = m.get("bwd_non_empty_bins")
        rows.append({
            "label": r["label"],
            "factor": r["factor"],
            "ne0": c.get("ne0"),
            "Te0_ev": c.get("Te0_ev"),
            "coulomb_log": c.get("coulomb_log"),
            "backward_tmax_factor": m.get("backward_tmax_factor"),
            "tmax_backward_s": m.get("tmax_backward_s"),
            "N": n,
            "Q_hat": m.get("Q_hat"),
            "standard_error": m.get("standard_error"),
            "cv_estimator": m.get("cv_estimator"),
            "effective_sample_size": ess,
            "ess_fraction": (
                float(ess) / float(n)
                if ess is not None and n not in (None, 0) else np.nan
            ),
            "N_wall_hits": m.get("N_wall_hits"),
            "n_pilot": n_pilot,
            "n_backward_success": n_succ,
            "backward_success_fraction": (
                float(n_succ) / float(n_pilot)
                if n_succ is not None and n_pilot not in (None, 0) else np.nan
            ),
            "n_backward_success_valid": n_valid,
            "valid_success_fraction": (
                float(n_valid) / float(n_succ)
                if n_valid is not None and n_succ not in (None, 0) else np.nan
            ),
            "bwd_non_empty_bins": non_empty,
            "s_score_nbins": n_bins,
            "non_empty_bin_fraction": (
                float(non_empty) / float(n_bins)
                if non_empty is not None and n_bins not in (None, 0) else np.nan
            ),
            "pool_markers_nonzero_score": m.get("pool_markers_nonzero_score"),
            "frac_pool_nonzero_score": m.get("frac_pool_nonzero_score"),
            "w_min": m.get("w_min"),
            "w_max": m.get("w_max"),
            "w_mean": m.get("w_mean"),
            "w_std": m.get("w_std"),
        })
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def finish_x_axis(ax, x, labels):
    maybe_log_x(ax, x)
    ax.set_xlabel("slowdown time scale factor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, alpha=0.3)


def plot_qhat_vs_tau(runs, out_path):
    x, labels = factors_and_labels(runs)
    q = metric_array(runs, "Q_hat")
    se = metric_array(runs, "standard_error")
    ok = np.isfinite(q) & np.isfinite(se)
    if not np.any(ok):
        print("  qhat_vs_tau: no Q_hat/SE data")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x[ok], q[ok], yerr=2.0 * se[ok], fmt="o-", capsize=5,
                color="C0")
    ax.set_ylabel("Q_hat +/- 2 SE")
    ax.set_title("Backward-informed estimate vs slowdown scale")
    finish_x_axis(ax, x, labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_efficiency_vs_tau(runs, out_path):
    x, labels = factors_and_labels(runs)
    ess = metric_array(runs, "effective_sample_size")
    n = metric_array(runs, "N")
    cv = metric_array(runs, "cv_estimator")
    ess_frac = np.where(n > 0, ess / n, np.nan)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(x, ess_frac, "o-", color="C2", label="ESS / N")
    ax1.set_ylabel("ESS / N")
    ax1.set_ylim(bottom=0.0)
    ax2 = ax1.twinx()
    ax2.plot(x, cv, "s--", color="C3", label="cv_estimator")
    ax2.set_ylabel("cv_estimator")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("Estimator efficiency vs slowdown scale")
    finish_x_axis(ax1, x, labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_backward_health_vs_tau(runs, out_path):
    x, labels = factors_and_labels(runs)
    n_pilot = metric_array(runs, "n_pilot")
    n_succ = metric_array(runs, "n_backward_success")
    n_valid = metric_array(runs, "n_backward_success_valid")
    success_frac = np.where(n_pilot > 0, n_succ / n_pilot, np.nan)
    valid_frac = np.where(n_succ > 0, n_valid / n_succ, np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, success_frac, "o-", label="backward success / pilot")
    ax.plot(x, valid_frac, "s-", label="valid Boozer / success")
    ax.set_ylabel("fraction")
    ax.set_ylim(bottom=0.0)
    ax.set_title("Backward pilot health vs slowdown scale")
    ax.legend()
    finish_x_axis(ax, x, labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_proposal_health_vs_tau(runs, out_path):
    x, labels = factors_and_labels(runs)
    non_empty = metric_array(runs, "bwd_non_empty_bins")
    n_bins = metric_array(runs, "s_score_nbins")
    non_empty_frac = np.where(n_bins > 0, non_empty / n_bins, np.nan)
    pool_frac = metric_array(runs, "frac_pool_nonzero_score")
    w_max = metric_array(runs, "w_max")

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(x, non_empty_frac, "o-", color="C0",
             label="non-empty score bins / bins")
    ax1.plot(x, pool_frac, "s-", color="C1",
             label="pool markers with nonzero score")
    ax1.set_ylabel("fraction")
    ax1.set_ylim(bottom=0.0)

    ax2 = ax1.twinx()
    ax2.plot(x, w_max, "^--", color="C4", label="w_max")
    ax2.set_ylabel("max IS weight")
    if np.all(np.isfinite(w_max)) and np.nanmax(w_max) > 0:
        ax2.set_yscale("log")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("Proposal support and weight scale")
    finish_x_axis(ax1, x, labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_weight_hists(runs, out_path):
    weights = []
    for r in runs:
        w = maybe_load(r["dir"] / "is_weights.npy")
        if w is None:
            continue
        w = w[np.isfinite(w) & (w > 0)]
        if w.size:
            weights.append((r, w))
    if not weights:
        print("  weight_hists: no positive weights")
        return

    all_w = np.concatenate([w for _, w in weights])
    bins = np.logspace(np.log10(all_w.min()), np.log10(all_w.max()), 50)
    colors = colors_for(len(weights))

    fig, ax = plt.subplots(figsize=(8, 5))
    for color, (r, w) in zip(colors, weights):
        ess = r["metrics"].get("effective_sample_size")
        n = r["metrics"].get("N")
        label = r["label"]
        if ess is not None and n not in (None, 0):
            label += f" (ESS/N={float(ess) / float(n):.2f})"
        ax.hist(w, bins=bins, density=True, histtype="step", lw=1.8,
                color=color, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("IS weight w")
    ax.set_ylabel("density")
    ax.set_title("IS weight distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_backward_score_hist_overlay(runs, out_path):
    curves = []
    for r in runs:
        edges = maybe_load(r["dir"] / "s_edges.npy")
        hist = maybe_load(r["dir"] / "backward_s_hist.npy")
        if edges is None or hist is None or hist.sum() <= 0:
            continue
        centers = 0.5 * (edges[:-1] + edges[1:])
        density = hist.astype(float) / float(hist.sum())
        curves.append((r, centers, density))
    if not curves:
        print("  backward_score_hist_overlay: no histograms")
        return

    score_coord = curves[0][0]["metrics"].get("score_coordinate", "s")
    colors = colors_for(len(curves))
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, (r, centers, density) in zip(colors, curves):
        ax.plot(centers, density, "o-", ms=3, lw=1.5,
                color=color, label=r["label"])
    ax.set_xlabel(f"backward score coordinate ({score_coord})")
    ax.set_ylabel("fraction of backward successes")
    ax.set_title("Backward-success score distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_sampled_s_hist_overlay(runs, out_path, hist_bins):
    curves = []
    for r in runs:
        pool_s = maybe_load(r["dir"] / "pool_s.npy")
        sample_idx = maybe_load(r["dir"] / "sample_idx.npy")
        if pool_s is None or sample_idx is None or sample_idx.size == 0:
            continue
        sampled_s = pool_s[sample_idx]
        sampled_s = sampled_s[np.isfinite(sampled_s)]
        if sampled_s.size:
            curves.append((r, sampled_s))
    if not curves:
        print("  sampled_s_hist_overlay: no sampled s data")
        return

    bins = np.linspace(0.0, 1.0, hist_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    colors = colors_for(len(curves))
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, (r, sampled_s) in zip(colors, curves):
        hist, _ = np.histogram(np.clip(sampled_s, 0.0, 1.0), bins=bins,
                               density=True)
        ax.plot(centers, hist, lw=1.7, color=color, label=r["label"])
    ax.set_xlabel("sampled birth s")
    ax.set_ylabel("density")
    ax.set_title("Sampled birth distribution in Boozer s")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def plot_loss_time_hist_overlay(runs, out_path):
    curves = []
    for r in runs:
        fr = maybe_load(r["dir"] / "forward_results.npy")
        if fr is None or fr.size == 0:
            continue
        stop = fr[:, 6].astype(int)
        t_loss = fr[stop == 1, 0]
        t_loss = t_loss[np.isfinite(t_loss) & (t_loss > 0)]
        if t_loss.size:
            curves.append((r, t_loss))
    if not curves:
        print("  loss_time_hist_overlay: no wall-hit times")
        return

    all_t = np.concatenate([t for _, t in curves])
    bins = np.logspace(np.log10(all_t.min()), np.log10(all_t.max()), 45)
    colors = colors_for(len(curves))
    fig, ax = plt.subplots(figsize=(8, 5))
    for color, (r, t_loss) in zip(colors, curves):
        ax.hist(t_loss, bins=bins, density=True, histtype="step", lw=1.8,
                color=color, label=f"{r['label']} (n={len(t_loss)})")
    ax.set_xscale("log")
    ax.set_xlabel("wall-hit time t_final [s]")
    ax.set_ylabel("density")
    ax.set_title("Forward wall-hit loss-time distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def per_sample_y(run):
    fr = maybe_load(run["dir"] / "forward_results.npy")
    w = maybe_load(run["dir"] / "is_weights.npy")
    if fr is None or w is None:
        return None
    stop = fr[:, 6].astype(int)
    a = (stop == 1).astype(np.float64)
    return a * w


def plot_running_qhat(runs, out_path, thin):
    colors = colors_for(len(runs))
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for color, r in zip(colors, runs):
        y = per_sample_y(r)
        if y is None or y.size == 0:
            continue
        n = len(y)
        k = np.arange(1, n + 1)
        csum = np.cumsum(y)
        csq = np.cumsum(y * y)
        mean = csum / k
        var = np.where(k > 1,
                       (csq / k - mean ** 2) * k / np.maximum(k - 1, 1),
                       0.0)
        se = np.sqrt(np.maximum(var / k, 0.0))
        step = max(int(thin), 1)
        idx = np.arange(step - 1, n, step)
        ax.plot(k[idx], mean[idx], lw=1.5, color=color, label=r["label"])
        ax.fill_between(k[idx], mean[idx] - 2 * se[idx],
                        mean[idx] + 2 * se[idx], color=color, alpha=0.12)
        plotted = True
    if not plotted:
        print("  running_qhat: no per-sample data")
        plt.close(fig)
        return
    ax.set_xscale("log")
    ax.set_xlabel("sample index k")
    ax.set_ylabel("running Q_hat(1:k) +/- 2 SE")
    ax.set_title("Running estimator convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"ERROR: run_dir does not exist: {run_dir}", file=sys.stderr)
        sys.exit(1)

    runs, incomplete = discover_runs(run_dir)
    for factor, d in incomplete:
        print(f"  skipping incomplete {d.name}: no metrics_summary.csv")
    if not runs:
        print("No completed tau_x* runs found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("Loaded completed tau runs:")
    for r in runs:
        m = r["metrics"]
        print(f"  {r['label']}: Q_hat={m.get('Q_hat', '?')}, "
              f"SE={m.get('standard_error', '?')}, "
              f"ESS={m.get('effective_sample_size', '?')}")

    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting plots to: {out_dir}")

    write_summary_csv(runs, out_dir / "tau_sweep_summary.csv")
    print("  wrote tau_sweep_summary.csv")
    plot_qhat_vs_tau(runs, out_dir / "qhat_vs_tau.png")
    plot_efficiency_vs_tau(runs, out_dir / "efficiency_vs_tau.png")
    plot_backward_health_vs_tau(runs, out_dir / "backward_health_vs_tau.png")
    plot_proposal_health_vs_tau(runs, out_dir / "proposal_health_vs_tau.png")
    plot_weight_hists(runs, out_dir / "weight_hists.png")
    plot_backward_score_hist_overlay(
        runs, out_dir / "backward_score_hist_overlay.png"
    )
    plot_sampled_s_hist_overlay(
        runs, out_dir / "sampled_s_hist_overlay.png", args.hist_bins
    )
    plot_loss_time_hist_overlay(runs, out_dir / "loss_time_hist_overlay.png")
    plot_running_qhat(runs, out_dir / "running_qhat.png", args.thin)
    print("Done.")


if __name__ == "__main__":
    main()