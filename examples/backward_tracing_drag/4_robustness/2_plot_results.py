"""
Plotting script for the coil-perturbation robustness study.

Reads all output files written by 1_trace_perturbed.py from the output/
directory and produces four figures:

  loss_fraction_ensemble.png  — loss fraction per ensemble member;
                                 baseline highlighted in red.
  loss_rate_distribution.png  — histogram of loss fractions across the
                                 perturbed ensemble.
  loss_vs_s.png               — birth Boozer-s distribution of lost
                                 particles for every ensemble member
                                 (grey) and baseline (red), plus ensemble
                                 mean (blue).
  bn_stats.png                — mean and max |B·n|/|B| per run, showing
                                 how each perturbation shifts the coil
                                 field away from the target LCFS.

Run after collecting all ensemble results:
  python 2_plot_results.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).parent.resolve()
OUT_DIR  = THIS_DIR / "output"

# ── Load loss summaries ────────────────────────────────────────────────────────
summary_files = sorted(OUT_DIR.glob("loss_summary_*.npy"))
if not summary_files:
    raise FileNotFoundError(
        f"No loss_summary_*.npy files found in {OUT_DIR}\n"
        "Run 1_trace_perturbed.py first."
    )

summaries = np.array([np.load(f) for f in summary_files])
# columns: pert_id, nparticles, n_lost, loss_fraction, sigma
pert_ids       = summaries[:, 0].astype(int)
n_particles    = summaries[:, 1].astype(int)
n_lost         = summaries[:, 2].astype(int)
loss_fractions = summaries[:, 3]
sigmas         = summaries[:, 4]

baseline_mask  = pert_ids == 0
perturbed_mask = ~baseline_mask

baseline_lf   = loss_fractions[baseline_mask][0] if baseline_mask.any() else None
perturbed_lfs = loss_fractions[perturbed_mask]
perturbed_ids = pert_ids[perturbed_mask]

print(f"Loaded {len(summaries)} runs  "
      f"({perturbed_mask.sum()} perturbed + {baseline_mask.sum()} baseline)")
if baseline_lf is not None:
    print(f"  Baseline loss fraction : {baseline_lf * 100:.3f}%")
if len(perturbed_lfs) > 0:
    sigma_val = sigmas[perturbed_mask][0]
    print(f"  Perturbation sigma     : {sigma_val * 1e3:.1f} mm")
    print(f"  Perturbed loss fracs   : "
          f"mean={perturbed_lfs.mean()*100:.3f}%  "
          f"std={perturbed_lfs.std()*100:.3f}%  "
          f"min={perturbed_lfs.min()*100:.3f}%  "
          f"max={perturbed_lfs.max()*100:.3f}%")


# ── Figure 1: loss fraction per ensemble member ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))

ax.bar(perturbed_ids, perturbed_lfs * 100,
       color="steelblue", label="Perturbed runs", zorder=2)

if baseline_lf is not None:
    ax.axhline(baseline_lf * 100, color="crimson", lw=2, ls="--",
               label=f"Baseline ({baseline_lf * 100:.3f}%)")

ax.set_xlabel("Perturbation ID (RNG seed)")
ax.set_ylabel("Loss fraction [%]")
npart = int(n_particles[0])
sigma_mm = sigmas[perturbed_mask][0] * 1e3 if perturbed_mask.any() else 0
ax.set_title(
    f"Alpha-particle loss vs coil perturbation ensemble\n"
    f"({npart:,} particles per run,  σ = {sigma_mm:.1f} mm)"
)
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "loss_fraction_ensemble.png", dpi=150)
plt.close(fig)
print("Saved  loss_fraction_ensemble.png")


# ── Figure 2: histogram of loss fractions ─────────────────────────────────────
if len(perturbed_lfs) > 1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(perturbed_lfs * 100, bins="auto",
            color="steelblue", edgecolor="white", label="Perturbed ensemble")
    if baseline_lf is not None:
        ax.axvline(baseline_lf * 100, color="crimson", lw=2, ls="--",
                   label=f"Baseline ({baseline_lf * 100:.3f}%)")
    ax.set_xlabel("Loss fraction [%]")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of loss fractions across ensemble")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_rate_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved  loss_rate_distribution.png")


# ── Figure 3: birth Boozer-s of lost particles ─────────────────────────────────
# The s-coordinate tells you *where* in the plasma lost particles were born.
# Outer-flux-surface losses (large s) are expected; inner losses (small s)
# indicate that a perturbation is opening new loss channels deep in the plasma.

s_bins    = np.linspace(0, 1, 21)
s_centers = 0.5 * (s_bins[:-1] + s_bins[1:])

fig, ax = plt.subplots(figsize=(8, 5))

# Grey lines: individual perturbed runs
s_hists = []
for pid in perturbed_ids:
    fpath = OUT_DIR / f"lost_initial_boozer_{pid:04d}.npy"
    if not fpath.exists():
        continue
    lost_b = np.load(fpath)
    if len(lost_b) == 0:
        s_hists.append(np.zeros(len(s_centers)))
        continue
    counts, _ = np.histogram(lost_b[:, 0], bins=s_bins)
    s_hists.append(counts)
    ax.plot(s_centers, counts, color="grey", alpha=0.35, lw=1)

# Blue line: ensemble mean
if s_hists:
    mean_hist = np.mean(s_hists, axis=0)
    ax.plot(s_centers, mean_hist, color="steelblue", lw=2,
            label="Perturbed ensemble mean")

# Red line: baseline
if baseline_lf is not None:
    fpath = OUT_DIR / "lost_initial_boozer_0000.npy"
    if fpath.exists():
        lost_b = np.load(fpath)
        if len(lost_b) > 0:
            counts, _ = np.histogram(lost_b[:, 0], bins=s_bins)
            ax.plot(s_centers, counts, color="crimson", lw=2.5, label="Baseline")

ax.set_xlabel("Boozer s (particle birth location)")
ax.set_ylabel("Number of lost particles")
ax.set_title("Birth location of lost particles vs coil perturbation")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "loss_vs_s.png", dpi=150)
plt.close(fig)
print("Saved  loss_vs_s.png")


# ── Figure 4: B·n quality per run ─────────────────────────────────────────────
# Mean and max |B·n|/|B| quantify how far each perturbation shifts the
# magnetic field on the LCFS from the ideal vacuum equilibrium.  A larger
# B·n usually means more open field lines → more losses.

bn_files = sorted(OUT_DIR.glob("bn_stats_*.npy"))
if bn_files:
    bn_pids  = []
    bn_means = []
    bn_maxs  = []
    for f in bn_files:
        pid = int(f.stem.split("_")[-1])
        arr = np.load(f)
        bn_pids.append(pid)
        bn_means.append(arr[0])
        bn_maxs.append(arr[1])
    bn_pids  = np.array(bn_pids)
    bn_means = np.array(bn_means)
    bn_maxs  = np.array(bn_maxs)

    pert_m = bn_pids > 0
    base_m = bn_pids == 0

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for arr, label, ax in [(bn_means, "mean |B·n|/|B|", axes[0]),
                            (bn_maxs,  "max |B·n|/|B|",  axes[1])]:
        ax.bar(bn_pids[pert_m], arr[pert_m], color="steelblue", label="Perturbed")
        if base_m.any():
            ax.axhline(arr[base_m][0], color="crimson", lw=2, ls="--",
                       label=f"Baseline ({arr[base_m][0]:.2e})")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    axes[1].set_xlabel("Perturbation ID (RNG seed)")
    fig.suptitle("Coil-to-surface flux quality vs perturbation\n"
                 "(larger value = coils deviate more from target LCFS)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bn_stats.png", dpi=150)
    plt.close(fig)
    print("Saved  bn_stats.png")

print(f"\nAll figures written to {OUT_DIR}/")
