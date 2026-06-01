#!/usr/bin/env python3
"""
Plot histograms of the initial Boozer s coordinate for lost vs all particles.

Reads from output/ (produced by tracing_gpu.py) and saves:
  output/loss_distribution_s.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

out_dir = Path(__file__).parent / "output"

required = ["initial_boozer.npy", "lost_initial_boozer.npy", "final_time.npy"]
missing  = [f for f in required if not (out_dir / f).exists()]
if missing:
    print(f"Missing files in {out_dir}:\n  " + "\n  ".join(missing))
    print("Run tracing_gpu.py first.")
    sys.exit(1)

s_all  = np.load(out_dir / "initial_boozer.npy")[:, 0]   # s for all traced particles
s_lost = np.load(out_dir / "lost_initial_boozer.npy")[:, 0]  # s for lost particles

final_time = np.load(out_dir / "final_time.npy")
tmax       = final_time.max()
n_total    = len(s_all)
n_lost     = len(s_lost)
print(f"Particles: {n_total} total, {n_lost} lost ({100*n_lost/n_total:.1f}%)")

# ── Plot ──────────────────────────────────────────────────────────────────────
bins = np.linspace(0, 1, 41)   # 40 bins across s ∈ [0, 1]

fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(s_all,  bins=bins, density=True, alpha=0.5, label="all traced")
ax.hist(s_lost, bins=bins, density=True, alpha=0.7, label=f"lost ({n_lost}/{n_total})")

ax.set_xlabel("s  (Boozer flux-surface label)")
ax.set_ylabel("probability density")
ax.set_title("Birth location of lost particles")
ax.legend()
ax.set_xlim(0, 1)

fig.tight_layout()
out_path = out_dir / "loss_distribution_s.png"
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
