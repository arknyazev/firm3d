#!/usr/bin/env python3
"""
Histogram of sampled Boozer s values vs the theoretical reactivity profile.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

IC_DIR = Path(__file__).parent / "outputs"

ic = np.loadtxt(IC_DIR / "initial_conditions_boozer.txt", comments="#")
s_sampled = ic[:, 0]
print(f"Loaded {len(s_sampled)} particles,  s range: [{s_sampled.min():.4f}, {s_sampled.max():.4f}]")

# ── Theoretical reactivity profile ───────────────────────────────────────────
def sigmav(T):
    return T**(-2/3) * np.exp(-19.94 * T**(-1/3)) if T > 0 else 0.0

nD         = lambda s: 1 - s**5
Tfunc      = lambda s: 11.5 * (1 - s)
reactivity = lambda s: nD(s)**2 * sigmav(Tfunc(s))

s_th = np.linspace(0, 1, 500)
r_th = np.array([reactivity(s) for s in s_th])

# ── Plot ──────────────────────────────────────────────────────────────────────
bins = np.linspace(0, 1, 51)
bin_centres = 0.5 * (bins[:-1] + bins[1:])

counts, _ = np.histogram(s_sampled, bins=bins)

fig, ax = plt.subplots(figsize=(7, 4))

ax.bar(bin_centres, counts, width=bins[1] - bins[0],
       color="steelblue", alpha=0.7, label="Sampled particles")

# Scale theoretical curve to match histogram area
r_th_norm = r_th / np.trapezoid(r_th, s_th) * len(s_sampled) * (bins[1] - bins[0])
ax.plot(s_th, r_th_norm, "r-", lw=2, label="Theoretical reactivity (normalised)")

ax.set_xlabel("Boozer $s$")
ax.set_ylabel("Number of particles / bin")
ax.set_title(f"Fusion birth distribution in $s$  (N={len(s_sampled)})")
ax.legend()
fig.tight_layout()

out = IC_DIR / "s_distribution.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
