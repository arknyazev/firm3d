#!/usr/bin/env python3
"""
Visualize particle trajectories using matplotlib.

Reads GPU tracing output from ./output/tol_*/{forward,backward}/
and produces overhead (X-Y), poloidal (R-Z), and 3D trajectory plots.

Usage examples:
  python visualize_trajectories_matplotlib.py run_2026-03-06_15-30
  python visualize_trajectories_matplotlib.py run_2026-03-06_15-30 tol_1p00em09
  python visualize_trajectories_matplotlib.py run_2026-03-06_15-30 tol_1p00em09 both
  python visualize_trajectories_matplotlib.py run_2026-03-06_15-30 tol_1p00em09 forward
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from trajectory_utils import (
    load_all_trajectories,
    load_initial_positions,
    truncate_lost_trajectories,
)


# ── Parameters ───────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent

if len(sys.argv) < 2:
    raise ValueError(
        'Usage: python visualize_trajectories_matplotlib.py <run_name> [tol_dir] [forward|backward|both]'
    )

run_name = sys.argv[1]
tol_dir = sys.argv[2] if len(sys.argv) >= 3 else None
which = sys.argv[3].lower() if len(sys.argv) >= 4 else "both"

if which not in ("forward", "backward", "both"):
    raise ValueError('which must be "forward", "backward", or "both"')

output_root = script_dir / "output" / run_name
output_root.mkdir(parents=True, exist_ok=True)

dpi = 300

tol_dir = sys.argv[1] if len(sys.argv) >= 2 else None
which = sys.argv[2].lower() if len(sys.argv) >= 3 else "both"
if which not in ("forward", "backward", "both"):
    raise ValueError('which must be "forward", "backward", or "both"')

runs = []
if which in ("forward", "both"):
    runs.append("forward")
if which in ("backward", "both"):
    runs.append("backward")

# ── Load initial positions (from tol folder) ─────────────────────────────
try:
    initial_xyz = load_initial_positions(output_root=str(output_root), tol_dir=tol_dir)
except FileNotFoundError:
    print("Warning: initial positions file not found in tol folder, skipping markers")
    initial_xyz = None

# ── Load trajectories for each run ───────────────────────────────────────
run_data = {}  # run_subdir -> dict with trajectories, lost_mask, tmax_values, final_times, tol_dir_full
print("Loading trajectory data...")
tol_dir_full_ref = None

for run_subdir in runs:
    trajectories, tmax_values, lost_mask, final_times, run_dir, tol_dir_full = load_all_trajectories(
        output_root=str(output_root),
        tol_dir=tol_dir,
        run_subdir=run_subdir,
    )
    trajectories = truncate_lost_trajectories(trajectories, final_times, tmax_values)

    if tol_dir_full_ref is None:
        tol_dir_full_ref = tol_dir_full
    else:
        if tol_dir_full != tol_dir_full_ref:
            raise RuntimeError("Internal error: tol_dir_full mismatch across runs")

    n_lost = int(np.sum(lost_mask))
    print(f"  tol={os.path.basename(tol_dir_full)} run={run_subdir}: "
          f"{trajectories.shape[0]} particles (confined: {trajectories.shape[0]-n_lost}, lost: {n_lost})")

    run_data[run_subdir] = {
        "traj": trajectories,
        "tmax": tmax_values,
        "lost": lost_mask,
    }

# ── Legend ───────────────────────────────────────────────────────────────
legend_elements = []
if "forward" in run_data:
    legend_elements.extend([
        Line2D([0], [0], color="b", alpha=0.5, label="Forward (confined)"),
        Line2D([0], [0], color="r", alpha=0.5, label="Forward (lost)"),
    ])
if "backward" in run_data:
    legend_elements.extend([
        Line2D([0], [0], color="k", alpha=0.5, linestyle="--", label="Backward (confined)"),
        Line2D([0], [0], color="0.5", alpha=0.5, linestyle="--", label="Backward (lost)"),
    ])
if initial_xyz is not None:
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="g",
               markersize=5, label="Initial", linestyle="")
    )

def plot_xy(ax, traj, lost_mask, style):
    for i in range(traj.shape[0]):
        if style == "forward":
            color = "r" if lost_mask[i] else "b"
            ls = "-"
        else:
            color = "0.5" if lost_mask[i] else "k"
            ls = "--"
        ax.plot(traj[i, :, 0], traj[i, :, 1], color=color, alpha=0.3, linewidth=0.5, linestyle=ls)

def plot_rz(ax, traj, lost_mask, style):
    R_traj = np.sqrt(traj[:, :, 0]**2 + traj[:, :, 1]**2)
    Z_traj = traj[:, :, 2]
    for i in range(traj.shape[0]):
        if style == "forward":
            color = "r" if lost_mask[i] else "b"
            ls = "-"
        else:
            color = "0.5" if lost_mask[i] else "k"
            ls = "--"
        ax.plot(R_traj[i], Z_traj[i], color=color, alpha=0.3, linewidth=0.5, linestyle=ls)

def plot_3d(ax, traj, lost_mask, style):
    for i in range(traj.shape[0]):
        if style == "forward":
            color = "r" if lost_mask[i] else "b"
            ls = "-"
        else:
            color = "0.5" if lost_mask[i] else "k"
            ls = "--"
        ax.plot(traj[i, :, 0], traj[i, :, 1], traj[i, :, 2], color=color, alpha=0.3, linewidth=0.5, linestyle=ls)

# ── 1. Overhead view (X-Y) ──────────────────────────────────────────────
print("Plotting overhead view...")
fig, ax = plt.subplots(figsize=(10, 10))

for run_subdir, data in run_data.items():
    plot_xy(ax, data["traj"], data["lost"], run_subdir)

if initial_xyz is not None:
    ax.scatter(initial_xyz[:, 0], initial_xyz[:, 1], c="green", s=10, alpha=0.5, zorder=5)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title(f"Particle Trajectories - Overhead View (X-Y) [{os.path.basename(tol_dir_full_ref)}]")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_elements, loc="best")

plt.tight_layout()
outfile = os.path.join(tol_dir_full_ref, f"trajectories_overhead_{which}.png")
plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
plt.close()
print(f"  Saved {outfile}")

# ── 2. Poloidal view (R-Z) ──────────────────────────────────────────────
print("Plotting poloidal view...")
fig, ax = plt.subplots(figsize=(10, 8))

for run_subdir, data in run_data.items():
    plot_rz(ax, data["traj"], data["lost"], run_subdir)

if initial_xyz is not None:
    R_init = np.sqrt(initial_xyz[:, 0]**2 + initial_xyz[:, 1]**2)
    Z_init = initial_xyz[:, 2]
    ax.scatter(R_init, Z_init, c="green", s=10, alpha=0.5, zorder=5)

ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")
ax.set_title(f"Particle Trajectories - Poloidal View (R-Z) [{os.path.basename(tol_dir_full_ref)}]")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_elements, loc="best")

plt.tight_layout()
outfile = os.path.join(tol_dir_full_ref, f"trajectories_poloidal_{which}.png")
plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
plt.close()
print(f"  Saved {outfile}")

# ── 3. 3D view ──────────────────────────────────────────────────────────
print("Plotting 3D view...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

for run_subdir, data in run_data.items():
    plot_3d(ax, data["traj"], data["lost"], run_subdir)

if initial_xyz is not None:
    ax.scatter(initial_xyz[:, 0], initial_xyz[:, 1], initial_xyz[:, 2], c="green", s=10, alpha=0.5)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
ax.set_title(f"Particle Trajectories - 3D View [{os.path.basename(tol_dir_full_ref)}]")
ax.legend(handles=legend_elements, loc="best")

plt.tight_layout()
outfile = os.path.join(tol_dir_full_ref, f"trajectories_3d_{which}.png")
plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
plt.close()
print(f"  Saved {outfile}")

print("\nDone!")