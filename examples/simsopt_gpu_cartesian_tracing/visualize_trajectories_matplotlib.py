#!/usr/bin/env python3

"""
Visualize particle trajectories using matplotlib.

Reads GPU tracing output from ./output/ and produces overhead (X-Y),
poloidal (R-Z), and 3D trajectory plots.

Usage:
    python visualize_trajectories_matplotlib.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from trajectory_utils import (
    load_all_trajectories,
    load_initial_positions,
    truncate_lost_trajectories,
)

# ── Parameters ───────────────────────────────────────────────────────────
output_dir = './output'
dpi = 300

# ── Load data ────────────────────────────────────────────────────────────
print("Loading trajectory data...")
trajectories, tmax_values, lost_mask, final_times = load_all_trajectories(output_dir)

nparticles = trajectories.shape[0]
n_lost = int(np.sum(lost_mask))
print(f"  {nparticles} particles (confined: {nparticles - n_lost}, lost: {n_lost})")

# Truncate lost trajectories to show only up to loss point
trajectories = truncate_lost_trajectories(trajectories, final_times, tmax_values)

# Load initial positions
try:
    initial_xyz = load_initial_positions(output_dir)
except FileNotFoundError:
    print("Warning: initial positions file not found, skipping markers")
    initial_xyz = None

# ── Shared legend ────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], color='b', alpha=0.5, label='Confined'),
    Line2D([0], [0], color='r', alpha=0.5, label='Lost'),
]
if initial_xyz is not None:
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g',
               markersize=5, label='Initial', linestyle=''))


# ── 1. Overhead view (X-Y) ──────────────────────────────────────────────
print("Plotting overhead view...")
fig, ax = plt.subplots(figsize=(10, 10))

for i in range(trajectories.shape[0]):
    color = 'r' if lost_mask[i] else 'b'
    ax.plot(trajectories[i, :, 0], trajectories[i, :, 1],
            color=color, alpha=0.3, linewidth=0.5)

if initial_xyz is not None:
    ax.scatter(initial_xyz[:, 0], initial_xyz[:, 1],
               c='green', s=10, alpha=0.5, zorder=5)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Particle Trajectories - Overhead View (X-Y)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_elements, loc='best')

plt.tight_layout()
outfile = os.path.join(output_dir, 'trajectories_overhead.png')
plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"  Saved {outfile}")


# ── 2. Poloidal view (R-Z) ──────────────────────────────────────────────
print("Plotting poloidal view...")
fig, ax = plt.subplots(figsize=(10, 8))

R_traj = np.sqrt(trajectories[:, :, 0]**2 + trajectories[:, :, 1]**2)
Z_traj = trajectories[:, :, 2]

for i in range(trajectories.shape[0]):
    color = 'r' if lost_mask[i] else 'b'
    ax.plot(R_traj[i], Z_traj[i], color=color, alpha=0.3, linewidth=0.5)

if initial_xyz is not None:
    R_init = np.sqrt(initial_xyz[:, 0]**2 + initial_xyz[:, 1]**2)
    Z_init = initial_xyz[:, 2]
    ax.scatter(R_init, Z_init, c='green', s=10, alpha=0.5, zorder=5)

ax.set_xlabel('R [m]')
ax.set_ylabel('Z [m]')
ax.set_title('Particle Trajectories - Poloidal View (R-Z)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(handles=legend_elements, loc='best')

plt.tight_layout()
outfile = os.path.join(output_dir, 'trajectories_poloidal.png')
plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"  Saved {outfile}")


# ── 3. 3D view ──────────────────────────────────────────────────────────
print("Plotting 3D view...")
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for i in range(trajectories.shape[0]):
    color = 'r' if lost_mask[i] else 'b'
    ax.plot(trajectories[i, :, 0], trajectories[i, :, 1],
            trajectories[i, :, 2], color=color, alpha=0.3, linewidth=0.5)

if initial_xyz is not None:
    ax.scatter(initial_xyz[:, 0], initial_xyz[:, 1], initial_xyz[:, 2],
               c='green', s=10, alpha=0.5)

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.set_title('Particle Trajectories - 3D View')
ax.legend(handles=legend_elements, loc='best')

plt.tight_layout()
outfile = os.path.join(output_dir, 'trajectories_3d.png')
plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
plt.close()
print(f"  Saved {outfile}")

print("\nDone!")
