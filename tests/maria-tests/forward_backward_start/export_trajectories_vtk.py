#!/usr/bin/env python3
"""
Export particle trajectories to VTK format for Paraview visualization.

Reads GPU tracing output from ./output/tol_*/{forward,backward}/
and writes a single .vtu file with polylines that can be loaded alongside
the surface VTK files.

Usage examples:
  python export_trajectories_vtk.py
  python export_trajectories_vtk.py tol_1p00em09
  python export_trajectories_vtk.py tol_1p00em09 forward
  python export_trajectories_vtk.py tol_1p00em09 both
"""

import os
import sys
import numpy as np
import xml.etree.ElementTree as ET

from trajectory_utils import (
    load_all_trajectories,
    truncate_lost_trajectories,
)

# ── Parameters ───────────────────────────────────────────────────────────
output_root = "./output"

# Optional CLI:
#   argv[1] = tol_dir (e.g. "tol_1p00em09" or "./output/tol_1p00em09")
#   argv[2] = which: "forward" | "backward" | "both"
tol_dir = sys.argv[1] if len(sys.argv) >= 2 else None
which = sys.argv[2].lower() if len(sys.argv) >= 3 else "both"
if which not in ("forward", "backward", "both"):
    raise ValueError('which must be "forward", "backward", or "both"')

runs = []
if which in ("forward", "both"):
    runs.append(("forward", 0))
if which in ("backward", "both"):
    runs.append(("backward", 1))

# ── Load data ────────────────────────────────────────────────────────────
all_points = []
all_lines = []
all_times = []
all_particle_ids = []
all_lost_status = []
all_run_ids = []

point_offset = 0
nparticles_ref = None

print("Loading trajectory data...")
for run_subdir, run_id in runs:
    trajectories, tmax_values, lost_mask, final_times, run_dir, tol_dir_full = load_all_trajectories(
        output_root=output_root,
        tol_dir=tol_dir,
        run_subdir=run_subdir,
    )
    trajectories = truncate_lost_trajectories(trajectories, final_times, tmax_values)

    nparticles = trajectories.shape[0]
    n_lost = int(np.sum(lost_mask))
    print(f"  tol={os.path.basename(tol_dir_full)} run={run_subdir}  "
          f"{nparticles} particles (confined: {nparticles - n_lost}, lost: {n_lost})")

    if nparticles_ref is None:
        nparticles_ref = nparticles
    elif nparticles != nparticles_ref:
        raise ValueError("forward/backward nparticles mismatch; cannot export in one file")

    for i in range(nparticles):
        valid = ~np.isnan(trajectories[i, :, 0])
        traj = trajectories[i, valid]
        times = tmax_values[valid]

        if len(traj) == 0:
            continue

        npts = len(traj)
        all_points.append(traj)
        all_times.append(times)
        all_lines.append(list(range(point_offset, point_offset + npts)))
        point_offset += npts

        all_particle_ids.append(float(i))
        all_lost_status.append(1.0 if lost_mask[i] else 0.0)
        all_run_ids.append(float(run_id))

points = np.vstack(all_points) if len(all_points) else np.zeros((0, 3), dtype=np.float32)
times = np.hstack(all_times) if len(all_times) else np.zeros((0,), dtype=np.float32)

# ── Write VTK unstructured grid ──────────────────────────────────────────
print("Writing VTK file...")

root = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1", byte_order="LittleEndian")
ugrid = ET.SubElement(root, "UnstructuredGrid")
piece = ET.SubElement(ugrid, "Piece")
piece.set("NumberOfPoints", str(len(points)))
piece.set("NumberOfCells", str(len(all_lines)))

# Points
pts_elem = ET.SubElement(piece, "Points")
pts_arr = ET.SubElement(pts_elem, "DataArray", type="Float32", NumberOfComponents="3", format="ascii")
pts_arr.text = " ".join(f"{x:.6e} {y:.6e} {z:.6e}" for x, y, z in points)

# Cells (polylines)
cells = ET.SubElement(piece, "Cells")

conn = ET.SubElement(cells, "DataArray", type="Int32", Name="connectivity", format="ascii")
conn.text = " ".join(str(idx) for line in all_lines for idx in line)

offsets = ET.SubElement(cells, "DataArray", type="Int32", Name="offsets", format="ascii")
cumsum = 0
offset_vals = []
for line in all_lines:
    cumsum += len(line)
    offset_vals.append(cumsum)
offsets.text = " ".join(map(str, offset_vals))

types = ET.SubElement(cells, "DataArray", type="UInt8", Name="types", format="ascii")
types.text = " ".join(["4"] * len(all_lines))  # 4 = VTK_POLY_LINE

# Point data: time
pdata = ET.SubElement(piece, "PointData")
time_arr = ET.SubElement(pdata, "DataArray", type="Float32", Name="time", format="ascii")
time_arr.text = " ".join(f"{t:.6e}" for t in times)

# Cell data
cdata = ET.SubElement(piece, "CellData")

for name, values in [
    ("particle_id", all_particle_ids),
    ("is_lost", all_lost_status),
    ("run_id", all_run_ids),  # 0=forward, 1=backward
]:
    arr = ET.SubElement(cdata, "DataArray", type="Float32", Name=name, format="ascii")
    arr.text = " ".join(f"{v:.6e}" for v in values)

# Write
# Save into the tol directory we actually used (derive from last loaded tol_dir_full)
outfile = os.path.join(tol_dir_full, f"trajectories_{which}.vtu")
tree = ET.ElementTree(root)
ET.indent(tree, space="  ")
tree.write(outfile, encoding="utf-8", xml_declaration=True)

print(f"  Saved {outfile}")
print(f"\nOpen in Paraview alongside {tol_dir_full}/../surface.vts (surface is still in ./output/)")