#!/usr/bin/env python3

"""
Export particle trajectories to VTK format for Paraview visualization.

Reads GPU tracing output from ./output/ and writes a single .vtu file
with polylines that can be loaded alongside the surface VTK files.

Usage:
    python export_trajectories_vtk.py
"""

import os
import numpy as np
import xml.etree.ElementTree as ET

from trajectory_utils import (
    load_all_trajectories,
    truncate_lost_trajectories,
)

# ── Parameters ───────────────────────────────────────────────────────────
output_dir = './output'

# ── Load data ────────────────────────────────────────────────────────────
print("Loading trajectory data...")
trajectories, tmax_values, lost_mask, final_times = load_all_trajectories(output_dir)
trajectories = truncate_lost_trajectories(trajectories, final_times, tmax_values)

nparticles = trajectories.shape[0]
n_lost = int(np.sum(lost_mask))
print(f"  {nparticles} particles (confined: {nparticles - n_lost}, lost: {n_lost})")


# ── Build VTK polyline data ──────────────────────────────────────────────
all_points = []
all_lines = []
all_times = []
all_particle_ids = []
all_lost_status = []

point_offset = 0
for i in range(nparticles):
    # Skip NaN points (truncated lost trajectories)
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

    all_particle_ids.append(i)
    all_lost_status.append(1.0 if lost_mask[i] else 0.0)

points = np.vstack(all_points)
times = np.hstack(all_times)


# ── Write VTK unstructured grid ──────────────────────────────────────────
print("Writing VTK file...")

root = ET.Element('VTKFile')
root.set('type', 'UnstructuredGrid')
root.set('version', '0.1')
root.set('byte_order', 'LittleEndian')

ugrid = ET.SubElement(root, 'UnstructuredGrid')
piece = ET.SubElement(ugrid, 'Piece')
piece.set('NumberOfPoints', str(len(points)))
piece.set('NumberOfCells', str(len(all_lines)))

# Points
pts_elem = ET.SubElement(piece, 'Points')
pts_arr = ET.SubElement(pts_elem, 'DataArray')
pts_arr.set('type', 'Float32')
pts_arr.set('NumberOfComponents', '3')
pts_arr.set('format', 'ascii')
pts_arr.text = ' '.join(f'{x:.6e} {y:.6e} {z:.6e}' for x, y, z in points)

# Cells (polylines)
cells = ET.SubElement(piece, 'Cells')

conn = ET.SubElement(cells, 'DataArray')
conn.set('type', 'Int32')
conn.set('Name', 'connectivity')
conn.set('format', 'ascii')
conn.text = ' '.join(str(idx) for line in all_lines for idx in line)

offsets = ET.SubElement(cells, 'DataArray')
offsets.set('type', 'Int32')
offsets.set('Name', 'offsets')
offsets.set('format', 'ascii')
cumsum = 0
offset_vals = []
for line in all_lines:
    cumsum += len(line)
    offset_vals.append(cumsum)
offsets.text = ' '.join(map(str, offset_vals))

types = ET.SubElement(cells, 'DataArray')
types.set('type', 'UInt8')
types.set('Name', 'types')
types.set('format', 'ascii')
types.text = ' '.join(['4'] * len(all_lines))  # 4 = VTK_POLY_LINE

# Point data: time
pdata = ET.SubElement(piece, 'PointData')
time_arr = ET.SubElement(pdata, 'DataArray')
time_arr.set('type', 'Float32')
time_arr.set('Name', 'time')
time_arr.set('format', 'ascii')
time_arr.text = ' '.join(f'{t:.6e}' for t in times)

# Cell data: particle_id and is_lost
cdata = ET.SubElement(piece, 'CellData')
for name, values in [('particle_id', all_particle_ids),
                     ('is_lost', all_lost_status)]:
    arr = ET.SubElement(cdata, 'DataArray')
    arr.set('type', 'Float32')
    arr.set('Name', name)
    arr.set('format', 'ascii')
    arr.text = ' '.join(f'{v:.6e}' for v in values)

# Write
outfile = os.path.join(output_dir, 'trajectories_all.vtu')
tree = ET.ElementTree(root)
ET.indent(tree, space='  ')
tree.write(outfile, encoding='utf-8', xml_declaration=True)

print(f"  Saved {outfile}")
print(f"\nOpen in Paraview alongside {output_dir}/surface.vts")
