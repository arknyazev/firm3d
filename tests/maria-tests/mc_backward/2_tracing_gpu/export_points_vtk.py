#!/usr/bin/env python3
"""
Export initial and final particle positions as VTK point clouds for Paraview.

Reads from the output/ directory produced by tracing_gpu.py and writes:
  points_initial.vtu          — all initial positions, coloured by vpar and s (Boozer)
  points_final_lost.vtu       — final positions of lost particles, coloured by t_final
  points_final_confined.vtu   — final positions of confined particles
  points_initial_outside.vtu  — particles removed before tracing (outside LCFS), if any

Load alongside surface_LPQH.vts / coils_LPQH.vtu in Paraview.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def write_points_vtu(filename, xyz, point_data=None):
    """
    Write a VTK unstructured grid (.vtu) file containing a point cloud.

    Args:
        filename:   output path (str or Path)
        xyz:        (N, 3) float array of Cartesian positions
        point_data: dict of {name: (N,) array} for scalar attributes
    """
    npts = len(xyz)

    root = ET.Element("VTKFile")
    root.set("type", "UnstructuredGrid")
    root.set("version", "0.1")
    root.set("byte_order", "LittleEndian")

    ugrid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(ugrid, "Piece")
    piece.set("NumberOfPoints", str(npts))
    piece.set("NumberOfCells", str(npts))

    pts_elem = ET.SubElement(piece, "Points")
    arr = ET.SubElement(pts_elem, "DataArray")
    arr.set("type", "Float64")
    arr.set("NumberOfComponents", "3")
    arr.set("format", "ascii")
    arr.text = " ".join(f"{x:.8e} {y:.8e} {z:.8e}" for x, y, z in xyz)

    # One VTK_VERTEX cell per point
    cells = ET.SubElement(piece, "Cells")
    for name_, data_, dtype in [
        ("connectivity", range(npts),       "Int32"),
        ("offsets",      range(1, npts + 1), "Int32"),
        ("types",        ["1"] * npts,       "UInt8"),   # 1 = VTK_VERTEX
    ]:
        da = ET.SubElement(cells, "DataArray")
        da.set("type", dtype)
        da.set("Name", name_)
        da.set("format", "ascii")
        da.text = " ".join(map(str, data_))

    if point_data:
        pdata = ET.SubElement(piece, "PointData")
        for name, data in point_data.items():
            da = ET.SubElement(pdata, "DataArray")
            da.set("type", "Float64")
            da.set("Name", name)
            da.set("format", "ascii")
            da.text = " ".join(f"{v:.8e}" for v in data)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(filename), encoding="utf-8", xml_declaration=True)
    print(f"Written {filename}  ({npts} points)")


def rphiz_to_xyz(rphiz):
    """Convert (R, phi, Z) columns to Cartesian (X, Y, Z)."""
    R, phi, Z = rphiz[:, 0], rphiz[:, 1], rphiz[:, 2]
    return np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])


def main():
    out_dir = Path(__file__).parent / "output"

    required = [
        "initial_xyz.npy", "initial_rphiz.npy", "initial_vtang.npy", "initial_boozer.npy",
        "final_xyz.npy", "final_vpar.npy", "final_time.npy",
    ]
    missing = [f for f in required if not (out_dir / f).exists()]
    if missing:
        print(f"Missing files in {out_dir}:\n  " + "\n  ".join(missing))
        print("Run tracing_gpu.py first.")
        sys.exit(1)

    initial_xyz    = np.load(out_dir / "initial_xyz.npy")
    initial_rphiz  = np.load(out_dir / "initial_rphiz.npy")
    initial_vtang  = np.load(out_dir / "initial_vtang.npy")
    initial_boozer = np.load(out_dir / "initial_boozer.npy")   # s, theta, zeta
    final_xyz      = np.load(out_dir / "final_xyz.npy")
    final_vpar     = np.load(out_dir / "final_vpar.npy")
    final_time     = np.load(out_dir / "final_time.npy")

    tmax      = final_time.max()
    lost_mask = final_time < 0.99 * tmax
    n_lost    = int(lost_mask.sum())
    n_total   = len(final_time)
    print(f"Loaded {n_total} particles: {n_lost} lost, {n_total - n_lost} confined")

    # ── Particles removed before tracing (outside LCFS) ──────────────────────
    outside_file = out_dir / "initial_outside_rphiz.npy"
    if outside_file.exists():
        outside_rphiz = np.load(outside_file)
        outside_xyz   = rphiz_to_xyz(outside_rphiz)
        write_points_vtu(out_dir / "points_initial_outside.vtu", outside_xyz)

    # ── All initial positions ─────────────────────────────────────────────────
    write_points_vtu(
        out_dir / "points_initial.vtu",
        initial_xyz,
        point_data={
            "vpar":  initial_vtang,
            "s":     initial_boozer[:, 0],
            "R":     initial_rphiz[:, 0],
            "Z":     initial_rphiz[:, 2],
        },
    )

    # ── Final positions of lost particles ─────────────────────────────────────
    if n_lost > 0:
        write_points_vtu(
            out_dir / "points_final_lost.vtu",
            final_xyz[lost_mask],
            point_data={
                "t_final": final_time[lost_mask],
                "vpar":    final_vpar[lost_mask],
            },
        )
    else:
        print("No lost particles — skipping points_final_lost.vtu")

    # ── Final positions of confined particles ─────────────────────────────────
    n_confined = n_total - n_lost
    if n_confined > 0:
        write_points_vtu(
            out_dir / "points_final_confined.vtu",
            final_xyz[~lost_mask],
            point_data={
                "t_final": final_time[~lost_mask],
                "vpar":    final_vpar[~lost_mask],
            },
        )
    else:
        print("No confined particles — skipping points_final_confined.vtu")


if __name__ == "__main__":
    main()
