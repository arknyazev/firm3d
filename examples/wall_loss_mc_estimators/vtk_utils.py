"""Paraview-ready VTK/VTU writers + trajectory snapshot tracing, shared by
the three MC-comparison workflows.
"""
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from simsopt.geo import curves_to_vtk

from perturbed_field_utils import flatten_stz


def write_coils_and_surface_vtk(out_dir, curves, s_input,
                                coils_stem="coils_LPQH",
                                surface_stem="surface_LPQH"):
    out_dir = Path(out_dir)
    curves_to_vtk(curves, str(out_dir / coils_stem), close=True)
    s_input.to_vtk(str(out_dir / surface_stem))


def write_points_vtu(filename, xyz, point_data=None):
    npts = len(xyz)
    if npts == 0:
        print(f"  (skip {Path(filename).name}: no points)")
        return

    root = ET.Element("VTKFile",
                      type="UnstructuredGrid", version="0.1",
                      byte_order="LittleEndian")
    ugrid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(ugrid, "Piece",
                          NumberOfPoints=str(npts), NumberOfCells=str(npts))

    pts_elem = ET.SubElement(piece, "Points")
    arr = ET.SubElement(pts_elem, "DataArray",
                        type="Float64", NumberOfComponents="3", format="ascii")
    arr.text = " ".join(f"{x:.8e} {y:.8e} {z:.8e}" for x, y, z in xyz)

    cells = ET.SubElement(piece, "Cells")
    for name_, data_, dtype in [
        ("connectivity", range(npts),         "Int32"),
        ("offsets",      range(1, npts + 1),  "Int32"),
        ("types",        ["1"] * npts,        "UInt8"),  # 1 = VTK_VERTEX
    ]:
        da = ET.SubElement(cells, "DataArray",
                           type=dtype, Name=name_, format="ascii")
        da.text = " ".join(map(str, data_))

    if point_data:
        pdata = ET.SubElement(piece, "PointData")
        for name, data in point_data.items():
            da = ET.SubElement(pdata, "DataArray",
                               type="Float64", Name=name, format="ascii")
            da.text = " ".join(f"{v:.8e}" for v in data)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(filename), encoding="utf-8", xml_declaration=True)
    print(f"  wrote {Path(filename).name} ({npts} points)")


def write_polylines_vtu(filename, pts_per_poly, pts,
                        point_data=None, cell_data=None):
    n_poly = pts.shape[0]
    if n_poly == 0:
        print(f"  (skip {Path(filename).name}: no polylines)")
        return
    flat    = pts.reshape(-1, 3)
    n_total = flat.shape[0]

    root = ET.Element("VTKFile",
                      type="UnstructuredGrid", version="0.1",
                      byte_order="LittleEndian")
    ugrid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(ugrid, "Piece",
                          NumberOfPoints=str(n_total),
                          NumberOfCells=str(n_poly))

    pts_elem = ET.SubElement(piece, "Points")
    da = ET.SubElement(pts_elem, "DataArray",
                       type="Float64", NumberOfComponents="3", format="ascii")
    da.text = " ".join(f"{x:.6e} {y:.6e} {z:.6e}" for x, y, z in flat)

    cells = ET.SubElement(piece, "Cells")
    conn = ET.SubElement(cells, "DataArray",
                         type="Int32", Name="connectivity", format="ascii")
    conn.text = " ".join(str(i) for i in range(n_total))
    off = ET.SubElement(cells, "DataArray",
                        type="Int32", Name="offsets", format="ascii")
    off.text = " ".join(str(pts_per_poly * (k + 1)) for k in range(n_poly))
    typ = ET.SubElement(cells, "DataArray",
                        type="UInt8", Name="types", format="ascii")
    typ.text = " ".join(["4"] * n_poly)  # 4 = VTK_POLY_LINE

    if point_data:
        pdata = ET.SubElement(piece, "PointData")
        for name, data in point_data.items():
            d = ET.SubElement(pdata, "DataArray",
                              type="Float64", Name=name, format="ascii")
            d.text = " ".join(f"{v:.6e}" for v in np.asarray(data).ravel())

    if cell_data:
        cdata = ET.SubElement(piece, "CellData")
        for name, data in cell_data.items():
            d = ET.SubElement(cdata, "DataArray",
                              type="Int32", Name=name, format="ascii")
            d.text = " ".join(str(int(v)) for v in data)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(filename), encoding="utf-8", xml_declaration=True)
    print(f"  wrote {Path(filename).name} ({n_poly} polylines, "
          f"{pts_per_poly} pts each)")


def trace_snapshots(tracer, field, R_init, phi_init, Z_init, vpar_init,
                    H_init, mass, charge, speed_ref, coulomb_log, Te_in_eV,
                    tmax, tol, n_snapshots, H_stop, use_energy_stop,
                    label="snapshots"):
    """Run tracer (either cartesian_gpu_tracing_drag or
    cartesian_gpu_tracing_backward_drag) n_snapshots times with
    tmax_i = (i+1)/n_snapshots * tmax and repack the results into
    time-resolved per-particle arrays.
    """
    n = len(R_init)
    stz = flatten_stz(R_init, phi_init, Z_init)
    tmax_snaps = np.linspace(tmax / n_snapshots, tmax, n_snapshots)

    xyz  = np.zeros((n, n_snapshots, 3), dtype=np.float64)
    vpar = np.zeros((n, n_snapshots),    dtype=np.float64)
    H    = np.zeros((n, n_snapshots),    dtype=np.float64)
    t_   = np.zeros((n, n_snapshots),    dtype=np.float64)
    stop = np.zeros((n, n_snapshots),    dtype=np.int32)

    t0 = time.time()
    for i, tmax_i in enumerate(tmax_snaps):
        out = tracer(
            field["cell_quad_pts"],
            np.ascontiguousarray(field["r_range"],   dtype=np.float64),
            np.ascontiguousarray(field["phi_range"], dtype=np.float64),
            np.ascontiguousarray(field["z_range"],   dtype=np.float64),
            np.ascontiguousarray(stz,                dtype=np.float64),
            float(mass), float(charge), float(speed_ref),
            np.ascontiguousarray(vpar_init, dtype=np.float64),
            np.ascontiguousarray(H_init,    dtype=np.float64),
            float(coulomb_log), bool(Te_in_eV),
            float(tmax_i), float(tol), int(n),
            float(H_stop), bool(use_energy_stop),
        )
        arr = np.asarray(out, dtype=np.float64).reshape(n, 7)
        t_[:, i]     = arr[:, 0]
        xyz[:, i, 0] = arr[:, 1]
        xyz[:, i, 1] = arr[:, 2]
        xyz[:, i, 2] = arr[:, 3]
        vpar[:, i]   = arr[:, 4]
        H[:, i]      = arr[:, 5]
        stop[:, i]   = arr[:, 6].astype(np.int32)
    print(f"  {label}: {n_snapshots} snapshots x {n} particles in "
          f"{time.time() - t0:.2f}s")
    return xyz, vpar, H, t_, stop, tmax_snaps


def write_trajectory_polylines(filename, initial_xyz, snap_xyz,
                               snap_time, snap_vpar, snap_H,
                               initial_vpar, initial_H,
                               particle_ids=None):
    n_traj, n_snap, _ = snap_xyz.shape
    pts_per = 1 + n_snap

    pts = np.empty((n_traj, pts_per, 3), dtype=np.float64)
    pts[:, 0, :]  = initial_xyz
    pts[:, 1:, :] = snap_xyz

    t_pt    = np.empty((n_traj, pts_per), dtype=np.float64)
    vpar_pt = np.empty((n_traj, pts_per), dtype=np.float64)
    H_pt    = np.empty((n_traj, pts_per), dtype=np.float64)
    t_pt[:, 0]     = 0.0
    t_pt[:, 1:]    = snap_time
    vpar_pt[:, 0]  = initial_vpar
    vpar_pt[:, 1:] = snap_vpar
    H_pt[:, 0]     = initial_H
    H_pt[:, 1:]    = snap_H

    pdata = {"time": t_pt, "vpar": vpar_pt, "H": H_pt}
    cdata = ({"particle_id": np.asarray(particle_ids).astype(np.int32)}
             if particle_ids is not None else None)
    write_polylines_vtu(filename, pts_per_poly=pts_per, pts=pts,
                        point_data=pdata, cell_data=cdata)