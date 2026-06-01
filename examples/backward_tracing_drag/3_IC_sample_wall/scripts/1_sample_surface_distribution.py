"""
Sample initial conditions for particles born uniformly on a given flux surface,
in Boozer coordinates.

Convert to cylindrical and Cartesian coordinates for GPU guiding-centre tracing
and Paraview visualisation.

Inputs:   boozmn.nc  (BOOZ_XFORM output for the equilibrium of interest)
Outputs:  outputs/initial_conditions_surface_boozer.txt      (s, theta, zeta)
          outputs/initial_conditions_surface_cylindrical.txt  (R, phi, Z)
          outputs/points_initial_surface.vtu                  (Paraview point cloud)
"""

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.coordinates import boozer_to_cylindrical
from firm3d.field.tracing_helpers import initialize_position_uniform_surf
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_world


# ── Input parameters ─────────────────────────────────────────────────────────

@dataclass
class Inputs:
    boozmn_file:   str   = "boozmn.nc"  # BOOZ_XFORM of the equilibrium
    nparticles:    int   = 10_000_000    # number of samples to generate
    resolution:    int   = 48           # grid points per dimension for field interpolation
    radial_order:  int   = 3            # spline order for radial (1-D) interpolation
    spline_degree: int   = 3            # spline degree for 3-D interpolation
    s_surface:     float = 1.0          # normalised toroidal flux of the surface to sample
    ntheta_max:    int   = 100          # theta grid points for Jacobian computation
    nzeta_max:     int   = 100          # zeta  grid points for Jacobian computation

inputs = Inputs()

# ── Logging ──────────────────────────────────────────────────────────────────

setup_logging(f"stdout_surface_s{inputs.s_surface}_p{inputs.nparticles}_res{inputs.resolution}.txt")
t_start = time.time()

# ── Build magnetic field interpolant ─────────────────────────────────────────

bri = BoozerRadialInterpolant(
    inputs.boozmn_file, inputs.radial_order, no_K=True, comm=comm_world
)
field = InterpolatedBoozerField(
    bri,
    inputs.spline_degree,
    ns_interp=inputs.resolution,
    ntheta_interp=inputs.resolution,
    nzeta_interp=inputs.resolution,
)


# ── Sample initial conditions on the flux surface ────────────────────────────
# Positions are drawn uniformly with respect to the volume element on the
# surface s = inputs.s_surface in Boozer coordinates (theta, zeta).

points_boozer = initialize_position_uniform_surf(
    field,
    inputs.nparticles,
    inputs.s_surface,
    ntheta_max=inputs.ntheta_max,
    nzeta_max=inputs.nzeta_max,
    comm=comm_world,
)

proc0_print(f"IC generation time: {time.time() - t_start:.2f}s")


# ── Convert to cylindrical and Cartesian coordinates ─────────────────────────

points_cyl = boozer_to_cylindrical(field, points_boozer)

R, phi, Z = points_cyl[:, 0], points_cyl[:, 1], points_cyl[:, 2]
points_xyz = np.column_stack([R * np.cos(phi), R * np.sin(phi), Z])


# ── VTK helper ───────────────────────────────────────────────────────────────

def write_points_vtu(filename, xyz, point_data=None):
    """Write a VTK unstructured grid (.vtu) point cloud for Paraview."""
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

    cells = ET.SubElement(piece, "Cells")
    for name_, data_, dtype in [
        ("connectivity", range(npts),       "Int32"),
        ("offsets",      range(1, npts + 1), "Int32"),
        ("types",        ["1"] * npts,       "UInt8"),  # 1 = VTK_VERTEX
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
    proc0_print(f"Written {filename}  ({npts} points)")


# ── Save ─────────────────────────────────────────────────────────────────────

output_dir = Path(__file__).parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

np.savetxt(
    output_dir / "initial_conditions_surface_boozer.txt",
    points_boozer,
    header=f"s_init theta_init zeta_init  [surface s={inputs.s_surface}]",
)
np.savetxt(
    output_dir / "initial_conditions_surface_cylindrical.txt",
    points_cyl,
    header=f"R_init phi_init Z_init  [surface s={inputs.s_surface}]",
)

write_points_vtu(
    output_dir / "points_initial_surface.vtu",
    points_xyz,
    point_data={
        "s": points_boozer[:, 0],
        "R": R,
        "Z": Z,
    },
)

proc0_print(f"Saved {inputs.nparticles} initial conditions to {output_dir}")
