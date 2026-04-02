"""
Check that wall-surface initial conditions are not immediately lost by the
SurfaceClassifier used in GPU tracing.

Builds the coil field and surface classifier identically to tracing_gpu.py,
then evaluates the signed distance for every particle from
outputs/initial_conditions_surface_cylindrical.txt.

No particle tracing is performed.

Inputs:   LandremanPaulQH_coils/  (coils + VMEC input)
          outputs/initial_conditions_surface_cylindrical.txt
Outputs:  outputs/classifier_check_sd.npy   — signed distance per particle
          outputs/points_inside.vtu          — particles inside LCFS  (Paraview)
          outputs/points_outside.vtu         — particles outside LCFS (Paraview)
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simsopt.field import (
    BiotSavart,
    Current,
    SurfaceClassifier,
    coils_via_symmetries,
)
from simsopt.field.coil import load_coils_from_makegrid_file
from simsopt.geo import SurfaceRZFourier


# ── Paths ─────────────────────────────────────────────────────────────────────

THIS_DIR  = Path(__file__).parent.resolve()
REPO_ROOT = THIS_DIR.parent.parent
COILS_DIR = REPO_ROOT / "LandremanPaulQH_coils"
IC_DIR    = THIS_DIR.parent / "outputs"
OUT_DIR   = IC_DIR


# ── Input parameters ──────────────────────────────────────────────────────────

@dataclass
class Inputs:
    coil_file:       Path  = COILS_DIR / "coils.curves_22_7_21"
    vmec_input_file: Path  = COILS_DIR / "input.vmec"
    ic_file_cyl:     Path  = IC_DIR / "initial_conditions_surface_cylindrical.txt"

    # Equilibrium
    nfp:        int   = 4
    ncoils:     int   = 5
    current:    float = 1.27797548115612e7  # [A]
    coil_order: int   = 20

    # Surface resolution (for classifier and VTK)
    nphi_surf:   int = 128
    ntheta_surf: int = 64

    # SurfaceClassifier
    sc_h: float = 0.05  # grid spacing [m]
    sc_p: int   = 2     # interpolant degree


inp = Inputs()
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load coils ─────────────────────────────────────────────────────────────

all_coils     = load_coils_from_makegrid_file(str(inp.coil_file), order=inp.coil_order)
base_curves   = [all_coils[i].curve for i in range(inp.ncoils)]
base_currents = [Current(inp.current) for _ in range(inp.ncoils)]
coils         = coils_via_symmetries(base_curves, base_currents, inp.nfp, stellsym=True)
bs            = BiotSavart(coils)


# ── 2. Load plasma boundary ───────────────────────────────────────────────────

s_input = SurfaceRZFourier.from_vmec_input(
    str(inp.vmec_input_file), range="full torus",
    nphi=inp.nphi_surf, ntheta=inp.ntheta_surf,
)


# ── 3. Build SurfaceClassifier ────────────────────────────────────────────────

sc_particle = SurfaceClassifier(s_input, h=inp.sc_h, p=inp.sc_p)


# ── 4. Load initial conditions ────────────────────────────────────────────────
# Columns: R_init  phi_init  Z_init

ic_cyl = np.loadtxt(inp.ic_file_cyl, comments="#")
R, phi, Z = ic_cyl[:, 0], ic_cyl[:, 1], ic_cyl[:, 2]
nparticles = len(R)
print(f"Loaded {nparticles} particles from {inp.ic_file_cyl.name}")


# ── 5. Evaluate signed distance ───────────────────────────────────────────────
# sd > 0  →  inside LCFS (good)
# sd < 0  →  outside LCFS (would be immediately discarded by the tracer)

sd = sc_particle.evaluate_rphiz(np.column_stack([R, phi, Z])).ravel()

inside  = sd >= 0
n_in    = int(inside.sum())
n_out   = int((~inside).sum())
print(f"\nSurfaceClassifier results:")
print(f"  Inside  LCFS: {n_in}/{nparticles} ({100*n_in/nparticles:.1f}%)")
print(f"  Outside LCFS: {n_out}/{nparticles} ({100*n_out/nparticles:.1f}%)")
print(f"  sd  min={sd.min():.4f}  mean={sd.mean():.4f}  max={sd.max():.4f}")

np.save(OUT_DIR / "classifier_check_sd.npy", sd)


# ── 6. Export to VTK for Paraview ─────────────────────────────────────────────

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
        ("connectivity", range(npts),        "Int32"),
        ("offsets",      range(1, npts + 1),  "Int32"),
        ("types",        ["1"] * npts,         "UInt8"),
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


X = R * np.cos(phi)
Y = R * np.sin(phi)
xyz = np.column_stack([X, Y, Z])

if n_in > 0:
    write_points_vtu(
        OUT_DIR / "points_inside.vtu",
        xyz[inside],
        point_data={"sd": sd[inside], "R": R[inside], "Z": Z[inside]},
    )

if n_out > 0:
    write_points_vtu(
        OUT_DIR / "points_outside.vtu",
        xyz[~inside],
        point_data={"sd": sd[~inside], "R": R[~inside], "Z": Z[~inside]},
    )

print(f"\nOutputs saved to {OUT_DIR}")
