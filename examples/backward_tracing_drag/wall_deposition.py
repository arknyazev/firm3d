#!/usr/bin/env python3
"""
Backward-only wall deposition estimate.

1. Load wall IC points.
2. Sample wall energy and pitch.
3. Trace backward with drag until H reaches H_fusion.
4. Treat stop_code == 2 particles as wall-depositing samples whose forward
   counterpart would have hit the same wall start with energy H_wall.
5. Bin the wall starts and compute a simple heat-load (/heat-flux) proxy.

The wall IC points are area-uniform samples on one field period of the s=1
surface. For a simple map, the bin area is estimated from the number of wall
samples in each bin:

    area_bin ~= area_one_period * n_wall_in_bin / n_wall_total

This makes the output a deposition pattern on the sampled wall domain. By
default, the energy pattern is normalized to 1 W total deposited power, so the
heat-flux map is W/m^2 per watt of deposited alpha power. Set
deposition_power_W to the physical total deposited power to get absolute W/m^2.
"""

import csv
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simsopt.field import (
    BiotSavart,
    Current,
    InterpolatedField,
    SurfaceClassifier,
    coils_via_symmetries,
)
from simsopt.field.coil import load_coils_from_makegrid_file
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as H_FUSION,
    ONE_EV,
)

from firm3d.util.gpu_utils import cartesian_interpolant_drag
from firm3dpp import cartesian_gpu_tracing_backward_drag


THIS_DIR = Path(__file__).resolve().parent


@dataclass
class Inputs:
    # Files
    coil_file: Path = THIS_DIR / "LandremanPaulQH_coils" / "coils.curves_22_7_21"
    vmec_input_file: Path = THIS_DIR / "LandremanPaulQH_coils" / "input.vmec"
    wall_ic_file: Path = Path(
        "/pscratch/sd/m/mariagar/projects/mc_proj/IC/"
        "initial_conditions_surface_cylindrical.txt"
    )
    wall_boozer_file: Path = Path(
        "/pscratch/sd/m/mariagar/projects/mc_proj/IC/"
        "initial_conditions_surface_boozer.txt"
    )

    # Equilibrium / coils
    nfp: int = 4
    ncoils: int = 5
    current: float = 1.27797548115612e7
    coil_order: int = 20

    # Interpolant grid
    n_r: int = 64
    n_phi: int = 128
    n_z: int = 64
    degree: int = 3
    nphi_surf: int = 128
    ntheta_surf: int = 64

    # SurfaceClassifier
    sc_h: float = 0.05
    sc_p: int = 2

    # Wall deposition bins. If wall_boozer_file exists, use theta/zeta bins.
    # Otherwise fall back to phi/Z bins.
    n_theta_bins: int = 48
    n_zeta_bins: int = 48
    n_phi_bins: int = 48
    n_Z_bins: int = 48

    # Energies
    mass: float = MASS
    charge: float = CHARGE
    H_low: float = 3.0e6 * ONE_EV
    H_high: float = H_FUSION
    H_fusion: float = H_FUSION
    coulomb_log: float = 17.0
    Te_in_eV: bool = True
    ne0: float = 1e21
    Te0_ev: float = 100.0

    # Normalize the successful deposition pattern to this power
    deposition_power_W: float = 1.0

    # Tracing
    n_wall: int = 1000000
    tmax_backward: float = 5e-5
    tol: float = 1e-9
    seed: int = 57

    # Finite-difference step for outward normal
    normal_fd_eps: float = 1e-4


inp = Inputs()


def _slowing_down_time(ne, Te_eV, mass, coulomb_log):
    eps0 = 8.8541878128e-12
    e_ch = 1.602176634e-19
    m_e = 9.1093837015e-31
    Z_alpha = 2.0
    Te_J = Te_eV * e_ch
    num = 3.0 * (2.0 * np.pi) ** 1.5 * eps0 ** 2 * mass * Te_J ** 1.5
    den = Z_alpha ** 2 * e_ch ** 4 * np.sqrt(m_e) * ne * coulomb_log
    return num / den


_tau_s = _slowing_down_time(inp.ne0, inp.Te0_ev, inp.mass, inp.coulomb_log)
_tmax_required = np.log(inp.H_fusion / inp.H_low) * _tau_s
inp.tmax_backward = float(1.2 * _tmax_required)
print(f"tau_s = {_tau_s:.4e} s, tmax_backward = {inp.tmax_backward:.4e} s "
      f"(= 1.2 * ln(H_fusion/H_low) * tau_s)")


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = (
    Path("/pscratch/sd/m/mariagar/projects/mc_proj/results/backward_wall_deposition")
    / timestamp
)
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "plots").mkdir(parents=True, exist_ok=True)
print(f"Writing outputs to {out_dir}")


def wrap_phi(phi, phi_min, phi_max):
    period = phi_max - phi_min
    return (phi - phi_min) % period + phi_min


def summarize_stop_codes(stop_codes):
    uniq, counts = np.unique(stop_codes.astype(int), return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}


def ne_fun(rphiz):
    return np.full(rphiz.shape[0], inp.ne0, dtype=np.float64)


def Te_fun(rphiz):
    return np.full(rphiz.shape[0], inp.Te0_ev, dtype=np.float64)


def write_points_vtu(filename, xyz, point_data=None):
    npts = len(xyz)
    if npts == 0:
        print(f"  (skip {filename.name}: no points)")
        return

    root = ET.Element(
        "VTKFile",
        type="UnstructuredGrid",
        version="0.1",
        byte_order="LittleEndian",
    )
    ugrid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(
        ugrid, "Piece", NumberOfPoints=str(npts), NumberOfCells=str(npts)
    )

    pts_elem = ET.SubElement(piece, "Points")
    arr = ET.SubElement(
        pts_elem, "DataArray",
        type="Float64", NumberOfComponents="3", format="ascii",
    )
    arr.text = " ".join(f"{x:.8e} {y:.8e} {z:.8e}" for x, y, z in xyz)

    cells = ET.SubElement(piece, "Cells")
    for name_, data_, dtype in [
        ("connectivity", range(npts), "Int32"),
        ("offsets", range(1, npts + 1), "Int32"),
        ("types", ["1"] * npts, "UInt8"),
    ]:
        da = ET.SubElement(
            cells, "DataArray", type=dtype, Name=name_, format="ascii"
        )
        da.text = " ".join(map(str, data_))

    if point_data:
        pdata = ET.SubElement(piece, "PointData")
        for name, data in point_data.items():
            da = ET.SubElement(
                pdata, "DataArray", type="Float64", Name=name, format="ascii"
            )
            da.text = " ".join(f"{v:.8e}" for v in np.asarray(data).ravel())

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(filename), encoding="utf-8", xml_declaration=True)
    print(f"  wrote {filename.name} ({npts} points)")


def surface_area_from_mesh(surface):
    """Approximate full-torus surface area from the VTK mesh points."""
    if hasattr(surface, "area"):
        try:
            return float(surface.area())
        except Exception:
            pass

    gamma = np.asarray(surface.gamma(), dtype=np.float64)
    nphi, ntheta, _ = gamma.shape
    area = 0.0
    for i in range(nphi):
        ip = (i + 1) % nphi
        for j in range(ntheta):
            jp = (j + 1) % ntheta
            p00 = gamma[i, j]
            p10 = gamma[ip, j]
            p01 = gamma[i, jp]
            p11 = gamma[ip, jp]
            area += 0.5 * np.linalg.norm(np.cross(p10 - p00, p01 - p00))
            area += 0.5 * np.linalg.norm(np.cross(p11 - p10, p01 - p10))
    return float(area)


def load_wall_boozer_if_aligned(path, n_expected):
    if not path.exists():
        print(f"  wall Boozer file not found: {path}")
        return None
    arr = np.loadtxt(str(path), comments="#")
    if arr.ndim != 2 or arr.shape[1] < 3:
        print(f"  wall Boozer file has unexpected shape {arr.shape}; ignoring")
        return None
    if arr.shape[0] < n_expected:
        print(f"  wall Boozer file has {arr.shape[0]} rows but cylindrical file "
              f"has {n_expected}; ignoring")
        return None
    return arr


def deposition_binning(
    wall_xyz,
    R_wall,
    phi_wall,
    Z_wall,
    H_wall,
    success_mask,
    total_area_m2,
    wall_boozer=None,
):
    if wall_boozer is not None:
        theta = np.mod(wall_boozer[:, 1], 2.0 * np.pi)
        zeta = np.mod(wall_boozer[:, 2], 2.0 * np.pi / inp.nfp)
        x = theta
        y = zeta
        x_edges = np.linspace(0.0, 2.0 * np.pi, inp.n_theta_bins + 1)
        y_edges = np.linspace(0.0, 2.0 * np.pi / inp.nfp, inp.n_zeta_bins + 1)
        coord_name = "theta_zeta"
        x_label = "theta"
        y_label = "zeta"
    else:
        x = phi_wall
        y = Z_wall
        x_edges = np.linspace(0.0, 2.0 * np.pi / inp.nfp, inp.n_phi_bins + 1)
        y_pad = 1.0e-12 + 0.01 * max(float(np.ptp(Z_wall)), 1.0)
        y_edges = np.linspace(float(Z_wall.min() - y_pad),
                              float(Z_wall.max() + y_pad),
                              inp.n_Z_bins + 1)
        coord_name = "phi_Z"
        x_label = "phi"
        y_label = "Z [m]"

    wall_counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    success_counts, _, _ = np.histogram2d(
        x[success_mask], y[success_mask], bins=[x_edges, y_edges]
    )
    energy_J, _, _ = np.histogram2d(
        x[success_mask],
        y[success_mask],
        bins=[x_edges, y_edges],
        weights=H_wall[success_mask],
    )

    area_m2 = total_area_m2 * wall_counts / max(float(len(wall_xyz)), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        heat_load_J_m2 = np.where(area_m2 > 0.0, energy_J / area_m2, 0.0)
        success_fraction = np.where(
            wall_counts > 0.0, success_counts / wall_counts, 0.0
        )

    total_energy_J = float(energy_J.sum())
    if inp.deposition_power_W > 0.0 and total_energy_J > 0.0:
        power_W = inp.deposition_power_W * energy_J / total_energy_J
        heat_flux_W_m2 = np.where(area_m2 > 0.0, power_W / area_m2, 0.0)
    else:
        power_W = np.zeros_like(energy_J)
        heat_flux_W_m2 = np.zeros_like(energy_J)

    return {
        "coord_name": coord_name,
        "x_label": x_label,
        "y_label": y_label,
        "x": x,
        "y": y,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "wall_counts": wall_counts,
        "success_counts": success_counts,
        "success_fraction": success_fraction,
        "area_m2": area_m2,
        "energy_J": energy_J,
        "power_W": power_W,
        "heat_load_J_m2": heat_load_J_m2,
        "heat_flux_W_m2": heat_flux_W_m2,
    }


def write_deposition_csv(path, dep):
    x_edges = dep["x_edges"]
    y_edges = dep["y_edges"]
    fields = [
        "i",
        "j",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "area_m2",
        "n_wall",
        "n_success",
        "success_fraction",
        "energy_J",
        "energy_MeV",
        "power_W",
        "heat_load_J_m2",
        "heat_flux_W_m2",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        nx, ny = dep["energy_J"].shape
        for i in range(nx):
            for j in range(ny):
                energy_J = float(dep["energy_J"][i, j])
                writer.writerow({
                    "i": i,
                    "j": j,
                    "x_min": float(x_edges[i]),
                    "x_max": float(x_edges[i + 1]),
                    "y_min": float(y_edges[j]),
                    "y_max": float(y_edges[j + 1]),
                    "area_m2": float(dep["area_m2"][i, j]),
                    "n_wall": int(dep["wall_counts"][i, j]),
                    "n_success": int(dep["success_counts"][i, j]),
                    "success_fraction": float(dep["success_fraction"][i, j]),
                    "energy_J": energy_J,
                    "energy_MeV": energy_J / ONE_EV / 1e6,
                    "power_W": float(dep["power_W"][i, j]),
                    "heat_load_J_m2": float(dep["heat_load_J_m2"][i, j]),
                    "heat_flux_W_m2": float(dep["heat_flux_W_m2"][i, j]),
                })


def plot_map(path, dep, key, title, cbar_label):
    fig, ax = plt.subplots(figsize=(7, 5))
    data = dep[key].T
    mesh = ax.pcolormesh(dep["x_edges"], dep["y_edges"], data, shading="auto")
    ax.set_xlabel(dep["x_label"])
    ax.set_ylabel(dep["y_label"])
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


print("\nBuilding coils + field...")
all_coils = load_coils_from_makegrid_file(str(inp.coil_file), order=inp.coil_order)
base_curves = [all_coils[i].curve for i in range(inp.ncoils)]
base_currents = [Current(inp.current) for _ in range(inp.ncoils)]

coils = coils_via_symmetries(base_curves, base_currents, inp.nfp, stellsym=True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)

s_input = SurfaceRZFourier.from_vmec_input(
    str(inp.vmec_input_file),
    range="full torus",
    nphi=inp.nphi_surf,
    ntheta=inp.ntheta_surf,
)

sc_particle = SurfaceClassifier(s_input, h=inp.sc_h, p=inp.sc_p)

rs = np.linalg.norm(s_input.gamma()[:, :, 0:2], axis=2)
z_max = np.max(np.abs(s_input.gamma()[:, :, 2]))

rrange = (np.min(rs), np.max(rs), inp.n_r)
phirange = (0.0, 2.0 * np.pi / inp.nfp, inp.n_phi)
zrange = (0.0, z_max, inp.n_z)
phi_min, phi_max = phirange[0], phirange[1]

bsh = InterpolatedField(
    bs, inp.degree, rrange, phirange, zrange, True, nfp=inp.nfp, stellsym=True
)

curves_to_vtk(curves, str(out_dir / "coils_LPQH"), close=True)
s_input.to_vtk(str(out_dir / "surface_LPQH"))

surface_area_full = surface_area_from_mesh(s_input)
surface_area_period = surface_area_full / inp.nfp
print(f"Surface area full torus: {surface_area_full:.6e} m^2")
print(f"Surface area one field period: {surface_area_period:.6e} m^2")

t0 = time.time()
r_range, phi_range, z_range, cell_quad_pts = cartesian_interpolant_drag(
    field=bsh,
    sc_particle=sc_particle,
    ne_fun=ne_fun,
    Te_fun=Te_fun,
    nfp=inp.nfp,
    n_metagrid_pts=inp.n_r,
)
print(f"GPU drag interpolant built in {time.time() - t0:.1f}s")


print("\n--- STEP 1: load wall IC and sample pitch + energy ---")
wall_ic = np.loadtxt(inp.wall_ic_file, comments="#")
wall_boozer_all = load_wall_boozer_if_aligned(inp.wall_boozer_file, len(wall_ic))

R_all = wall_ic[:, 0]
phi_all = wall_ic[:, 1]
Z_all = wall_ic[:, 2]
n_avail = len(R_all)
n_wall = min(inp.n_wall, n_avail)
print(f"  available: {n_avail}, using {n_wall}")

rng = np.random.default_rng(inp.seed)
idx = rng.choice(n_avail, size=n_wall, replace=False)
R_wall, phi_wall, Z_wall = R_all[idx], phi_all[idx], Z_all[idx]
wall_boozer = wall_boozer_all[idx] if wall_boozer_all is not None else None

sd = sc_particle.evaluate_rphiz(
    np.column_stack([R_wall, phi_wall, Z_wall])
).ravel()
inside = sd >= 0.0
n_outside = int((~inside).sum())
if n_outside:
    print(f"  {n_outside} wall points outside LCFS - dropped")
    R_wall, phi_wall, Z_wall = R_wall[inside], phi_wall[inside], Z_wall[inside]
    if wall_boozer is not None:
        wall_boozer = wall_boozer[inside]
    n_wall = int(inside.sum())

phi_wall = wrap_phi(phi_wall, phi_min, phi_max)


def outward_unit_normal(xyz_wall, eps):
    def sd_xyz(xyz_arr):
        R = np.sqrt(xyz_arr[:, 0] ** 2 + xyz_arr[:, 1] ** 2)
        phi = np.arctan2(xyz_arr[:, 1], xyz_arr[:, 0])
        Z = xyz_arr[:, 2]
        return sc_particle.evaluate_rphiz(np.column_stack([R, phi, Z])).ravel()

    grad_sd = np.zeros_like(xyz_wall)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        grad_sd[:, k] = (sd_xyz(xyz_wall + d) - sd_xyz(xyz_wall - d)) / (2 * eps)

    grad_norm = np.linalg.norm(grad_sd, axis=1, keepdims=True)
    grad_norm[grad_norm == 0.0] = 1.0
    return -grad_sd / grad_norm


wall_xyz = np.column_stack([
    R_wall * np.cos(phi_wall),
    R_wall * np.sin(phi_wall),
    Z_wall,
])

n_out_hat = outward_unit_normal(wall_xyz, inp.normal_fd_eps)

bs.set_points(wall_xyz)
B_xyz = np.asarray(bs.B())
B_mag = np.linalg.norm(B_xyz, axis=1, keepdims=True)
B_mag[B_mag == 0.0] = 1.0
b_hat = B_xyz / B_mag

b_dot_n = np.einsum("ij,ij->i", b_hat, n_out_hat)

H_wall = rng.uniform(inp.H_low, inp.H_high, size=n_wall)
v_total_w = np.sqrt(2.0 * H_wall / inp.mass)

lam_abs = rng.uniform(0.0, 1.0, size=n_wall)
lam_wall = -np.sign(b_dot_n) * lam_abs
mask_zero = np.isclose(b_dot_n, 0.0)
lam_wall[mask_zero] = rng.uniform(-1.0, 1.0, size=int(np.count_nonzero(mask_zero)))
vtang_w = lam_wall * v_total_w

_vn_par = lam_wall * v_total_w * b_dot_n
print(f"  (v_par b_hat).n_out after sampling: max={_vn_par.max():.3e} "
      f"mean={_vn_par.mean():.3e} (all <= 0)")
print(f"  b_hat.n_out approx 0 fallbacks: {int(mask_zero.sum())}")

stz_init = np.empty(3 * n_wall, dtype=np.float64)
stz_init[0::3] = R_wall
stz_init[1::3] = phi_wall
stz_init[2::3] = Z_wall

np.save(out_dir / "wall_starts_xyz.npy", wall_xyz)
np.save(
    out_dir / "wall_starts_state.npy",
    np.column_stack([wall_xyz[:, 0], wall_xyz[:, 1], wall_xyz[:, 2], vtang_w, H_wall]),
)
if wall_boozer is not None:
    np.save(out_dir / "wall_starts_boozer.npy", wall_boozer)


print("\n--- STEP 2: backward GPU tracing (drag) ---")
speed_ref = float(sqrt(2.0 * inp.H_fusion / inp.mass))

t0 = time.time()
out = cartesian_gpu_tracing_backward_drag(
    cell_quad_pts,
    np.ascontiguousarray(r_range, dtype=np.float64),
    np.ascontiguousarray(phi_range, dtype=np.float64),
    np.ascontiguousarray(z_range, dtype=np.float64),
    np.ascontiguousarray(stz_init, dtype=np.float64),
    float(inp.mass),
    float(inp.charge),
    speed_ref,
    np.ascontiguousarray(vtang_w, dtype=np.float64),
    np.ascontiguousarray(H_wall, dtype=np.float64),
    float(inp.coulomb_log),
    bool(inp.Te_in_eV),
    float(inp.tmax_backward),
    float(inp.tol),
    int(n_wall),
    float(inp.H_fusion),
    True,
)
bwd = np.asarray(out, dtype=np.float64).reshape(n_wall, 7)
print(f"  tracing done in {time.time() - t0:.2f}s")

np.save(out_dir / "backward_results.npy", bwd)
stop_codes = bwd[:, 6].astype(int)
print(f"  stop codes: {summarize_stop_codes(stop_codes)}")

success = stop_codes == 2
M = int(success.sum())
frac_success = M / max(n_wall, 1)
deposit_energy_J = np.where(success, H_wall, 0.0)
print(f"  reached H_fusion (stop_code==2): {M}/{n_wall} "
      f"({100.0 * frac_success:.2f}%)")
print(f"  deposited energy proxy: {deposit_energy_J.sum():.6e} J "
      f"({deposit_energy_J.sum() / ONE_EV / 1e6:.6e} MeV)")


print("\n--- STEP 3: wall deposition bins ---")
dep = deposition_binning(
    wall_xyz=wall_xyz,
    R_wall=R_wall,
    phi_wall=phi_wall,
    Z_wall=Z_wall,
    H_wall=H_wall,
    success_mask=success,
    total_area_m2=surface_area_period,
    wall_boozer=wall_boozer,
)
print(f"  coordinate grid: {dep['coord_name']}")
print(f"  nonempty bins: {int((dep['wall_counts'] > 0).sum())}/"
      f"{dep['wall_counts'].size}")
print(f"  bins with deposition: {int((dep['energy_J'] > 0).sum())}/"
      f"{dep['energy_J'].size}")
print(f"  max heat-load proxy: {float(dep['heat_load_J_m2'].max()):.6e} J/m^2")
print(f"  normalized deposition power: {inp.deposition_power_W:.6e} W")
print(f"  max heat flux: {float(dep['heat_flux_W_m2'].max()):.6e} W/m^2")

np.savez(
    out_dir / "wall_deposition_maps.npz",
    coord_name=np.array(dep["coord_name"]),
    x_edges=dep["x_edges"],
    y_edges=dep["y_edges"],
    wall_counts=dep["wall_counts"],
    success_counts=dep["success_counts"],
    success_fraction=dep["success_fraction"],
    area_m2=dep["area_m2"],
    energy_J=dep["energy_J"],
    power_W=dep["power_W"],
    heat_load_J_m2=dep["heat_load_J_m2"],
    heat_flux_W_m2=dep["heat_flux_W_m2"],
)
write_deposition_csv(out_dir / "wall_deposition_bins.csv", dep)

with open(out_dir / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    w.writerow(["n_wall", n_wall])
    w.writerow(["n_success_stop_code_2", M])
    w.writerow(["frac_success", frac_success])
    w.writerow(["H_low_MeV", inp.H_low / ONE_EV / 1e6])
    w.writerow(["H_high_MeV", inp.H_high / ONE_EV / 1e6])
    w.writerow(["H_fusion_MeV", inp.H_fusion / ONE_EV / 1e6])
    w.writerow(["total_surface_area_full_torus_m2", surface_area_full])
    w.writerow(["sampled_surface_area_one_period_m2", surface_area_period])
    w.writerow(["deposition_coordinate_grid", dep["coord_name"]])
    w.writerow(["total_deposit_energy_J", float(dep["energy_J"].sum())])
    w.writerow(["total_deposit_energy_MeV", float(dep["energy_J"].sum() / ONE_EV / 1e6)])
    w.writerow(["max_heat_load_J_m2", float(dep["heat_load_J_m2"].max())])
    w.writerow(["deposition_power_W", inp.deposition_power_W])
    w.writerow(["max_heat_flux_W_m2", float(dep["heat_flux_W_m2"].max())])
    w.writerow(["tmax_backward_s", inp.tmax_backward])
    for k, v in summarize_stop_codes(stop_codes).items():
        w.writerow([f"stop_code_{k}_count", v])


print("\n--- STEP 4: VTK exports ---")
write_points_vtu(
    out_dir / "wall_starts_deposition.vtu",
    wall_xyz,
    point_data={
        "H_wall": H_wall,
        "vpar_init": vtang_w,
        "stop_code": stop_codes.astype(np.float64),
        "deposit_success": success.astype(np.float64),
        "deposit_energy_J": deposit_energy_J,
        "deposit_energy_MeV": deposit_energy_J / ONE_EV / 1e6,
    },
)

success_xyz = wall_xyz[success]
write_points_vtu(
    out_dir / "wall_deposition_successes.vtu",
    success_xyz,
    point_data={
        "H_wall": H_wall[success],
        "deposit_energy_MeV": H_wall[success] / ONE_EV / 1e6,
    },
)


print("\n--- STEP 5: plots ---")
pdir = out_dir / "plots"
plot_map(
    pdir / "wall_counts.png",
    dep,
    "wall_counts",
    "Wall samples per bin",
    "count",
)
plot_map(
    pdir / "success_fraction.png",
    dep,
    "success_fraction",
    "Backward success fraction per wall bin",
    "success fraction",
)
plot_map(
    pdir / "deposited_energy.png",
    dep,
    "energy_J",
    "Deposited wall energy proxy",
    "J",
)
plot_map(
    pdir / "heat_load_proxy.png",
    dep,
    "heat_load_J_m2",
    "Heat-load proxy",
    "J/m^2",
)
plot_map(
    pdir / "heat_flux.png",
    dep,
    "heat_flux_W_m2",
    "Heat flux",
    "W/m^2",
)

fig, ax = plt.subplots(figsize=(7, 4))
if M:
    ax.hist(H_wall[success] / ONE_EV / 1e6, bins=40, alpha=0.75, color="C1")
ax.set_xlabel("wall energy [MeV]")
ax.set_ylabel("successful wall samples")
ax.set_title("Energy of backward-successful wall samples")
fig.tight_layout()
fig.savefig(pdir / "successful_wall_energy_hist.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(wall_xyz[:, 0], wall_xyz[:, 1], s=1, alpha=0.15, color="grey",
           label=f"wall starts ({n_wall})")
if M:
    ax.scatter(wall_xyz[success, 0], wall_xyz[success, 1],
               s=4, alpha=0.55, color="C1", label=f"success ({M})")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_aspect("equal")
ax.legend()
ax.set_title("Wall deposition points, XY")
fig.tight_layout()
fig.savefig(pdir / "wall_deposition_xy.png", dpi=150)
plt.close(fig)

print(f"  wrote plots to {pdir}")
print(f"  success fraction: {100.0 * frac_success:.2f}%")
print(f"  total deposit energy proxy: {float(dep['energy_J'].sum()):.6e} J")
print(f"Outputs at: {out_dir}")