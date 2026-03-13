#!/usr/bin/env python3
"""
Distribution tracking test:
- sample particles on the wall
- sample pitch uniformly
- sample wall energy uniformly in [H_low, H0]
- trace backward with drag until either:
    (a) wall/loss/timeout -> f = 0
    (b) H reaches H0      -> f = S(r_birth)
- produce plots of wall distribution and birth points
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from simsopt.configs import get_data
from simsopt.field import InterpolatedField, SurfaceClassifier
from simsopt.geo import SurfaceRZFourier
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV

from firm3d.util.gpu_utils import cartesian_interpolant_drag   # <- new function
from firm3dpp import cartesian_gpu_tracing_backward_drag


# ============================================================================
# User parameters
# ============================================================================

nparticles = 10     # INCREASE THIS
seed = 7

mass = PROTON_MASS
charge = ELEMENTARY_CHARGE

H0 = 3.5e6 * ONE_EV        # birth energy [J]
H_low = 0.3e6 * ONE_EV     # wall-energy sampling lower bound [J]

coulomb_log = 17.0
Te_in_eV = True

tmax = 5e-5
tol = 1e-9

degree = 3
n = 16

# plotting
point_size = 10


# ============================================================================
# Output directory
# ============================================================================

script_dir = Path(__file__).resolve().parent
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = script_dir / "outputs_distribution_tracking" / timestamp
out_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Helpers
# ============================================================================

def wrap_phi(phi, phi_min, phi_max):
    period = phi_max - phi_min
    return (phi - phi_min) % period + phi_min


def xyz_to_rphiz(xyz, phi_min, phi_max):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.sqrt(x * x + y * y)
    phi = wrap_phi(np.arctan2(y, x), phi_min, phi_max)
    return np.column_stack([r, phi, z])


def flatten_rphiz(arr):
    out = np.empty(3 * arr.shape[0], dtype=np.float64)
    out[0::3] = arr[:, 0]
    out[1::3] = arr[:, 1]
    out[2::3] = arr[:, 2]
    return out


def total_speed_from_H(H, m):
    return np.sqrt(2.0 * H / m)


def sample_wall_points(surface, nparticles, rng):
    gamma = surface.gamma().reshape(-1, 3)
    idx = rng.integers(0, gamma.shape[0], size=nparticles)
    return gamma[idx]


def birth_source_placeholder(xyz):
    """
    Placeholder source S(r_birth).
    """
    R = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    Z = xyz[:, 2]
    return np.exp(-((R - np.mean(R)) ** 2) / (2 * (0.15 ** 2))) * np.exp(-(Z ** 2) / (2 * (0.15 ** 2)))


def ne_field(rphiz):
    return np.full(rphiz.shape[0], 5e19)

def Te_field(rphiz):
    return np.full(rphiz.shape[0], 5e3)


def run_backward_drag(
    cell_quad_pts_drag,
    r_range,
    phi_range,
    z_range,
    stz_init,
    mass,
    charge,
    vtotal_ref,
    vtang,
    H_init,
    coulomb_log,
    Te_in_eV,
    tmax,
    tol,
    H_stop,
):
    out = cartesian_gpu_tracing_backward_drag(
        cell_quad_pts_drag,
        np.array(r_range, dtype=np.float64),
        np.array(phi_range, dtype=np.float64),
        np.array(z_range, dtype=np.float64),
        np.asarray(stz_init, dtype=np.float64),
        float(mass),
        float(charge),
        float(vtotal_ref),
        np.asarray(vtang, dtype=np.float64),
        np.asarray(H_init, dtype=np.float64),
        float(coulomb_log),
        bool(Te_in_eV),
        float(tmax),
        float(tol),
        int(len(H_init)),
        float(H_stop),
        True,
    )
    return np.asarray(out, dtype=np.float64).reshape(len(H_init), 7)


# ============================================================================
# Build field / geometry / interpolants
# ============================================================================

base_curves, base_currents, ma, nfp, bs = get_data("ncsx")

mpol = 5
ntor = 5
stellsym = True
surface = SurfaceRZFourier.from_nphi_ntheta(
    mpol=mpol,
    ntor=ntor,
    stellsym=stellsym,
    nfp=nfp,
    range="full torus",
    nphi=64,
    ntheta=24,
)
surface.fit_to_curve(ma, 0.20, flip_theta=False)

sc_particle = SurfaceClassifier(surface, h=0.1, p=2)

rs = np.linalg.norm(surface.gamma()[:, :, 0:2], axis=2)
zs = surface.gamma()[:, :, 2]

r_range = (np.min(rs), np.max(rs), n)
phi_range = (0, 2 * np.pi / nfp, n * 2)
z_range = (np.min(zs), np.max(zs), n)

bsh = InterpolatedField(
    bs,
    degree,
    r_range,
    phi_range,
    z_range,
    True,
    nfp=nfp,
    stellsym=True,
)


r_range, phi_range, z_range, cell_quad_pts_drag = cartesian_interpolant_drag(
    field=bsh,
    sc_particle=sc_particle,
    ne_fun=ne_field,
    Te_fun=Te_field,
    nfp=nfp,
    n_metagrid_pts=n,
)

# ============================================================================
# Initial wall distribution
# ============================================================================

rng = np.random.default_rng(seed)

xyz_wall = sample_wall_points(surface, nparticles, rng)
rphiz_wall = xyz_to_rphiz(xyz_wall, phi_range[0], phi_range[1])
stz_wall = flatten_rphiz(rphiz_wall)

# Sample wall energies uniformly
H_wall = rng.uniform(H_low, H0, size=nparticles)

# Sample pitch lambda = vpar / v uniformly in [-1, 1]
lam = rng.uniform(-1.0, 1.0, size=nparticles)

# Per-particle total speed from wall energy
v_total_each = total_speed_from_H(H_wall, mass)
vtang = lam * v_total_each

# For error scaling / tolerances, use the maximum reference speed
vtotal_ref = float(np.sqrt(2.0 * H0 / mass))


# ============================================================================
# Backward tracing
# ============================================================================

res = run_backward_drag(
    cell_quad_pts_drag=cell_quad_pts_drag,
    r_range=r_range,
    phi_range=phi_range,
    z_range=z_range,
    stz_init=stz_wall,
    mass=mass,
    charge=charge,
    vtotal_ref=vtotal_ref,
    vtang=vtang,
    H_init=H_wall,
    coulomb_log=coulomb_log,
    Te_in_eV=Te_in_eV,
    tmax=tmax,
    tol=tol,
    H_stop=H0,
)

t_final = res[:, 0]
xyz_birth_or_end = res[:, 1:4]
vpar_final = res[:, 4]
H_final = res[:, 5]
stop_code = res[:, 6].astype(int)

hit_birth = (stop_code == 2)

# Distribution value on wall:
f_wall = np.zeros(nparticles, dtype=np.float64)
if np.any(hit_birth):
    f_wall[hit_birth] = birth_source_placeholder(xyz_birth_or_end[hit_birth])

print("Stop codes:", {int(k): int(v) for k, v in zip(*np.unique(stop_code, return_counts=True))})
print(f"Particles reaching birth energy: {np.sum(hit_birth)}/{nparticles}")


# ============================================================================
# Save arrays
# ============================================================================

np.save(out_dir / "xyz_wall.npy", xyz_wall)
np.save(out_dir / "rphiz_wall.npy", rphiz_wall)
np.save(out_dir / "H_wall.npy", H_wall)
np.save(out_dir / "lambda_wall.npy", lam)
np.save(out_dir / "results_backward.npy", res)
np.save(out_dir / "f_wall.npy", f_wall)
np.save(out_dir / "hit_birth.npy", hit_birth)


# ============================================================================
# Plots
# ============================================================================

# 1) Wall points colored by f
plt.figure(figsize=(7, 6))
sc = plt.scatter(
    xyz_wall[:, 0], xyz_wall[:, 2],
    c=f_wall,
    s=point_size,
)
plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.title("Wall samples colored by f_wall")
plt.colorbar(sc, label="f_wall")
plt.tight_layout()
plt.savefig(out_dir / "wall_distribution_xz.png", dpi=200)
plt.close()

# 2) Wall points in (phi, z)
plt.figure(figsize=(7, 6))
sc = plt.scatter(
    rphiz_wall[:, 1], rphiz_wall[:, 2],
    c=f_wall,
    s=point_size,
)
plt.xlabel("phi [rad]")
plt.ylabel("z [m]")
plt.title("Wall distribution in (phi, z)")
plt.colorbar(sc, label="f_wall")
plt.tight_layout()
plt.savefig(out_dir / "wall_distribution_phiz.png", dpi=200)
plt.close()

# 3) Birth points for successful trajectories
if np.any(hit_birth):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(
        xyz_birth_or_end[hit_birth, 0],
        xyz_birth_or_end[hit_birth, 2],
        c=f_wall[hit_birth],
        s=point_size,
    )
    plt.xlabel("x_birth [m]")
    plt.ylabel("z_birth [m]")
    plt.title("Birth points (stop_code == 2)")
    plt.colorbar(sc, label="S(r_birth)")
    plt.tight_layout()
    plt.savefig(out_dir / "birth_points_xz.png", dpi=200)
    plt.close()

# 4) Histogram of lambda for successful trajectories
plt.figure(figsize=(7, 5))
plt.hist(lam[hit_birth], bins=40)
plt.xlabel(r"$\lambda = v_\parallel / v$")
plt.ylabel("count")
plt.title("Pitch distribution of wall samples that reach birth energy")
plt.tight_layout()
plt.savefig(out_dir / "lambda_hit_birth_hist.png", dpi=200)
plt.close()

# 5) Histogram of wall energy for successful trajectories
plt.figure(figsize=(7, 5))
plt.hist(H_wall[hit_birth] / ONE_EV / 1e6, bins=40)
plt.xlabel("wall energy [MeV]")
plt.ylabel("count")
plt.title("Wall-energy distribution of successful backward traces")
plt.tight_layout()
plt.savefig(out_dir / "Hwall_hit_birth_hist.png", dpi=200)
plt.close()

# 6) 2D map in (lambda, H_wall)
plt.figure(figsize=(7, 6))
plt.hist2d(
    lam,
    H_wall / ONE_EV / 1e6,
    bins=50,
    weights=f_wall,
)
plt.xlabel(r"$\lambda = v_\parallel / v$")
plt.ylabel("wall energy [MeV]")
plt.title("Weighted wall distribution in (lambda, H_wall)")
plt.colorbar(label="weighted count")
plt.tight_layout()
plt.savefig(out_dir / "distribution_lambda_Hwall.png", dpi=200)
plt.close()

print(f"Saved outputs to {out_dir}")