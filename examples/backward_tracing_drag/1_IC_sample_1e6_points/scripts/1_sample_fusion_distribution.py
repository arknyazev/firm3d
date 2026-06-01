"""
Sample initial conditions for fusion-born alpha particles,
from a D-T reactivity profile in Boozer coordinates.

Сonvert to cylindrical coordinates for GPU guiding-centre tracing.

Inputs:   boozmn.nc  (BOOZ_XFORM output for the equilibrium of interest)
Outputs:  outputs/initial_conditions_boozer.txt      (s, theta, zeta, vpar)
          outputs/initial_conditions_cylindrical.txt  (R, phi, Z, vpar)
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.coordinates import boozer_to_cylindrical
from firm3d.field.tracing_helpers import (
    initialize_position_profile,
    initialize_velocity_uniform,
)
from firm3d.util.constants import (
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from firm3d.util.functions import proc0_print, setup_logging
from firm3d.util.mpi import comm_world


# ── Input parameters ─────────────────────────────────────────────────────────

@dataclass
class Inputs:
    boozmn_file:   str = "boozmn.nc"  # BOOZ_XFORM of the equilibrium
    nparticles:    int = 1_000_000    # number of samples to generate
    resolution:    int = 48           # grid points per dimension for field interpolation
    radial_order:  int = 3            # spline order for radial (1-D) interpolation
    spline_degree: int = 3            # spline degree for 3-D interpolation

inputs = Inputs()

# ── Logging ──────────────────────────────────────────────────────────────────

setup_logging(f"stdout_p{inputs.nparticles}_res{inputs.resolution}.txt")
t_start = time.time()

# ── Build magnetic field interpolant ─────────────────────────────────────────
# BoozerRadialInterpolant reads the BOOZ_XFORM file and builds a 1-D spline in
# the flux-surface label s.  InterpolatedBoozerField extends this to a full 3-D
# (s, theta, zeta) interpolant on a regular grid.

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


# ── D-T fusion birth distribution ────────────────────────────────────────────
# Reactivity  ∝  n_D(s) · n_T(s) · <σv>(T(s))
# Particles are sampled with probability proportional to reactivity(s).

def sigmav(T_keV):
    """
    D-T rate coefficient; 
    Following Bader et al., Nucl. Fusion 61 (2021) 116060.
    """
    if T_keV > 0:
        return T_keV ** (-2 / 3) * np.exp(-19.94 * T_keV ** (-1 / 3))
    return 0.0

nD         = lambda s: 1 - s**5          # normalised deuterium density
nT         = nD                          # equal deuterium/tritium densities
T_keV      = lambda s: 11.5 * (1 - s)    # temperature profile [keV]
reactivity = lambda s: nD(s) * nT(s) * sigmav(T_keV(s))


# ── Sample initial conditions ────────────────────────────────────────────────
# Positions are drawn from the reactivity-weighted distribution over the plasma
# volume in Boozer coordinates (s, theta, zeta).
# Parallel velocities are drawn uniformly on [-vpar0, +vpar0], consistent with
# an isotropic birth distribution projected onto the parallel direction.

vpar0  = np.sqrt(2 * FUSION_ALPHA_PARTICLE_ENERGY / ALPHA_PARTICLE_MASS)

points_boozer = initialize_position_profile(
    field, inputs.nparticles, reactivity, comm=comm_world
)
vpar_init = initialize_velocity_uniform(vpar0, inputs.nparticles, comm=comm_world)

proc0_print(f"IC generation time: {time.time() - t_start:.2f}s")


# ── Convert to cylindrical coordinates ───────────────────────────────────────
# The GPU tracing kernel expects positions as (R, phi, Z) in metres.

points_cyl = boozer_to_cylindrical(field, points_boozer)


# ── Save ─────────────────────────────────────────────────────────────────────

output_dir = Path(__file__).parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

np.savetxt(
    output_dir / "initial_conditions_boozer.txt",
    np.column_stack([points_boozer, vpar_init]),
    header="s_init theta_init zeta_init vpar_init",
)
np.savetxt(
    output_dir / "initial_conditions_cylindrical.txt",
    np.column_stack([points_cyl, vpar_init]),
    header="R_init phi_init Z_init vpar_init",
)
proc0_print(f"Saved {inputs.nparticles} initial conditions to {output_dir}")
