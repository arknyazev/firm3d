#!/usr/bin/env python3
"""
Algorithm:
----------
BACKWARD PHASE (build IS proposal):
  1. Load wall states from 1_sample_surface_distribution outputs
     (R, phi, Z) + sample uniform pitch lambda and uniform wall
     energy H_w in [H_low, H_0].
  2. Trace backward with drag from each wall state.
     Keep only stop_code == 2 (reached birth energy H_0).
  3. Collect successful birth endpoints {x_j} = (X, Y, Z, v_par, H_0).
  4. Score s_j via 5-D histogram over (X, Y, Z, v_par, H) in birth space.
  5. Compute biased distribution  pi_j (q)  proportional to  p_0(x_j) * s_j
     where p_0 is the fusion reactivity evaluated at x_j (in Boozer s).

FORWARD PHASE (IS estimator + baseline comparison):
  IS forward:
    - Resample N birth states from pi.
    - Trace forward with drag.
    - Compute weighted estimator  Q_hat = (1/N) sum A(X_i) * (a_i / pi_i).

  Baseline forward:
    - Load birth states sampled directly from p_0 (fusion distribution outputs).
    - Trace forward with drag.
    - Estimate loss fraction naively as  mean(A(X_i)).

Inputs (paths set in Inputs dataclass below):
  - surface IC file:   outputs from 1_sample_surface_distribution.py
                       (initial_conditions_surface_cylindrical.txt)
  - fusion IC file:    outputs from 1_sample_fusion_distribution.py
                       (initial_conditions_cylindrical.txt)
  - boozmn.nc:         for evaluating p_0 in Boozer coordinates

Outputs (written to out_dir):
  - backward_results.npy          raw backward tracer output (n_wall x 7)
  - birth_endpoints.npy           successful birth points (M x 5): X,Y,Z,vpar,H
  - birth_endpoints_rphiz.npy     same in cylindrical: R,phi,Z,vpar,H
  - birth_endpoints_boozer.npy    Boozer s for each birth endpoint (M,)
  - scores.npy                    histogram score s_j for each birth endpoint (M,)
  - p0_at_births.npy              p_0(x_j) for each birth endpoint (M,)
  - pi_weights.npy                normalised pi_j = p0 * s / Z  (M,)
  - forward_is_results.npy        IS forward tracer output (N x 7)
  - forward_baseline_results.npy  baseline forward tracer output (N_base x 7)
  - summary.csv                   Q_hat_IS, Q_hat_baseline, N, M, n_wall, etc.
"""

import os
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np

from simsopt.configs import get_data
from simsopt.field import InterpolatedField, SurfaceClassifier,  BiotSavart, Current, coils_via_symmetries
from simsopt.field.coil import load_coils_from_makegrid_file
from simsopt.geo import SurfaceRZFourier, curves_to_vtk
from simsopt.util import proc0_print
from simsopt.util.constants import ONE_EV
from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE  as CHARGE,
    ALPHA_PARTICLE_MASS    as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
)

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.coordinates import boozer_to_cylindrical, cylindrical_to_boozer
from firm3d.util.gpu_utils import cartesian_interpolant_drag
from firm3dpp import (
    cartesian_gpu_tracing_drag,
    cartesian_gpu_tracing_backward_drag,
)


# =============================================================================
# Inputs
# =============================================================================

@dataclass
class Inputs:

    # Files
    coil_file: Path        = "LandremanPaulQH_coils" / "coils.curves_22_7_21"
    vmec_input_file: Path   = "LandremanPaulQH_coils" / "input.vmec"

    # Equilibrium
    nfp:     int   = 4                   # number of field periods
    ncoils:       int   = 5                   # unique base coil shapes per half field period
    current:      float = 1.27797548115612e7  # coil current [A] — from extcur.curves_22_7_21
    coil_order:   int   = 20                  # Fourier order for coil curve representation

    # Interpolation grid
    n_r:    int = 64   # grid cells in R
    n_phi:  int = 128  # grid cells in φ
    n_z:    int = 64   # grid cells in Z
    degree: int = 3    # spline degree (must be 3 for the GPU CUDA kernel)
    nphi_surf:   int = 128  # surface resolution used for B·n check and VTK output
    ntheta_surf: int = 64

    # SurfaceClassifier (loss criterion)
    sc_h: float = 0.05  # grid spacing [m] — smaller is more accurate near the boundary
    sc_p: int   = 2     # interpolant degree


    ######################### end of config from 2_tracing_gpu

    # ── Equilibrium / field ───────────────────────────────────────────────────
    boozmn_file: str = "1_IC_sample_1e6_points/inputs/boozmn.nc"        # BOOZ_XFORM output
    radial_order: int = 3                  # spline order for radial interpolation
    spline_degree: int = 3                 # 3-D interpolant degree
    resolution: int = 48                   # grid pts per dim for field interpolant

    # ── Wall IC file (from 1_sample_surface_distribution.py) ─────────────────
    # Columns: R_init  phi_init  Z_init   (no vpar — we sample that here)
    wall_ic_file: str = "3_IC_sample_wall/outputs/initial_conditions_surface_cylindrical.txt"
    n_wall: int = 50_000       # wall states to use for backward phase

    # ── Fusion birth IC file (from 1_sample_fusion_distribution.py) ──────────
    # Columns: R_init  phi_init  Z_init  vpar_init
    fusion_ic_file: str = "1_IC_sample_1e6_points/outputs/initial_conditions_cylindrical.txt"
    n_baseline: int = 50_000   # birth states for standard baseline forward run

    # ── Physics ───────────────────────────────────────────────────────────────
    mass: float = MASS
    charge: float = CHARGE
    H_0: float = 3.5e6 * ONE_EV    # fusion birth energy [J]
    H_low: float = 0.3e6 * ONE_EV  # lower bound for uniform wall-energy sampling [J]
    coulomb_log: float = 17.0
    Te_in_eV: bool = True
    ne0: float = 5e19              # m^-3  (will be uniform profile for now)
    Te0_ev: float = 5e3            # eV    (will be uniform profile for now)

    # ── Tracing ───────────────────────────────────────────────────────────────
    tmax_backward: float = 5e-5    # max backward integration time [s]
    tmax_forward: float = 1e-2     # max forward integration time [s]
    n_interp: int = 16             # interpolant meta-grid pts

    # ── IS forward sample size ────────────────────────────────────────────────
    N_is: int = 10_000             # number of IS forward samples to draw

    # ── Histogram binning (5-D: X, Y, Z, v_par, H) ───────────────────────────
    n_bins_xyz: int = 10           # bins per spatial axis
    n_bins_vpar: int = 8
    n_bins_H: int = 1              # H is always H_0 at birth — keep as 1

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 57

    # ── Support-preserving mixture proposal ───────────────────────────────
    eta_mix: float = 0.05   # pi = (1-eta)*pi_tilde + eta*a

    # ── Diagnostics for birth-space representation ────────────────────────
    n_diag_fusion: int = 50_000
    diag_n_bins_xyz: int = 10
    diag_n_bins_vpar: int = 8
    diag_n_bins_s: int = 20


inp = Inputs()

# =============================================================================
# Output directory
# =============================================================================

script_dir = Path(__file__).resolve().parent
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = script_dir / "outputs_backward_is_mc" / timestamp
out_dir.mkdir(parents=True, exist_ok=True)

proc0_print("=" * 60)
proc0_print(f"Saving outputs to: {out_dir}")
proc0_print(f"Birth energy H_0 = {inp.H_0/ONE_EV/1e6:.3f} MeV")
proc0_print(f"Wall energy range [{inp.H_low/ONE_EV/1e6:.3f}, {inp.H_0/ONE_EV/1e6:.3f}] MeV")


# =============================================================================
# Helpers
# =============================================================================

def wrap_phi(phi: np.ndarray, phi_min: float, phi_max: float) -> np.ndarray:
    period = phi_max - phi_min
    return (phi - phi_min) % period + phi_min


def rphiz_to_xyz(R, phi, Z):
    return R * np.cos(phi), R * np.sin(phi), Z


def xyz_to_rphiz(x, y, z):
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return R, phi, z


def flatten_stz(R, phi, Z):
    """Pack (R, phi, Z) arrays into the flat [R0,phi0,Z0, R1,phi1,Z1,...] layout
    expected by the GPU tracer's stz_init argument."""
    n = len(R)
    out = np.empty(3 * n, dtype=np.float64)
    out[0::3] = R
    out[1::3] = phi
    out[2::3] = Z
    return out


def summarize_stop_codes(stop_codes: np.ndarray) -> dict:
    uniq, counts = np.unique(stop_codes.astype(int), return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq, counts)}


def ne_fun(rphiz):
    return np.full(rphiz.shape[0], inp.ne0, dtype=np.float64)


def Te_fun(rphiz):
    return np.full(rphiz.shape[0], inp.Te0_ev, dtype=np.float64)


def run_backward_drag(
    cell_quad_pts, r_range, phi_range, z_range,
    stz_init, vtang, H_init, tmax, nparticles,
):
    """Wrapper around cartesian_gpu_tracing_backward_drag.
    Returns array of shape (nparticles, 7):
      col 0: elapsed time
      col 1-3: X, Y, Z
      col 4: v_par
      col 5: H
      col 6: stop_code  (0=tmax, 1=wall, 2=energy, 3=invalid)
    """
    speed_ref = float(sqrt(2.0 * inp.H_0 / inp.mass))
    out = cartesian_gpu_tracing_backward_drag(
        cell_quad_pts,
        np.array(r_range, dtype=np.float64),
        np.array(phi_range, dtype=np.float64),
        np.array(z_range, dtype=np.float64),
        np.asarray(stz_init, dtype=np.float64),
        float(inp.mass),
        float(inp.charge),
        speed_ref,
        np.asarray(vtang, dtype=np.float64),
        np.asarray(H_init, dtype=np.float64),
        float(inp.coulomb_log),
        bool(inp.Te_in_eV),
        float(tmax),
        float(inp.tol),
        int(nparticles),
        float(inp.H_0),       # H_stop = H_0 for backward
        True,                  # use_energy_stop
    )
    return np.asarray(out, dtype=np.float64).reshape(nparticles, 7)


def run_forward_drag(
    cell_quad_pts, r_range, phi_range, z_range,
    stz_init, vtang, H_init, tmax, nparticles,
):
    """Wrapper around cartesian_gpu_tracing_drag.
    Returns array of shape (nparticles, 7).
    Stop code 1 = wall hit (lost), 0 = reached tmax (confined).
    """
    speed_ref = float(sqrt(2.0 * inp.H_0 / inp.mass))
    out = cartesian_gpu_tracing_drag(
        cell_quad_pts,
        np.array(r_range, dtype=np.float64),
        np.array(phi_range, dtype=np.float64),
        np.array(z_range, dtype=np.float64),
        np.asarray(stz_init, dtype=np.float64),
        float(inp.mass),
        float(inp.charge),
        speed_ref,
        np.asarray(vtang, dtype=np.float64),
        np.asarray(H_init, dtype=np.float64),
        float(inp.coulomb_log),
        bool(inp.Te_in_eV),
        float(tmax),
        float(inp.tol),
        int(nparticles),
        0.0,   # H_stop unused — no energy stop in forward run
        False, # use_energy_stop = False
    )
    return np.asarray(out, dtype=np.float64).reshape(nparticles, 7)


# new helpers for unbiasedness
def normalize_probabilities(w: np.ndarray, name: str) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError(f"{name} must be 1D")
    if np.any(w < 0.0):
        raise ValueError(f"{name} contains negative entries")
    total = w.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError(f"{name}: weights do not sum to a positive finite value")
    p = w / total
    # final clean renormalization
    p /= p.sum()
    return p


def make_mixture_proposal(
    a_weights: np.ndarray,
    scores: np.ndarray,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build support-preserving proposal:
        pi_tilde_j ∝ a_j * s_j
        pi_mix_j   = (1-eta) * pi_tilde_j + eta * a_j

    Returns:
        pi_tilde, pi_mix
    """
    if not (0.0 < eta <= 1.0):
        raise ValueError(f"eta must satisfy 0 < eta <= 1, got {eta}")

    a_weights = normalize_probabilities(a_weights, "a_weights")
    scores = np.asarray(scores, dtype=np.float64)

    if scores.ndim != 1:
        raise ValueError("scores must be 1D")
    if len(scores) != len(a_weights):
        raise ValueError("scores and a_weights must have same length")
    if np.any(scores < 0.0):
        raise ValueError("scores must be nonnegative")

    # If all scores are zero, just fall back to baseline.
    if np.all(scores == 0.0):
        return a_weights.copy(), a_weights.copy()

    unnorm = a_weights * scores
    pi_tilde = normalize_probabilities(unnorm, "pi_tilde")
    pi_mix = (1.0 - eta) * pi_tilde + eta * a_weights
    pi_mix = normalize_probabilities(pi_mix, "pi_mix")
    return pi_tilde, pi_mix


def _histogram_prob(
    points: np.ndarray,
    edges: list[np.ndarray],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    hist, _ = np.histogramdd(points, bins=edges, weights=weights)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0.0:
        hist /= total
    return hist


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-300) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], eps))))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def birth_distribution_representation_diagnostics(
    fusion_points: np.ndarray,          # columns [X, Y, Z, vpar]
    backward_points: np.ndarray,        # columns [X, Y, Z, vpar]
    fusion_weights: np.ndarray | None = None,
    backward_weights: np.ndarray | None = None,
    n_bins_xyz: int = 10,
    n_bins_vpar: int = 8,
    s_fusion: np.ndarray | None = None,
    s_backward: np.ndarray | None = None,
    n_bins_s: int = 20,
) -> dict:
    """
    Empirical diagnostics for how the backward-recovered cloud covers
    the birth distribution.

    Main outputs:
      - missing_baseline_mass_4d
      - js_divergence_4d
      - optional Boozer-s analogues

    Interpretation:
      This does NOT ask whether the backward cloud should equal p0.
      It asks whether large chunks of baseline birth mass are completely
      uncovered by the discretized backward-informed support.
    """
    fusion_points = np.asarray(fusion_points, dtype=np.float64)
    backward_points = np.asarray(backward_points, dtype=np.float64)

    if fusion_points.ndim != 2 or fusion_points.shape[1] != 4:
        raise ValueError("fusion_points must have shape (N,4)")
    if backward_points.ndim != 2 or backward_points.shape[1] != 4:
        raise ValueError("backward_points must have shape (M,4)")

    n_fusion = fusion_points.shape[0]
    n_backward = backward_points.shape[0]

    if fusion_weights is None:
        fusion_weights = np.full(n_fusion, 1.0 / n_fusion, dtype=np.float64)
    else:
        fusion_weights = normalize_probabilities(fusion_weights, "fusion_weights")

    if backward_weights is None:
        backward_weights = np.full(n_backward, 1.0 / n_backward, dtype=np.float64)
    else:
        backward_weights = normalize_probabilities(backward_weights, "backward_weights")

    mins = np.minimum(fusion_points.min(axis=0), backward_points.min(axis=0))
    maxs = np.maximum(fusion_points.max(axis=0), backward_points.max(axis=0))

    edges = [
        np.linspace(mins[0], maxs[0], n_bins_xyz + 1),
        np.linspace(mins[1], maxs[1], n_bins_xyz + 1),
        np.linspace(mins[2], maxs[2], n_bins_xyz + 1),
        np.linspace(mins[3], maxs[3], n_bins_vpar + 1),
    ]

    P = _histogram_prob(fusion_points, edges, weights=fusion_weights)
    Q = _histogram_prob(backward_points, edges, weights=backward_weights)

    # fraction of empirical fusion-birth mass in bins where backward cloud has zero support
    missing_mass_4d = float(P[Q == 0.0].sum())

    out = {
        "n_fusion_points": int(n_fusion),
        "n_backward_points": int(n_backward),
        "missing_baseline_mass_4d": missing_mass_4d,
        "js_divergence_4d": jensen_shannon_divergence(P.ravel(), Q.ravel()),
        "covered_bins_4d": int(np.sum(Q > 0.0)),
        "total_bins_4d": int(Q.size),
        "covered_fraction_4d": float(np.sum(Q > 0.0) / Q.size),
    }

    if s_fusion is not None and s_backward is not None:
        s_fusion = np.asarray(s_fusion, dtype=np.float64).ravel()
        s_backward = np.asarray(s_backward, dtype=np.float64).ravel()

        s_min = min(float(s_fusion.min()), float(s_backward.min()))
        s_max = max(float(s_fusion.max()), float(s_backward.max()))
        s_edges = [np.linspace(s_min, s_max, n_bins_s + 1)]

        P_s = _histogram_prob(s_fusion[:, None], s_edges, weights=fusion_weights)
        Q_s = _histogram_prob(s_backward[:, None], s_edges, weights=backward_weights)

        out["missing_baseline_mass_s"] = float(P_s[Q_s == 0.0].sum())
        out["js_divergence_s"] = jensen_shannon_divergence(P_s.ravel(), Q_s.ravel())
        out["covered_bins_s"] = int(np.sum(Q_s > 0.0))
        out["total_bins_s"] = int(Q_s.size)
        out["covered_fraction_s"] = float(np.sum(Q_s > 0.0) / Q_s.size)

    return out


def nearest_neighbor_coverage_diagnostics(
    fusion_points: np.ndarray,          # columns [X, Y, Z, vpar]
    backward_points: np.ndarray,        # columns [X, Y, Z, vpar]
    fusion_weights: np.ndarray | None = None,
) -> dict:
    """
    Distance from each fusion-birth marker to the nearest backward-recovered point,
    measured in coordinates normalized by the fusion-cloud stddev.

    This is a geometric coverage diagnostic, not a probability metric.
    """
    fusion_points = np.asarray(fusion_points, dtype=np.float64)
    backward_points = np.asarray(backward_points, dtype=np.float64)

    if fusion_points.ndim != 2 or fusion_points.shape[1] != 4:
        raise ValueError("fusion_points must have shape (N,4)")
    if backward_points.ndim != 2 or backward_points.shape[1] != 4:
        raise ValueError("backward_points must have shape (M,4)")

    n_fusion = fusion_points.shape[0]
    if fusion_weights is None:
        fusion_weights = np.full(n_fusion, 1.0 / n_fusion, dtype=np.float64)
    else:
        fusion_weights = normalize_probabilities(fusion_weights, "fusion_weights")

    scale = fusion_points.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)

    F = fusion_points / scale
    B = backward_points / scale

    # O(NM), okay for debug sizes
    d2 = (
        np.sum(F[:, None, :] ** 2, axis=2)
        + np.sum(B[None, :, :] ** 2, axis=2)
        - 2.0 * (F @ B.T)
    )
    d2 = np.maximum(d2, 0.0)
    dmin = np.sqrt(np.min(d2, axis=1))

    order = np.argsort(dmin)
    d_sorted = dmin[order]
    w_sorted = fusion_weights[order]
    cdf = np.cumsum(w_sorted)

    def weighted_quantile(q: float) -> float:
        idx = np.searchsorted(cdf, q, side="left")
        idx = min(idx, len(d_sorted) - 1)
        return float(d_sorted[idx])

    return {
        "nn_dist_mean": float(np.sum(fusion_weights * dmin)),
        "nn_dist_q50": weighted_quantile(0.50),
        "nn_dist_q90": weighted_quantile(0.90),
        "nn_dist_q99": weighted_quantile(0.99),
        "nn_dist_max": float(dmin.max()),
    }

# =============================================================================
# Fusion birth distribution  p_0(x)  — in Boozer s
# =============================================================================

def sigmav(T_keV: float) -> float:
    """D-T rate coefficient (Bader et al. 2021)."""
    if T_keV > 0:
        return T_keV ** (-2.0 / 3.0) * np.exp(-19.94 * T_keV ** (-1.0 / 3.0))
    return 0.0


def fusion_reactivity(s: np.ndarray) -> np.ndarray:
    """Unnormalised fusion birth probability as a function of Boozer s."""
    nD = 1.0 - s ** 5
    T = 11.5 * (1.0 - s)
    sv = np.array([sigmav(float(t)) for t in T])
    return nD ** 2 * sv   # nD == nT assumed


# =============================================================================
# Build magnetic field and GPU interpolant
# =============================================================================

proc0_print("\nBuilding magnetic field...")

# FPP coils loading from 2_tracing_gpu
# ── 1. Load coils ─────────────────────────────────────────────────────────────
# load_coils_from_makegrid_file returns ncoils × nfp coils, omitting
# stellarator-symmetric images.  coils_via_symmetries reconstructs the full
# set with stellarator symmetry.

all_coils     = load_coils_from_makegrid_file(str(inp.coil_file), order=inp.coil_order)
base_curves   = [all_coils[i].curve for i in range(inp.ncoils)]
base_currents = [Current(inp.current) for _ in range(inp.ncoils)]

coils  = coils_via_symmetries(base_curves, base_currents, inp.nfp, stellsym=True)
curves = [c.curve for c in coils]
bs     = BiotSavart(coils)


# ── 2. Load plasma boundary from VMEC input ──────────────────────────────────
# The VMEC boundary represents first wall in this example

s_input = SurfaceRZFourier.from_vmec_input(
    str(inp.vmec_input_file), range="full torus",
    nphi=inp.nphi_surf, ntheta=inp.ntheta_surf,
)


# ── 3. B·n check ─────────────────────────────────────────────────────────────
# |B·n̂|/|B| should be small on VMEC's LCFS for precise coils.
# A large value means the coil field does not reproduce the target equilibrium.

bs.set_points(s_input.gamma().reshape((-1, 3)))
B   = bs.B().reshape((inp.nphi_surf, inp.ntheta_surf, 3))
BN  = np.sum(B * s_input.unitnormal(), axis=2)
rel = np.abs(BN) / np.linalg.norm(B, axis=2)
print(f"B·n check: mean |B·n|/|B| = {rel.mean():.2e},  max = {rel.max():.2e}")


# ── 4. Save geometry to VTK ───────────────────────────────────────────────────

bs.set_points(s_input.gamma().reshape((-1, 3)))
B_on_surf = bs.B().reshape((inp.nphi_surf, inp.ntheta_surf, 3))
B_N       = np.sum(B_on_surf * s_input.unitnormal(), axis=2)
absB      = np.linalg.norm(B_on_surf, axis=2)

curves_to_vtk(curves, OUT_DIR + "coils_LPQH", close=True)
s_input.to_vtk(OUT_DIR + "surface_LPQH", extra_data={
    "B_N":            B_N[:, :, None],
    "abs_B_N_over_B": (np.abs(B_N) / absB)[:, :, None],
})
print(f"VTK files written to {OUT_DIR}")


# ── 5. Build interpolated field on a cylindrical grid ────────────────────────

sc_particle = SurfaceClassifier(s_input, h=inp.sc_h, p=inp.sc_p)

rs    = np.linalg.norm(s_input.gamma()[:, :, 0:2], axis=2)
z_max = np.max(np.abs(s_input.gamma()[:, :, 2]))

rrange   = (np.min(rs), np.max(rs), inp.n_r)
phirange = (0, 2 * np.pi / inp.nfp, inp.n_phi)
phi_min = phirange[0]
phi_max = phirange[1]
zrange   = (0, z_max, inp.n_z)

bsh = InterpolatedField(
    bs, inp.degree, rrange, phirange, zrange, True, nfp=inp.nfp, stellsym=True
)
print(f"Interpolation grid: {inp.n_r}(R) × {inp.n_phi}(φ) × {inp.n_z}(Z)")
print("  error in B:       ", bsh.estimate_error_B(1000))
print("  error in GradAbsB:", bsh.estimate_error_GradAbsB(1000))


# ── 6. Build the GPU interpolant ──────────────────────────────────────────────
t0 = time.time()
r_range, phi_range, z_range, cell_quad_pts = cartesian_interpolant_drag(
    field=bsh,
    sc_particle=sc_particle,
    ne_fun=ne_fun,
    Te_fun=Te_fun,
    nfp=inp.nfp,
    n_metagrid_pts=inp.n_r,
)
print(f"GPU interpolant built in {time.time()-t0:.1f}s")



# =============================================================================
# STEP 1 — Load wall states from surface IC file
# =============================================================================

proc0_print("\n--- STEP 1: Loading wall states ---")

wall_ic = np.loadtxt(inp.wall_ic_file, comments="#")
# Columns: R  phi  Z  (no vpar in surface distribution file)
if wall_ic.shape[1] >= 4:
    # file has vpar column — ignore it, we resample pitch below
    R_wall_all = wall_ic[:, 0]
    phi_wall_all = wall_ic[:, 1]
    Z_wall_all = wall_ic[:, 2]
else:
    R_wall_all = wall_ic[:, 0]
    phi_wall_all = wall_ic[:, 1]
    Z_wall_all = wall_ic[:, 2]

n_available = len(R_wall_all)
n_wall = min(inp.n_wall, n_available)
proc0_print(f"  Using {n_wall}/{n_available} wall states from {inp.wall_ic_file}")

rng = np.random.default_rng(inp.seed)
idx_wall = rng.choice(n_available, size=n_wall, replace=False)

R_wall = R_wall_all[idx_wall]
phi_wall = phi_wall_all[idx_wall]
Z_wall = Z_wall_all[idx_wall]

# Filter to particles inside the LCFS (surface IC may include points just
# outside due to Boozer->cylindrical conversion imprecision)
sd = sc_particle.evaluate_rphiz(
    np.column_stack([R_wall, phi_wall, Z_wall])
).ravel()
inside = sd >= 0
n_outside = int(np.sum(~inside))
if n_outside > 0:
    proc0_print(f"  Removing {n_outside} wall points outside LCFS")
    R_wall = R_wall[inside]
    phi_wall = phi_wall[inside]
    Z_wall = Z_wall[inside]
    n_wall = int(np.sum(inside))

# Wrap phi into [phi_min, phi_max]
phi_wall = wrap_phi(phi_wall, phi_min, phi_max)

# Sample uniform pitch  lambda = v_par / v  in [-1, 1]
lam_wall = rng.uniform(-1.0, 1.0, size=n_wall)

# Sample uniform wall energy in [H_low, H_0]
H_wall = rng.uniform(inp.H_low, inp.H_0, size=n_wall)

# Compute v_par from pitch and per-particle total speed
v_total_wall = np.sqrt(2.0 * H_wall / inp.mass)
vtang_wall = lam_wall * v_total_wall

proc0_print(f"  Wall states ready: {n_wall}")


# =============================================================================
# STEP 2 — Backward tracing
# =============================================================================

proc0_print("\n--- STEP 2: Backward tracing ---")

stz_wall = flatten_stz(R_wall, phi_wall, Z_wall)

t0 = time.time()
bwd_results = run_backward_drag(
    cell_quad_pts, r_range, phi_range, z_range,
    stz_wall, vtang_wall, H_wall,
    inp.tmax_backward, n_wall,
)
proc0_print(f"  Done in {time.time() - t0:.2f}s")

np.save(out_dir / "backward_results.npy", bwd_results)

stop_codes_bwd = bwd_results[:, 6].astype(int)
proc0_print(f"  Stop codes: {summarize_stop_codes(stop_codes_bwd)}")


# =============================================================================
# STEP 3 — Keep successful backward traces (stop_code == 2)
# =============================================================================

proc0_print("\n--- STEP 3: Filtering successful backward traces ---")

hit_birth = (stop_codes_bwd == 2)
n_success = int(np.sum(hit_birth))
proc0_print(f"  Successful (stop_code=2): {n_success}/{n_wall} "
            f"({100.*n_success/n_wall:.1f}%)")

if n_success == 0:
    raise RuntimeError(
        "No successful backward traces. "
        "Try increasing tmax_backward, decreasing H_low, or increasing n_wall."
    )

# Birth endpoints in Cartesian + state space
X_birth = bwd_results[hit_birth, 1]
Y_birth = bwd_results[hit_birth, 2]
Z_birth = bwd_results[hit_birth, 3]
vpar_birth = bwd_results[hit_birth, 4]
H_birth = bwd_results[hit_birth, 5]   # should be ~= H_0 for all

# Also store cylindrical for convenience / Boozer evaluation
R_birth, phi_birth, Zb = xyz_to_rphiz(X_birth, Y_birth, Z_birth)
phi_birth_wrapped = wrap_phi(phi_birth, phi_min, phi_max)

birth_xyz_state = np.column_stack([X_birth, Y_birth, Z_birth, vpar_birth, H_birth])
birth_rphiz_state = np.column_stack([R_birth, phi_birth_wrapped, Zb, vpar_birth, H_birth])

np.save(out_dir / "birth_endpoints.npy", birth_xyz_state)
np.save(out_dir / "birth_endpoints_rphiz.npy", birth_rphiz_state)
proc0_print(f"  Saved birth endpoints ({n_success} points)")


# =============================================================================
# STEP 4 — Score via 5-D histogram in (X, Y, Z, v_par, H) birth space
# =============================================================================

proc0_print("\n--- STEP 4: Computing histogram scores ---")

# Build bin edges for each dimension.
# H is always ~H_0 so one bin is enough there; we keep it for generality.
bins_X    = np.linspace(X_birth.min(), X_birth.max(), inp.n_bins_xyz + 1)
bins_Y    = np.linspace(Y_birth.min(), Y_birth.max(), inp.n_bins_xyz + 1)
bins_Z    = np.linspace(Z_birth.min(), Z_birth.max(), inp.n_bins_xyz + 1)
bins_vpar = np.linspace(vpar_birth.min(), vpar_birth.max(), inp.n_bins_vpar + 1)
bins_H    = np.array([H_birth.min() - 1.0, H_birth.max() + 1.0])  # single bin

data_5d = np.column_stack([X_birth, Y_birth, Z_birth, vpar_birth, H_birth])
hist, edges = np.histogramdd(
    data_5d,
    bins=[bins_X, bins_Y, bins_Z, bins_vpar, bins_H],
)

# For each birth point find which bin it falls in and assign that bin's count
# as the score.  We clamp indices to handle edge points.
def bin_index(vals, edges):
    idx = np.searchsorted(edges, vals, side="right") - 1
    return np.clip(idx, 0, len(edges) - 2)

iX    = bin_index(X_birth,    bins_X)
iY    = bin_index(Y_birth,    bins_Y)
iZ    = bin_index(Z_birth,    bins_Z)
ivpar = bin_index(vpar_birth, bins_vpar)
iH    = np.zeros(n_success, dtype=int)   # single H bin

scores = hist[iX, iY, iZ, ivpar, iH].astype(np.float64)

# Normalise scores to [0, 1] to keep numerics clean (doesn't change pi shape)
scores /= scores.max()

np.save(out_dir / "scores.npy", scores)
proc0_print(f"  Score range: [{scores.min():.3e}, {scores.max():.3e}]")
proc0_print(f"  Non-zero score bins: {int(np.sum(hist > 0))} / {hist.size}")


# =============================================================================
# STEP 5 — Evaluate p_0(x_j) at birth endpoints (fusion reactivity in Boozer s)
# =============================================================================

proc0_print("\n--- STEP 5: Evaluating p_0 at birth endpoints ---")

# Convert birth (R, phi, Z) to Boozer s via the BOOZ_XFORM field.
# We use firm3d's cylindrical_to_boozer if available, otherwise approximate
# with the signed-distance-based s estimate.
try:
    bri = BoozerRadialInterpolant(
        inp.boozmn_file, inp.radial_order, no_K=True
    )
    boozer_field = InterpolatedBoozerField(
        bri, inp.spline_degree,
        ns_interp=inp.resolution,
        ntheta_interp=inp.resolution,
        nzeta_interp=inp.resolution,
    )
    rphiz_birth = np.column_stack([R_birth, phi_birth, Zb])
    boozer_coords = cylindrical_to_boozer(boozer_field, rphiz_birth)
    s_birth = boozer_coords[:, 0]
    proc0_print("  Converted birth points to Boozer s via BOOZ_XFORM.")
except Exception as exc:
    proc0_print(f"  WARNING: cylindrical_to_boozer failed ({exc}).")
    exit

np.save(out_dir / "birth_endpoints_boozer.npy", s_birth)

p0_birth = fusion_reactivity(s_birth)
# Zero out any negative values from extrapolation
p0_birth = np.maximum(p0_birth, 0.0)

np.save(out_dir / "p0_at_births.npy", p0_birth)
proc0_print(f"  p_0 range: [{p0_birth.min():.3e}, {p0_birth.max():.3e}]")


# =============================================================================
# STEP 6 — Mixed proposal pi = (1-eta) pi_tilde + eta a
# =============================================================================

proc0_print("\n--- STEP 6: Computing mixed biased proposal pi ---")

# At this stage, a_weights is still the cloud-restricted discrete reference law:
# a_j ∝ p0_birth(x_j) over the successful backward endpoints only.
# The eta-mix fixes support on this discrete state space.
# So it does not by itself fix any deeper mismatch between this cloud and the full p0.

a_weights = normalize_probabilities(p0_birth, "a_weights")

pi_tilde, pi_weights = make_mixture_proposal(
    a_weights=a_weights,
    scores=scores,
    eta=inp.eta_mix,
)

# Useful global diagnostics
is_weights_all = a_weights / pi_weights

np.save(out_dir / "a_weights.npy", a_weights)
np.save(out_dir / "pi_tilde.npy", pi_tilde)
np.save(out_dir / "pi_weights.npy", pi_weights)
np.save(out_dir / "all_importance_weights.npy", is_weights_all)

proc0_print(f"  eta_mix                  = {inp.eta_mix:.3e}")
proc0_print(f"  pure proposal support    = {int(np.sum(pi_tilde > 0))}/{n_success}")
proc0_print(f"  mixed proposal support   = {int(np.sum(pi_weights > 0))}/{n_success}")
proc0_print(f"  max IS weight            = {is_weights_all.max():.6e}")
proc0_print(f"  mean IS weight           = {is_weights_all.mean():.6e}")
proc0_print(f"  bound 1/eta             = {1.0 / inp.eta_mix:.6e}")


# =============================================================================
# STEP 6.5 — Diagnostics: how badly does the backward cloud represent birth space?
# =============================================================================

proc0_print("\n--- STEP 6.5: Birth-distribution representation diagnostics ---")

fusion_ic_diag = np.loadtxt(inp.fusion_ic_file, comments="#")
n_fus_available = len(fusion_ic_diag)
n_diag = min(inp.n_diag_fusion, n_fus_available)

idx_diag = rng.choice(n_fus_available, size=n_diag, replace=False)

R_fus_diag = fusion_ic_diag[idx_diag, 0]
phi_fus_diag = wrap_phi(fusion_ic_diag[idx_diag, 1], phi_min, phi_max)
Z_fus_diag = fusion_ic_diag[idx_diag, 2]
vpar_fus_diag = fusion_ic_diag[idx_diag, 3].astype(np.float64)

# Filter to inside-LCFS
sd_fus_diag = sc_particle.evaluate_rphiz(
    np.column_stack([R_fus_diag, phi_fus_diag, Z_fus_diag])
).ravel()
inside_fus_diag = sd_fus_diag >= 0.0

R_fus_diag = R_fus_diag[inside_fus_diag]
phi_fus_diag = phi_fus_diag[inside_fus_diag]
Z_fus_diag = Z_fus_diag[inside_fus_diag]
vpar_fus_diag = vpar_fus_diag[inside_fus_diag]

X_fus_diag = R_fus_diag * np.cos(phi_fus_diag)
Y_fus_diag = R_fus_diag * np.sin(phi_fus_diag)

fusion_diag_points = np.column_stack([X_fus_diag, Y_fus_diag, Z_fus_diag, vpar_fus_diag])
backward_diag_points = np.column_stack([X_birth, Y_birth, Z_birth, vpar_birth])

# Optional Boozer-s on fusion diagnostic points
try:
    rphiz_fus_diag = np.column_stack([R_fus_diag, phi_fus_diag, Z_fus_diag])
    boozer_fus_diag = cylindrical_to_boozer(boozer_field, rphiz_fus_diag)
    s_fus_diag = boozer_fus_diag[:, 0]
except Exception as exc:
    proc0_print(f"  WARNING: cylindrical_to_boozer failed on fusion diagnostic sample ({exc})")
    s_fus_diag = None

birth_diag = birth_distribution_representation_diagnostics(
    fusion_points=fusion_diag_points,
    backward_points=backward_diag_points,
    fusion_weights=None,      # equal-weight empirical birth sample
    backward_weights=None,    # equal-weight empirical backward cloud
    n_bins_xyz=inp.diag_n_bins_xyz,
    n_bins_vpar=inp.diag_n_bins_vpar,
    s_fusion=s_fus_diag,
    s_backward=s_birth,
    n_bins_s=inp.diag_n_bins_s,
)

nn_diag = nearest_neighbor_coverage_diagnostics(
    fusion_points=fusion_diag_points,
    backward_points=backward_diag_points,
)

for k, v in birth_diag.items():
    proc0_print(f"  {k:28s} = {v}")
for k, v in nn_diag.items():
    proc0_print(f"  {k:28s} = {v}")

np.save(out_dir / "fusion_diag_points.npy", fusion_diag_points)
np.save(out_dir / "backward_diag_points.npy", backward_diag_points)

if s_fus_diag is not None:
    np.save(out_dir / "fusion_diag_s.npy", s_fus_diag)
np.save(out_dir / "backward_diag_s.npy", s_birth)

with open(out_dir / "birth_representation_diagnostics.csv", "w", newline="") as f:
    diag_row = {**birth_diag, **nn_diag}
    writer = csv.DictWriter(f, fieldnames=list(diag_row.keys()))
    writer.writeheader()
    writer.writerow(diag_row)


# =============================================================================
# STEP 7 — IS forward: resample birth states from pi, trace forward
# =============================================================================

proc0_print("\n--- STEP 7: IS forward tracing ---")

N_is = min(inp.N_is, n_success)   # can't draw more than available markers
sampled_idx = rng.choice(n_success, size=N_is, p=pi_weights)

X_is  = X_birth[sampled_idx]
Y_is  = Y_birth[sampled_idx]
Z_is  = Z_birth[sampled_idx]
vpar_is = vpar_birth[sampled_idx]
H_is  = np.full(N_is, inp.H_0, dtype=np.float64)

# Convert Cartesian birth coords back to cylindrical for tracer
R_is  = np.sqrt(X_is**2 + Y_is**2)
phi_is = wrap_phi(np.arctan2(Y_is, X_is), phi_min, phi_max)
stz_is = flatten_stz(R_is, phi_is, Z_is)

t0 = time.time()
fwd_is_results = run_forward_drag(
    cell_quad_pts, r_range, phi_range, z_range,
    stz_is, vpar_is, H_is,
    inp.tmax_forward, N_is,
)
proc0_print(f"  IS forward done in {time.time() - t0:.2f}s")

np.save(out_dir / "forward_is_results.npy", fwd_is_results)

stop_codes_is = fwd_is_results[:, 6].astype(int)
proc0_print(f"  IS forward stop codes: {summarize_stop_codes(stop_codes_is)}")

# Wall-hit indicator A: stop_code == 1
A_is = (stop_codes_is == 1).astype(np.float64)

# IS weights  w_j = a_j / pi_j  for the sampled indices
# We use the normalised forms so the ratio is dimensionless.
a_sampled  = a_weights[sampled_idx]
pi_sampled = pi_weights[sampled_idx]
# Guard against zero pi (shouldn't happen since we sampled from pi, but be safe)
safe_pi = np.where(pi_sampled > 0, pi_sampled, np.inf)
IS_weights = a_sampled / safe_pi

Q_hat_IS = float(np.mean(A_is * IS_weights))
Q_hat_IS_std = float(np.std(A_is * IS_weights) / sqrt(N_is))

proc0_print(f"  Q_hat_IS  = {Q_hat_IS:.6e}  ±  {Q_hat_IS_std:.3e}  (N={N_is})")


# =============================================================================
# STEP 8 — Naive baseline forward tracing from fusion distribution
# =============================================================================

proc0_print("\n--- STEP 8: Baseline forward tracing ---")

fusion_ic = np.loadtxt(inp.fusion_ic_file, comments="#")
# Columns: R  phi  Z  vpar
n_fus_available = len(fusion_ic)
n_base = min(inp.n_baseline, n_fus_available)

idx_base = rng.choice(n_fus_available, size=n_base, replace=False)
R_base   = fusion_ic[idx_base, 0]
phi_base = fusion_ic[idx_base, 1]
Z_base   = fusion_ic[idx_base, 2]
vpar_base = fusion_ic[idx_base, 3].astype(np.float64)
H_base   = np.full(n_base, inp.H_0, dtype=np.float64)

# Filter to particles inside LCFS
sd_base = sc_particle.evaluate_rphiz(
    np.column_stack([R_base, phi_base, Z_base])
).ravel()
inside_base = sd_base >= 0
n_out_base = int(np.sum(~inside_base))
if n_out_base > 0:
    proc0_print(f"  Removing {n_out_base} baseline particles outside LCFS")
    R_base    = R_base[inside_base]
    phi_base  = phi_base[inside_base]
    Z_base    = Z_base[inside_base]
    vpar_base = vpar_base[inside_base]
    H_base    = H_base[inside_base]
    n_base    = int(inside_base.sum())

phi_base = wrap_phi(phi_base, phi_min, phi_max)
stz_base = flatten_stz(R_base, phi_base, Z_base)

t0 = time.time()
fwd_base_results = run_forward_drag(
    cell_quad_pts, r_range, phi_range, z_range,
    stz_base, vpar_base, H_base,
    inp.tmax_forward, n_base,
)
proc0_print(f"  Baseline forward done in {time.time() - t0:.2f}s")

np.save(out_dir / "forward_baseline_results.npy", fwd_base_results)

stop_codes_base = fwd_base_results[:, 6].astype(int)
proc0_print(f"  Baseline stop codes: {summarize_stop_codes(stop_codes_base)}")

A_base = (stop_codes_base == 1).astype(np.float64)
Q_hat_base = float(np.mean(A_base))
Q_hat_base_std = float(np.std(A_base) / sqrt(n_base))

proc0_print(f"  Q_hat_baseline = {Q_hat_base:.6e}  ±  {Q_hat_base_std:.3e}  (N={n_base})")


# =============================================================================
# Summary
# =============================================================================

proc0_print("\n--- Summary ---")
proc0_print(f"  n_wall              = {n_wall}")
proc0_print(f"  n_backward_success  = {n_success}")
proc0_print(f"  backward_success_%  = {100.*n_success/n_wall:.1f}")
proc0_print(f"  N_IS_forward        = {N_is}")
proc0_print(f"  N_baseline_forward  = {n_base}")
proc0_print(f"  Q_hat_IS            = {Q_hat_IS:.6e} ± {Q_hat_IS_std:.3e}")
proc0_print(f"  Q_hat_baseline      = {Q_hat_base:.6e} ± {Q_hat_base_std:.3e}")

summary = {
    "n_wall": n_wall,
    "n_backward_success": n_success,
    "backward_success_pct": 100.0 * n_success / n_wall,
    "N_IS_forward": N_is,
    "N_baseline_forward": n_base,
    "Q_hat_IS": Q_hat_IS,
    "Q_hat_IS_std": Q_hat_IS_std,
    "Q_hat_baseline": Q_hat_base,
    "Q_hat_baseline_std": Q_hat_base_std,
    "H_0_MeV": inp.H_0 / ONE_EV / 1e6,
    "H_low_MeV": inp.H_low / ONE_EV / 1e6,
    "tmax_backward": inp.tmax_backward,
    "tmax_forward": inp.tmax_forward,
    "n_bins_xyz": inp.n_bins_xyz,
    "n_bins_vpar": inp.n_bins_vpar,
}

with open(out_dir / "summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)

proc0_print(f"\nDone. All outputs in: {out_dir}")