"""Build the perturbed Biot-Savart field used by the three MC-comparison
workflows (forward MC, uniform-s IS, backward-informed IS).

The perturbation model is identical to backward_tracing_drag/4_robustness/1_trace_perturbed.py:

  * Layer 1 — same Gaussian sample applied to every base
    curve before stellarator-symmetry expansion.
  * Layer 2 — an independent sample applied to every coil
    after symmetry expansion.

Both layers are seeded by ``perturbation_id`` via ``PCG64DXSM`` so the same
``perturbation_id`` reproduces the same perturbed field across the three
workflows.
"""
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from numpy.random import PCG64DXSM, Generator

from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    InterpolatedField,
    SurfaceClassifier,
    coils_via_symmetries,
)
from simsopt.field.coil import load_coils_from_makegrid_file
from simsopt.geo import (
    CurvePerturbed,
    GaussianSampler,
    PerturbationSample,
    SurfaceRZFourier,
)

from firm3d.util.gpu_utils import cartesian_interpolant_drag


@dataclass
class FieldConfig:
    coil_file: Path
    vmec_input_file: Path
    # Equilibrium / coils
    nfp: int = 4
    ncoils: int = 5
    current: float = 1.27797548115612e7
    coil_order: int = 20
    sigma: float = 1e-3
    length: float = 0.5
    # Interpolation grid
    n_r: int = 64
    n_phi: int = 128
    n_z: int = 64
    degree: int = 3
    nphi_surf: int = 128
    ntheta_surf: int = 64
    # SurfaceClassifier
    sc_h: float = 0.05
    sc_p: int = 2


def _build_perturbed_coils(cfg, perturbation_id):
    all_coils     = load_coils_from_makegrid_file(str(cfg.coil_file),
                                                  order=cfg.coil_order)
    base_curves   = [all_coils[i].curve for i in range(cfg.ncoils)]
    base_currents = [Current(cfg.current) for _ in range(cfg.ncoils)]

    if perturbation_id == 0:
        coils = coils_via_symmetries(base_curves, base_currents,
                                     cfg.nfp, stellsym=True)
        print("Using exact (unperturbed) coils.")
        return coils

    rg = Generator(PCG64DXSM(perturbation_id))
    sampler = GaussianSampler(
        base_curves[0].quadpoints, cfg.sigma, cfg.length, n_derivs=1,
    )
    # Layer 1
    base_curves_pert = [
        CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg))
        for c in base_curves
    ]
    coils_sym = coils_via_symmetries(
        base_curves_pert, base_currents, cfg.nfp, stellsym=True,
    )
    # Layer 2
    coils = [
        Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)),
             c.current)
        for c in coils_sym
    ]
    print(f"Gaussian perturbation applied: sigma={cfg.sigma:.1e} m, "
          f"L={cfg.length:.2f} m, seed={perturbation_id}")
    return coils


def build_perturbed_field(cfg, perturbation_id, ne_fun, Te_fun):
    """Construct the Biot-Savart field, LCFS classifier, and GPU drag
    interpolant for the requested perturbation.

    Returns a dict containing every object the three workflows need:
        bs, coils, curves, s_input, sc_particle, bsh,
        r_range, phi_range, z_range, cell_quad_pts,
        phi_min, phi_max, bn_stats.
    """
    coils  = _build_perturbed_coils(cfg, perturbation_id)
    curves = [c.curve for c in coils]
    bs     = BiotSavart(coils)

    s_input = SurfaceRZFourier.from_vmec_input(
        str(cfg.vmec_input_file), range="full torus",
        nphi=cfg.nphi_surf, ntheta=cfg.ntheta_surf,
    )

    # B·n diagnostic on the LCFS
    bs.set_points(s_input.gamma().reshape((-1, 3)))
    B  = bs.B().reshape((cfg.nphi_surf, cfg.ntheta_surf, 3))
    BN = np.sum(B * s_input.unitnormal(), axis=2)
    rel = np.abs(BN) / np.linalg.norm(B, axis=2)
    print(f"B*n check: mean |B*n|/|B| = {rel.mean():.4e}, max = {rel.max():.4e}")
    bn_stats = np.array([float(rel.mean()), float(rel.max())])

    sc_particle = SurfaceClassifier(s_input, h=cfg.sc_h, p=cfg.sc_p)

    rs    = np.linalg.norm(s_input.gamma()[:, :, 0:2], axis=2)
    z_max = float(np.max(np.abs(s_input.gamma()[:, :, 2])))

    rrange   = (float(np.min(rs)), float(np.max(rs)), cfg.n_r)
    phirange = (0.0, 2.0 * np.pi / cfg.nfp, cfg.n_phi)
    zrange   = (0.0, z_max, cfg.n_z)

    bsh = InterpolatedField(
        bs, cfg.degree, rrange, phirange, zrange, True,
        nfp=cfg.nfp, stellsym=True,
    )

    t0 = time.time()
    r_range, phi_range, z_range, cell_quad_pts = cartesian_interpolant_drag(
        field=bsh, sc_particle=sc_particle,
        ne_fun=ne_fun, Te_fun=Te_fun,
        nfp=cfg.nfp, n_metagrid_pts=cfg.n_r,
    )
    print(f"GPU drag interpolant built in {time.time() - t0:.1f}s")

    return {
        "bs": bs, "coils": coils, "curves": curves,
        "s_input": s_input, "sc_particle": sc_particle, "bsh": bsh,
        "r_range": r_range, "phi_range": phi_range, "z_range": z_range,
        "cell_quad_pts": cell_quad_pts,
        "phi_min": phirange[0], "phi_max": phirange[1],
        "bn_stats": bn_stats,
    }


def wrap_phi(phi, phi_min, phi_max):
    period = phi_max - phi_min
    return (phi - phi_min) % period + phi_min


def flatten_stz(R, phi, Z):
    n = len(R)
    out = np.empty(3 * n, dtype=np.float64)
    out[0::3] = R
    out[1::3] = phi
    out[2::3] = Z
    return out