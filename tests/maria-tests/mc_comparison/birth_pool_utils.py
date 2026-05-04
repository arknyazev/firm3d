"""Load the fusion birth pool, filter it against the LCFS, and resolve its
Boozer ``s`` coordinate.

The three MC-comparison workflows share a single "valid fusion birth pool":

    1. Read ``fusion_ic_file`` columns (R, phi, Z, vpar).
    2. Optionally restrict to the first ``n_particles`` rows (so methods 1/2/3
       all see the same pool even when ``n_particles`` is small).
    3. Drop rows outside the VMEC LCFS (same ``SurfaceClassifier`` used by
       ``4_robustness/1_trace_perturbed.py``).
    4. Attach Boozer ``s`` values.  If a pre-computed ``fusion_boozer_file``
       exists alongside ``fusion_ic_file`` we use those directly (they were
       produced by the same Boozer interpolant that we would otherwise call);
       otherwise we invert (R, phi, Z) with a chunked, recursive fallback that
       handles isolated conversion failures without poisoning whole chunks.
    5. Drop any remaining rows whose Boozer ``s`` is NaN / not in [0, 1].

The resulting pool is deterministic for a given ``(fusion_ic_file,
n_particles, VMEC input, boozmn.nc)``, so all three methods operate on the
identical pool indexing and hence on the same target measure Q.
"""
import time
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.coordinates import cylindrical_to_boozer


def load_fusion_pool(fusion_ic_file, n_particles=None,
                     fusion_boozer_file=None):
    """Load the raw fusion IC file and (optionally) its pre-computed Boozer
    counterpart.  Returns a dict with keys R, phi, Z, vpar and — if the
    Boozer file is present and aligned — s_pre, theta_pre, zeta_pre."""
    cyl = np.loadtxt(str(fusion_ic_file), comments="#")
    n_avail = len(cyl)
    if n_particles is not None and 0 < n_particles < n_avail:
        cyl = cyl[:n_particles]
    out = {
        "R":    cyl[:, 0].astype(np.float64),
        "phi":  cyl[:, 1].astype(np.float64),
        "Z":    cyl[:, 2].astype(np.float64),
        "vpar": cyl[:, 3].astype(np.float64),
    }
    if fusion_boozer_file is not None:
        bz_path = Path(fusion_boozer_file)
        if bz_path.exists():
            bz = np.loadtxt(str(bz_path), comments="#")
            if n_particles is not None and 0 < n_particles < len(bz):
                bz = bz[:n_particles]
            if len(bz) != len(cyl):
                raise RuntimeError(
                    f"fusion cyl/boozer IC files disagree: "
                    f"len(cyl)={len(cyl)} vs len(boozer)={len(bz)}"
                )
            out["s_pre"]     = bz[:, 0].astype(np.float64)
            out["theta_pre"] = bz[:, 1].astype(np.float64)
            out["zeta_pre"]  = bz[:, 2].astype(np.float64)
    return out


def subset_pool(pool, mask):
    return {k: (v[mask] if isinstance(v, np.ndarray) else v)
            for k, v in pool.items()}


def filter_inside_lcfs(pool, sc_particle):
    """Return a boolean mask of pool rows inside (signed distance >= 0) the
    LCFS represented by ``sc_particle``."""
    rphiz = np.column_stack([pool["R"], pool["phi"], pool["Z"]])
    sd = sc_particle.evaluate_rphiz(rphiz).ravel()
    return sd >= 0


def build_boozer_interpolant(boozmn_file, radial_order=3, boozer_degree=3,
                             boozer_res=48):
    bri = BoozerRadialInterpolant(str(boozmn_file), radial_order, no_K=True)
    bf = InterpolatedBoozerField(
        bri, boozer_degree,
        ns_interp=boozer_res,
        ntheta_interp=boozer_res,
        nzeta_interp=boozer_res,
    )
    # Keep bri alive for as long as bf lives.  InterpolatedBoozerField does
    # not hold a strong Python reference to its source BoozerRadialInterpolant
    # (only a C++ pointer), so without this attach the Python `bri` wrapper
    # gets garbage-collected when this function returns and subsequent calls
    # like bf.R() fall back to the base-class `_R_impl` stub
    # ("_R_impl was not implemented").
    bf._bri = bri
    return bf


def _cyl_to_boozer_chunked(boozer_field, rphiz, chunk=1_000, progress=True,
                           transformer=None):
    """Convert (R, phi, Z) to (s, theta, zeta) with chunked recursive
    subdivision on conversion failure, so a single bad point only pays its own
    cost rather than poisoning a whole chunk.

    If ``transformer`` (a ``BoozerCoordinateTransformer``) is provided, its
    method is called directly so the underlying coordinate grid is built
    once and reused across calls.  Otherwise the module-level
    ``cylindrical_to_boozer`` is used and a fresh grid is built per call.
    """
    n = len(rphiz)
    out = np.full_like(rphiz, np.nan)
    failed = 0
    idx_all = np.arange(n)

    def _recurse(pts, idx):
        nonlocal failed
        if len(pts) == 0:
            return
        try:
            if transformer is not None:
                out[idx] = transformer.cylindrical_to_boozer(pts)
            else:
                out[idx] = cylindrical_to_boozer(boozer_field, pts)
        except RuntimeError:
            if len(pts) == 1:
                failed += 1
                return
            mid = len(pts) // 2
            _recurse(pts[:mid], idx[:mid])
            _recurse(pts[mid:], idx[mid:])

    t0 = time.time()
    done = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _recurse(rphiz[s:e], idx_all[s:e])
        done += (e - s)
        if progress:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-9)
            eta = (n - done) / max(rate, 1e-9)
            print(f"    Boozer convert: {done}/{n} "
                  f"({100 * done / n:5.1f}%, "
                  f"elapsed {elapsed:6.1f}s, "
                  f"rate {rate:7.1f} pts/s, "
                  f"ETA {eta:6.0f}s, "
                  f"failures so far {failed})",
                  flush=True)
    return out, failed


def ensure_valid_pool(pool, sc_particle, boozer_field, transformer=None):
    """Apply the common filters to produce the "valid fusion birth pool":
    LCFS-inside + finite Boozer s in [0, 1].  Returns (valid_pool,
    s_pool, theta_pool, zeta_pool, diagnostics_dict).

    If ``pool`` already has ``s_pre`` from a pre-computed Boozer IC file we
    reuse it; otherwise we invert (R, phi, Z).  Pass ``transformer`` to
    reuse a pre-built ``BoozerCoordinateTransformer`` across calls.
    """
    n_input = len(pool["R"])

    # (1) LCFS filter
    inside = filter_inside_lcfs(pool, sc_particle)
    n_outside = int((~inside).sum())
    pool = subset_pool(pool, inside)
    print(f"  LCFS filter: dropped {n_outside}/{n_input} outside, "
          f"kept {len(pool['R'])}")

    # (2) Boozer s
    if "s_pre" in pool:
        s_all     = pool["s_pre"]
        theta_all = pool["theta_pre"]
        zeta_all  = pool["zeta_pre"]
        n_bz_failed = 0
        print("  using pre-computed Boozer coordinates")
    else:
        R, phi, Z = pool["R"], pool["phi"], pool["Z"]
        rphiz = np.column_stack([R, phi, Z])
        print(f"  converting {len(rphiz)} points to Boozer...")
        t0 = time.time()
        out, n_bz_failed = _cyl_to_boozer_chunked(
            boozer_field, rphiz, transformer=transformer,
        )
        print(f"  Boozer conversion done in {time.time() - t0:.1f}s; "
              f"failures: {n_bz_failed}/{len(rphiz)}")
        s_all     = out[:, 0]
        theta_all = out[:, 1]
        zeta_all  = out[:, 2]

    # (3) Keep only points with valid s in [0, 1]
    valid = np.isfinite(s_all) & (s_all >= 0.0) & (s_all <= 1.0)
    n_bad = int((~valid).sum())
    pool = subset_pool(pool, valid)
    s_pool     = s_all[valid]
    theta_pool = theta_all[valid]
    zeta_pool  = zeta_all[valid]
    print(f"  Boozer validity filter: dropped {n_bad}, kept {len(s_pool)}")

    diag = {
        "n_input":      int(n_input),
        "n_outside":    int(n_outside),
        "n_bz_failed":  int(n_bz_failed),
        "n_bz_invalid": int(n_bad),
        "n_pool":       int(len(s_pool)),
    }
    return pool, s_pool, theta_pool, zeta_pool, diag


def convert_successes_to_boozer(rphiz, sc_particle, boozer_field,
                                chunk=1_000, transformer=None):
    """Used by the backward pilot to convert backward-success endpoints to
    Boozer.  Does the same LCFS pre-filter + chunked recursive inversion and
    returns the full-length arrays (with NaN where invalid) plus a validity
    mask in [0, 1].  Pass ``transformer`` to reuse a pre-built
    ``BoozerCoordinateTransformer`` across calls."""
    n = len(rphiz)
    s_all     = np.full(n, np.nan)
    theta_all = np.full(n, np.nan)
    zeta_all  = np.full(n, np.nan)

    sd = sc_particle.evaluate_rphiz(rphiz).ravel()
    inside = sd >= 0
    n_outside = int((~inside).sum())
    inside_idx = np.where(inside)[0]
    pts = rphiz[inside_idx]

    n_bz_failed = 0
    if len(pts):
        print(f"  converting {len(pts)} backward successes to Boozer...")
        t0 = time.time()
        out, n_bz_failed = _cyl_to_boozer_chunked(boozer_field, pts,
                                                  chunk=chunk,
                                                  transformer=transformer)
        print(f"  done in {time.time() - t0:.1f}s; "
              f"failures: {n_bz_failed}/{len(pts)}")
        s_all[inside_idx]     = out[:, 0]
        theta_all[inside_idx] = out[:, 1]
        zeta_all[inside_idx]  = out[:, 2]

    valid = np.isfinite(s_all) & (s_all >= 0.0) & (s_all <= 1.0)
    return s_all, theta_all, zeta_all, valid, {
        "n_total":      int(n),
        "n_outside":    int(n_outside),
        "n_bz_failed":  int(n_bz_failed),
        "n_valid":      int(valid.sum()),
    }
