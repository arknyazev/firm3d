"""Load the fusion birth pool, filter it against the LCFS, and resolve its
Boozer s coordinate.

The resulting pool is deterministic for a given (fusion_ic_file,
n_particles, VMEC input, boozmn.nc), so all three methods operate on the
identical pool indexing and hence on the same target measure Q.
"""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.coordinates import (
    BoozerCoordinateTransformer,
    cylindrical_to_boozer,
)


def load_fusion_pool(fusion_ic_file, n_particles=None,
                     fusion_boozer_file=None):
    """Load the raw fusion IC file and (optionally) its pre-computed Boozer counterpart"""
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
    """Return a boolean mask of pool rows inside (signed distance >= 0) the LCFS"""
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
    # Keep bri alive for as long as bf lives
    bf._bri = bri
    return bf


def _worker_convert_chunk(args):
    """Top-level worker for the parallel Boozer-conversion path."""
    (boozmn_file_str, radial_order, boozer_degree, boozer_res,
     pts, chunk) = args

    bri = BoozerRadialInterpolant(boozmn_file_str, radial_order, no_K=True)
    bf  = InterpolatedBoozerField(
        bri, boozer_degree,
        ns_interp=boozer_res, ntheta_interp=boozer_res, nzeta_interp=boozer_res,
    )
    bf._bri = bri  # keep bri alive (s.t. it is not garbage-collected)
    transformer = BoozerCoordinateTransformer(bf, grid_resolution=(50, 50, 50))

    n = len(pts)
    out = np.full_like(pts, np.nan)
    failed = 0
    idx_all = np.arange(n)

    def _recurse(p, idx):
        nonlocal failed
        if len(p) == 0:
            return
        try:
            out[idx] = transformer.cylindrical_to_boozer(p)
        except RuntimeError:
            if len(p) == 1:
                failed += 1
                return
            mid = len(p) // 2
            _recurse(p[:mid], idx[:mid])
            _recurse(p[mid:], idx[mid:])

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _recurse(pts[s:e], idx_all[s:e])

    return out, failed


def _cyl_to_boozer_chunked(boozer_field, rphiz, chunk=1_000, progress=True,
                           transformer=None, n_workers=1, boozmn_file=None,
                           radial_order=3, boozer_degree=3, boozer_res=48):
    """Convert (R, phi, Z) to (s, theta, zeta) with chunked recursive
    subdivision on conversion failure, so a single bad point only pays its
    own cost rather than poisoning a whole chunk.
    """
    n = len(rphiz)
    if n == 0:
        return np.full((0, 3), np.nan), 0

    # ── Parallel path ────────────────────────────────────────────────────
    if n_workers > 1 and boozmn_file is not None:
        n_workers = min(n_workers, n)
        chunks = np.array_split(rphiz, n_workers)
        args_list = [
            (str(boozmn_file), radial_order, boozer_degree, boozer_res,
             c, chunk)
            for c in chunks
        ]

        if progress:
            sizes = [len(c) for c in chunks]
            print(f"  Boozer convert (parallel): {n_workers} workers on "
                  f"{n} points, slice sizes "
                  f"{min(sizes)}-{max(sizes)}/worker; building "
                  f"per-worker boozer_field...",
                  flush=True)

        t0 = time.time()
        results = [None] * n_workers
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            future_to_idx = {
                ex.submit(_worker_convert_chunk, args): i
                for i, args in enumerate(args_list)
            }
            done_count = 0
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                results[i] = fut.result()
                done_count += 1
                if progress:
                    elapsed = time.time() - t0
                    print(f"    worker {i:>2}/{n_workers} done "
                          f"({done_count}/{n_workers} workers finished, "
                          f"elapsed {elapsed:6.1f}s, "
                          f"slice failures {results[i][1]})",
                          flush=True)

        out = np.concatenate([r[0] for r in results], axis=0)
        failed = int(sum(r[1] for r in results))
        if progress:
            elapsed = time.time() - t0
            rate = n / max(elapsed, 1e-9)
            print(f"  Boozer convert: {n}/{n} done in {elapsed:.1f}s "
                  f"(rate {rate:.1f} pts/s, total failures {failed})",
                  flush=True)
        assert out.shape == rphiz.shape, (
            f"parallel concat shape mismatch: out={out.shape} "
            f"vs rphiz={rphiz.shape}")
        return out, failed

    # ── Sequential path ──────────────────────────────────────────────────
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


def ensure_valid_pool(pool, sc_particle, boozer_field, transformer=None,
                      n_workers=1, boozmn_file=None,
                      radial_order=3, boozer_degree=3, boozer_res=48):
    """Apply the common filters to produce the "valid fusion birth pool":
    LCFS-inside + finite Boozer s in [0, 1].  Returns (valid_pool,
    s_pool, theta_pool, zeta_pool, diagnostics_dict).
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
            n_workers=n_workers, boozmn_file=boozmn_file,
            radial_order=radial_order,
            boozer_degree=boozer_degree,
            boozer_res=boozer_res,
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
                                chunk=1_000, transformer=None,
                                n_workers=1, boozmn_file=None,
                                radial_order=3, boozer_degree=3,
                                boozer_res=48):
    """Used by the backward pilot to convert backward-success endpoints to Boozer"""
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
        out, n_bz_failed = _cyl_to_boozer_chunked(
            boozer_field, pts,
            chunk=chunk,
            transformer=transformer,
            n_workers=n_workers,
            boozmn_file=boozmn_file,
            radial_order=radial_order,
            boozer_degree=boozer_degree,
            boozer_res=boozer_res,
        )
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