#!/usr/bin/env python3
"""Standalone Boozer-conversion test.

Loads the birth_rphiz array previously dumped by either
backward_tracing_only.py (BWO_PATH) or backward_informed_mc_s.py (MCS_PATH),
builds the Boozer interpolant from scratch in a fresh Python process, and
runs cylindrical_to_boozer.

Purpose: factor out whatever "in-script global state" mc_comparison may be
introducing by replicating only the minimum required setup.  If this
standalone run succeeds at the backward_tracing_only success rate, the bug
is something polluting boozer_field state inside backward_informed_mc_s.py.
If it fails 100% just like the in-script run, the bug is upstream of the
script-vs-script comparison and we need a different theory.

Run:
    python standalone_boozer_convert.py                # uses ~/mcs_birth_rphiz.npy
    INPUT=~/bwo_birth_rphiz.npy python standalone_boozer_convert.py

No GPU required; pure-CPU Boozer inversion.
"""
import os
import time
from pathlib import Path

import numpy as np

from firm3d.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from firm3d.field.coordinates import cylindrical_to_boozer

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_BOOZMN = (THIS_DIR.parent / "mc_backward" /
                  "LandremanPaulQH_coils" / "boozmn.nc")

INPUT      = Path(os.environ.get("INPUT",
                                  str(Path.home() / "mcs_birth_rphiz.npy")))
BOOZMN     = Path(os.environ.get("BOOZMN", str(DEFAULT_BOOZMN)))
RADIAL_ORD = int(os.environ.get("RADIAL_ORD", 3))
BZ_DEGREE  = int(os.environ.get("BZ_DEGREE", 3))
BZ_RES     = int(os.environ.get("BZ_RES", 48))


def main():
    if not INPUT.exists():
        raise SystemExit(f"missing input: {INPUT}")
    if not BOOZMN.exists():
        raise SystemExit(f"missing boozmn: {BOOZMN}")

    rphiz = np.load(INPUT)
    print(f"loaded: {INPUT}  shape={rphiz.shape}")

    print(f"building boozer interpolant from {BOOZMN}")
    print(f"  radial_order={RADIAL_ORD}  boozer_degree={BZ_DEGREE}  "
          f"boozer_res={BZ_RES}")
    t0 = time.time()
    bri = BoozerRadialInterpolant(str(BOOZMN), RADIAL_ORD, no_K=True)
    boozer_field = InterpolatedBoozerField(
        bri, BZ_DEGREE,
        ns_interp=BZ_RES, ntheta_interp=BZ_RES, nzeta_interp=BZ_RES,
    )
    print(f"  built in {time.time() - t0:.2f}s")

    n = len(rphiz)
    out = np.full_like(rphiz, np.nan)
    failed = 0

    def _recurse(pts, idx):
        nonlocal failed
        if len(pts) == 0:
            return
        try:
            out[idx] = cylindrical_to_boozer(boozer_field, pts)
        except RuntimeError:
            if len(pts) == 1:
                failed += 1
                return
            mid = len(pts) // 2
            _recurse(pts[:mid], idx[:mid])
            _recurse(pts[mid:], idx[mid:])

    chunk = 1000
    idx_all = np.arange(n)
    t0 = time.time()
    done = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        _recurse(rphiz[s:e], idx_all[s:e])
        done += (e - s)
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-9)
        print(f"  Boozer convert: {done}/{n} "
              f"({100 * done / n:5.1f}%, "
              f"elapsed {elapsed:6.1f}s, "
              f"rate {rate:7.1f} pts/s, "
              f"failures so far {failed})",
              flush=True)

    s = out[:, 0]
    valid = np.isfinite(s) & (s >= 0.0) & (s <= 1.0)
    print(f"\nfailures   : {failed}/{n}  ({100 * failed / n:.1f}%)")
    print(f"valid s    : {int(valid.sum())}/{n}")
    if valid.any():
        print(f"s range    : [{s[valid].min():.3f}, {s[valid].max():.3f}]")


if __name__ == "__main__":
    main()
