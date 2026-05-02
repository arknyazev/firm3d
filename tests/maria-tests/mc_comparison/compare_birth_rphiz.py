#!/usr/bin/env python3
"""Compare birth_rphiz arrays dumped by backward_tracing_only and
backward_informed_mc_s, to see whether the two scripts produce identical
backward-birth points before the cylindrical->Boozer conversion.

Run after both scripts have been executed and have written the .npy files
to /tmp.  Either path can be overridden via env var:

    BWO_PATH=/some/where/bwo.npy MCS_PATH=/some/where/mcs.npy \
        python compare_birth_rphiz.py
"""
import os
import sys
from pathlib import Path

import numpy as np

BWO_PATH = Path(os.environ.get("BWO_PATH", "/tmp/bwo_birth_rphiz.npy"))
MCS_PATH = Path(os.environ.get("MCS_PATH", "/tmp/mcs_birth_rphiz.npy"))


def _load(p):
    if not p.exists():
        sys.exit(f"missing: {p} — did the script that writes it actually "
                 f"run to completion?")
    return np.load(p)


def main():
    a = _load(BWO_PATH)
    b = _load(MCS_PATH)

    print(f"bwo path      : {BWO_PATH}")
    print(f"mcs path      : {MCS_PATH}")
    print(f"bwo shape     : {a.shape}")
    print(f"mcs shape     : {b.shape}")

    if a.shape != b.shape:
        print("\nSHAPE MISMATCH — different number of birth points.")
        print(f"first 3 rows of bwo:\n{a[:3]}")
        print(f"first 3 rows of mcs:\n{b[:3]}")
        return

    diff = a - b
    abs_diff = np.abs(diff)
    print(f"max |diff|    : {abs_diff.max():.6e}")
    print(f"mean |diff|   : {abs_diff.mean():.6e}")
    print(f"per-column max: R={abs_diff[:, 0].max():.3e}  "
          f"phi={abs_diff[:, 1].max():.3e}  "
          f"Z={abs_diff[:, 2].max():.3e}")
    print(f"exactly equal : {np.array_equal(a, b)}")

    n_show = min(5, a.shape[0])
    print(f"\nfirst {n_show} rows of bwo (R, phi, Z):")
    print(a[:n_show])
    print(f"\nfirst {n_show} rows of mcs (R, phi, Z):")
    print(b[:n_show])
    if n_show > 0:
        print(f"\nfirst {n_show} row diffs (bwo - mcs):")
        print(diff[:n_show])


if __name__ == "__main__":
    main()
