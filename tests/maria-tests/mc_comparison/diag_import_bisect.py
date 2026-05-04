#!/usr/bin/env python3
"""Bisect which import in mc_comparison's stack breaks BoozerInterpolant
virtual dispatch.

Run all four cases — each one is a *fresh Python process*.  The first one
that fails on R() with `_R_impl was not implemented` tells us which import
introduces the bug.

Run:
    cd tests/maria-tests/mc_comparison/
    python diag_import_bisect.py 1
    python diag_import_bisect.py 2
    python diag_import_bisect.py 3
    python diag_import_bisect.py 4

(no GPU needed)
"""
import sys
from pathlib import Path

CASE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
print(f"=== CASE {CASE} ===")

if CASE == 1:
    # Bare-minimum: just firm3d boozermagneticfield, like backward_tracing_only.
    from firm3d.field.boozermagneticfield import (
        BoozerRadialInterpolant, InterpolatedBoozerField,
    )

elif CASE == 2:
    # Add: firm3dpp + firm3d coordinates (matching mc_comparison's top-of-file).
    from firm3dpp import (
        cartesian_gpu_tracing_backward_drag,
        cartesian_gpu_tracing_drag,
    )
    from firm3d.field.coordinates import (
        BoozerCoordinateTransformer,
        boozer_to_cylindrical,
        cylindrical_to_boozer,
    )
    from firm3d.field.boozermagneticfield import (
        BoozerRadialInterpolant, InterpolatedBoozerField,
    )

elif CASE == 3:
    # Add: birth_pool_utils.
    from firm3dpp import (
        cartesian_gpu_tracing_backward_drag,
        cartesian_gpu_tracing_drag,
    )
    from firm3d.field.coordinates import (
        BoozerCoordinateTransformer,
        boozer_to_cylindrical,
        cylindrical_to_boozer,
    )
    from birth_pool_utils import build_boozer_interpolant
    from firm3d.field.boozermagneticfield import (
        BoozerRadialInterpolant, InterpolatedBoozerField,
    )

elif CASE == 4:
    # Add: perturbed_field_utils.
    from firm3dpp import (
        cartesian_gpu_tracing_backward_drag,
        cartesian_gpu_tracing_drag,
    )
    from firm3d.field.coordinates import (
        BoozerCoordinateTransformer,
        boozer_to_cylindrical,
        cylindrical_to_boozer,
    )
    from birth_pool_utils import build_boozer_interpolant
    from perturbed_field_utils import (
        FieldConfig, build_perturbed_field, flatten_stz, wrap_phi,
    )
    from firm3d.field.boozermagneticfield import (
        BoozerRadialInterpolant, InterpolatedBoozerField,
    )

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
BOOZMN = THIS_DIR.parent / "mc_backward" / "LandremanPaulQH_coils" / "boozmn.nc"

print(f"  building from {BOOZMN}")
bri = BoozerRadialInterpolant(str(BOOZMN), 3, no_K=True)
bf = InterpolatedBoozerField(bri, 3, ns_interp=48, ntheta_interp=48, nzeta_interp=48)

print(f"  type: {type(bf).__name__}  field_type: {bf.field_type!r}")
print(f"  MRO: {[c.__name__ for c in type(bf).__mro__]}")

bf.set_points(np.array([[0.5, 0.0, 0.0]]))
try:
    R = bf.R()
    print(f"  R() OK: {R[0, 0]:.6f}")
except RuntimeError as e:
    print(f"  R() FAILED: {e}")
