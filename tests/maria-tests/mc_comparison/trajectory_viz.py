#!/usr/bin/env python3
"""Stand-alone high-resolution trajectory visualizer for Paraview.

Forward-traces a curated list of pool markers (selected once by inspecting
a prior forward_mc run) through the perturbed coil field and writes a
Paraview-ready VTU polyline file.

Because the drag tracer is deterministic, the same (R, phi, Z, vpar)
initial condition produces the same trajectory regardless of which
estimator "drew" it, so we trace **once** here — not three times.

Output layout (under ``--out_dir``):
    coils_LPQH.vtu              perturbed coils
    surface_LPQH.vts            VMEC LCFS
    trajectories_forward.vtu    N polylines, cell data:
                                    particle_id  (pool index)
                                    outcome      (0 confined, 1 lost)
                                    t_final      (loss time for lost,
                                                  tmax otherwise)
                                    s_boozer     (Boozer s of birth)
                                point data:
                                    time, vpar, H

In Paraview: open the VTU, Color By → outcome (coolwarm), or → s_boozer
(viridis), or → t_final.  The three estimator scripts' outputs live in
sibling directories and can be overlaid.
"""
import argparse
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np

from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE as CHARGE,
    ALPHA_PARTICLE_MASS as MASS,
    FUSION_ALPHA_PARTICLE_ENERGY as H_FUSION,
)
from firm3dpp import cartesian_gpu_tracing_drag

from perturbed_field_utils import (
    FieldConfig, build_perturbed_field, flatten_stz, wrap_phi,
)
from birth_pool_utils import (
    build_boozer_interpolant, ensure_valid_pool, load_fusion_pool,
)
from vtk_utils import trace_snapshots, write_coils_and_surface_vtk


THIS_DIR = Path(__file__).resolve().parent
COILS_DIR = THIS_DIR.parent / "mc_backward" / "LandremanPaulQH_coils"

# Baked-in default selection: 5 lost + 5 confined pool indices spanning s,
# curated from forward_mc/ outputs of the 2026-04-22 500k run.  Override on
# the command line with --viz_indices if you want a different set.
DEFAULT_VIZ_INDICES = (
    # Lost (5), s ascending
    25075, 27494, 24219, 41312, 7053,
    # Confined (5), s ascending
    45416, 15618, 36970, 34223, 24074,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--perturbation_id", type=int, default=57)
    p.add_argument("--n_pool", type=int, default=50_000,
                   help="Max rows read from fusion_ic_file; must match the "
                        "value used by the comparison run so pool indices "
                        "line up.")
    p.add_argument("--viz_indices", type=str,
                   default=",".join(str(i) for i in DEFAULT_VIZ_INDICES),
                   help="Comma-separated pool indices to trace.")
    p.add_argument("--tmax_trajectory", type=float, default=1e-3,
                   help="Integration horizon for visualisation (s).  "
                        "Default 1 ms — long enough for lost particles to "
                        "actually hit the wall, short enough that snapshot "
                        "dt resolves drift orbits.")
    p.add_argument("--n_snapshots", type=int, default=2000,
                   help="Snapshots per trajectory (~1000 with tmax=1e-3 "
                        "gives dt=1 us = ~15 gyro-periods).")
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--ne0", type=float, default=1e21)
    p.add_argument("--Te0_ev", type=float, default=100.0)
    p.add_argument("--coulomb_log", type=float, default=17.0)
    p.add_argument("--coil_file", type=Path,
                   default=COILS_DIR / "coils.curves_22_7_21")
    p.add_argument("--vmec_input_file", type=Path,
                   default=COILS_DIR / "input.vmec")
    p.add_argument("--boozmn_file", type=Path,
                   default=COILS_DIR / "boozmn.nc")
    p.add_argument("--fusion_ic_file", type=Path,
                   default=Path("/pscratch/sd/m/mariagar/projects/mc_proj/IC/"
                                "initial_conditions_cylindrical.txt"))
    p.add_argument("--fusion_boozer_file", type=Path,
                   default=Path("/pscratch/sd/m/mariagar/projects/mc_proj/IC/"
                                "initial_conditions_boozer.txt"))
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()


def parse_indices(s):
    s = s.strip()
    if not s:
        return np.empty(0, dtype=int)
    return np.array([int(tok) for tok in s.split(",") if tok.strip()],
                    dtype=int)


def write_trajectory_vtu_colored(filename, n_snapshots,
                                 initial_xyz, snap_xyz, snap_time,
                                 snap_vpar, snap_H,
                                 initial_vpar, initial_H,
                                 particle_id, outcome, t_final, s_boozer):
    """Polyline VTU with per-cell categorical fields ready for Paraview
    coloring.  Each polyline has n_snapshots+1 points (initial + snaps).

    Per-cell fields:
        particle_id  Int32
        outcome      Int32    (0 confined, 1 lost)
        t_final      Float64
        s_boozer     Float64

    Per-point fields:
        time  Float64
        vpar  Float64
        H     Float64
    """
    n_traj = snap_xyz.shape[0]
    pts_per = 1 + n_snapshots

    pts = np.empty((n_traj, pts_per, 3), dtype=np.float64)
    pts[:, 0, :]  = initial_xyz
    pts[:, 1:, :] = snap_xyz
    flat = pts.reshape(-1, 3)
    n_total = flat.shape[0]

    t_pt    = np.empty((n_traj, pts_per), dtype=np.float64)
    vpar_pt = np.empty((n_traj, pts_per), dtype=np.float64)
    H_pt    = np.empty((n_traj, pts_per), dtype=np.float64)
    t_pt[:, 0]     = 0.0
    t_pt[:, 1:]    = snap_time
    vpar_pt[:, 0]  = initial_vpar
    vpar_pt[:, 1:] = snap_vpar
    H_pt[:, 0]     = initial_H
    H_pt[:, 1:]    = snap_H

    root = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1",
                      byte_order="LittleEndian")
    ugrid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(ugrid, "Piece",
                          NumberOfPoints=str(n_total),
                          NumberOfCells=str(n_traj))

    # Points
    pelem = ET.SubElement(piece, "Points")
    da = ET.SubElement(pelem, "DataArray",
                       type="Float64", NumberOfComponents="3", format="ascii")
    da.text = " ".join(f"{x:.6e} {y:.6e} {z:.6e}" for x, y, z in flat)

    # Cells: one polyline per particle
    cells = ET.SubElement(piece, "Cells")
    conn = ET.SubElement(cells, "DataArray",
                         type="Int32", Name="connectivity", format="ascii")
    conn.text = " ".join(str(i) for i in range(n_total))
    off = ET.SubElement(cells, "DataArray",
                        type="Int32", Name="offsets", format="ascii")
    off.text = " ".join(str(pts_per * (k + 1)) for k in range(n_traj))
    typ = ET.SubElement(cells, "DataArray",
                        type="UInt8", Name="types", format="ascii")
    typ.text = " ".join(["4"] * n_traj)  # 4 = VTK_POLY_LINE

    # Per-point data
    pdata = ET.SubElement(piece, "PointData")
    for name, arr in [("time", t_pt), ("vpar", vpar_pt), ("H", H_pt)]:
        d = ET.SubElement(pdata, "DataArray",
                          type="Float64", Name=name, format="ascii")
        d.text = " ".join(f"{v:.6e}" for v in arr.ravel())

    # Per-cell data (one value per polyline)
    cdata = ET.SubElement(piece, "CellData")
    for name, arr, dtype in [
        ("particle_id", particle_id, "Int32"),
        ("outcome",     outcome,     "Int32"),
        ("t_final",     t_final,     "Float64"),
        ("s_boozer",    s_boozer,    "Float64"),
    ]:
        d = ET.SubElement(cdata, "DataArray",
                          type=dtype, Name=name, format="ascii")
        if dtype == "Int32":
            d.text = " ".join(str(int(v)) for v in arr)
        else:
            d.text = " ".join(f"{v:.6e}" for v in arr)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(filename), encoding="utf-8", xml_declaration=True)
    print(f"  wrote {Path(filename).name} "
          f"({n_traj} polylines, {pts_per} pts each)")


def main():
    args = parse_args()

    viz_idx = parse_indices(args.viz_indices)
    if viz_idx.size == 0:
        raise SystemExit("No --viz_indices provided; nothing to trace.")

    out_dir = args.out_dir or (
        Path("/pscratch/sd/m/mariagar/projects/mc_proj/results/mc_comparison")
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        / "trajectory_viz"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {out_dir}")
    print(f"viz_indices ({len(viz_idx)}): {viz_idx.tolist()}")

    # Field + pool -----------------------------------------------------------
    cfg = FieldConfig(coil_file=args.coil_file,
                      vmec_input_file=args.vmec_input_file)

    def ne_fun(rphiz): return np.full(rphiz.shape[0], args.ne0,
                                      dtype=np.float64)
    def Te_fun(rphiz): return np.full(rphiz.shape[0], args.Te0_ev,
                                      dtype=np.float64)

    field = build_perturbed_field(cfg, args.perturbation_id, ne_fun, Te_fun)
    write_coils_and_surface_vtk(out_dir, field["curves"], field["s_input"])

    print("\n--- Loading fusion birth pool (for viz_indices lookup) ---")
    raw_pool = load_fusion_pool(args.fusion_ic_file, args.n_pool,
                                fusion_boozer_file=args.fusion_boozer_file)
    boozer_field = build_boozer_interpolant(args.boozmn_file)
    pool, s_pool, _, _, _ = ensure_valid_pool(
        raw_pool, field["sc_particle"], boozer_field,
    )
    N_pool = len(pool["R"])
    print(f"  N_pool = {N_pool}")

    bad = viz_idx[(viz_idx < 0) | (viz_idx >= N_pool)]
    if bad.size:
        raise SystemExit(
            f"ERROR: viz_indices {bad.tolist()} are out of range for "
            f"N_pool={N_pool}.  Did the comparison run use a different "
            f"--n_pool?  Re-run the inspector on the current pool if so."
        )

    R_v    = pool["R"][viz_idx]
    phi_v  = wrap_phi(pool["phi"][viz_idx], field["phi_min"], field["phi_max"])
    Z_v    = pool["Z"][viz_idx]
    vpar_v = pool["vpar"][viz_idx]
    H_v    = np.full(len(viz_idx), H_FUSION, dtype=np.float64)
    s_v    = s_pool[viz_idx]

    X_v = R_v * np.cos(phi_v); Y_v = R_v * np.sin(phi_v)
    initial_xyz = np.column_stack([X_v, Y_v, Z_v])

    # Forward trace with snapshots ------------------------------------------
    print(f"\n--- Forward tracing {len(viz_idx)} particles "
          f"({args.n_snapshots} snapshots, tmax={args.tmax_trajectory:.2e} s) ---")
    speed_ref = float(sqrt(2.0 * H_FUSION / MASS))
    t0 = time.time()
    snap_xyz, snap_vpar, snap_H, snap_time, snap_stop, tmax_snaps = (
        trace_snapshots(
            tracer=cartesian_gpu_tracing_drag, field=field,
            R_init=R_v, phi_init=phi_v, Z_init=Z_v,
            vpar_init=vpar_v, H_init=H_v,
            mass=MASS, charge=CHARGE, speed_ref=speed_ref,
            coulomb_log=args.coulomb_log, Te_in_eV=True,
            tmax=args.tmax_trajectory, tol=args.tol,
            n_snapshots=int(args.n_snapshots),
            H_stop=0.0, use_energy_stop=False,
            label="viz forward snapshots",
        )
    )
    print(f"  done in {time.time() - t0:.2f}s")

    # Per-particle outcome + loss time:
    # outcome = 1 iff the particle ever stopped with stop_code==1 at any
    # snapshot; t_final = first snapshot time where stop_code != 0 (for
    # stopping codes; confined → last snapshot's t).
    hit_any = (snap_stop == 1).any(axis=1)
    outcome = hit_any.astype(np.int32)
    # t_final: for each particle, the snapshot t at which the tracer first
    # stopped.  With use_energy_stop=False, stop_code == 1 fires only on
    # wall hit; once stopped the tracer reports a frozen state, so we
    # use the *final* snapshot's reported t (which equals the wall-hit
    # time for lost particles and the full tmax for confined).
    t_final = snap_time[:, -1].astype(np.float64)

    print("\n--- Per-particle summary ---")
    print(f"{'pool_idx':>10} {'outcome':>8} {'s':>7} {'t_final [s]':>14}")
    for i in range(len(viz_idx)):
        print(f"{int(viz_idx[i]):>10} "
              f"{'LOST' if outcome[i] else 'conf':>8} "
              f"{s_v[i]:>7.3f} {t_final[i]:>14.3e}")

    # Save arrays (for downstream scripting) + VTU ---------------------------
    np.save(out_dir / "viz_indices.npy", viz_idx.astype(np.int64))
    np.save(out_dir / "traj_xyz.npy",    snap_xyz)
    np.save(out_dir / "traj_vpar.npy",   snap_vpar)
    np.save(out_dir / "traj_H.npy",      snap_H)
    np.save(out_dir / "traj_time.npy",   snap_time)
    np.save(out_dir / "traj_stop.npy",   snap_stop)
    np.save(out_dir / "traj_tmax.npy",   tmax_snaps)
    np.save(out_dir / "traj_outcome.npy", outcome)
    np.save(out_dir / "traj_s.npy",      s_v)
    np.save(out_dir / "traj_t_final.npy", t_final)

    print("\n--- Writing VTU ---")
    write_trajectory_vtu_colored(
        out_dir / "trajectories_forward.vtu",
        n_snapshots=int(args.n_snapshots),
        initial_xyz=initial_xyz,
        snap_xyz=snap_xyz, snap_time=snap_time,
        snap_vpar=snap_vpar, snap_H=snap_H,
        initial_vpar=vpar_v, initial_H=H_v,
        particle_id=viz_idx.astype(np.int32),
        outcome=outcome,
        t_final=t_final,
        s_boozer=s_v,
    )

    # Config snapshot for reproducibility
    with open(out_dir / "run_config.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k} = {v}\n")

    print(f"\nDone.  Paraview: open {out_dir / 'trajectories_forward.vtu'} "
          f"and color by 'outcome' (coolwarm), 's_boozer' (viridis), or "
          f"'t_final'.")


if __name__ == "__main__":
    main()
