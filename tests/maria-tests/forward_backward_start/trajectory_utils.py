#!/usr/bin/env python3
"""
Shared utilities for loading and processing particle trajectory data
from GPU tracing output files.
Accepts new directory format.
"""

import os
import re
import glob
import numpy as np


def parse_tmax_from_filename(filename: str) -> float:
    """
    Parse tmax value from encoded filename.
    Example: 'particles_final_xyz_tmax_1p00em04.npy' -> 1.00e-04
    """
    match = re.search(r'tmax_([^.]+)', filename)
    if not match:
        raise ValueError(f"Could not extract tmax from filename: {filename}")

    tmax_str = match.group(1)
    tmax_str = tmax_str.replace('p', '.').replace('m', '-')
    return float(tmax_str)


def find_trajectory_files(output_dir: str, pattern: str) -> list[tuple[float, str]]:
    """
    Find all trajectory files matching the pattern, sorted by tmax.
    Returns list of (tmax, filepath) tuples.
    """
    files = glob.glob(os.path.join(output_dir, pattern))
    tmax_files = []
    for filepath in files:
        try:
            tmax = parse_tmax_from_filename(os.path.basename(filepath))
            tmax_files.append((tmax, filepath))
        except ValueError:
            continue
    tmax_files.sort(key=lambda x: x[0])
    return tmax_files


def list_tol_dirs(output_root: str = "./output") -> list[str]:
    """
    List tol_* directories under output_root, sorted.
    Returns full paths.
    """
    tol_dirs = sorted(
        d for d in glob.glob(os.path.join(output_root, "tol_*"))
        if os.path.isdir(d)
    )
    return tol_dirs


def pick_tol_dir(output_root: str = "./output", tol_dir: str | None = None) -> str:
    """
    If tol_dir is provided, use it (absolute or relative).
    Otherwise pick the last tol_* folder under output_root.
    """
    if tol_dir is not None:
        # allow passing "tol_1p00em09" or "./output/tol_..."
        if os.path.isdir(tol_dir):
            return tol_dir
        candidate = os.path.join(output_root, tol_dir)
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(f"tol_dir not found: {tol_dir}")

    tol_dirs = list_tol_dirs(output_root)
    if len(tol_dirs) == 0:
        raise FileNotFoundError(f"No tol_* folders found under {output_root}")
    return tol_dirs[-1]


def load_all_trajectories(output_root: str = "./output",
                          tol_dir: str | None = None,
                          run_subdir: str = "forward"):
    """
    Load all trajectory snapshots for a given tol folder and run_subdir.

    Folder layout expected:
      output_root/
        tol_<tag>/
          run_subdir/particles_final_xyz_tmax_*.npy
          run_subdir/particles_final_time_tmax_*.npy
          (optional) tmax_values.npy

    Args:
        output_root: top-level output directory (default "./output")
        tol_dir: specific tol directory name or path; if None uses latest tol_*
        run_subdir: "forward" or "backward"

    Returns:
        trajectories: (nparticles, ntimes, 3) array of XYZ positions
        tmax_values:   (ntimes,) array of tmax values (sorted)
        lost_mask:     (nparticles,) boolean array (True if particle was lost at last tmax)
        final_times:   (nparticles, ntimes) array of actual final times
        run_dir:       full path to the run directory used
        tol_dir_full:  full path to the tol directory used
    """
    tol_dir_full = pick_tol_dir(output_root, tol_dir)
    run_dir = os.path.join(tol_dir_full, run_subdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    xyz_files = find_trajectory_files(run_dir, "particles_final_xyz_tmax_*.npy")
    time_files = find_trajectory_files(run_dir, "particles_final_time_tmax_*.npy")

    if len(xyz_files) == 0:
        raise FileNotFoundError(f"No trajectory xyz files found in {run_dir}")
    if len(xyz_files) != len(time_files):
        raise ValueError(f"Mismatch in {run_dir}: {len(xyz_files)} xyz files but {len(time_files)} time files")

    tmax_values = np.array([tmax for tmax, _ in xyz_files], dtype=np.float64)

    first_xyz = np.load(xyz_files[0][1])
    nparticles = first_xyz.shape[0]
    ntimes = len(xyz_files)

    trajectories = np.zeros((nparticles, ntimes, 3), dtype=np.float64)
    final_times = np.zeros((nparticles, ntimes), dtype=np.float64)

    for i, ((tmax_xyz, xyz_file), (tmax_time, time_file)) in enumerate(zip(xyz_files, time_files)):
        if abs(tmax_xyz - tmax_time) > 0:
            raise ValueError(f"tmax mismatch between xyz and time at index {i}: {tmax_xyz} vs {tmax_time}")
        trajectories[:, i, :] = np.load(xyz_file)
        final_times[:, i] = np.load(time_file)

    # Lost if it didn't reach final tmax of THIS run
    lost_mask = final_times[:, -1] < 0.99 * tmax_values[-1]

    return trajectories, tmax_values, lost_mask, final_times, run_dir, tol_dir_full


def load_initial_positions(output_root: str = "./output",
                           tol_dir: str | None = None):
    """
    Load initial particle positions (nparticles, 3) XYZ array from tol folder.

    Looks in:
      tol_dir_full/particles_initial_xyz.npy
    """
    tol_dir_full = pick_tol_dir(output_root, tol_dir)
    path = os.path.join(tol_dir_full, "particles_initial_xyz.npy")
    return np.load(path)


def truncate_lost_trajectories(trajectories, final_times, tmax_values):
    """
    For lost particles, set trajectory points after loss to NaN.
    """
    trajectories_truncated = trajectories.copy()
    for i in range(trajectories.shape[0]):
        for j in range(trajectories.shape[1]):
            if final_times[i, j] < 0.99 * tmax_values[j]:
                trajectories_truncated[i, j:, :] = np.nan
                break
    return trajectories_truncated