
#!/usr/bin/env python3

"""
Shared utilities for loading and processing particle trajectory data
from GPU tracing output files.
"""

import os
import re
import glob
import numpy as np


def parse_tmax_from_filename(filename):
    """
    Parse tmax value from encoded filename.
    Example: 'particles_final_xyz_tmax_1p00em04.npy' -> 1.00e-04
    """
    match = re.search(r'tmax_([^.]+)', filename)
    if not match:
        raise ValueError(f"Could not extract tmax from filename: {filename}")

    tmax_str = match.group(1)
    # Convert: '1p00em04' -> '1.00e-04'
    tmax_str = tmax_str.replace('p', '.').replace('m', '-')
    return float(tmax_str)


def find_trajectory_files(output_dir, pattern='particles_final_xyz_tmax_*.npy'):
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


def load_all_trajectories(output_dir='./output'):
    """
    Load all trajectory snapshots and construct full trajectory arrays.

    Returns:
        trajectories: (nparticles, ntimes, 3) array of XYZ positions
        tmax_values:   (ntimes,) array of tmax values
        lost_mask:     (nparticles,) boolean array (True if particle was lost)
        final_times:   (nparticles, ntimes) array of actual final times
    """
    xyz_files = find_trajectory_files(output_dir, 'particles_final_xyz_tmax_*.npy')
    time_files = find_trajectory_files(output_dir, 'particles_final_time_tmax_*.npy')

    if len(xyz_files) == 0:
        raise FileNotFoundError(f"No trajectory files found in {output_dir}")
    if len(xyz_files) != len(time_files):
        raise ValueError(f"Mismatch: {len(xyz_files)} xyz files but {len(time_files)} time files")

    tmax_values = np.array([tmax for tmax, _ in xyz_files])

    first_xyz = np.load(xyz_files[0][1])
    nparticles = first_xyz.shape[0]
    ntimes = len(xyz_files)

    trajectories = np.zeros((nparticles, ntimes, 3))
    final_times = np.zeros((nparticles, ntimes))

    for i, ((tmax_xyz, xyz_file), (tmax_time, time_file)) in enumerate(zip(xyz_files, time_files)):
        trajectories[:, i, :] = np.load(xyz_file)
        final_times[:, i] = np.load(time_file)

    # A particle is "lost" if it didn't reach the final tmax
    lost_mask = final_times[:, -1] < 0.99 * tmax_values[-1]

    return trajectories, tmax_values, lost_mask, final_times


def load_initial_positions(output_dir='./output'):
    """Load initial particle positions (nparticles, 3) XYZ array."""
    return np.load(os.path.join(output_dir, 'particles_initial_xyz.npy'))


def truncate_lost_trajectories(trajectories, final_times, tmax_values):
    """
    For lost particles, set trajectory points after loss to NaN.
    This allows plotting to show trajectories only up to the loss point.
    """
    trajectories_truncated = trajectories.copy()

    for i in range(trajectories.shape[0]):
        for j in range(trajectories.shape[1]):
            if final_times[i, j] < 0.99 * tmax_values[j]:
                trajectories_truncated[i, j:, :] = np.nan
                break

    return trajectories_truncated


