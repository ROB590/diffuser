#!/usr/bin/env python3
"""
visualize_trajectories.py

Loads all .npz trajectory files from a directory and plots the 2D path
(trace of state[:,0] vs. state[:,1]) for each trajectory in one figure.
"""

import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

def visualize_all(root_dir, suffix="*.npz", dim_x=0, dim_y=1, alpha=0.5):
    """
    Plot all trajectories in `root_dir` matching `suffix`.

    Args:
        root_dir (str): Path to directory containing .npz files.
        suffix   (str): Glob pattern (default '*.npz').
        dim_x    (int): State dimension to plot on x-axis (default 0).
        dim_y    (int): State dimension to plot on y-axis (default 1).
        alpha    (float): Line transparency for overlap visibility.
    """
    pattern = os.path.join(root_dir, suffix)
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files found in {root_dir} matching {suffix}")

    plt.figure(figsize=(8, 8))
    for traj_file in files:
        data = np.load(traj_file)
        if 'state' not in data:
            print(f"  Skipping {traj_file!r}: no 'state' key.")
            continue

        states = data['state']  # shape (T, S)
        if states.ndim != 2 or states.shape[1] <= max(dim_x, dim_y):
            print(f"  Skipping {traj_file!r}: state has shape {states.shape}")
            continue

        x = states[:, dim_x]
        y = states[:, dim_y]
        label = os.path.basename(traj_file)
        plt.plot(x, y, linewidth=1, alpha=alpha, label=label)

    plt.xlabel(f"state[:,{dim_x}]")
    plt.ylabel(f"state[:,{dim_y}]")
    plt.title("All Trajectory Paths")
    plt.grid(True)
    # Only show legend if there are few trajectories
    if len(files) <= 20:
        plt.legend(fontsize="small", loc="best", ncol=2)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Visualize all 2D trajectories in a directory of .npz files."
    )
    parser.add_argument(
        "root_dir",
        help="Directory containing your trajectory .npz files"
    )
    parser.add_argument(
        "--pattern", "-p",
        default="*.npz",
        help="Glob pattern to match files (default: '*.npz')"
    )
    parser.add_argument(
        "--dim-x", type=int, default=0,
        help="Index of state dimension for x-axis (default: 0)"
    )
    parser.add_argument(
        "--dim-y", type=int, default=1,
        help="Index of state dimension for y-axis (default: 1)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Line transparency for overlap (default: 0.5)"
    )
    args = parser.parse_args()

    visualize_all(
        root_dir=args.root_dir,
        suffix=args.pattern,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        alpha=args.alpha
    )

if __name__ == "__main__":
    main()
