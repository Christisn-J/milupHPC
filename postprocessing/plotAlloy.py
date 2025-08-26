#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import os
import argparse
import glob
import logging

# Use non-interactive backend for environments without display
matplotlib.use('Agg')

# === Achsenlimits (optional anpassbar) ===
AX_LIMITS = {
    'x': (-0.05, 0.05),
    'y': (-0.05, 0.05),
    'z': (-0.5, 0.5)
}

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
    )

def parse_args():
    parser = argparse.ArgumentParser(description="3D plot of particle positions.")
    parser.add_argument("--data", "-d", type=str, default="output", help="Input directory")
    parser.add_argument("--output", "-o", type=str, default="output", help="Output directory")
    parser.add_argument("--maxpoints", "-m", type=int, default=10e9, help="Max points for plotting")
    parser.add_argument("--plot_type", "-p", type=int, choices=[0, 1, 2, 3], required=True,
                        help="Plot type: [0]: rho; [1]: positions by process; [2]: rho, p, e; [3]: rho, p, e, noi")
    return parser.parse_args()

def apply_axis_limits(ax):
    ax.set_xlim(*AX_LIMITS['x'])
    ax.set_ylim(*AX_LIMITS['y'])
    # ax.set_zlim(*AX_LIMITS['z'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

def plot_2d_projection(x, y, rho, axis_labels, title, filename, output_dir, cmap="viridis", dpi=150):
    """
    Erzeugt eine 2D-Scatter-Projektion mit Farbkodierung (z. B. nach Dichte).
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    sc = ax.scatter(x, y, c=rho, cmap=cmap, s=1.0)
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Density")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
def plot_2d_slice(x, y, z, rho, plane="xy", eps=0.005, output_dir=".", filename="slice.png", title="", cmap="viridis", dpi=150):
    """
    Plot a 2D slice of the particle distribution near a selected plane by showing only particles close to that plane.

    Parameters:
        x, y, z       : Arrays of particle positions
        rho           : Array of particle densities (or other scalar to color)
        plane         : "xy", "xz", or "yz"
        eps           : Slice thickness (half-width) around plane (e.g. |z|<eps for xy slice)
        output_dir    : Directory to save plot
        filename      : Output image filename
        title         : Plot title
        cmap          : Matplotlib colormap
        dpi           : Plot resolution
    """
    if plane == "xy":
        mask = np.abs(z) < eps
        x_proj, y_proj = x[mask], y[mask]
        c_proj = rho[mask]
        xlabel, ylabel = "x", "y"
    elif plane == "xz":
        mask = np.abs(y) < eps
        x_proj, y_proj = x[mask], z[mask]
        c_proj = rho[mask]
        xlabel, ylabel = "x", "z"
    elif plane == "yz":
        mask = np.abs(x) < eps
        x_proj, y_proj = y[mask], z[mask]
        c_proj = rho[mask]
        xlabel, ylabel = "y", "z"
    else:
        raise ValueError("Invalid plane: choose 'xy', 'xz', or 'yz'")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    sc = ax.scatter(x_proj, y_proj, c=c_proj, cmap=cmap, s=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Density (ρ)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
def main():
    setup_logging()
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)
    file_list = sorted(glob.glob(os.path.join(args.data, "*.h5")), key=os.path.basename)

    for i, h5file in enumerate(file_list):
        logging.info(f"Processing {h5file}")
        with h5py.File(h5file, 'r') as data:
            positions = np.array(data["x"][:])
            process = np.array(data["proc"][:])
            N = positions.shape[0]

            rho = np.array(data["rho"][:])
            p = np.array(data["p"][:]) if "p" in data else None
            e = np.array(data["e"][:]) if "e" in data else None
            noi = np.array(data["noi"][:]) if "noi" in data else None

        if N > args.maxpoints:
            logging.info(f"Downsampling from {N} to {args.maxpoints}")
            idx = np.random.choice(N, args.maxpoints, replace=False)
            positions = positions[idx]
            process = process[idx]
            rho = rho[idx]
            if p is not None: p = p[idx]
            if e is not None: e = e[idx]
            if noi is not None: noi = noi[idx]


        logging.info(f"Plotting timestep {i}")
        out_path = os.path.join(args.output, f"Process3D_type{args.plot_type}_{i:06d}.png")

        # Plot Types
        if args.plot_type == 0:
            fig = plt.figure(dpi=300, figsize=(10, 7))
            ax = fig.add_subplot(projection='3d')
            sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                            c=rho, cmap='viridis', s=0.5, alpha=0.8)
            apply_axis_limits(ax)
            fig.colorbar(sc, ax=ax, pad=0.1).set_label('Density (rho)')
            ax.set_title(f"Density (rho) - Timestep {i}")

        elif args.plot_type == 1:

            basename = os.path.splitext(os.path.basename(h5file))[0]
            x=positions[:, 0]
            y=positions[:, 1]
            z=positions[:, 2]
            # 2D-Projektionen
            plot_2d_projection(x, y, rho, ("x", "y"),
                               f"{basename} (x-y projection)",
                               f"{basename}_proj_xy.png",
                               args.output)
            plot_2d_projection(x, z, rho, ("x", "z"),
                               f"{basename} (x-z projection)",
                               f"{basename}_proj_xz.png",
                               args.output)
            plot_2d_projection(y, z, rho, ("y", "z"),
                               f"{basename} (y-z projection)",
                               f"{basename}_proj_yz.png",
                               args.output)

            slice_eps = 0.005  # tolerance for being in plane

            # x, y, z, rho already loaded earlier
            plot_2d_slice(x, y, z, rho, plane="xy", eps=slice_eps,
                          output_dir=args.output,
                          filename=f"{basename}_slice_xy.png",
                          title=f"Slice in x-y (|z|<{slice_eps}), t={i:.2e}")

            plot_2d_slice(x, y, z, rho, plane="xz", eps=slice_eps,
                          output_dir=args.output,
                          filename=f"{basename}_slice_xz.png",
                          title=f"Slice in x-z (|y|<{slice_eps}), t={i:.2e}")

            plot_2d_slice(x, y, z, rho, plane="yz", eps=slice_eps,
                          output_dir=args.output,
                          filename=f"{basename}_slice_yz.png",
                          title=f"Slice in y-z (|x|<{slice_eps}), t={i:.2e}")

            fig = plt.figure(dpi=300, figsize=(10, 7))
            ax = fig.add_subplot(projection='3d')
            vmin = np.min(process)
            vmax = np.max(process)

            cmap = plt.get_cmap('tab20', vmax - vmin + 1)  # diskrete Farben
            sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                            c=process, cmap=cmap, s=0.5,
                            vmin=vmin - 0.5, vmax=vmax + 0.5,
                            alpha=0.8)
            apply_axis_limits(ax)
            fig.colorbar(sc, ax=ax, pad=0.1, label='Process ID')
            ax.set_title(f"Particle Positions by Process - Timestep {i}")

        elif args.plot_type == 2:
            fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(18, 6), dpi=300)
            datasets = {'Density (rho)': rho, 'Pressure (p)': p, 'Energy (e)': e}
            cmaps = {'Density (rho)': 'viridis', 'Pressure (p)': 'plasma', 'Energy (e)': 'inferno'}

            for idx, (label, data_array) in enumerate(datasets.items()):
                ax = axs[idx]
                if data_array is None:
                    ax.text(0.5, 0.5, 0.5, f"No data for {label}", ha='center', va='center')
                    ax.set_axis_off()
                    continue
                sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                c=data_array, cmap=cmaps[label], s=0.5, alpha=0.8)
                apply_axis_limits(ax)
                ax.set_title(f"{label} - Timestep {i}")
                fig.colorbar(sc, ax=ax, pad=0.1)

        elif args.plot_type == 3:
            fig, axs = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(18, 14), dpi=300)
            axs = axs.flatten()
            datasets = {'Density (rho)': rho, 'Pressure (p)': p, 'Energy (e)': e, 'Noise (noi)': noi}
            cmaps = {'Density (rho)': 'viridis', 'Pressure (p)': 'plasma', 'Energy (e)': 'inferno', 'Noise (noi)': 'magma'}

            for idx, (label, data_array) in enumerate(datasets.items()):
                ax = axs[idx]
                if data_array is None:
                    ax.text(0.5, 0.5, 0.5, f"No data for {label}", ha='center', va='center')
                    ax.set_axis_off()
                    continue
                sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                                c=data_array, cmap=cmaps[label], s=0.5, alpha=0.8)
                apply_axis_limits(ax)
                ax.set_title(f"{label} - Timestep {i}")
                fig.colorbar(sc, ax=ax, pad=0.1)

        else:
            logging.warning(f"Unknown plot_type {args.plot_type}. Skipping.")
            continue

        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
