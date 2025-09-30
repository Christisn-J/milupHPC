#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import os
import argparse
import glob
import logging
import re

# Use non-interactive backend (for environments without display)
matplotlib.use('Agg')

# Predefined selections of keys for plotting
SELECTS = {
    0: {"name": "mass & velocity",                                  "keys": ["m", "v"]},
    1: {"name": "density rate",                                     "keys": ["drhodt"]},
    2: {"name": "density, pressure, energy, speed of sound",        "keys": ["rho", "p", "e", "cs"]},
    3: {"name": "process, smoothing length, number of interactions","keys": ["proc", "sml", "noi"]},
    4: {"name": "stress components",                                "keys": ["Sxx", "Sxy", "Sxz", "Syz"]}
}

# Metadata for each field: display name, LaTeX symbol, unit, and colormap
FIELD_META = {
    "x":    {"name": "Position",         "symbol": r"$\vec{r}$",       "unit": r"$\mathrm{m}$",                                         "cmap": "gray"},
    "m":    {"name": "Mass",             "symbol": r"$m$",             "unit": r"$\mathrm{kg}$",                              "cmap": "viridis"},
    "v":    {"name": "Velocity",         "symbol": r"$\vec{v}$",       "unit": r"$\frac{\mathrm{m}}{\mathrm{s}}$",            "cmap": "plasma"},
    "rho":  {"name": "Density",          "symbol": r"$\rho$",          "unit": r"$\frac{\mathrm{kg}}{\mathrm{m}^3}$",         "cmap": "viridis"},
    "p":    {"name": "Pressure",         "symbol": r"$p$",             "unit": r"$\mathrm{Pa}$",                              "cmap": "plasma"},
    "e":    {"name": "Energy",           "symbol": r"$\epsilon$",      "unit": r"$\mathrm{J}$",                               "cmap": "inferno"},
    "cs":   {"name": "Speed of Sound",   "symbol": r"$c_s$",           "unit": r"$\frac{\mathrm{m}}{\mathrm{s}}$",            "cmap": "magma"},
    "proc": {"name": "Process",          "symbol": r"$\mathrm{proc}$", "unit": r"$-$",                                         "cmap": "tab20"},
    "sml":  {"name": "Smoothing Length", "symbol": r"$h$",             "unit": r"$\mathrm{m}$",                               "cmap": "viridis"},
    "noi":  {"name": "Number of Interactions","symbol": r"$\mathrm{noi}$",  "unit": r"$-$",                                         "cmap": "plasma"},
    "Sxx":  {"name": "Stress XX",        "symbol": r"$\sigma_{xx}$",   "unit": r"$\mathrm{Pa}$",                              "cmap": "coolwarm"},
    "Sxy":  {"name": "Stress XY",        "symbol": r"$\sigma_{xy}$",   "unit": r"$\mathrm{Pa}$",                              "cmap": "coolwarm"},
    "Sxz":  {"name": "Stress XZ",        "symbol": r"$\sigma_{xz}$",   "unit": r"$\mathrm{Pa}$",                              "cmap": "coolwarm"},
    "Syz":  {"name": "Stress YZ",        "symbol": r"$\sigma_{yz}$",   "unit": r"$\mathrm{Pa}$",                              "cmap": "coolwarm"},
    "drhodt": {"name": "Density Rate", "symbol": r"$\frac{d\rho}{dt}$", "unit": r"$\frac{\mathrm{kg}}{\mathrm{m}^3\cdot\mathrm{s}}$", "cmap": "cividis"}
}


FRAME_PADDING = 0.1
PLANES = [("x", "y"), ("x", "z"), ("y", "z")]
extension=".png"

# Optional axis limits and labels configuration
AXES_CONFIG = {
    "x": {"limits": (-1, 1), "labels": f'x [{FIELD_META["x"]["unit"]}]', "index": 0},
    "y": {"limits": (-1, 1), "labels": f'y [{FIELD_META["x"]["unit"]}]', "index": 1},
    "z": {"limits": (-1, 1), "labels": f'z [{FIELD_META["x"]["unit"]}]', "index": 2}
}

def setup_logging():
    """
    Configures the logging format and level.
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
    )
def get_global_extrema(file_list, key, is_vector=False):
    """
    Finds the global minimum and maximum values for a given key
    across multiple HDF5 files.

    Args:
        file_list (list of str): List of HDF5 file paths
        key (str): Dataset key to search in files
        is_vector (bool): Whether the data is vector-valued (e.g. x, v)

    Returns:
        tuple: (mins, maxs), where each is a list for vector fields or a scalar for scalar fields
    """
    if is_vector:
        ndim = None
        mins = None
        maxs = None
    else:
        global_min = np.inf
        global_max = -np.inf

    for h5file in file_list:
        with h5py.File(h5file, 'r') as data_h5:
            if key in data_h5:
                arr = np.array(data_h5[key][:])
                if is_vector:
                    if ndim is None:
                        ndim = arr.shape[1]
                        mins = [np.inf] * ndim
                        maxs = [-np.inf] * ndim
                    for i in range(ndim):
                        mins[i] = min(mins[i], np.min(arr[:, i]))
                        maxs[i] = max(maxs[i], np.max(arr[:, i]))
                else:
                    global_min = min(global_min, np.min(arr))
                    global_max = max(global_max, np.max(arr))
    if is_vector:
        return mins, maxs
    else:
        return global_min, global_max
def slice_particles(coords, data, plane="xy", eps=0.005):
    if plane == "xy":
        mask = np.abs(coords[2]) < eps
    elif plane == "xz":
        mask = np.abs(coords[1]) < eps
    elif plane == "yz":
        mask = np.abs(coords[0]) < eps
    else:
        raise ValueError("Invalid plane: choose 'xy', 'xz', or 'yz'")

    return tuple(coord[mask] for coord in coords), [d[mask] if d is not None else None for d in data]
def plot_2D_slice(planes, coords, data, filename=None, title=None, delta=1e-2, **kwargs):
    logging.info(f"Plotting 2D slice {planes}")
    for ax1, ax2 in planes:

        i1 = AXES_CONFIG[ax1]["index"]
        i2 = AXES_CONFIG[ax2]["index"]
        plane_name = f"{ax1}{ax2}"

        sliced_coords, sliced_data = slice_particles(coords, data, plane=plane_name, eps=delta)
        coords_plane = (sliced_coords[i1], sliced_coords[i2])

        if plane_name == "xy":
            ax3 = "z"
        elif plane_name == "xz":
            ax3 = "y"
        elif plane_name == "yz":
            ax3 = "x"
        else:
            raise ValueError("Invalid plane: choose 'xy', 'xz', or 'yz'")

        plot_2D_scatter(
            coords=coords_plane,  # extract plane
            datas=sliced_data,
            title=f"{title} | Plane {plane_name} thickness: |{ax3} <= {delta:.3e}|",
            filename=f"{filename}_slice_{plane_name}",
            **kwargs
        )
def plot_2D_projection(planes, coords, data, filename=None, title=None, **kwargs):
    logging.info(f"Plotting 2D projection {planes}")
    for ax1, ax2 in planes:
        i1 = AXES_CONFIG[ax1]["index"]
        i2 = AXES_CONFIG[ax2]["index"]
        coords_plane = (coords[i1], coords[i2])
        plane_name = f"{ax1}{ax2}"

        plot_2D_scatter(
            coords_plane,
            data,
            title=f"{title} | Plane {plane_name}",
            filename=f"{filename}_projection_{plane_name}",
            **kwargs
        )
def plot_3D_scatter(coords, datas, filename=None, labels=None, title=None, cmaps=["viridis"], dpi=300, point_size=0.5, axis_config=AXES_CONFIG, vmin=None, vmax=None):
    """
    Creates a 3D scatter plot for one or more scalar fields.

    Args:
        coords (tuple): Tuple of three arrays (x, y, z)
        datas (list): List of scalar data arrays
        labels (list): List of labels for each dataset
        title (str): Overall plot title
        cmaps (list): List of colormaps
        dpi (int): Plot resolution
        point_size (float): Size of scatter points
        axis_config (dict): Axis limits and labels
        vmin (list): Minimum color limits for each dataset
        vmax (list): Maximum color limits for each dataset
    """
    logging.info(f"Plotting 3D scatter")
    n = len(datas)
    fig, axs = plt.subplots(1, n, subplot_kw={'projection': '3d'}, figsize=(6 * n, 6), dpi=dpi)

    # Make sure axs is iterable even for n=1
    if n == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        label = labels[i] if labels else f"Data {i}"
        data = datas[i]
        cmap = cmaps[i] if i < len(cmaps) else "viridis"

        if data is None:
            ax.text(0.5, 0.5, 0.5, f"No data for {label}", ha='center', va='center')
            ax.set_axis_off()
            continue

        sc = ax.scatter(
            coords[0], coords[1], coords[2],
            c=data,
            cmap=cmap,
            vmin=vmin[i] if vmin else None,
            vmax=vmax[i] if vmax else None,
            s=point_size,
            alpha=0.8
        )
        ax.set_xlim(*axis_config["x"]["limits"])
        ax.set_ylim(*axis_config["y"]["limits"])
        ax.set_zlim(*axis_config["z"]["limits"])

        ax.set_xlabel(axis_config["x"]["labels"])
        ax.set_ylabel(axis_config["y"]["labels"])
        ax.set_zlabel(axis_config["z"]["labels"])

        # Try to extract label content inside parentheses for colorbar label
        match = re.search(r'\((.*?)\)', label)
        cbar_label = match.group(1) if match else label

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_title(label)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(f"{filename}{extension}", bbox_inches='tight')
    plt.close(fig)
def plot_2D_scatter(coords, datas, filename=None, labels=None, title=None, cmaps=["viridis"], dpi=300, point_size=0.5, axis_config=AXES_CONFIG, vmin=None, vmax=None):
    """
    Creates a 2D scatter plot for one or more scalar fields.

    Args:
        coords (tuple): Tuple of two arrays (x, y)
        datas (list): List of scalar data arrays
        labels (list): Labels for each dataset
        title (str): Overall plot title
        cmaps (list): List of colormaps
        dpi (int): Plot resolution
        point_size (float): Size of scatter points
        axis_config (dict): Axis limits and labels
        vmin (list): Minimum color limits for each dataset
        vmax (list): Maximum color limits for each dataset
    """
    logging.info(f"Plotting 2D scatter {filename[-2:]}")
    n = len(datas)
    fig, axs = plt.subplots(1, n, figsize=(6 * n, 6), dpi=dpi)

    # Ensure axs is iterable
    if n == 1:
        axs = [axs]

    for i, (ax, data, label, cmap) in enumerate(zip(axs, datas, labels, cmaps)):
        if data is None:
            ax.text(0.5, 0.5, f"No data for {label}", ha='center', va='center')
            ax.set_axis_off()
            continue

        sc = ax.scatter(
            coords[0], coords[1],
            c=data,
            cmap=cmap,
            s=point_size,
            vmin=vmin[i] if vmin else None,
            vmax=vmax[i] if vmax else None,
            alpha=0.8
        )

        ax.set_xlim(*axis_config["x"]["limits"])
        ax.set_ylim(*axis_config["y"]["limits"])

        ax.set_xlabel(axis_config["x"]["labels"])
        ax.set_ylabel(axis_config["y"]["labels"])

        # Try to extract label content inside parentheses for colorbar label
        match = re.search(r'\((.*?)\)', label)
        cbar_label = match.group(1) if match else label

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_title(label)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.savefig(f"{filename}{extension}", bbox_inches='tight')
    plt.close(fig)
def main(args):
    """
    Main function to process input files and generate plots.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Determine keys to plot based on user input
    if args.key:
        keys = args.key
    elif args.plot_type is not None:
        keys = SELECTS[args.plot_type]["keys"]
    else:
        logging.error("You must specify either --plot_type or --key.")
        return

    # Create output directory if it does not exist
    os.makedirs(args.output, exist_ok=True)

    # Find and sort HDF5 files in the input directory
    file_list = sorted(glob.glob(os.path.join(args.data, "*.h5")), key=os.path.basename)

    # Automatically adjust AXES_CONFIG based on global x min/max
    logging.info("Computing global extrema for 'x' to adjust axis limits...")
    x_mins, x_maxs = get_global_extrema(file_list, "x", is_vector=True)

    for i, label in zip(range(args.dim), AXES_CONFIG.keys()):
        AXES_CONFIG[label]["limits"] = (x_mins[i] - FRAME_PADDING, x_maxs[i] + FRAME_PADDING)
        logging.info(f"Updated axis '{label}' limits: {AXES_CONFIG[label]['limits']}")

    if args.slice:
        # Compute average data extent (range) for slice delta
        extents = [x_maxs[i] - x_mins[i] for i in range(args.dim)]
        logging.info(f"Extents: {extents}")
        avg_extent = np.mean(extents)
        thickness = 0.01 * avg_extent  # e.g. 1% of average range
        logging.info(f"Using slice delta (thickness): {thickness:.5g}")


# If requested, compute global extrema for color normalization
    if args.extrema:
        vmins = {}
        vmaxs = {}

        for key in keys:
            if key != "noi":  # Skip 'noi' as it may be categorical
                vmin, vmax = get_global_extrema(file_list, key)
                logging.info(f"Global min/max for '{key}': {vmin} / {vmax}")
                vmins[key], vmaxs[key] = vmin, vmax

    # Loop over files and generate plots
    for i, h5file in enumerate(file_list):
        logging.info(f"Plotting timestep {i}")
        logging.info(f"Processing file: {h5file}")

        with h5py.File(h5file, 'r') as data_h5:
            # Load positions (required for all plots)
            key = "x"
            if key in data_h5:
                positions = np.array(data_h5[key][:])
                logging.info(f"Loaded '{key}' with shape {positions.shape}")
            else:
                logging.error(f"Key '/{key}' not found in file {h5file}")
                continue

            # Load requested data keys
            data_select = {}

            for key in keys:
                if key in data_h5:
                    arr = np.array(data_h5[key][:], dtype=np.float64)

                    # If vector data (2D or 3D), compute norm
                    if arr.ndim == 2 and arr.shape[1] in [2, 3]:
                        logging.info(f"'{key}' is a vector field. Computing norm.")
                        arr = np.linalg.norm(arr, axis=1)

                    data_select[key] = arr
                    logging.info(f"Loaded '{key}' with shape {arr.shape}")
                else:
                    logging.warning(f"Key '/{key}' not found in file {h5file}")
                    data_select[key] = None  # Mark missing data as None

        # Set output filename and extension
        filename = f"ts{i:06d}_{SELECTS[args.plot_type]["name"]}"
        title = f"Timestep {i}"
        data = [data_select.get(key) for key in keys]
        cmaps = [FIELD_META[key]["cmap"] for key in keys]
        labels = [f'{FIELD_META[key]["name"]} ({FIELD_META[key]["symbol"]} in [{FIELD_META[key]["unit"]}])' for key in keys]
        extremes = {
            "min": [vmins.get(key) if args.extrema else None for key in keys],
            "max": [vmaxs.get(key) if args.extrema else None for key in keys]
        }

        # Plot in 3D or 2D based on argument
        if args.dim == 3:
            coords = (positions[:, 0], positions[:, 1], positions[:, 2])
            funk = plot_3D_scatter
            planes=PLANES
        elif args.dim == 2:
            coords = (positions[:, 0], positions[:, 1], np.zeros_like(positions[:, 1]))
            funk=plot_2D_scatter
            planes=[PLANES[0]]
        else:
            continue

        funk(
            coords,
            data,
            labels=labels,
            title=title,
            cmaps=cmaps,
            vmin=extremes["min"],
            vmax=extremes["max"],
            filename=os.path.join(args.output, f"{filename}")
        )

        # --- Projektionen in 2D (für 3D-Daten) ---
        if args.projection:
            plot_2D_projection(
                planes,
                coords,
                data,
                labels=labels,
                title=title,
                cmaps=cmaps,
                vmin=extremes["min"],
                vmax=extremes["max"],
                filename=os.path.join(args.output, f"{filename}")
            )
        # --- Slices in 2D (für 3D-Daten) ---
        if args.slice:
            plot_2D_slice(
                planes,
                coords,
                data,
                delta=thickness,
                labels=labels,
                title=title,
                cmaps=cmaps,
                vmin=extremes["min"],
                vmax=extremes["max"],
                filename=os.path.join(args.output, f"{filename}")
            )

        logging.info(f"Saved plot to {args.output}")

if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(description="2D/3D plotting tool for particle data from HDF5 files.")
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="output",
        help="Input directory containing HDF5 files."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory for saving plots."
    )
    parser.add_argument(
        "--plot_type", "-p",
        type=int,
        choices=range(len(SELECTS.keys())),
        help="Plot type selection: "
             f"[0]: {SELECTS[0]["name"]}; "
             f"[1]: {SELECTS[1]["name"]}; "
             f"[2]: {SELECTS[2]["name"]}; "
             f"[3]: {SELECTS[3]["name"]}; "
             f"[4]: {SELECTS[4]["name"]}."
    )
    parser.add_argument(
        "--key", "-k",
        type=str,
        nargs='+',
        help="List of keys to plot (overrides --plot_type)."
    )
    parser.add_argument(
        "--dim", "-D",
        type=int,
        choices=[2, 3],
        default=3,
        help="Problem dimensionality: 2 or 3 (default: 3)."
    )
    parser.add_argument(
        "--extrema", "-e",
        action='store_true',
        help="Enable global extrema computation for color scaling."
    )

    parser.add_argument(
        "--slice",
        action='store_true',
        help="Enable 2D slices through coordinate planes."
    )

    parser.add_argument(
        "--projection",
        action='store_true',
        help="Enable 2D projection plots onto coordinate planes."
    )


    args = parser.parse_args()

    if args.plot_type not in range(len(SELECTS.keys())) and not args.key:
        parser.print_help()
        exit(1)

    main(args)

