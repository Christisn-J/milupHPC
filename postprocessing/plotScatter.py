#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import h5py
import os, sys
import argparse
import glob
import logging
import re

import utility
from constants import si_prefixes, FIELD_META
from help import find_resource_files, read_material

utility.setup_dynamic_plotStyle()
utility.setup_global_latex()

# Use non-interactive backend (for environments without display)
matplotlib.use('Agg')
plt.rcParams["agg.path.chunksize"] = 10000
plt.rcParams["path.simplify"] = True
plt.rcParams["path.simplify_threshold"] = 1.0

reverse = True

AXES_CONFIG = utility.AXES_CONFIG
FRAME_PADDING = utility.FRAME_PADDING
PLANES = utility.PLANES
EXTENSION = utility.EXTENSION
DTYPE = utility.EXTENSION
SCALE = utility.SCALE
DEFAULT_MARKER_ATTRS = utility.DEFAULT_MARKER_ATTRS

# Predefined selections of keys for plotting
SELECTS = {
    0: {"name": "mechanics", "slug": "mechanics", "keys": ["x", "m", "v", "E"]},
    1: {"name": "change rates", "slug": "change_rates", "keys": ["drhodt", "dedt", "a"]},
    2: {"name": "hydro", "slug": "hydro", "keys": ["rho", "p", "e", "cs"]},
    3: {"name": "process", "slug": "process", "keys": ["proc", "sml", "noi", "matId"]},
    4: {"name": "stress", "slug": "stress", "keys": ["Sxx", "Sxy", "Sxz", "Syz"]},
    5: {"name": "velocity", "slug": "v", "keys": ["v"]},
    6: {"name": "material id", "slug": "matId", "keys": ["matId"]},
    7: {"name": "rho", "slug": "rho", "keys": ["rho"]},
    8: {"name": "process", "slug": "proc", "keys": ["proc"]},
    9: {"name": "number of interactions", "slug": "noi", "keys": ["noi"]},
    10: {"name": "pressure", "slug": "p", "keys": ["p"]},
    11: {"name": "specific energy", "slug": "e", "keys": ["e"]},
    12: {"name": "speed of sound", "slug": "cs", "keys": ["cs"]}
}


def plot_2D_slice(planes, coords, data, filename=None, title=None, delta=1e-2, **kwargs):
    logging.info(f"Plotting 2D slice {planes}")
    for ax1, ax2 in planes:
        plane_name = f"{ax1}{ax2}"

        sliced_coords, sliced_data = utility.slice_particles(coords, data, plane=plane_name, eps=delta)
        coords_plane = (sliced_coords[AXES_CONFIG[ax1]["index"]], sliced_coords[AXES_CONFIG[ax2]["index"]])

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
            title=utility.format_math_text(rf"{title} | Plane {plane_name} | $\Delta_{{{ax3}}} \le {delta:.3e}$"),
            filename=f"{filename}_slice_{plane_name}",
            **kwargs
        )


def plot_2D_projection(planes, coords, data, filename=None, title=None, **kwargs):
    logging.info(f"Plotting 2D projection {planes}")
    for ax1, ax2 in planes:
        coords_plane = (coords[AXES_CONFIG[ax1]["index"]], coords[AXES_CONFIG[ax2]["index"]])
        plane_name = f"{ax1}{ax2}"

        plot_2D_scatter(
            coords_plane,
            data,
            title=utility.format_math_text(f"{title} | Plane {plane_name}"),
            filename=f"{filename}_projection_{plane_name}",
            **kwargs
        )


def plot_3D_scatter(coords, datas, filename=None, labels=None, keys=None, title=None, cmaps=["viridis"], dpi=300, axis_config=AXES_CONFIG, vmin=None, vmax=None, marker_atts=DEFAULT_MARKER_ATTRS,
                    material_atts=None, note=None, extension=None):
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
    utility.setup_dynamic_plotStyle(n=n, dim=3)

    fig, axs = plt.subplots(1, n, subplot_kw={'projection': '3d'}, figsize=plt.rcParams["figure.figsize"], dpi=dpi)

    # Make sure axs is iterable even for n=1
    if n == 1:
        axs = [axs]

    for i, ax in enumerate(axs):
        label = utility.format_math_text(labels[i]) if labels else f"Data {i}"
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

            s=marker_atts["s"],
            alpha=marker_atts["alpha"],
            rasterized=marker_atts["rasterized"],
            linewidths=marker_atts["linewidths"],
            marker=marker_atts["marker"]
        )
        # --- View-Winkel setzen ---
        ax.view_init(elev=30, azim=-60)

        # --- Z-Achse nach links verschieben ---
        ax.zaxis._axinfo["juggled"] = (1, 2, 0)

        ax.set_box_aspect([1, 1, 1])

        ax.set_xlim(*axis_config["x"]["limits"])
        ax.set_ylim(*axis_config["y"]["limits"])
        ax.set_zlim(*axis_config["z"]["limits"])

        ax.set_xlabel(utility.format_math_text(axis_config["x"]["labels"]), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
        ax.set_ylabel(utility.format_math_text(axis_config["y"]["labels"]), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
        ax.set_zlabel(utility.format_math_text(axis_config["z"]["labels"]), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)

        # Try to extract label content inside parentheses for colorbar label
        match = re.search(r'\((.*?)\)', label)
        plt.rcParams.update({
            "xtick.minor.visible": True,
            "ytick.minor.visible": True
        })
        cbar_label = match.group(1) if match else label

        if cmap == "tab10" or (isinstance(cmap, str) and cmap.startswith("tab")):
            if keys[i] == 'matId':
                cbar = utility.setup_discrete_colormap(fig, ax, sc, data, base_cmap_name=cmap, material_names=material_atts)
            else:
                cbar = utility.setup_discrete_colormap(fig, ax, sc, data, base_cmap_name=cmap)

        else:
            cbar = fig.colorbar(sc, ax=ax, pad=0.05)

        cbar.set_label(cbar_label)
        utility.style_colorbar(cbar, n=n, dim=3)
        ax.set_title(utility.format_math_text(label), fontsize=plt.rcParams["axes.titlesize"])

    if title is not None:
        if n==1:
            fig.suptitle(utility.format_math_text(title), fontsize=plt.rcParams["figure.titlesize"])
        else:
            fig.suptitle(utility.format_math_text(title), fontsize=plt.rcParams["figure.titlesize"], y=1.0)

    # --- Optionale Fußnote unter dem Plot ---
    if note:
        fig.text(0.5, 0.01, note, ha="center", va="bottom")

    plt.tight_layout()
    for e in extension:
        plt.savefig(f"{filename}.{e}", dpi=dpi, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)


def plot_2D_scatter(coords, datas, filename=None, labels=None, keys=None, title=None, cmaps=["viridis"], axis_config=AXES_CONFIG, vmin=None, vmax=None, dpi=300, marker_atts=DEFAULT_MARKER_ATTRS,
                    material_atts=None, note=None, extension=None):
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
    if filename and ("slice" in filename or "projection" in filename):
        logging.info(f"Plotting 2D scatter {filename[-2:]}")
    else:
        logging.info(f"Plotting 2D scatter")

    n = len(datas)
    utility.setup_dynamic_plotStyle(n=n, dim=2)

    fig, axs = plt.subplots(1, n, figsize=plt.rcParams["figure.figsize"], dpi=dpi)

    # Ensure axs is iterable
    if n == 1:
        axs = [axs]

    for i, (ax, data, label, cmap) in enumerate(zip(axs, datas, labels, cmaps)):
        if data is None:
            ax.text(0.5, 0.5, utility.format_math_text(f"No data for {label}"), ha='center', va='center')
            ax.set_axis_off()
            continue

        sc = ax.scatter(
            coords[0], coords[1],
            c=data,
            cmap=cmap,
            vmin=vmin[i] if vmin else None,
            vmax=vmax[i] if vmax else None,

            s=marker_atts["s"],
            alpha=marker_atts["alpha"],
            rasterized=marker_atts["rasterized"],
            linewidths=marker_atts["linewidths"],
            marker=marker_atts["marker"]
        )

        ax.set_xlim(*axis_config["x"]["limits"])
        ax.set_ylim(*axis_config["y"]["limits"])

        # --- ACHSENVERHÄLTNIS FESTLEGEN ---
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel(utility.format_math_text(axis_config["x"]["labels"]), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
        ax.set_ylabel(utility.format_math_text(axis_config["y"]["labels"]), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)

        # Try to extract label content inside parentheses for colorbar label
        match = re.search(r'\((.*?)\)', label)
        plt.rcParams.update({
            "xtick.minor.visible": True,
            "ytick.minor.visible": True
        })
        cbar_label = match.group(1) if match else label

        if cmap == "tab10" or (isinstance(cmap, str) and cmap.startswith("tab")):
            if keys[i] == 'matId':
                cbar = utility.setup_discrete_colormap(fig, ax, sc, data, base_cmap_name=cmap, material_names=material_atts)
            else:
                cbar = utility.setup_discrete_colormap(fig, ax, sc, data, base_cmap_name=cmap)
        else:
            cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)
        utility.style_colorbar(cbar, n=n)
        ax.set_title(utility.format_math_text(label), fontsize=plt.rcParams["axes.titlesize"])

    if title is not None:
        fig.suptitle(utility.format_math_text(title), fontsize=plt.rcParams["figure.titlesize"])#, y=1.02)

    # --- Optionale Fußnote unter dem Plot ---
    if note:
        fig.text(0.5, 0.01, note, ha="center", va="bottom")

    plt.tight_layout()
    for e in extension:
        plt.savefig(f"{filename}.{e}", dpi=dpi, bbox_inches='tight', pad_inches=0.2)
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
        slug = "_".join(keys)
    elif args.plotType is not None:
        keys = SELECTS[args.plotType]["keys"]
        slug = SELECTS[args.plotType]['slug']
    else:
        logging.error("You must specify either --plotType or --key.")
        return

    # Create output directory if it does not exist
    os.makedirs(args.output, exist_ok=True)

    # Find and sort HDF5 files in the input directory
    file_list = sorted(glob.glob(os.path.join(args.data, f"*{EXTENSION['H5']}")), key=os.path.basename, reverse=reverse)

    # Automatically adjust AXES_CONFIG based on global x min/max
    logging.info("Computing global extrema for 'x' to adjust axis limits...")
    x_mins, x_maxs = utility.get_global_extrema(file_list, "x", is_vector=True)
    logging.info(file_list)

    for i, axis in zip(range(args.dim), AXES_CONFIG.keys()):
        if i < len(x_mins) and i < len(x_maxs):
            logging.info(f"'{axis}' {x_mins} {x_maxs}")

            AXES_CONFIG[axis]["limits"] = ((x_mins[i] - FRAME_PADDING) / si_prefixes[args.scale["x"]]["factor"], (x_maxs[i] + FRAME_PADDING) / si_prefixes[args.scale["x"]]["factor"])
            logging.info(f"Updated axis '{axis}' limits: {AXES_CONFIG[axis]['limits']}")

            AXES_CONFIG[axis]['labels'] = utility.format_math_text(f'${axis}$ [${si_prefixes[args.scale["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]')
            logging.info(f"Updated axis '{axis}' labels: {AXES_CONFIG[axis]['labels']}")

        else:
            logging.warning(f"Skipping axis '{axis}' due to insufficient extrema data.")

    if args.slice:
        # Compute average data extent (range) for slice delta
        extents = [x_maxs[i] - x_mins[i] for i in range(args.dim)]
        logging.info(f"Extents: {extents}")
        avg_extent = np.mean(extents)
        thickness = 0.01 * avg_extent  # e.g. 1% of average range
        logging.info(f"Using slice Δ = {thickness:.5g}")

    # If requested, compute global extrema for color normalization
    if args.extrema:
        vmins = {}
        vmaxs = {}

        for key in keys:
            if key != "noi":  # Skip 'noi' as it may be categorical
                vmin, vmax = utility.get_global_extrema(file_list, key)
                logging.info(f"Global min/max for '{key}': {vmin} / {vmax}")
                vmins[key], vmaxs[key] = vmin, vmax

    if args.config:
        t_0 = float(utility.read_from_file(args.config, "t_0", fallback=0.0))
        t_end = float(utility.read_from_file(args.config, "timeEnd", fallback=0.0))
        if args.resource:
            steps = int(utility.read_from_file(args.resource, "NSTEPs", fallback=None))
        else:
            steps = len(file_list)

        delta_t = t_end / steps  # oder aus Resource-Datei lesen
        time = np.linspace(t_0 + delta_t, t_end, steps) / si_prefixes[args.scale["t"]]["factor"]

    # Loop over files and generate plots
    for idx, h5file in enumerate(file_list):
        if reverse:
            i = len(file_list) - idx - 1
        else:
            i = idx

        logging.info(f"Plotting timestep {i}")
        logging.info(f"Processing file: {h5file}")

        with h5py.File(h5file, 'r') as data_h5:
            # Load positions (required for all plots)
            key = "x"
            if key in data_h5:
                positions = np.array(data_h5[key][:]) / si_prefixes[args.scale["x"]]["factor"]
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
        filename = f"ts{i:06d}_{slug}"
        if args.config and i <= len(time):
            title = utility.format_math_text(
                rf"Time: $N_{{\mathrm{{snap}}}}={i:06d}$, $t={time[i]:.3f}$ ${si_prefixes[args.scale['t']]['abbr']}{FIELD_META['t']['unit'].strip('$')}$"
                "\n"
                rf"Particles: $N_{{\mathrm{{par, tot}}}}={positions.shape[0]:0.3e}$"
            )
        else:
            title = utility.format_math_text(
                rf"Time: $N_{{\mathrm{{snap}}}} = {i:06d}$"
                "\n"
                rf"Particles: $N_{{\mathrm{{par, tot}}}} = {positions.shape[0]:0.3e}$"
            )

        data = [data_select.get(key) for key in keys]
        cmaps = [FIELD_META[key]["cmap"] for key in keys]
        labels = [utility.format_math_text(f'{FIELD_META[key]["name"]} ({FIELD_META[key]["symbol"]} [{FIELD_META[key]["unit"]}])') for key in keys]
        extremes = {"min": [vmins.get(key) if args.extrema else None for key in keys], "max": [vmaxs.get(key) if args.extrema else None for key in keys]}

        if args.dynamicRender:
            marker_atts = utility.dynamic_render_config(positions.shape[0], args.dim)
            logging.info(f"Dynamic rendering enabled for N={positions.shape[0]}: {marker_atts}")
            positions = positions[::marker_atts["skip"]]
            for key, data_array in data_select.items():
                if data_array is not None:
                    data_select[key] = data_array[::marker_atts["skip"]]
        else:
            logging.info(f"Dynamic rendering disabled; using full data with N={positions.shape[0]}")
            marker_atts = DEFAULT_MARKER_ATTRS

        material_atts = None
        if args.material:
            materials = read_material(args.material)
            material_atts = {int(m["ID"]): m["name"] for m in materials if "ID" in m and "name" in m}

        # Plot in 3D or 2D based on argument
        if args.dim == 3:
            coords = (positions[:, 0], positions[:, 1], positions[:, 2])
            funk = plot_3D_scatter
            planes = PLANES
        elif args.dim == 2:
            coords = (positions[:, 0], positions[:, 1], np.zeros_like(positions[:, 1]))
            funk = plot_2D_scatter
            planes = [PLANES[0]]
        else:
            continue

        funk(
            coords,
            data,
            labels=labels,
            keys=keys,
            title=title,
            cmaps=cmaps,
            vmin=extremes["min"],
            vmax=extremes["max"],
            filename=os.path.join(args.output, f"{filename}"),
            marker_atts=marker_atts,
            material_atts=material_atts,
            dpi=args.dpi,
            extension=args.extension
        )

        # --- Projektionen in 2D (für 3D-Daten) ---
        if args.projection:
            plot_2D_projection(
                planes,
                coords,
                data,
                labels=labels,
                keys=keys,
                title=title,
                cmaps=cmaps,
                vmin=extremes["min"],
                vmax=extremes["max"],
                filename=os.path.join(args.output, f"{filename}"),
                marker_atts=marker_atts,
                material_atts=material_atts,
                dpi=args.dpi,
                extension=args.extension
            )
        # --- Slices in 2D (für 3D-Daten) ---
        if args.slice:
            plot_2D_slice(
                planes,
                coords,
                data,
                delta=thickness,
                labels=labels,
                keys=keys,
                title=title,
                cmaps=cmaps,
                vmin=extremes["min"],
                vmax=extremes["max"],
                filename=os.path.join(args.output, f"{filename}"),
                marker_atts=marker_atts,
                material_atts=material_atts,
                dpi=args.dpi,
                extension=args.extension
            )

        logging.info(f"Saved plot to {args.output}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="2D/3D plotting tool for particle data from HDF5 files.")
    parser.add_argument("--data", "-p", type=str, default="output/timestep/", help="Input directory containing HDF5 files.")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config file")
    parser.add_argument("--resource", "-r", type=str, default=None, help="Resource file")
    parser.add_argument("--material", "-m", type=str, default=None, help="Material file")
    parser.add_argument("--output", "-o", type=str, default="output/visualized/", help="Output directory for saving plots.")
    parser.add_argument("--verbose", "-v", type=int, choices=[1, 2, 3], default=3, help="Enable verbose output")
    parser.add_argument("--dim", "-d", type=int, choices=[2, 3], default=3, help="Problem dimensionality (default: 3).")
    parser.add_argument("--extrema", "-e", action='store_true', help="Enable global extrema computation for color scaling.")
    parser.add_argument("--dynamicRender", "-dR", action="store_true", help="Enable dynamic rendering, skipping points for better performance.")
    parser.add_argument("--slice", action='store_true', help="Enable 2D slices through coordinate planes.")
    parser.add_argument("--projection", action='store_true', help="Enable 2D projection plots onto coordinate planes.")
    parser.add_argument("--select", "-s", type=int, nargs='+', help="List of particle IDs to highlight")
    parser.add_argument("--scale", nargs="+", default=[], metavar="axis=factor", help="Scaling for axes, e.g. --scale x=centi t=kilo")
    parser.add_argument("--key", "-k", type=str, nargs='+', help="List of keys to plot (overrides --plotType).")
    parser.add_argument("--plotType", "-pT", type=int, choices=SELECTS.keys(), help="Plot type selection: " + "; ".join(f"[{k}]: {SELECTS[k]['name']}" for k in sorted(SELECTS)))
    parser.add_argument("--dpi", type=int, default=300, help="Set DPI for all output plots (default: 300).")
    parser.add_argument("--extension", nargs="+", default=["png"], choices=["png", "pdf", "svg", "jpg"], help="Output file formats (default: png). Example: -e png pdf svg")

    args = parser.parse_args()
    args.extension=["png", "pdf"]
    # Set logging level based on verbosity flag
    if args.verbose >= 3:
        log_level = logging.DEBUG
    elif args.verbose == 2:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    utility.setup_logging()
    # logging.basicConfig(
    #     level=log_level,
    #     format='[%(levelname)s] %(message)s',
    #     handlers=[logging.StreamHandler(sys.stdout)]
    # )

    base = os.path.abspath(args.data)
    configured_path = os.path.join(base, "configured")

    # --- config ---
    if args.config is None:
        args.config = sorted([os.path.join(configured_path, f) for f in os.listdir(configured_path) if f.endswith(".info")])[0]

    # --- material ---
    if args.material is None:
        args.material = sorted([os.path.join(configured_path, f) for f in os.listdir(configured_path) if f.endswith(".cfg")])[0]

    # --- resource ---
    if args.resource is None:
        args.resource = sorted([os.path.join(configured_path, f) for f in os.listdir(configured_path) if f.endswith(".res")])[0]

    # --------------------------------------------------
    # logging
    # --------------------------------------------------
    logging.info(f"Using config:   {args.config or 'None'}")
    logging.info(f"Using material:{args.material or 'None'}")
    logging.info(f"Using resource:{args.resource or 'None'}")

    for item in args.scale:
        if "=" not in item:
            raise ValueError(f"Invalid scale format '{item}', expected axis=factor")

        axis, factor = item.split("=", 1)

        if axis not in SCALE:
            raise ValueError(f"Unknown scale axis '{axis}', allowed: {list(SCALE.keys())}")

        if factor not in si_prefixes:
            raise ValueError(
                f"Unknown scale factor '{factor}', allowed: {list(si_prefixes.keys())}"
            )

        SCALE[axis] = factor
    args.scale = SCALE

    if args.plotType not in SELECTS and not args.key:
        parser.print_help()
        exit(1)

    main(args)
