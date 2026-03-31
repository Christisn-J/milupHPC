import numpy as np
from constants import si_prefixes, FIELD_META

SPECIFY_DTYPE = 32  # oder 64
DTYPE = {"float": np.float32 if SPECIFY_DTYPE == 32 else np.float64, "int": np.int32 if SPECIFY_DTYPE == 32 else np.int64}
EXTENSION={"PNG": ".png", "PDF": ".pdf", "H5": ".h5"}
FRAME_PADDING = 0.01
PLANES = [("x", "y"), ("x", "z"), ("y", "z")]
SCALE = {"x": "none", "t": "none", "m": "none"}

AXES_CONFIG = {
    "x": {"limits": (-1 / si_prefixes[SCALE["x"]]["factor"], 1 / si_prefixes[SCALE["x"]]["factor"]), "labels": rf'$x$ [${si_prefixes[SCALE["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]', "index": 0},
    "y": {"limits": (-1 / si_prefixes[SCALE["x"]]["factor"], 1 / si_prefixes[SCALE["x"]]["factor"]), "labels": rf'$y$ [${si_prefixes[SCALE["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]', "index": 1},
    "z": {"limits": (-1 / si_prefixes[SCALE["x"]]["factor"], 1 / si_prefixes[SCALE["x"]]["factor"]), "labels": rf'$z$ [${si_prefixes[SCALE["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]', "index": 2}
}

DEFAULT_MARKER_ATTRS = {"s": 1, "alpha": 1, "marker": ".", "linewidths": 1, "rasterized": False, "skip": 1}
def dynamic_render_config(N, dim=3):
    if dim == 3:
        if N > 5e7: return {"s": 0.1, "alpha": 1, "marker": ".", "linewidths": 1.0, "rasterized": True, "skip": 1}
        if N > 1e7: return {"s": 0.5, "alpha": 1, "marker": ".", "linewidths": 1.0, "rasterized": True, "skip": 1}
        if N > 1e6: return {"s": 1.0, "alpha": 1, "marker": ".", "linewidths": 1.0, "rasterized": False, "skip": 1}
        if N > 1e5: return {"s": 2.0, "alpha": 1, "marker": ".", "linewidths": 1.0, "rasterized": False, "skip": 1}
    else:  # dim == 2
        if N > 5e7: return {"s": 0.1, "alpha": 1, "marker": ".", "linewidths": 0.031, "rasterized": True, "skip": 1}
        if N > 1e7: return {"s": 0.2, "alpha": 1, "marker": ".", "linewidths": 0.062, "rasterized": True, "skip": 1}
        if N > 1e6: return {"s": 0.4, "alpha": 1, "marker": ".", "linewidths": 0.125, "rasterized": False, "skip": 1}
        if N > 1e5: return {"s": 1.0, "alpha": 1, "marker": ".", "linewidths": 0.250, "rasterized": False, "skip": 1}
    return DEFAULT_MARKER_ATTRS


import re
import matplotlib as mpl

def format_math_text(text: str) -> str:
    r"""
    Bereitet einen Text für Matplotlib vor, wenn text.usetex = False.
    - \pmb{…} → \mathbf{…}
    - \num{…} → nur der Inhalt
    """
    if not mpl.rcParams.get("text.usetex", False):
        # \pmb{…} → \mathbf{…}
        text = re.sub(r"\\pmb\{([^{}]+)\}", r"\\mathbf{\1}", text)
        # \num{…} → …
        text = re.sub(r"\\num\{([^{}]+)\}", r"\1", text)
    return text

import matplotlib as mpl

def setup_global_latex():

    mpl.rcParams["text.usetex"] = False  # deaktiviert LaTeX
    mpl.rcParams["font.family"] = "serif"  # optional für ähnliche Schrift

    # mpl.rcParams["text.usetex"] = True
    # mpl.rcParams["text.latex.preamble"] = r"""
    # \usepackage{amsmath}
    # \usepackage{siunitx}
    # """

import matplotlib.pyplot as plt

def style_colorbar(cbar, n=1, dim=2, factor=1):
    base_size = plt.rcParams["font.size"]

    if dim==2 and n==1:
        factor=2
        f=2*0.75
    elif dim==2 and n==2:
        factor=2.5
        f=2*0.75
    elif dim==3 and n==1:
        factor=2
        f=2*0.75
    elif dim==3 and n==2:
        factor=2.8
        f=2*0.8
    else:
        factor=factor
        f=factor

    cbar.ax.tick_params(labelsize=base_size*f)

    # Label Größe setzen
    cbar.set_label(cbar.ax.get_ylabel(), fontsize=base_size*factor)

    # Falls Titel existiert
    if cbar.ax.get_title():
        cbar.ax.set_title(cbar.ax.get_title(), fontsize=base_size*factor)

def setup_dynamic_plotStyle(n=1, dim=None, factor=1.0):
    """
    Set global matplotlib style for scientific plots.

    Args:
        n_axes (int): Number of subplots in the figure.
                      Adjusts font sizes for readability.
    """

    base_size = 11*factor
    if dim==2 and n==1:
        factor=2
        ax_title_size = base_size*factor
        fig_title_size = base_size*factor
        legend_size = int(base_size*factor/2)
        ax_label_size = base_size*factor
        tick_label_size= base_size*0.75*factor
        figsize=(8*n, 8)
    elif dim==2 and n==2:
        factor=2.5
        ax_title_size = base_size*factor
        fig_title_size = base_size*factor
        legend_size = int(base_size*factor/2)
        ax_label_size = base_size*factor
        tick_label_size= base_size*0.75*factor
        figsize=(8*n, 8)
    elif dim==3 and n==1:
        factor=2
        ax_title_size = base_size*factor
        fig_title_size = base_size*factor
        legend_size = int(base_size*factor/2)
        ax_label_size = base_size*factor
        tick_label_size= base_size*0.75*factor
        figsize=(8*n, 8)
    elif dim==3 and n==2:
        factor=2.5
        ax_title_size = base_size*factor
        fig_title_size = base_size*factor
        legend_size = int(base_size*factor/2)
        ax_label_size = base_size*factor
        tick_label_size= ax_label_size-base_size
        figsize = (9*n, 8)
    else:
        ax_title_size = base_size + int(base_size * (n-1))
        fig_title_size = base_size + int(base_size * (n-1))
        legend_size = base_size + int((base_size/2) * (n-1))
        ax_label_size = base_size + int(base_size * (n-1))
        tick_label_size= ax_label_size
        figsize=(8*n, 8)

    # Update rcParams
    plt.rcParams.update({
        "font.size": base_size,
        "axes.labelsize": ax_label_size,
        "xtick.labelsize": tick_label_size,
        "ytick.labelsize": tick_label_size,
        "axes.titlesize": ax_title_size,
        "figure.titlesize": fig_title_size,
        "legend.fontsize": 11,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
        "axes.formatter.use_mathtext": True,

        # Grid & Minor-Ticks global
        "axes.grid": True,
        # "grid.which": "both",       # major + minor
        "grid.linestyle": "--",      # Major-Linien
        "grid.color": "gray",
        "grid.alpha": 0.5,
        "grid.linewidth": 0.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,


        "figure.figsize": figsize, # Default figure size skaliert mit der Anzahl der Achsen
    })

    if dim==3:
        plt.rcParams.update({
            "xtick.minor.visible": False,
            "ytick.minor.visible": False
        })
    else:
        plt.rcParams.update({
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5
        })
        # Globale Aktivierung der Minor-Ticks
        plt.minorticks_on()

        # Optional: Customize minor grid for alle Achsen automatisch
        # Das geht über den aktuellen Figure Manager
        for fig in plt.get_fignums():
            for ax in plt.figure(fig).get_axes():
                ax.minorticks_on()
                ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

import logging
def setup_logging(time=False, level=logging.INFO):
    """
    Configure logging format and level.

    Parameters
    ----------
    time : bool
        Include timestamps in log output.
    level : int
        Logging level (default: logging.INFO).
    """
    log_format = (
        "%(asctime)s [%(levelname)s] %(message)s"
        if time else
        "[%(levelname)s] %(message)s"
    )

    logging.basicConfig(
        format=log_format,
        level=level
    )

def read_from_file(file_path, variable, fallback):
    """
    Liest eine Variable aus einer Config-Datei.
    Unterstützt:
      - key value
      - key = value
      - key=value
      - key="value"
    """
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith(";") or line.startswith("#"):
                continue

            # Entferne Inline-Kommentare
            line = line.split("#", 1)[0].strip()

            if "=" in line:
                key, value = map(str.strip, line.split("=", 1))
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key, value = parts[0], parts[1]

            # Quotes entfernen
            value = value.strip('"').strip("'")

            if key == variable:
                return value

    return fallback

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
def setup_discrete_colormap(fig, ax, sc, data, base_cmap_name="tab10", material_names=None):
    """
    Setup discrete colormap with stable mapping and unlimited colors.

    Handles:
        • many discrete values (>64)
        • sparse IDs
        • stable color assignment
    """

    # ---- unique sorted values ----
    unique_vals = np.unique(data)
    n_colors = len(unique_vals)

    # ---- stable mapping value -> index ----
    value_to_index = {v: i for i, v in enumerate(unique_vals)}
    mapped_data = np.vectorize(value_to_index.get)(data)

    # ---- continuous sampling of cmap ----
    # ---- choose discrete cmap if possible ----
    if n_colors <= 10:
        base_cmap = plt.get_cmap("tab10")
        colors = base_cmap(np.arange(n_colors))
    elif n_colors <= 20:
        base_cmap = plt.get_cmap("tab20")
        colors = base_cmap(np.arange(n_colors))
    else:
        base_cmap = plt.get_cmap(base_cmap_name)
        colors = base_cmap(np.linspace(0, 1, n_colors))
    new_cmap = ListedColormap(colors)

    # ---- discrete normalization ----
    norm = BoundaryNorm(
        boundaries=np.arange(n_colors + 1) - 0.5,
        ncolors=n_colors
    )

    # ---- update scatter ----
    sc.set_array(mapped_data)
    sc.set_cmap(new_cmap)
    sc.set_norm(norm)

    # ---- colorbar ----
    ticks = np.arange(n_colors)
    cbar = fig.colorbar(sc, ax=ax, ticks=ticks)

    labels = []
    for v in unique_vals:
        if material_names and int(v) in material_names:
            labels.append(f"{int(v)} – {material_names[int(v)]}")
        else:
            labels.append(str(int(v)))

    cbar.ax.set_yticklabels(labels)

    if material_names:
        cbar.ax.tick_params(labelrotation=90)
        for label in cbar.ax.get_yticklabels():
            label.set_va("center")

    return cbar

import h5py
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


