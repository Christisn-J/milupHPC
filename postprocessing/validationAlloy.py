#!/usr/bin/env python3

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import glob

from sklearn.cluster import DBSCAN

import utility

from constants import si_prefixes, FIELD_META

FACTOR=1.5
utility.setup_dynamic_plotStyle(factor=FACTOR)
utility.setup_global_latex()

scale = {"x": "centi", "t": "micro"}
ID = 0

reverse = True

EXTRA = True
FRAME_PADDING = utility.FRAME_PADDING
EXTENSION = utility.EXTENSION

# -----------------------------------
# Tuning parameters (histogram selection + geometric/clustering parameters)
# -----------------------------------
parameters = {
    "2D": {
        "slab": {
            "n_leading_bins": 5,
            "dbscan_eps": None,  # epsilon for DBSCAN clustering
            "dbscan_min_samples": None,  # minimum samples for DBSCAN
            "mean_particle_spacing": None,
            "delta_cut": np.array([0.25, 0.25, 0.25]),  # slab cut thickness (x,y,z)
            "histogram_quantile_default": 1.0  # default quantile for histogram selection
        },
        "cylinder": {
            "n_leading_bins": 1,
            "dbscan_eps": None,  # epsilon for DBSCAN clustering
            "dbscan_min_samples": None,  # minimum samples for DBSCAN
            "mean_particle_spacing": None,
            "delta_cut": np.array([0.25, 0.25, 0.25]),
            "histogram_quantile_default": 1.0
        }
    },
    "3D": {
        "slab": {
            "n_leading_bins": 6,
            "dbscan_eps": 0.2275,  # epsilon for DBSCAN clustering
            "dbscan_min_samples": 10,  # minimum samples for DBSCAN
            "mean_particle_spacing": None,
            "delta_cut": np.array([0.25, 0.25, 0.25]),
            "histogram_quantile_default": 1.0
        },
        "cylinder": {
            "n_leading_bins": 1,
            "dbscan_eps": 0.2275,  # epsilon for DBSCAN clustering
            "dbscan_min_samples": 10,  # minimum samples for DBSCAN
            "mean_particle_spacing": None,
            "delta_cut": np.array([0.2275, 0.2275, 0.2275]),
            "histogram_quantile_default": 1.0
        }
    }
}


# -----------------------------------
# Daten laden
# -----------------------------------
def load_h5_data(file_path, matId=ID):
    """
    Lädt die HDF5-Datei und gibt die relevanten Daten zurück.
    Fügt bei 2D-Daten automatisch eine z-Spalte = 0 hinzu.
    """
    with h5py.File(file_path, 'r') as df:
        coor = df['x'][:]
        mask = (df['matId'][:] == matId)
        coor = coor[mask]

        # sicherstellen, dass 3 Spalten vorliegen (x,y,z)
        if coor.shape[1] == 2:
            coor = np.hstack([coor, np.zeros((coor.shape[0], 1))])

    return coor / si_prefixes[scale["x"]]["factor"]


# -----------------------------------
# Dateien sortieren
# -----------------------------------
def get_h5_files_in_order(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    files.sort(key=lambda x: int(x[2:8]))  # Annahme Zeitstempel in Zeichen 2-7
    return [os.path.join(folder_path, f) for f in files]


import re


def read_sml_from_material(material_file, mat_id=0):
    with open(material_file, "r") as f:
        content = f.read()

    # Kommentare entfernen
    # alles nach # bis Zeilenende löschen
    content = re.sub(r"#.*", "", content)

    # Materialblock mit ID finden
    pattern_block = re.compile(
        r"\{\s*ID\s*=\s*" + str(mat_id) + r"\s*;.*?\}",
        re.DOTALL
    )

    block_match = pattern_block.search(content)
    if block_match is None:
        raise ValueError(f"Material with ID={mat_id} not found.")

    block = block_match.group(0)

    # aktives sml lesen
    sml_match = re.search(r"\bsml\s*=\s*([0-9.eE+-]+)", block)
    if sml_match is None:
        raise ValueError("Active sml not found in material block.")

    return float(sml_match.group(1))


# -----------------------------------
# 3D
# -----------------------------------
def select_cylinder_3D(pos, reference=None, delta=None, dim=None):
    r = np.sqrt((pos[:, 0] - reference[0]) ** 2 + (pos[:, 1] - reference[1]) ** 2)
    cyl_mask = r < max(delta[0], delta[1])

    return pos[cyl_mask], np.empty((0, pos.shape[1]))


def select_slab_3D(pos, reference=None, delta=None, dim=None, eps=None, min_samples=None):
    mask_z = np.abs(pos[:, 2] - reference[2]) < delta[2]
    slab_particles = pos[mask_z]

    # nichts im slab
    if len(slab_particles) == 0:
        logging.warning("no slab")
        return np.empty((0, pos.shape[1])), np.empty((0, pos.shape[1]))

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(slab_particles[:, :2])
    labels = clustering.labels_

    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

    # kein Cluster gefunden
    if len(unique_labels) == 0:
        logging.warning("no cluster")
        return np.empty((0, pos.shape[1])), slab_particles

    largest_label = unique_labels[np.argmax(counts)]

    cluster = slab_particles[labels == largest_label]
    fragments = slab_particles[labels != largest_label]
    return cluster, fragments


def plot_section_3D(pos, section, reference=None, delta=None, dim=None, marker_atts=utility.DEFAULT_MARKER_ATTRS, snap=None, time=None, dpi=300):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    dx, dy, dz = delta[0], delta[1], delta[2]
    x0, y0, z0 = reference[0], reference[1], reference[2]

    utility.setup_dynamic_plotStyle(n=1, dim=dim)
    fig = plt.figure(figsize=plt.rcParams["figure.figsize"], dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # --- Alle Partikel als Hintergrund ---
    ax.scatter(x, y, z,
               c="lightgray",
               s=marker_atts["s"] * 0.3,
               alpha=marker_atts["alpha"] * 0.01,
               marker=marker_atts["marker"],
               linewidths=marker_atts["linewidths"],
               label=utility.format_math_text("Remaining particles"))

    if len(section["slab"][0]) > 0:
        ax.scatter(
            section["slab"][0][:, 0],
            section["slab"][0][:, 1],
            section["slab"][0][:, 2],
            c="red",
            s=marker_atts["s"],
            alpha=0.6,
            label=utility.format_math_text(rf"Slab cut (largest cluster), $\Delta_z = \num{{{delta[2] * 2:.1f}}}$ ${si_prefixes[scale['x']]['abbr']}{FIELD_META['x']['unit'].strip('$')}$")
        )

    if len(section["slab"][1]) > 0:
        ax.scatter(section["slab"][1][:, 0], section["slab"][1][:, 1], section["slab"][1][:, 2],
                   c="green", s=marker_atts["s"], alpha=0.4, label=utility.format_math_text("Fragments"))

    if EXTRA:
        # Zusätzlich: visualisierung als Quader (halbtransparent)
        x_slab = np.array([x.min(), x.max()])
        y_slab = np.array([y.min(), y.max()])
        X_slab, Y_slab = np.meshgrid(x_slab, y_slab)
        Z_slab = np.full_like(X_slab, reference[2])
        ax.plot_surface(X_slab, Y_slab, Z_slab, color='red', alpha=0.15)

    # --- Cylinder ---
    ax.scatter(section["cylinder"][0][:, 0], section["cylinder"][0][:, 1], section["cylinder"][0][:, 2],
               c="blue", s=marker_atts["s"],
               alpha=0.3,
               label=utility.format_math_text(rf"Cylinder cut, $r=\num{{{max(delta[0], delta[1]):.2f}}}$ ${si_prefixes[scale['x']]['abbr']}{FIELD_META['x']['unit'].strip('$')}$"))

    if EXTRA:
        # Cylinder-Fläche
        phi = np.linspace(0, 2 * np.pi, 50)
        z_cyl = np.linspace(pos[:, 2].min(), pos[:, 2].max() + FRAME_PADDING, 2)
        PHI, ZC = np.meshgrid(phi, z_cyl)
        X_cyl = reference[0] + max(delta[0], delta[1]) * np.cos(PHI)
        Y_cyl = reference[1] + max(delta[0], delta[1]) * np.sin(PHI)
        ax.plot_surface(X_cyl, Y_cyl, ZC, color='blue', alpha=0.15)

    # --- referencepunkt ---
    ax.scatter(reference[0], reference[1], reference[2],
               c="black", s=15, marker=marker_atts["marker"],
               label=utility.format_math_text(rf"Reference Point ($\num{{{x0:.1f}}}$, $\num{{{y0:.1f}}}$, $\num{{{z0:.1f}}}$) ${si_prefixes[scale['x']]['abbr']}{FIELD_META['x']['unit'].strip('$')}$"))

    # --- View-Winkel setzen ---
    ax.view_init(elev=30, azim=-60)

    # --- Z-Achse nach links verschieben ---
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)

    ax.set_box_aspect([1, 1, 1])

    # Achsen + Titel
    ax.set_xlabel(utility.format_math_text(rf'$x$ [${si_prefixes[scale["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]'), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    ax.set_ylabel(utility.format_math_text(rf'$y$ [${si_prefixes[scale["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]'), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    ax.set_zlabel(utility.format_math_text(rf'$z$ [${si_prefixes[scale["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]'), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    ax.set_title(utility.format_math_text(
        rf"Analysis ${dim}D$, Time: $N_{{\mathrm{{snap}}}}=\num{{{snap:06d}}}$, $t=\num{{{time:.3f}}}$ ${si_prefixes[scale['t']]['abbr']}{FIELD_META['t']['unit'].strip('$')}$"
        "\n"
        rf"Particles: $N_{{targ}}=\num{{{pos.shape[0]:0.2e}}}$ (matId==0)")
        , fontsize=plt.rcParams["figure.titlesize"])  # \\ Slab & Cylinder Cuts")

    ax.legend(loc="upper right", fontsize=plt.rcParams["legend.fontsize"])
    plt.tight_layout()

    return fig, ax


def calculate_histograms(pos, reference=None, delta=None, dim=None, quantile=1.0):
    """
    Erstellt Histogramme der Partikelabstände in der Slab für jede Datei,
    inkl. Verteilungskurve (KDE), Mittelwert und Median.
    """

    # Abstände berechnen
    if dim == 2:
        distances = np.sqrt((pos[:, 0] - reference[0]) ** 2 + (pos[:, 1] - reference[1]) ** 2)
    elif dim == 3:
        distances = np.sqrt((pos[:, 0] - reference[0]) ** 2 + (pos[:, 1] - reference[1]) ** 2 + (pos[:, 2] - reference[2]) ** 2)
    else:
        logging.error(f"{dim}D not supported!")
        exit()

    # Nur  `quantile` auswählen
    r_threshold = np.quantile(distances, quantile)
    distances = distances[distances <= r_threshold]

    # Dynamische Bins: Square-root rule
    bins = max(5, int(np.sqrt(len(distances))))  # min 5 Bins

    # Histogramm berechnen
    counts, bin_edges = np.histogram(distances, bins=bins)

    # Statistik
    mean_dist = distances.mean()
    median_dist = np.median(distances)

    return {"counts": counts, "bins": bin_edges, "mean": mean_dist, "median": median_dist}


def plot_histograms(counts, bin_edges, mean, median, N, quantile, dpi=300):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
    ax.bar((bin_edges[:-1] + bin_edges[1:]) / 2, counts, width=np.diff(bin_edges), color='skyblue', alpha=0.7, edgecolor='black')
    ax.axvline(mean, color='red', linestyle='--', lw=2, label=utility.format_math_text(rf"Mean$=\num{{{mean:.3f}}}$"))
    ax.axvline(median, color='green', linestyle='-.', lw=2, label=utility.format_math_text(rf"Median$=\num{{{median:.3f}}}$"))
    ax.set_xlabel(utility.format_math_text("distance [units]"), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    ax.set_ylabel(utility.format_math_text("Number of pairs"), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    ax.set_title(utility.format_math_text(rf"distances refrenxpint (largest cluster, $N=\num{{{N['total']:.2e}}}$) $\num{{{quantile:.2e}}}$ -> $\num{{{N['quantile']:.2e}}}$"),
                 fontsize=plt.rcParams["figure.titlesize"])
    ax.legend(loc="upper right", fontsize=plt.rcParams["legend.fontsize"])
    plt.tight_layout()


# -----------------------------------
# 2D
# -----------------------------------
def select_cylinder_2D(pos, reference=None, delta=None, dim=None):
    dx, dy = delta[0], delta[1]
    x0, y0 = reference[0], reference[1]
    x, y = pos[:, 0], pos[:, 1]

    mask_x = (np.abs(x - reference[0]) < dx) & (y < reference[1])
    return pos[mask_x], np.empty((0, pos.shape[1]))


def select_slab_2D(pos, reference=None, delta=None, dim=None):
    dx, dy = delta[0], delta[1]
    x0, y0 = reference[0], reference[1]
    x, y = pos[:, 0], pos[:, 1]

    mask_y = np.abs(y - y0) < dy
    if not np.any(mask_y):
        return np.nan

    return pos[mask_y], np.empty((0, pos.shape[1]))


def plot_section_2D(pos, section, reference=None, delta=None, dim=None, marker_atts=utility.DEFAULT_MARKER_ATTRS, snap=None, time=None, dpi=300):
    """
    Plottet die ausgeschnittenen Bereiche in 2D (x-y) für matId==0.
    Markiert vertikale (Depth) und horizontale (Radius) Schnitte.
    """
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    dx, dy, dz = delta[0], delta[1], delta[2]
    x0, y0, z0 = reference[0], reference[1], reference[2]

    utility.setup_dynamic_plotStyle(n=1, dim=dim)
    fig, ax = plt.subplots(figsize=plt.rcParams["figure.figsize"], dpi=dpi)

    ax.scatter(x, y, c="gray",
               s=marker_atts["s"],
               alpha=marker_atts["alpha"] * 0.1,
               rasterized=marker_atts["rasterized"],
               linewidths=marker_atts["linewidths"],
               marker=marker_atts["marker"],
               label=f"Remaining particles"
               )
    ax.scatter(section["slab"][0][:, 0], section["slab"][0][:, 1], c="red",
               s=marker_atts["s"],
               alpha=marker_atts["alpha"],
               rasterized=marker_atts["rasterized"],
               linewidths=marker_atts["linewidths"],
               marker=marker_atts["marker"]
               )
    ax.scatter(section["cylinder"][0][:, 0], section["cylinder"][0][:, 1], c="blue",
               s=marker_atts["s"],
               alpha=marker_atts["alpha"],
               rasterized=marker_atts["rasterized"],
               linewidths=marker_atts["linewidths"],
               marker=marker_atts["marker"]
               )
    ax.axhline(y=y0, color='black', linestyle='--', lw=1,
               label=utility.format_math_text(rf"Surface height at $\num{{{y0:.2f}}}$ ${si_prefixes[scale['x']]['abbr']}{FIELD_META['x']['unit'].strip('$')}$"))
    ax.axvline(x=x0, color='black', linestyle=':', lw=1)
    # Transparente Bereiche
    ax.axhspan(y0 - dy, y0 + dy,
               color="red", alpha=0.15,
               label=utility.format_math_text(rf"Horizontal cut, $\Delta_y=\num{{{2 * dy:.1f}}}$ ${si_prefixes[scale['x']]['abbr']}{FIELD_META['x']['unit'].strip('$')}$"))

    ax.axvspan(x0 - dx, x0 + dx,
               color="blue", alpha=0.15,
               label=utility.format_math_text(rf"Vertical cut, $\Delta_x=\num{{{2 * dx:.1f}}}$ ${si_prefixes[scale['x']]['abbr']}{FIELD_META['x']['unit'].strip('$')}$"))
    # Referenzpunkt
    ax.scatter(x0, y0,
               c="black", s=15,
               label=utility.format_math_text(rf"Reference Point ($\num{{{x0:.1f}}}$, $\num{{{y0:.1f}}}$) ${si_prefixes[scale['x']]['abbr']}{FIELD_META['x']['unit'].strip('$')}$"))

    # --- ACHSENVERHÄLTNIS FESTLEGEN ---
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel(utility.format_math_text(rf'$x$ [${si_prefixes[scale["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]'), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    ax.set_ylabel(utility.format_math_text(rf'$y$ [${si_prefixes[scale["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]'), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    ax.set_title(utility.format_math_text(
        rf"Analysis ${dim}D$, Time: $N_{{\mathrm{{snap}}}}=\num{{{snap:06d}}}$, $t=\num{{{time:.3f}}}$ ${si_prefixes[scale['t']]['abbr']}{FIELD_META['t']['unit'].strip('$')}$"
        "\n"
        rf"Particles: $N_{{targ}}=\num{{{pos.shape[0]:0.2e}}}$ (matId==0)")
        , fontsize=plt.rcParams["figure.titlesize"])  # \\ Vertical & Horizontal Cuts")

    ax.legend(loc="upper right", fontsize=plt.rcParams["legend.fontsize"])
    plt.tight_layout()


def calculate_depth(pos, reference=None, delta=None, dim=None):
    dx, dy, dz = delta[0], delta[1], delta[2]
    x0, y0, z0 = reference[0], reference[1], reference[2]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    distances = np.empty((0, pos.shape[1]))
    # if dim == 2:
    #     distances = np.sqrt((y - y0) ** 2)
    # elif dim == 3:
    #     distances = np.sqrt((z - z0) ** 2)

    if dim == 2:
        distances = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    elif dim == 3:
        distances = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

    return np.min(distances)
    # return np.mean(np.sort(distances)[:50:])

    # def calculate_depth_average(pos, reference=None, delta=None, dim=None, percentile=0.05):
    #     dx, dy, dz = delta[0], delta[1], delta[2]
    #     x0, y0, z0 = reference[0], reference[1], reference[2]
    #     x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    #
    #     distances=np.empty((0, pos.shape[1]))
    #     # if dim == 2:
    #     #     distances = np.sqrt((y - y0) ** 2)
    #     # elif dim == 3:
    #     #     distances = np.sqrt((z - z0) ** 2)
    #     # else:
    #     #     raise ValueError("dim muss 2 oder 3 sein")
    #     if dim == 2:
    #         distances = np.sqrt((x - x0) ** 2+ (y - y0) ** 2)
    #     elif dim == 3:
    #         distances = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    #
    #     if len(distances) == 0:
    #         return np.nan
    #
    #     # --- kleinste n% auswählen ---
    #     n_select = max(1, int(len(distances) * percentile))
    #     smallest_distances = np.partition(distances, n_select-1)[:n_select]

    return smallest_distances.mean()


def calculate_radius(pos, reference=None, delta=None, dim=None):
    dx, dy, dz = delta[0], delta[1], delta[2]
    x0, y0, z0 = reference[0], reference[1], reference[2]
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

    distances = np.empty((0, pos.shape[1]))
    # if dim == 2:
    #
    #     # left = x[x < x0]
    #     # right = x[x > x0]
    #     #
    #     # if len(left) == 0 or len(right) == 0:
    #     #     return np.nan
    #     #
    #     # dist_left = x0 - np.max(left)
    #     # dist_right = np.min(right) - x0
    #     #
    #     # return 0.5 * (dist_left + dist_right)
    #
    #     distances = np.sqrt((x - x0) ** 2)
    # elif dim == 3:
    #     distances = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    if dim == 2:
        distances = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    elif dim == 3:
        distances = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    return np.min(distances)
    # return np.mean(np.sort(distances)[:20:])


# def calculate_radii_average(pos, reference=None, delta=None, dim=None, percentile=0.05):
#     """
#     Berechnet den mittleren Abstand der Partikel zum Referenzpunkt,
#     nur über die kleinsten 'percentile' Anteil der Abstände.
#
#     Args:
#         pos (np.ndarray): Partikelpositionen (N x dim)
#         reference (array-like): Referenzpunkt
#         delta (array-like): optional, für Konsistenz mit anderen Funktionen
#         dim (int): Dimensionalität (2 oder 3)
#         percentile (float): Anteil der kleinsten Abstände (0 < percentile <= 1)
#
#     Returns:
#         float: Mittelwert über die kleinsten Abstände
#     """
#     x0, y0, z0 = reference[0], reference[1], reference[2]
#     x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
#
#     # --- Abstände berechnen ---
#     # if dim == 2:
#     #     distances = np.abs(x - x0)  # 2D: Abstand nur in x
#     # elif dim == 3:
#     #     distances = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)  # 3D: Abstand in xy
#     # else:
#     #     raise ValueError("dim muss 2 oder 3 sein")
#
#     if dim == 2:
#         distances = np.sqrt((x - x0) ** 2+ (y - y0) ** 2)
#     elif dim == 3:
#         distances = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
#
#     if len(distances) == 0:
#         return np.nan
#
#     # --- kleinste n% auswählen ---
#     n_select = max(1, int(len(distances) * percentile))
#     smallest_distances = np.partition(distances, n_select-1)[:n_select]
#
#     return smallest_distances.mean()


def calculate_distance(pos, reference=None, dim=None):
    reference = np.array(reference[:dim])
    coords = pos[:, :dim]
    distances = np.linalg.norm(coords - reference, axis=1)
    return distances.min()


# -----------------------------------
# Analyse über Ordner
# -----------------------------------
def analyze(args, file_list, time=None):
    # Initialize results arrays for depth, radii, and distance
    results = {
        "depth": np.full(len(file_list), np.nan, dtype=float),
        "radii": np.full(len(file_list), np.nan, dtype=float),
        "distance": np.full(len(file_list), np.nan, dtype=float)
    }

    # Initialize counters for total, slab, and cylinder particles
    N = {"total": 0.0, "slab": 0.0, "cylinder": 0.0}

    # Shortcut to parameters for the chosen dimension (2D or 3D)
    dim_params = parameters[f"{args.dim}D"]

    # Loop over all HDF5 files
    for idx, h5file in enumerate(file_list):
        if reverse:
            i = len(file_list) - idx - 1
        else:
            i = idx

        # Load particle positions from HDF5
        position = load_h5_data(h5file)
        N["total"] = position.shape[0]

        # Generate visualization attributes for dynamic rendering
        marker_atts = utility.dynamic_render_config(position.shape[0], args.dim)

        # Extract filename without extension for output naming
        filename = os.path.splitext(os.path.basename(h5file))[0]
        logging.info(f"Work at {filename}")

        # -------------------------------
        # 2D Analysis
        # -------------------------------
        if args.dim == 2:
            # Set reference point for distance calculations (scaled by SI factor)
            reference = np.array([0.0, 0.05, 0.0]) / si_prefixes[scale["x"]]["factor"]

            # Calculate minimum distance to reference point
            results["distance"][i] = calculate_distance(position, reference=reference, dim=args.dim)

            # Select slab particles within the delta cut region
            slab_particles, slab_fragments = select_slab_2D(
                position,
                reference=reference,
                delta=dim_params["slab"]["delta_cut"],
                dim=args.dim
            )
            if slab_particles is None or len(slab_particles) == 0:
                logging.warning("No slab particles found (2D)")
                continue

            # Update particle count
            N["slab"] = slab_particles.shape[0]

            # Compute histogram with default quantile
            res = calculate_histograms(
                slab_particles,
                reference=reference,
                delta=dim_params["slab"]["delta_cut"],
                dim=args.dim,
                quantile=dim_params["slab"]["histogram_quantile_default"]
            )
            # Plot and save the histogram
            plot_histograms(
                res["counts"], res["bins"], res["mean"], res["median"],
                {"total": slab_particles.shape[0], "quantile": slab_particles.shape[0]},
                dim_params["slab"]["histogram_quantile_default"], dpi=args.dpi
            )
            for e in args.extension:
                plt.savefig(os.path.join(args.output, f"{filename}_slab_histogram_{args.dim}D.{e}"), dpi=args.dpi, bbox_inches='tight')
            plt.close()

            # Calculate N-leading bins quantile for refined histogram
            n_bins = dim_params["slab"]["n_leading_bins"]
            q = np.sum(res['counts'][:n_bins]) / slab_particles.shape[0]
            res = calculate_histograms(
                slab_particles,
                reference=reference,
                delta=dim_params["slab"]["delta_cut"],
                dim=args.dim,
                quantile=q
            )
            plot_histograms(
                res["counts"], res["bins"], res["mean"], res["median"],
                {"total": slab_particles.shape[0], "quantile": np.sum(res['counts'][:n_bins])},
                q, dpi=args.dpi
            )
            for e in args.extension:
                plt.savefig(os.path.join(args.output, f"{filename}_slab_histogram_{args.dim}D_select.{e}"), dpi=args.dpi, bbox_inches='tight')
            plt.close()

            results["radii"][i] = res["mean"]

            # -------------------------------
            # Cylinder analysis in 2D
            # -------------------------------
            cylinder_particles, _ = select_cylinder_2D(
                position,
                reference=reference,
                delta=dim_params["cylinder"]["delta_cut"],
                dim=args.dim
            )
            if len(cylinder_particles) == 0:
                logging.warning("No cylinder particles found (2D)")
                continue

            N["cylinder"] = cylinder_particles.shape[0]

            # Histogram with default quantile
            res = calculate_histograms(
                cylinder_particles,
                reference=reference,
                delta=dim_params["cylinder"]["delta_cut"],
                dim=args.dim,
                quantile=dim_params["cylinder"]["histogram_quantile_default"]
            )
            plot_histograms(
                res["counts"], res["bins"], res["mean"], res["median"],
                {"total": cylinder_particles.shape[0], "quantile": cylinder_particles.shape[0]},
                dim_params["cylinder"]["histogram_quantile_default"], dpi=args.dpi
            )
            for e in args.extension:
                plt.savefig(os.path.join(args.output, f"{filename}_cylinder_histogram_{args.dim}D.{e}"), dpi=args.dpi, bbox_inches='tight')
            plt.close()

            # N-leading bins for cylinder histogram
            n_bins = dim_params["cylinder"]["n_leading_bins"]
            q = np.sum(res['counts'][:n_bins]) / cylinder_particles.shape[0]
            res = calculate_histograms(
                cylinder_particles,
                reference=reference,
                delta=dim_params["cylinder"]["delta_cut"],
                dim=args.dim,
                quantile=q
            )
            plot_histograms(
                res["counts"], res["bins"], res["mean"], res["median"],
                {"total": cylinder_particles.shape[0], "quantile": np.sum(res['counts'][:n_bins])},
                q, dpi=args.dpi
            )
            for e in args.extension:
                plt.savefig(os.path.join(args.output, f"{filename}_cylinder_histogram_{args.dim}D_select.{e}"), dpi=args.dpi, bbox_inches='tight')
            plt.close()

            results["depth"][i] = res["mean"]

            # Plot final 2D section with slab and cylinder particles
            section = {
                "slab": [slab_particles, slab_fragments],
                "cylinder": [cylinder_particles, np.empty((0, position.shape[1]))]
            }
            plot_section_2D(
                position,
                section,
                reference=reference,
                delta=dim_params["slab"]["delta_cut"],
                marker_atts=marker_atts,
                dim=args.dim,
                snap=i,
                time=time[i],
                dpi=args.dpi
            )
            for e in args.extension:
                plt.savefig(os.path.join(args.output, f"{filename}_cluster_{args.dim}D.{e}"), dpi=args.dpi, bbox_inches='tight')
            plt.close()

        # -------------------------------
        # 3D Analysis
        # -------------------------------
        elif args.dim == 3:
            reference = np.array([0.0, 0.0, 0.05]) / si_prefixes[scale["x"]]["factor"]
            results["distance"][i] = calculate_distance(position, reference=reference, dim=args.dim)

            # Slab clustering in 3D using DBSCAN with epsilon and min_samples
            slab_cluster_particles, slab_fragment_particles = select_slab_3D(
                position,
                reference=reference,
                delta=dim_params["slab"]["delta_cut"],
                eps=dim_params["slab"]["dbscan_eps"],
                min_samples=dim_params["slab"]["dbscan_min_samples"],
                dim=args.dim
            )
            N["slab"] = slab_cluster_particles.shape[0] + slab_fragment_particles.shape[0]

            # Compute radius for slab particles
            results["radii"][i] = calculate_radius(
                slab_cluster_particles,
                reference=reference,
                delta=dim_params["slab"]["delta_cut"],
                dim=args.dim
            )

            # Slab histogram with default quantile
            res = calculate_histograms(
                slab_cluster_particles,
                reference=reference,
                delta=dim_params["slab"]["delta_cut"],
                dim=args.dim,
                quantile=dim_params["slab"]["histogram_quantile_default"]
            )
            plot_histograms(
                res["counts"], res["bins"], res["mean"], res["median"],
                {"total": slab_cluster_particles.shape[0], "quantile": slab_cluster_particles.shape[0]},
                dim_params["slab"]["histogram_quantile_default"], dpi=args.dpi
            )
            for e in args.extension:
                plt.savefig(os.path.join(args.output, f"{filename}_slab_histogram_{args.dim}D.{e}"), dpi=args.dpi, bbox_inches='tight')
            plt.close()

            # N-leading bins quantile for 3D slab
            n_bins = dim_params["slab"]["n_leading_bins"]
            q = np.sum(res['counts'][:n_bins]) / slab_cluster_particles.shape[0]
            res = calculate_histograms(
                slab_cluster_particles,
                reference=reference,
                delta=dim_params["slab"]["delta_cut"],
                dim=args.dim,
                quantile=q
            )
            plot_histograms(
                res["counts"], res["bins"], res["mean"], res["median"],
                {"total": slab_cluster_particles.shape[0], "quantile": np.sum(res['counts'][:n_bins])},
                q, dpi=args.dpi
            )
            for e in args.extension:
                plt.savefig(os.path.join(args.output, f"{filename}_slab_histogram_{args.dim}D_select.{e}"), dpi=args.dpi, bbox_inches='tight')
            plt.close()

            results["radii"][i] = res["mean"]

            # Cylinder analysis in 3D
            cylinder_cluster_particles, _ = select_cylinder_3D(
                position,
                reference=reference,
                delta=dim_params["cylinder"]["delta_cut"],
                dim=args.dim
            )
            N["cylinder"] = cylinder_cluster_particles.shape[0]

            # Compute depth for cylinder particles
            results["depth"][i] = calculate_depth(
                cylinder_cluster_particles,
                reference=reference,
                delta=dim_params["cylinder"]["delta_cut"],
                dim=args.dim
            )

            # Cylinder histogram with default quantile
            res = calculate_histograms(
                cylinder_cluster_particles,
                reference=reference,
                delta=dim_params["cylinder"]["delta_cut"],
                dim=args.dim,
                quantile=dim_params["cylinder"]["histogram_quantile_default"]
            )
            plot_histograms(
                res["counts"], res["bins"], res["mean"], res["median"],
                {"total": cylinder_cluster_particles.shape[0], "quantile": cylinder_cluster_particles.shape[0]},
                dim_params["cylinder"]["histogram_quantile_default"], dpi=args.dpi
            )
            for e in args.extension:
                plt.savefig(os.path.join(args.output, f"{filename}_cylinder_histogram_{args.dim}D.{e}"), dpi=args.dpi, bbox_inches='tight')
            plt.close()

            # N-leading bins for 3D cylinder
            n_bins = dim_params["cylinder"]["n_leading_bins"]
            q = np.sum(res['counts'][:n_bins]) / cylinder_cluster_particles.shape[0]
            res = calculate_histograms(
                cylinder_cluster_particles,
                reference=reference,
                delta=dim_params["cylinder"]["delta_cut"],
                dim=args.dim,
                quantile=q
            )
            plot_histograms(
                res["counts"], res["bins"], res["mean"], res["median"],
                {"total": cylinder_cluster_particles.shape[0], "quantile": np.sum(res['counts'][:n_bins])},
                q, dpi=args.dpi
            )
            for e in args.extension:
                plt.savefig(os.path.join(args.output, f"{filename}_cylinder_histogram_{args.dim}D_select.{e}"), dpi=args.dpi, bbox_inches='tight')
            plt.close()

            results["depth"][i] = res["mean"]

            # Plot final 3D section with slab and cylinder particles
            section = {
                "slab": [slab_cluster_particles, slab_fragment_particles],
                "cylinder": [cylinder_cluster_particles, _]
            }
            plot_section_3D(
                position,
                section,
                reference=reference,
                delta=dim_params["slab"]["delta_cut"],
                marker_atts=marker_atts,
                dim=args.dim,
                snap=i,
                time=time[i],
                dpi=args.dpi
            )

            for e in args.extension:
                plt.savefig(os.path.join(args.output, f"{filename}_cluster_{args.dim}D.{e}"), dpi=args.dpi, bbox_inches='tight', pad_inches=0.5)
            plt.close()

    return results, N


# -----------------------------------
# Main
# -----------------------------------
def main(args):
    file_list = sorted(glob.glob(os.path.join(args.path, f"*{EXTENSION['H5']}")), key=os.path.basename, reverse=reverse)

    # TODO: nehem sml oder mean asu material.cfg um die 0.25 an ddie teilchen dicht anzupassen
    sml = read_sml_from_material(args.material, mat_id=ID)
    h = sml / si_prefixes[scale["x"]]["factor"]

    t_0 = float(utility.read_from_file(args.config, "t_0", fallback=0.0))
    t_end = float(utility.read_from_file(args.config, "timeEnd", fallback=0.0))

    if args.resource:
        steps = int(utility.read_from_file(args.resource, "NSTEPs", fallback=None))
    else:
        steps = len(file_list)

    # Warnung, falls Anzahl HDF5-Dateien und Schritte nicht übereinstimmen
    if len(file_list) != steps:
        logging.warning(f"Number of .h5 files ({len(file_list)}) does not match expected steps ({steps})")

    logging.info(f"found {len(file_list)} .h5 files")

    delta_t = t_end / steps  # oder aus Resource-Datei lesen
    time = np.linspace(t_0 + delta_t, t_end, steps) / si_prefixes[scale["t"]]["factor"]
    t_end /= si_prefixes[scale["t"]]["factor"]

    # Analyse
    res, N = analyze(args, file_list, time)

    # # Ergebnisarrays ebenfalls anpassen
    # time=time[1::]
    # res["radii"] = res["radii"][1:]
    # res["depth"] = res["depth"][1:]
    # res["distance"] = res["distance"][1:]

    # -------------------------------
    # Experimental data
    # -------------------------------
    err_exp = 0.5e-3 / si_prefixes[scale["x"]]["factor"]

    t_exp_R_6061 = np.array([2.845, 3.075, 3.083, 5.098, 5.607, 6.582, 6.402, 7.670,
                             9.334, 9.948, 10.297, 14.121, 15.082, 18.871, 25.25,
                             27.04, 40.808, 43.68, 52.176, 68.81]) * (si_prefixes["micro"]["factor"] / si_prefixes[scale["t"]]["factor"])
    R_exp_6061 = np.array([0.7505, 0.803, 0.7555, 0.8835, 0.976, 1.066, 1.048, 1.154,
                           1.213, 1.282, 1.273, 1.293, 1.522, 1.616, 1.304,
                           1.295, 1.281, 1.245, 1.322, 1.324]) * (si_prefixes["centi"]["factor"] / si_prefixes[scale["x"]]["factor"])

    t_exp_D_6061 = np.array([2.872, 2.994, 4.787, 5.216, 5.528, 6.462, 6.639, 6.418,
                             8.180, 7.779, 6.758, 8.709, 7.943, 9.721, 12.314,
                             15.08, 16.151, 18.658, 23.618, 24.683]) * (si_prefixes["micro"]["factor"] / si_prefixes[scale["t"]]["factor"])
    D_exp_6061 = np.array([0.825, 0.905, 0.869, 1.0255, 0.857, 1.125, 1.0505, 1.057,
                           1.0685, 1.256, 1.1065, 1.068, 1.1065, 1.111, 1.207,
                           1.2585, 1.3525, 1.3365, 1.437, 1.5175]) * (si_prefixes["centi"]["factor"] / si_prefixes[scale["x"]]["factor"])

    mask_exp_R = t_exp_R_6061 <= t_end
    R_exp_cut = R_exp_6061[mask_exp_R]
    t_exp_R_cut = t_exp_R_6061[mask_exp_R]

    mask_exp_D = t_exp_D_6061 <= t_end
    D_exp_cut = D_exp_6061[mask_exp_D]
    t_exp_D_cut = t_exp_D_6061[mask_exp_D]

    # -------------------------------
    # Plots
    # -------------------------------
    utility.setup_dynamic_plotStyle(factor=FACTOR)
    fig, axes = plt.subplots(len(args.analyse), 1, figsize=(8, 4 * len(args.analyse)), sharex=True)

    # falls nur ein Plot existiert
    if len(args.analyse) == 1:
        axes = [axes]

    # mantissa, exponent = f"{h:.3e}".split("e")
    # exponent = int(exponent)
    label_mesh = utility.format_math_text(rf"$\pm\,h=\num{{{h:.3e}}}$ ${si_prefixes[scale['x']]['abbr']}{FIELD_META['x']['unit'].strip('$')}$")

    # mantissa, exponent = f"{err_exp:.1e}".split("e")
    # exponent = int(exponent)
    label_exp = utility.format_math_text(rf"$\pm\,\num{{{err_exp:.1e}}}$ ${si_prefixes[scale['x']]['abbr']}{FIELD_META['x']['unit'].strip('$')}$ (Al6061-T6)")

    # Get delta from parameters for chosen dimension
    dim_params = parameters[f"{args.dim}D"]
    delta = dim_params["slab"]["delta_cut"]  # typically use slab delta for display

    if args.dim == 2:
        delta_str = rf"(\num{{{delta[0] * 2:.1f}}}, \num{{{delta[1] * 2:.1f}}})"
    elif args.dim == 3:
        delta_str = rf"(\num{{{delta[0] * 2:.1f}}}, \num{{{delta[1] * 2:.1f}}}, \num{{{delta[2] * 2:.1f}}})"
    else:
        raise ValueError(f"Unsupported dimension: {args.dim}")

    fig.suptitle(utility.format_math_text(
        rf"Validation ${args.dim}D$, Particles: $N_{{par, tot}}=\num{{{N['total']:0.2e}}}$"
        "\n"
        rf"Thickness: $\pmb{{\Delta}}_{{thick}}={delta_str}$ ${si_prefixes[scale['x']]['abbr']}{FIELD_META['x']['unit'].strip('$')}$"
        ), fontsize=plt.rcParams["figure.titlesize"], y=0.95
    )

    ax_i = 0

    # --- Depth ---
    if "depth" in args.analyse:
        ax = axes[ax_i]
        # ax.plot(time, res["depth"], lw=2, marker='o', label=utility.format_math_text(rf"$d_{{real}}$ {label_mesh}"))
        # ax.fill_between(time, res["depth"] - h, res["depth"] + h, alpha=0.3)
        ax.errorbar(time, res["depth"], yerr=h, fmt="x", capsize=4, label=utility.format_math_text(rf"$d_{{real}}$ {label_mesh}"))
        ax.errorbar(t_exp_D_cut, D_exp_cut, yerr=err_exp, fmt=".", capsize=4, c="g", label=utility.format_math_text(rf"$d_{{exp}}$ {label_exp}"))
        ax.set_ylabel(utility.format_math_text(rf'$d$ [${si_prefixes[scale["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]'), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
        ax.set_title(utility.format_math_text("Crater Depth"), fontsize=plt.rcParams["axes.titlesize"])
        ax.legend(loc="upper left", fontsize=plt.rcParams["legend.fontsize"])
        ax_i += 1

    # --- Radius ---
    if "radii" in args.analyse:
        ax = axes[ax_i]
        # ax.plot(time, res["radii"], lw=2, marker='o',label=utility.format_math_text(rf"$r_{{real}}$ {label_mesh}"))
        # ax.fill_between(time, res["radii"] - h, res["radii"] + h, alpha=0.3)
        ax.errorbar(time, res["radii"], yerr=h, fmt="x", capsize=4, label=utility.format_math_text(rf"$r_{{real}}$ {label_mesh}"))
        ax.errorbar(t_exp_R_cut, R_exp_cut, yerr=err_exp, fmt=".", capsize=4, label=utility.format_math_text(rf"$r_{{exp}}$ {label_exp}"))
        ax.set_ylabel(utility.format_math_text(rf'$r$ [${si_prefixes[scale["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]'), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
        ax.set_title(utility.format_math_text("Crater Radius"), fontsize=plt.rcParams["axes.titlesize"])
        ax.legend(loc="upper left", fontsize=plt.rcParams["legend.fontsize"])
        ax_i += 1

    # --- Distance ---
    if "distance" in args.analyse:
        ax = axes[ax_i]
        # ax.plot(time, res["distance"], lw=2, marker='o', label=utility.format_math_text(rf"$d_{{real}}$ {label_mesh}"))
        # ax.fill_between(time, res["distance"] - h, res["distance"] + h, alpha=0.3)
        ax.errorbar(time, res["distance"], yerr=h, fmt="x", capsize=4, label=utility.format_math_text(rf"$d_{{real}}$ {label_mesh}"))
        ax.errorbar(t_exp_R_cut, R_exp_cut, yerr=err_exp, fmt=".", capsize=4, label=utility.format_math_text(rf"$r_{{exp}}$ {label_exp}"))
        ax.errorbar(t_exp_D_cut, D_exp_cut, yerr=err_exp, fmt=".", capsize=4, c="g", label=utility.format_math_text(rf"$d_{{exp}}$ {label_exp}"))
        ax.set_ylabel(utility.format_math_text(rf' $d_{{min}}$ [${si_prefixes[scale["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]'), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
        ax.set_title(utility.format_math_text("Minimum Distance to Surface Point"), fontsize=plt.rcParams["axes.titlesize"])
        ax.legend(loc="upper left", fontsize=plt.rcParams["legend.fontsize"])
        ax_i += 1

    axes[-1].set_xlabel(utility.format_math_text(rf'$t$ [${si_prefixes[scale["t"]]["abbr"]}s$]'), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)

    # plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    for e in args.extension:
        plt.savefig(os.path.join(args.output, f"analysis_crater_{args.dim}D.{e}"), dpi=args.dpi, bbox_inches='tight')
    plt.close()


# -----------------------------------
# CLI
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Particle simulation for cube and impactor")
    parser.add_argument("--path", "-p", type=str, default="output", help="Input directory containing HDF5 files.")
    parser.add_argument("--dim", "-d", type=int, choices=[1, 2, 3], default=2, help="Number of spatial dimensions (1, 2 or 3)")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config file")
    parser.add_argument("--material", "-m", type=str, default=None, help="Config file")
    parser.add_argument("--resource", "-r", type=str, default=None, help="Resource file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory for saving plots.")
    parser.add_argument("--analyse", "-a", nargs="+", choices=["depth", "radii", "distance"], default=["depth", "radii"], help="Wähle eine oder mehrere Analysen aus.")
    parser.add_argument("--verbose", "-v", type=int, choices=[1, 2, 3], default=3, help="Enable verbose output")
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

    base = os.path.abspath(args.path)
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

    if args.output is None:
        args.output = os.path.join(base, "validation")

    # --------------------------------------------------
    # logging
    # --------------------------------------------------
    logging.info(f"Using config:   {args.config or 'None'}")
    logging.info(f"Using material: {args.material or 'None'}")
    logging.info(f"Using resource: {args.resource or 'None'}")
    logging.info(f"Using output:   {args.output or 'None'}")

    os.makedirs(args.output, exist_ok=True)
    main(args)
