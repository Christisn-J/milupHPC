#!/usr/bin/env python3
# ==================================================================================================
# Particle Simulation Script
#
# Description:
# This script initializes a particle system for simulations of colliding targets and impactors.
# Supports cube and sphere geometries via a SHAPES registry. Particles are generated based
# on the chosen geometry, mass, density, and spacing Δ.
#
# Features:
# - Cube and Sphere particle generators
# - Explicit velocity input for impactor
# - Neighbor statistics for SPH smoothing length estimation
# - Optional runtime optimization to skip sanity checks
# - Full 2D/3D visualization with scatter plots and slices
# - Saves initial particle state to HDF5
#
# Empirical Reference Table for Particle Resolution and Initial Separation:
# ------------------------------------------------------------------------
# The following table shows empirically determined grid spacing Δ, smoothing
# length h, and initial target-impactor gap d_gap for different total particle
# counts N in 2D and 3D simulations. The initial gap is chosen proportional
# to the particle spacing Δ to ensure that the smoothing length h is smaller
# than the separation distance.
#
#  +--------+----------------------+--------------------+----------------------+--------------------+--------------------+----------------------+
#  |        |        2D            |                    |                      |          3D        |                    |                      |
#  |  N     | Δ [m]                | h [m]              | d_gap = 5Δ [m]       | Δ [m]              | h [m]              | d_gap = 2.5Δ [m]    |
#  +--------+----------------------+--------------------+----------------------+--------------------+--------------------+----------------------+
#  | 1e4    | 1.0e-3               | 3.49e-3            | 5.0e-3               | 4.0e-3             | 9.10e-3            | 1.0e-2               |
#  | 1e5    | 3.0e-4               | 1.05e-3            | 1.5e-3               | 2.0e-3             | 4.55e-3            | 5.0e-3               |
#  | 1e6    | 1.0e-4               | 3.49e-4            | 5.0e-4               | 1.0e-3             | 2.28e-3            | 2.5e-3               |
#  | 1e7    | 3.0e-5               | 1.05e-4            | 1.5e-4               | 4.0e-4             | 9.10e-4            | 1.0e-3               |
#  | 1e8    | 1.0e-5               | 1.30e-5            | 5.0e-5               | 2.0e-4             | 2.60e-4            | 5.0e-4               |
# ------------------------------------------------------------------------
#
# Notes:
# - Δ defines particle spacing in each dimension.
# - h is the suggested smoothing length based on neighbor statistics.
# - d_gap sets the initial separation between target and impactor.
# - Values are empirical and may require adjustment based on simulation needs.
# ==================================================================================================

import sys, os
import numpy as np
import json
import logging
import h5py
import resource
import argparse
from datetime import datetime
from scipy.spatial import cKDTree

# Suppress matplotlib debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# --------------------------------------------------------------------------------------------------
# Local imports
# --------------------------------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../postprocessing")))
import plotScatter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../constants")))
import constants

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utility")))
import utility

from constants import si_prefixes

utility.setup_global_latex()

# ==================================================================================================
# Logging-Level "EXTRA" definieren (zwischen DEBUG=10 und INFO=20)
# ==================================================================================================
EXTRA_LEVEL_NUM = 15
logging.addLevelName(EXTRA_LEVEL_NUM, "EXTRA")


def extra(self, message, *args, **kws):
    """Log message at EXTRA level."""
    if self.isEnabledFor(EXTRA_LEVEL_NUM):
        self._log(EXTRA_LEVEL_NUM, message, args, **kws)


# Methode allen Loggern hinzufügen
logging.Logger.extra = extra


# Modulweite Funktion ermöglichen: logging.extra(...)
def _logging_extra(message, *args, **kws):
    logging.getLogger().extra(message, *args, **kws)


logging.extra = _logging_extra

# ==================================================================================================
# Global configuration
# ==================================================================================================
ETA = 1.3  # Scaling factor for smoothing length estimate
NEIGHBORS = (30, 180)  # target neighbor range for smoothing length search
GAP = {0: None, 1: 10, 2: 5, 3: 2.5}  # Initial gap multipliers based on dimension

FIELD_META = constants.FIELD_META
MATERIALS = constants.MATERIALS
AXES_CONFIG = utility.AXES_CONFIG
FRAME_PADDING = utility.FRAME_PADDING
PLANES = utility.PLANES
SCALE = utility.SCALE
EXTENSION = utility.EXTENSION
DTYPE = utility.DTYPE

# ==================================================================================================
# SHAPE REGISTRY
# Maps shape names to generation functions and geometric properties
# ==================================================================================================
SHAPES = {
    "cube": {
        "center": lambda o: o["center"],
        "volume": lambda o, dim: (2 * o["extent_max"]) ** dim,
        "generate": lambda o, **kw: generate_cube_particles(
            o["extent_max"],
            o["speed"],
            o["mass"],
            o["id"],
            o["material"]["density"],
            center=o["center"],
            **kw
        ),
    },
    "sphere": {
        "center": lambda o: o["center"],
        "volume": lambda o, dim: (
            (4 / 3 * np.pi * o["extent_max"] ** 3) if dim == 3
            else (np.pi * o["extent_max"] ** 2) if dim == 2
            else (2 * o["extent_max"])
        ),
        "generate": lambda o, **kw: generate_sphere_particles(
            o["extent_max"],
            o["speed"],
            o["mass"],
            o["id"],
            o["material"]["density"],
            center=o["center"],
            **kw
        ),
    },
}


# ==================================================================================================
# Helper functions
# ==================================================================================================
def shape_def(obj):
    """Return the shape definition from the SHAPES registry."""
    try:
        return SHAPES[obj["shape"]]
    except KeyError:
        raise ValueError(f"Unknown shape '{obj['shape']}'")


def find_sml_for_target_neighbors(tree, positions, target_range=(150, 180), h_initial=0.001, dim=2, optimize=False):
    h_min = h_initial * 0.5
    h_max = h_initial * 3.0
    best_h = None
    best_avg_neighbors = 0

    for _ in range(20):
        h_mid = 0.5 * (h_min + h_max)
        if optimize:
            # Query nur k = target_max_neighbors + 1 Nachbarn
            k = target_range[1] + 1
            distances, _ = tree.query(positions, k=k)
            avg_neighbors = np.mean(np.sum(distances[:, 1:] <= h_mid, axis=1))
        else:
            # Volle Radius-Abfrage
            neighbors_lens = np.array([len(n) - 1 for n in tree.query_ball_point(positions, r=h_mid)])
            avg_neighbors = np.mean(neighbors_lens)

        if target_range[0] <= avg_neighbors <= target_range[1]:
            best_h = h_mid
            best_avg_neighbors = avg_neighbors
            break
        if avg_neighbors < target_range[0]:
            h_min = h_mid
        else:
            h_max = h_mid

    return best_h, best_avg_neighbors


# ==================================================================================================
# Object definitions
# ==================================================================================================
target = {
    "id": 0,
    "name": "Target",
    "shape": "cube",
    "material": MATERIALS["AL6061"],
    "extent_min": 0.0,
    "extent_max": 5.0e-2,

    "center": np.zeros(3),
    "speed": np.zeros(3),
    "mass": None,
    "volume": None,
    "particles": None,
}

projectile = {
    "id": 1,
    "name": "Projectile",
    "shape": "sphere",
    "material": MATERIALS["AL6061"],
    "extent_min": 0.0,
    "extent_max": 0.5 * 6.35e-3,

    "center": np.zeros(3),
    "speed": np.zeros(3),
    "mass": None,
    "volume": None,
    "particles": None,
}


# ==================================================================================================
# Particle generators
# ==================================================================================================
def generate_cube_particles(extent_max, velocity, mass, material_id, density, center, delta, dim, optimize=False):
    """
    Generate particles in a cube efficiently.
    Uses numpy vectorization, low memory footprint, works for large N.
    """
    # Koordinatenbereiche pro Achse
    x = np.arange(center[0] - extent_max, center[0] + extent_max + delta, delta)
    y = np.arange(center[1] - extent_max, center[1] + extent_max + delta, delta) if dim >= 2 else np.array([center[1]])
    z = np.arange(center[2] - extent_max, center[2] + extent_max + delta, delta) if dim == 3 else np.array([center[2]])

    # Meshgrid erzeugen für alle Dimensionen
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pos = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Anzahl Partikel
    N = pos.shape[0]

    # Particle Array vorbereiten
    particles = np.zeros((N, 10))
    particles[:, :3] = pos
    particles[:, 3:3 + dim] = velocity[:dim]
    particles[:, 6] = mass
    particles[:, 7] = material_id
    particles[:, 8] = density

    return particles


def generate_sphere_particles(extent_max, velocity, mass, material_id, density, center, delta, dim, optimize=False):
    """
    Generate particles inside a sphere (or circle in 2D) efficiently.
    Uses numpy vectorization, low memory footprint, works for large N.
    """
    # Koordinatenbereiche vorbereiten (nur benötigte Punkte)
    x = np.arange(center[0] - extent_max, center[0] + extent_max + delta, delta)
    y = np.arange(center[1] - extent_max, center[1] + extent_max + delta, delta) if dim >= 2 else np.array([center[1]])
    z = np.arange(center[2] - extent_max, center[2] + extent_max + delta, delta) if dim == 3 else np.array([center[2]])

    # Minimaler Meshgrid-Ansatz, nur für Maske, nicht stacken
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    r2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2

    mask = r2 <= extent_max ** 2
    pos = np.column_stack([X[mask], Y[mask], Z[mask]])
    N = pos.shape[0]

    # Particle Array
    particles = np.zeros((N, 10))
    particles[:, :3] = pos
    particles[:, 3:3 + dim] = velocity[:dim]
    particles[:, 6] = mass
    particles[:, 7] = material_id
    particles[:, 8] = density

    return particles


# ==================================================================================================
# MAIN SIMULATION
# ==================================================================================================
def main(args):
    start_time = datetime.now()
    logging.extra(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    delta = args.delta
    speed = args.velocity

    # --- Setup materials and particle mass ----------------------------------------------
    for obj in (target, projectile):
        obj["material"] = MATERIALS[args.material[obj["id"] % len(args.material)]]
        obj["mass"] = obj["material"]["density"] * delta ** args.dim
        logging.extra(f"{obj['name'].capitalize()} | Material: {obj['material']['name']} | Mass per particle: {obj['mass']:.3e} kg")

    # --- Set impactor velocity ------------------------------------------------------------
    projectile["speed"][:] = 0.0
    projectile["speed"][args.dim - 1] = speed
    logging.extra(f"Projectile velocity set along dimension {args.dim}: {[f'{v:.2e}' for v in projectile['speed']]}")

    # --- Place impactor at initial gap ---------------------------------------------------
    offset = target["extent_max"] + GAP[args.dim] * delta + projectile["extent_max"]
    shape_def(projectile)["center"](projectile)[args.dim - 1] = offset
    logging.extra(f"Projectile initial center position: {[f'{v:.2e}' for v in projectile['center']]}")

    # --- Generate particle distributions --------------------------------------------------
    for obj in (target, projectile):
        obj["particles"] = shape_def(obj)["generate"](obj, delta=delta, dim=args.dim, optimize=args.optimize)
        obj["volume"] = shape_def(obj)["volume"](obj, args.dim, )
        logging.info(f"{obj['name'].capitalize()} generated: {len(obj['particles'])} particles | Volume: {obj['volume']:.3e} m³")

    total_particles = np.concatenate((target["particles"], projectile["particles"]))
    positions = total_particles[:, :args.dim]
    logging.info(f"Total particles in simulation: {len(total_particles)}")

    # --- Optional sanity check for duplicate positions -----------------------------------
    if not args.optimize:
        rounded_positions = np.round(positions, decimals=10)
        unique_positions = np.unique(rounded_positions, axis=0)
        if len(unique_positions) != len(rounded_positions):
            duplicates = len(rounded_positions) - len(unique_positions)
            logging.warning(f"{duplicates} duplicate particle positions detected!")
        else:
            logging.info("No duplicate particle positions found.")

    # --- Neighbor statistics and smoothing length ---------------------------------------
    tree = cKDTree(positions)
    distances, _ = tree.query(positions, k=2)
    avg_dist = np.mean(distances[:, 1])
    sml_estimate = ETA * avg_dist
    logging.info(f"N={len(total_particles):.2e}, Δ={delta:.2e}, estimated h≈{sml_estimate:.2e}")

    smoothing_length = sml_estimate
    if smoothing_length < delta:
        logging.warning("Suggested smoothing length is smaller than delta!")
    elif smoothing_length < 1.1 * delta:
        logging.warning("Smoothing length is only slightly larger than delta.")

    sml_optimized, avg_neighbors_optimized = None, None
    if not args.optimize:
        target_min_neighbors, target_max_neighbors = NEIGHBORS
        sml_optimized, avg_neighbors_optimized = find_sml_for_target_neighbors(
            tree, positions, target_range=(target_min_neighbors, target_max_neighbors),
            h_initial=smoothing_length, dim=args.dim, optimize=args.optimize
        )
        if sml_optimized is not None:
            logging.info(f"SML optimized: h ≈ {sml_optimized:.6e} m, avg neighbors ≈ {avg_neighbors_optimized:.1f}")
            smoothing_length = sml_optimized
        else:
            logging.warning("No suitable smoothing length found in tested range.")

    if args.pipeline:
        result = {
            "header": [
                "delta_particles",
                "N_tot",
                "N_target",
                "N_projectile",
                "SML_estimate",
                "SML_optimize",
                "avg_distance_particles",
                "avg_neighbors_optimized",
                "delta_gap"
            ],
            "data": {
                "delta_particles": "{:.6e}".format(float(delta)),
                "N_tot": "{:.6e}".format(int(total_particles.shape[0])),
                "N_target": "{:.6e}".format(int(target["particles"].shape[0])),
                "N_projectile": "{:.6e}".format(int(projectile["particles"].shape[0])),
                "SML_estimate": "{:.6e}".format(float(sml_estimate)),
                "SML_optimize": "{:.6e}".format(float(sml_optimized if sml_optimized is not None else np.nan)),
                "avg_distance_particles": "{:.6e}".format(float(avg_dist)),
                "avg_neighbors_optimized": "{:.6e}".format(float(avg_neighbors_optimized if avg_neighbors_optimized is not None else np.nan)),
                "delta_gap": "{:.6e}".format(float(GAP[args.dim] * delta))
            }
        }

        # print("PIPELINE_JSON_START")
        print(json.dumps(result))
        # print("PIPELINE_JSON_END")

    # ======================================================================
    # Visualization setup
    # ======================================================================
    date_str = datetime.now().strftime("%Y%m%d")
    basename = f"{date_str}_alloy_D{args.dim}_{projectile['shape'][0]}{target['shape'][0]}_N{len(total_particles):.1e}_DELTA{delta:.1e}_SML{smoothing_length:.2e}_V{speed:.2e}"

    if args.output == "./":
        args.output = os.path.join(os.getcwd(), f"output/{date_str}")
    # N = len(total_particles)
    # exponent = int(np.floor(np.log10(N)))  # Ganze Zahl des Exponenten
    # mantisse = N / 10 ** exponent  # Mantisse zwischen 1 und 10
    # args.output = os.path.join(args.output, f"N{exponent:02d}_{mantisse:.2f}")
    if not args.dry:
        os.makedirs(args.output, exist_ok=True)
    logging.info(f"Output directory: {args.output}")

    # Compute axis limits
    x_mins = [float(np.min(positions[:, i])) for i in range(args.dim)]
    x_maxs = [float(np.max(positions[:, i])) for i in range(args.dim)]

    for i, axis in zip(range(args.dim), AXES_CONFIG.keys()):
        if i < len(x_mins) and i < len(x_maxs):
            logging.info(f"'{axis}' {x_mins} {x_maxs}")

            AXES_CONFIG[axis]["limits"] = ((x_mins[i] - FRAME_PADDING) / si_prefixes[args.scale["x"]]["factor"], (x_maxs[i] + FRAME_PADDING) / si_prefixes[args.scale["x"]]["factor"])
            logging.info(f"Updated axis '{axis}' limits: {AXES_CONFIG[axis]['limits']}")

            AXES_CONFIG[axis]['labels'] = utility.format_math_text(f'${axis}$ [${si_prefixes[args.scale["x"]]["abbr"]}{FIELD_META["x"]["unit"].strip("$")}$]')
            logging.info(f"Updated axis '{axis}' labels: {AXES_CONFIG[axis]['labels']}")

        else:
            logging.warning(f"Skipping axis '{axis}' due to insufficient extrema data.")

    # Compute derived quantities for plotting
    velocity_magnitude = np.linalg.norm(total_particles[:, 3:3 + args.dim], axis=1)
    position_magnitude = np.linalg.norm(total_particles[:, :args.dim], axis=1)
    DATA_SOURCES = {
        "x": position_magnitude,
        "v": velocity_magnitude,
        "m": total_particles[:, 6],
        "matId": total_particles[:, 7],
        "rho": total_particles[:, 8],
        "e": total_particles[:, 9]
    }

    marker_atts = utility.dynamic_render_config(positions.shape[0], args.dim)
    logging.extra(f"Dynamic rendering configuration: { {k: (f'{v:.2e}' if isinstance(v, float) else v) for k, v in marker_atts.items()} }")
    skip = marker_atts["skip"]

    # --- Plot particles -------------------------------------------------------
    if args.dim == 3:
        coords = (positions[:, 0][::skip] / si_prefixes[args.scale["x"]]["factor"], positions[:, 1][::skip] / si_prefixes[args.scale["x"]]["factor"],
                  positions[:, 2][::skip] / si_prefixes[args.scale["x"]]["factor"])
        plot_func = plotScatter.plot_3D_scatter
        planes = PLANES
    else:
        coords = (positions[:, 0][::skip] / si_prefixes[args.scale["x"]]["factor"], positions[:, 1][::skip] / si_prefixes[args.scale["x"]]["factor"],
                  np.zeros_like(positions[:, 0][::skip] / si_prefixes[args.scale["x"]]["factor"]))
        plot_func = plotScatter.plot_2D_scatter
        planes = [PLANES[0]]

    datas = []
    for k in args.keys:

        data = DATA_SOURCES[k]
        if args.scale.get(k):
            data = data / si_prefixes[args.scale[k]]["factor"]

        datas.append(data[::skip])

    vector_fields = {"v", "a", "x"}
    labels = [
        utility.format_math_text(rf"{FIELD_META[k]['name']} "
                                 rf"({f'$|{FIELD_META[k]['symbol'].strip('$')}|$' if k in vector_fields else FIELD_META[k]['symbol']} "
                                 rf"[${si_prefixes[args.scale[k]]['abbr'] if args.scale.get(k) else ''}{FIELD_META[k]['unit'].strip('$')}$])")
        for k in args.keys
    ]
    cmaps = [FIELD_META[k]['cmap'] if k != 'matId' else "tab10" for k in args.keys]

    # Base title
    title = utility.format_math_text(
        rf"Initial Conditions ${args.dim}D$, Time: $t=\num{{{0:.3f}}}$ ${si_prefixes[args.scale['t']]['abbr']}{FIELD_META['t']['unit'].strip('$')}$ "  # Simulation Dimension
        "\n"
        rf"Particles: $N_{{par ,tot}}=\num{{{positions.shape[0]:.2e}}}$, $\Delta_{{par}}=\num{{{delta:.1e}}}$ {FIELD_META['x']['unit']} "
        "\n"
        rf"Velocity: $|{FIELD_META['v']['symbol'].strip('$')}_{{\mathrm{{proj}}}}|=\num{{{speed:.2e}}}$ {FIELD_META['v']['unit']}"
    )

    # Add material info if 'matId' is not in keys
    note = None
    material_atts = None
    if "matId" not in args.keys:
        note = utility.format_math_text(f"\n {target['name']}: {target['material']['alias']}, {projectile['name']}: {projectile['material']['alias']}")
    else:
        material_atts = {int(obj["id"]): utility.format_math_text(f"{obj['material']['alias']} ({obj['name']})") for obj in (target, projectile) if "id" in obj and "material" in obj}

    if not args.dry:
        plot_func(coords, datas=datas, labels=labels, keys=args.keys, cmaps=cmaps,
                  title=title, note=note,
                  filename=os.path.join(args.output, f"{basename}_{'_'.join(args.keys)}"),
                  dpi=args.dpi, marker_atts=marker_atts, material_atts=material_atts, axis_config=AXES_CONFIG, extension=args.extension
                  )
    if args.slice and not args.dry and not args.optimize:
        plotScatter.plot_2D_slice(planes, coords, data=datas, labels=labels, keys=args.keys, cmaps=cmaps,
                                  title=title, note=note,
                                  filename=os.path.join(args.output, f"{basename}_{'_'.join(args.keys)}"),
                                  dpi=args.dpi, marker_atts=marker_atts, material_atts=material_atts, axis_config=AXES_CONFIG, extension=args.extension
                                  )

    # --- Save HDF5 dataset ---------------------------------------------------
    h5_file_path = os.path.join(args.output, f"{basename}.h5")
    if not args.dry:
        logging.info(f"Saving HDF5 file to {h5_file_path}")
        with h5py.File(h5_file_path, "w") as h5f:
            h5f.create_dataset("x", data=total_particles[:, :args.dim].astype(DTYPE["float"]))
            h5f.create_dataset("v", data=total_particles[:, 3:3 + args.dim].astype(DTYPE["float"]))
            h5f.create_dataset("m", data=total_particles[:, 6].astype(DTYPE["float"]))
            h5f.create_dataset("materialId", data=total_particles[:, 7].astype(DTYPE["int"]))
            h5f.create_dataset("rho", data=total_particles[:, 8].astype(DTYPE["float"]))
            h5f.create_dataset("u", data=total_particles[:, 9].astype(DTYPE["float"]))

        # --- Log HDF5 summary ---------------------------------------------------
        with h5py.File(h5_file_path, "r") as f:
            logging.extra("HDF5 dataset contents:")
            for key in f.keys():
                dataset = f[key]
                logging.extra(f"{key}: shape={dataset.shape}, dtype={dataset.dtype}, min={np.min(dataset):.3e}, max={np.max(dataset):.3e}")

    # --- Memory usage -------------------------------------------------------
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_rss_gb = usage / (1024 ** 2 if sys.platform != "darwin" else 1024 ** 3)
    logging.info(f"Max memory usage: {max_rss_gb:.2e} GB")

    end_time = datetime.now()
    logging.extra(f"Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Total runtime: {end_time - start_time}")


# ==================================================================================================
# Command line interface
# ==================================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize particle system for target-impactor simulation.")
    parser.add_argument("-d", "--dim", type=int, default=2, choices=[1, 2, 3], help="Simulation dimension")
    parser.add_argument("--delta", type=float, default=1.0e-3, help="Particle spacing")
    parser.add_argument("--velocity", type=float, default=0.0, help="Impactor velocity along impact direction")
    parser.add_argument("--material", nargs="+", default=["AL6061"], choices=list(constants.MATERIALS.keys()), help=f"Available materials: {', '.join(list(constants.MATERIALS.keys()))}")
    parser.add_argument("--keys", "-k", nargs="+", default=["x", "v", "m", "matId", "rho", "e"], help="Keys for visualization")
    parser.add_argument("--slice", action="store_true", help="Generate 2D slice plots")
    parser.add_argument("--output", "-o", default="./", help="Output folder")
    parser.add_argument("-v", "--verbose", type=int, default=3, help="Logging level")
    parser.add_argument("--dry", action="store_true", help="Run without saving files")
    parser.add_argument("--optimize", action="store_true", help="Skip runtime checks for faster execution")
    parser.add_argument("--pipeline", action="store_true", help="Output compact JSON for pipeline usage")
    parser.add_argument("--scale", nargs="+", default=[], metavar="axis=factor", help="Scaling for axes, e.g. --scale x=centi t=micro")
    parser.add_argument("--dpi", type=int, default=300, help="Set DPI for all output plots (default: 300).")
    parser.add_argument("--extension", nargs="+", default=["png"], choices=["png", "pdf", "svg", "jpg"], help="Output file formats (default: png). Example: -e png pdf svg")

    args = parser.parse_args()

    # verbosity mapping
    LEVEL_MAP = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: EXTRA_LEVEL_NUM,
        4: logging.DEBUG,
    }

    log_level = LEVEL_MAP.get(args.verbose, logging.INFO)
    utility.setup_logging(time=False, level=log_level)
    if args.pipeline:
        logging.disable(logging.CRITICAL)  # disables all logging

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

    if not args.keys: args.keys = ["x"]
    main(args)
