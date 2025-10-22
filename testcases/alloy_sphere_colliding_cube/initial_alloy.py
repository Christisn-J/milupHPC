#!/usr/bin/env python3
import sys, os
import numpy as np
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import h5py
import argparse
from datetime import datetime
from scipy.spatial import cKDTree
import resource

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../postprocessing")))
import plotScatter

# Konstanten ===================================================================================================
OPTIMIZE_COMPUTATION=True
DTYPE = 32  # oder 64
FIELD_META = plotScatter.FIELD_META
TYPE = {"float": np.float32 if DTYPE == 32 else np.float64, "int": np.int32 if DTYPE == 32 else np.int64}
AXES_CONFIG = plotScatter.AXES_CONFIG
FRAME_PADDING = plotScatter.FRAME_PADDING
PLANES = plotScatter.PLANES
EXTENSION={"plot": ".png", "data": ".h5"}
ETA = 1.3
NEIGHBORS = (30, 180)

# === Material definitions ===
MATERIALS = {
    "AL6061": {"density": 2700.0, "unit": "kg/m³", "name": "Aluminum 6061"},
    "STEEL": {"density": 7850.0, "unit": "kg/m³", "name": "Steel"},
    "COPPER": {"density": 8960.0, "unit": "kg/m³", "name": "Copper"},
    "ICE": {"density": 917.0, "unit": "kg/m³", "name": "Ice"},
    "BASALT": {"density": 917.0, "unit": "kg/m³", "name": "Basalt"},
}

TARGET=  {"id":0, "shape": "cube",   "name": "Target",   "particles": None, "speed": [0,0,0], "material": MATERIALS["AL6061"], "mass": None, "volume": None, "cube": {"length": 5.0e-2, "center": None}, "sphere": {"radius": None, "center": None}}
IMPACTOR={"id":1, "shape": "sphere", "name": "Impactor", "particles": None, "speed": None,    "material": MATERIALS["AL6061"], "mass": None, "volume": None, "cube": {"length": None,   "center": None},   "sphere": {"radius": 0.5 * 6.35e-3, "center": None}}

# Parameter ===================================================================================================
EMPIRICAL_PARAMS={
    1: {"N": [1e+2, 1e+3, 1e+4, 1e+5, 1e+6], "delta": [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7 ], "sml":[None, None, None, None, None]},
    2: {"N": [1e+4, 1e+5, 1e+6, 1e+7, 1e+8], "delta": [1.0e-3, 0.3e-3, 1.0e-4, 0.3e-4, 1.0e-5 ], "sml":[3.49e-03, 1.05e-03, 3.49e-04, 1.05e-04, 1.30e-05]},
    3: {"N": [1e+4, 1e+5, 1e+6, 1e+7, 1e+8], "delta": [0.4e-2, 0.2e-2, 1.0e-3, 0.4e-3, 0.2e-3 ], "sml":[9.102e-3, 4.550e-3, 2.275e-3, 9.100e-4, 2.60e-04]}
}

SET = {
    "v": {"name": "Velocities", "values":[0.0, -5.9e-2, -1e0, -5.3e0, -23.9, -7e3], "unit": f'{FIELD_META["v"]["unit"]}'},
    "N": {"name": "Number of Particles", "values":None, "unit": r"$-$"}
}

# Funktionen ===================================================================================================
def find_sml_for_target_neighbors(tree, positions, target_range=(150, 180), h_initial=0.001, dim=3, tol=1):
    """
    Find a smoothing length h such that the average number of neighbors is within the target range.
    Uses binary search between h_min and h_max.

    Returns:
        best_h (float): smoothing length
        avg_neighbors (float): average number of neighbors at that h
    """
    h_min = h_initial * 0.5
    h_max = h_initial * 3.0
    best_h = None
    best_avg_neighbors = 0

    for _ in range(20):  # max 20 iterations
        h_mid = 0.5 * (h_min + h_max)
        neighbors_lens = np.array([len(n) - 1 for n in tree.query_ball_point(positions, r=h_mid)], dtype=TYPE["int"])
        avg_neighbors = np.mean(neighbors_lens)

        if target_range[0] <= avg_neighbors <= target_range[1]:
            best_h = h_mid
            best_avg_neighbors = avg_neighbors
            break  # found suitable h

        if avg_neighbors < target_range[0]:
            h_min = h_mid
        else:
            h_max = h_mid

    return best_h, best_avg_neighbors
def get_material_properties(material_key):
    if material_key not in MATERIALS:
        raise ValueError(f"Material '{material_key}' nicht definiert!")
    return MATERIALS[material_key]
def generate_cube_particles(edge_length, velocity, mass, material_id, density, center=None, delta=1e-3, dim=3):
    """
    Generate particles arranged in a cubic grid.

    Args:
        edge_length (float): Half edge length of the cube.
        velocity (list): Velocity vector of particles [vx, vy, vz].
        mass (float): Mass of each particle.
        material_id (int): Material identifier.
        density (float): Density of the material.
        center (list): Center coordinates of the cube.
        delta (float): Particle spacing.
        dim (int): Dimension (1, 2 or 3).

    Returns:
        np.ndarray: Array of particles with columns [x, y, z, vx, vy, vz, m, materialId, rho].
    """
    if center is None:
        center = [0] * dim

    ranges = [np.arange(center[i] - edge_length, center[i] + edge_length + delta, delta) for i in range(dim)]
    grids = np.meshgrid(*ranges, indexing='ij')
    coords = np.stack([g.ravel() for g in grids], axis=0)

    N = coords[0].size

    vx = np.full(N, velocity[0], dtype=TYPE["float"])
    vy = np.full(N, velocity[1], dtype=TYPE["float"]) if dim >= 2 else np.zeros(N)
    vz = np.full(N, velocity[2], dtype=TYPE["float"]) if dim == 3 else np.zeros(N)

    m = np.full(N, mass, dtype=TYPE["float"])
    material_ids = np.full(N, material_id, dtype=TYPE["int"])
    rho = np.full(N, density, dtype=TYPE["float"])
    e= np.full(N, 0.0, dtype=TYPE["float"])

    # Fill missing coords with zeros depending on dim
    if dim == 1:
        y = np.zeros(N)
        z = np.zeros(N)
        particles = np.vstack([coords[0], y, z, vx, vy, vz, m, material_ids, rho, e]).T
    elif dim == 2:
        z = np.zeros(N)
        particles = np.vstack([coords[0], coords[1], z, vx, vy, vz, m, material_ids, rho, e]).T
    else:
        particles = np.vstack([coords[0], coords[1], coords[2], vx, vy, vz, m, material_ids, rho, e]).T

    return particles
def generate_sphere_particles(radius, velocity, mass, material_id, density, center=None, delta=1e-3, dim=3):
    """
    Generate particles arranged inside a sphere.

    Args:
        radius (float):
        velocity (list): Velocity vector of particles [vx, vy, vz].
        mass (float): Mass of each particle.
        material_id (int): Material identifier.
        density (float): Density of the material.
        center (list): Center coordinates of the cube.
        delta (float): Particle spacing.
        dim (int): Dimension (1, 2 or 3).

    Returns:
        np.ndarray: Array of particles with columns [x, y, z, vx, vy, vz, m, materialId, rho].
    """
    if center is None:
        center = [0] * dim

    x = np.arange(center[0] - radius, center[0] + radius + delta, delta)
    y = np.arange(center[1] - radius, center[1] + radius + delta, delta) if dim >= 2 else np.array([center[1]])
    z = np.arange(center[2] - radius, center[2] + radius + delta, delta) if dim == 3 else np.array([center[2]])

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X_flat, Y_flat, Z_flat = X.ravel(), Y.ravel(), Z.ravel()

    distances = np.sqrt((X_flat - center[0]) ** 2 + (Y_flat - center[1]) ** 2 + (Z_flat - center[2]) ** 2)
    mask = distances <= radius

    X_valid = X_flat[mask]
    Y_valid = Y_flat[mask]
    Z_valid = Z_flat[mask]

    N = len(X_valid)

    vx = np.full(N, velocity[0], dtype=TYPE["float"])
    vy = np.full(N, velocity[1], dtype=TYPE["float"]) if dim >= 2 else np.zeros(N)
    vz = np.full(N, velocity[2], dtype=TYPE["float"]) if dim == 3 else np.zeros(N)

    m = np.full(N, mass, dtype=TYPE["float"])
    material_ids = np.full(N, material_id, dtype=TYPE["int"])
    rho = np.full(N, density, dtype=TYPE["float"])
    e= np.full(N, 0.0, dtype=TYPE["float"])

    particles = np.vstack([X_valid, Y_valid, Z_valid, vx, vy, vz, m, material_ids, rho, e]).T
    return particles

def main(dim, verbose, outDir, params, dry=False):
    speed=params["speed"]
    delta=params["delta"]
    if delta is None:
        logging.error("delta was not set. Exiting.")
        exit(1)

    logging.info("=== Simulation Parameters ===")
    for entity in (TARGET, IMPACTOR):
        # Wenn index außerhalb, nimm das erste Material
        if entity["id"] >= len(params["material"]):
            entity["material"] = get_material_properties(params["material"][0])
        else:
            entity["material"] = get_material_properties(params["material"][entity["id"]])
        logging.info(f"Gewähltes Material für {entity['name']} ID {entity['id']}: {entity['material']['name']} mit Dichte {entity['material']['density']} kg/m³")
        entity["mass"] = entity["material"]["density"] * delta ** dim

    logging.info(f"Dimensions: {dim}D")

    if dim == 2:
        IMPACTOR["sphere"]["center"] = np.array([0, TARGET["cube"]["length"] + 4 * delta + IMPACTOR["sphere"]["radius"], 0])
        IMPACTOR["speed"] = [0, speed, 0]
    elif dim == 3:
        IMPACTOR["sphere"]["center"] = np.array([0, 0, TARGET["cube"]["length"] + 4 * delta + IMPACTOR["sphere"]["radius"]])
        IMPACTOR["speed"]  = [0, 0, speed]
    else:
        IMPACTOR["sphere"]["center"] = np.array([TARGET["cube"]["length"] + 4 * delta + IMPACTOR["sphere"]["radius"], 0, 0])
        IMPACTOR["speed"]  = [speed, 0, 0]

    if 3 <= verbose :
        logging.debug(f"Output directory: {outDir}")
        logging.debug(f"Particle spacing (delta): {delta:.2e} m")
        for entity in (TARGET, IMPACTOR):
            logging.debug(f"{entity["name"]} parameters:")
            logging.debug(f"Material density: {entity["material"]["density"]:.1f} kg/m³")
            if entity["shape"] == "cube":
                logging.debug(f"Cube half-length: {entity[entity["shape"]]['length']:.3e} m")
            elif entity["shape"] == "sphere":
                logging.debug(f"Sphere radius: {entity[entity["shape"]]['radius']:.3e} m")
            logging.debug(f"Speed: {entity["speed"]} m/s")

    for entity in (TARGET, IMPACTOR):
        if entity["shape"] == "sphere":
            entity["particles"] = generate_sphere_particles(entity["sphere"]["radius"], entity["speed"], entity["mass"], entity["id"], entity["material"]["density"], delta=delta, dim=dim, center=entity["sphere"]["center"])
        elif entity["shape"] == "cube":
            entity["particles"] = generate_cube_particles(entity["cube"]["length"],     entity["speed"], entity["mass"], entity["id"], entity["material"]["density"], delta=delta, dim=dim)
        else:
            logging.warning(f"shape ist nicht fertgeletg!")

    if 3 <= verbose :
        for entity in (TARGET, IMPACTOR):
            logging.debug(f"Generated {len(entity['particles']):,} {entity["name"]} particles.")
            if entity["shape"] == "cube":
                entity["volume"]=(2 * entity["cube"]["length"]) ** dim
            elif entity["shape"] == "sphere":
                entity["volume"] = (4/3 * np.pi * entity["sphere"]["radius"]**3) if dim == 3 else (np.pi * entity["sphere"]["radius"]**2) if dim == 2 else (2 * entity["sphere"]["radius"])
            logging.debug(f"Cube volume: {entity['volume']:.4e} m³")
            logging.debug(f"Particles per m³ (cube): {len(entity['particles'])/entity['volume']:.2e}")

    total_particles = np.concatenate((TARGET["particles"], IMPACTOR["particles"]))
    logging.info(f"Generated Total {len(total_particles):.2e} particles.")
    logging.info(f"  Δ={delta:.2e} → N={len(total_particles):.0f} ≈ {len(total_particles):.2e} ")
    positions = total_particles[:, :dim]

    if not OPTIMIZE_COMPUTATION:
        # Überprüfung auf doppelte Positionen
        rounded_positions = np.round(positions, decimals=10)
        unique_positions = np.unique(rounded_positions, axis=0)

        if len(unique_positions) != len(rounded_positions):
            duplicates = len(rounded_positions) - len(unique_positions)
            logging.warning(f"{duplicates} doppelte Partikelposition(en) erkannt!")
        else:
            logging.info("Keine doppelten Partikelpositionen gefunden.")

    tree = cKDTree(positions)
    distances, indices = tree.query(positions, k=2)
    nearest_distances = distances[:, 1]
    average_distance = np.mean(nearest_distances)
    logging.info(f"Average particle nearest-neighbor distance: {average_distance:.6e} m ~ delta ={delta:.6e} m")

    if average_distance < 0.5 * delta:
        logging.warning("Average particle spacing is suspiciously low compared to delta!")

    if 3 <= verbose :
        logging.debug(f"Nearest-neighbor stats:")
        logging.debug(f"  min: {np.min(nearest_distances):.3e} m")
        logging.debug(f"  max: {np.max(nearest_distances):.3e} m")
        logging.debug(f"  mean: {average_distance:.3e} m")
        logging.debug(f"  std: {np.std(nearest_distances):.3e} m")

    # === SPH Smoothing Length Vorschlag ===
    eta = ETA  # Sicherheitsfaktor eta ∈ [1.2, 2.0]
    smoothing_length = eta * average_distance
    logging.info(f"Empfohlene Smoothing Length h ≈ {smoothing_length:.6e} m (η = {eta}, average_distance = {average_distance:.6e})")

    if smoothing_length < delta:
        logging.warning("Vorgeschlagene smoothing length ist kleiner als delta! SPH-Ergebnisse können ungenau sein.")
    elif smoothing_length < 1.1 * delta:
        logging.warning("Smoothing length ist nur minimal größer als delta – eventuell zu wenig Nachbarn.")

    if not OPTIMIZE_COMPUTATION:
        # === Automatische SML-Suche für Nachbarn ===
        target_min_neighbors, target_max_neighbors = NEIGHBORS
        best_h, best_avg_n = find_sml_for_target_neighbors(
            tree,
            positions,
            target_range=(target_min_neighbors, target_max_neighbors),
            h_initial=smoothing_length,
            dim=dim
        )

        if best_h is not None:
            logging.info(f"Gefundene SML für {target_min_neighbors}–{target_max_neighbors} Nachbarn:")
            logging.info(f"  h ≈ {best_h:.6e} m  → durchschnittlich {best_avg_n:.1f} Nachbarn")
            smoothing_length=best_h
        else:
            logging.warning("Keine geeignete SML im getesteten Bereich gefunden.")

        # === Berechne max. Anzahl an Nachbarn innerhalb der SML ===
        logging.info("Berechne Anzahl von Nachbarn pro Partikel innerhalb der smoothing length h...")

        neighbors_per_particle = tree.query_ball_point(positions, r=smoothing_length)
        num_neighbors = np.array([len(neighs) - 1 for neighs in neighbors_per_particle])  # -1: exclude self

        logging.info(f"Nachbarn innerhalb SML (h = {smoothing_length:.2e} m):")
        logging.info(f"  max:  {np.max(num_neighbors)}")
        logging.info(f"  min:  {np.min(num_neighbors)}")
        logging.info(f"  mean: {np.mean(num_neighbors):.2f}")
        logging.info(f"  std:  {np.std(num_neighbors):.2f}")

    # === Visualisierung auslagern ===
    date_str = datetime.now().strftime("%Y%m%d")
    name = "alloy"
    basename = (
        f"{date_str}_{name}"
        f"_N{len(total_particles):.1e}"
        f"_SML{smoothing_length:.2e}"
        f"_D{dim}"
        f"_V{np.linalg.norm(IMPACTOR['speed']):.2e}"
    )
    if not dry:
        logging.info(f"Saving HDF5 as {basename}{EXTENSION['data']}")
        # Save data to HDF5
        with h5py.File(os.path.join(outDir,f"{basename}.h5"), "w") as h5f:
            h5f.create_dataset("x", data=total_particles[:, :dim].astype(TYPE["float"]))  # only spatial coordinates
            h5f.create_dataset("v", data=total_particles[:, 3:3+dim].astype(TYPE["float"]))  # velocity components
            h5f.create_dataset("m", data=total_particles[:, 6].astype(TYPE["float"]))  # mass
            h5f.create_dataset("materialId", data=total_particles[:, 7].astype(TYPE["int"]))  # material id
            h5f.create_dataset("rho", data=total_particles[:, 8].astype(TYPE["float"]))  # density
            h5f.create_dataset("u", data=total_particles[:, 9].astype(TYPE["float"]))  # specific energy

        # Automatically adjust AXES_CONFIG based on global x min/max
        logging.info("Computing global extrema for 'x' to adjust axis limits...")
        x_mins = [float(np.min(positions[:, i])) for i in range(dim)]
        x_maxs = [float(np.max(positions[:, i])) for i in range(dim)]
        for i, label in zip(range(dim), AXES_CONFIG.keys()):
            if i < len(x_mins) and i < len(x_maxs):
                logging.debug(f"'{label}' mins: {[f'{v:.2f}' for v in x_mins]}, maxs: {[f'{v:.2f}' for v in x_maxs]}")
                AXES_CONFIG[label]["limits"] = (x_mins[i] - FRAME_PADDING, x_maxs[i] + FRAME_PADDING)
                limits = AXES_CONFIG[label]['limits']
                logging.info(f"Updated axis '{label}' limits: ({limits[0]:.2f}, {limits[1]:.2f})")
            else:
                logging.warning(f"Skipping axis '{label}' due to insufficient extrema data.")

        N = total_particles.shape[0]
        alpha, marker_size, skip = plotScatter.dynamic_render_config(N)

        # Optional: Logging
        logging.info(f"Using dynamic rendering config for N={N}: alpha={alpha}, marker_size={marker_size}, skip={skip}")

        # Plot in 3D or 2D based on argument
        if dim == 3:
            coords = (total_particles[:, 0][::skip], total_particles[:, 1][::skip], total_particles[:, 2][::skip])
            funk = plotScatter.plot_3D_scatter
            planes=PLANES
        elif dim == 2:
            coords = (total_particles[:, 0][::skip], total_particles[:, 1][::skip], np.zeros_like(total_particles[:, 1][::skip]))
            funk=plotScatter.plot_2D_scatter
            planes=[PLANES[0]]
        else:
            coords = (total_particles[:, 0][::skip],  np.zeros_like(total_particles[:, 0][::skip]), np.zeros_like(total_particles[:, 0][::skip]))
            funk=plotScatter.plot_2D_scatter

        velocity_magnitude = np.linalg.norm(total_particles[:, 3:3+dim], axis=1)

        logging.info(f"Saving Plots as {basename}{EXTENSION['plot']}")
        funk(
            coords,
            datas=[total_particles[:, 6][::skip], total_particles[:, 8][::skip], velocity_magnitude[::skip], total_particles[:, 9][::skip]],
            labels=[
                f"Mass (m) [{FIELD_META['m']['unit']}]",
                f"Velocity (|v|) [{FIELD_META['v']['unit']}]",
                f"Density (ρ) [{FIELD_META['rho']['unit']}]",
                f"specific Energy (e) [{FIELD_META['e']['unit']}]"
            ],
            cmaps=[FIELD_META['m']['cmap'], FIELD_META['v']['cmap'], FIELD_META['rho']['cmap'], FIELD_META['e']['cmap']],
            filename=os.path.join(outDir, f"{basename}_hydro"),
            dpi=300,
            point_size=marker_size,
            axis_config=AXES_CONFIG,
            alpha=alpha
        )

        if dim == 3:
            # --- Slices in 2D (für 3D-Daten) ---
            plotScatter.plot_2D_slice(
                planes,
                (total_particles[:, 0], total_particles[:, 1], total_particles[:, 2]),
                data=[total_particles[:, 6], total_particles[:, 8], velocity_magnitude, total_particles[:, 9]],
                labels=[
                    f"Mass (m) [{FIELD_META['m']['unit']}]",
                    f"Velocity (|v|) [{FIELD_META['v']['unit']}]",
                    f"Density (ρ) [{FIELD_META['rho']['unit']}]",
                    f"specific Energy (e) [{FIELD_META['e']['unit']}]"
                ],
                cmaps=[FIELD_META['m']['cmap'], FIELD_META['v']['cmap'], FIELD_META['rho']['cmap'], FIELD_META['e']['cmap']],
                filename=os.path.join(outDir, f"{basename}_hydro_slice"),
                dpi=300,
                point_size=marker_size,
                axis_config=AXES_CONFIG,
                alpha=alpha
            )

        funk(
            coords,
            datas=[total_particles[:, 7][::skip]],
            labels=["Material ID"],
            cmaps=["tab10"],
            filename=os.path.join(outDir, f"{basename}_id"),
            dpi=300,
            point_size=marker_size,
            axis_config=AXES_CONFIG,
            alpha=alpha,
        )

    if 3 < verbose:
        with h5py.File(os.path.join(outDir, f"{basename}.h5"), "r") as f:
            logging.debug("HDF5 dataset contents:")
            for key in f.keys():
                dataset = f[key]
                shape = dataset.shape
                dtype = dataset.dtype
                min_val = np.min(dataset)
                max_val = np.max(dataset)

                logging.debug(f"  {key}:")
                logging.debug(f"    shape = {shape}, dtype = {dtype}")
                logging.debug(f"    min = {min_val:.3e}, max = {max_val:.3e}")

                sample = dataset[:3]  # First 3 entries

                if sample.ndim == 1:

                    logging.debug(f"    sample: {sample} ...")
                else:
                    logging.debug(f"    sample:")
                    for i, row in enumerate(sample):
                        row_str = ", ".join([f"{val:.6f}" for val in row])
                        logging.debug(f"      [{i}] [{row_str}]")

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # macOS
        max_rss_gb = usage / (1024 ** 3)
    else:  # Linux & andere
        max_rss_gb = usage / (1024 ** 2)

    logging.info(f"Max memory usage: {max_rss_gb:.2f} GB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Particle simulation for cube and impactor")
    parser.add_argument("-d", "--dimensions", type=int, choices=[1, 2, 3], default=3,help="Number of spatial dimensions (1, 2 or 3)")
    parser.add_argument("-v", "--verbose", type=int, choices=[1, 2, 3], default=3,help="Enable verbose output")
    parser.add_argument("--output", "-o", type=str, default="./", help="Output directory")
    parser.add_argument("--delta", type=float, default=1e-3, help="Particle spacing (default: 1e-3m)")
    parser.add_argument("--dry", action="store_true", help="Run the script without saving any files.")
    parser.add_argument("--set", type=str, choices=list(SET.keys()))
    parser.add_argument("--constant", type=int, default=0, help="Index of constant parameter value (default: 0)")
    parser.add_argument("--material",nargs="+",type=str,default=["AL6061"],choices=MATERIALS.keys(),help="Materials used in simulation. Default: AL6061")

    args = parser.parse_args()

    # Set logging level based on verbosity flag
    if args.verbose >= 3:
        log_level = logging.DEBUG
    elif args.verbose == 2:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logging.info("=== Parser Parameters ===")

    if not os.path.exists(args.output):
        logging.error(f"Not existing output directory: {args.output}")
        sys.exit(1)

    if args.dry:
        logging.info("[Dry-run] Skipping file writes.")

    SET["N"]["values"] = EMPIRICAL_PARAMS[args.dimensions]["delta"]
    if args.set:
        set={"speed": SET['v']["values"], "delta": SET["N"]["values"], "N":EMPIRICAL_PARAMS[args.dimensions]["N"]}
        for i in range(len(SET[args.set]["values"])):
            logging.info("=== Set Parameters ===")
            if args.set.lower() == "v":
                dirName=f"N{set['N'][args.constant]:.0e}_v{set['speed'][i]:.2e}"
                params={"speed": set["speed"][i], "delta":set["delta"][args.constant], "material":args.material}
            elif args.set.lower() == "n":
                dirName=f"N{set['N'][i]:.0e}_v{set['speed'][args.constant]:.2e}"
                params={"speed": set["speed"][args.constant], "delta":set["delta"][i], "material":args.material}
            else:
                logging.warning(f"Unbekannter Set-Modus: {args.set}")
                continue

            out_dir = os.path.join(args.output, dirName)
            logging.info(f"{out_dir} {SET['N']['values']}")
            logging.debug(f"{os.path.basename(out_dir)}")
            os.makedirs(out_dir, exist_ok=True)
            main(args.dimensions, args.verbose, out_dir, params, args.dry)
    else:
        os.makedirs(args.output, exist_ok=True)
        params={"speed": SET["v"]["values"][args.constant], "delta":args.delta, "material":args.material}
        main(args.dimensions, args.verbose, args.output, params, args.dry)