#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import os
from datetime import datetime
from scipy.spatial import cKDTree
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)



# === Material definitions ===
MATERIALS = {
    "AL6061": {"density": 2700.0, "unit": "kg/m³", "name": "Aluminum 6061"},
    "STEEL": {"density": 7850.0, "unit": "kg/m³", "name": "Steel"},
    "COPPER": {"density": 8960.0, "unit": "kg/m³", "name": "Copper"},
}

CUBE_LENGTH = 5.0e-2           # half edge length of cube (m)
IMPACT_RADIUS = 0.5 * 6.35e-3  # sphere radius (m)
IMPACT_SPEED = -0.059          # initial speed of impactor (m/s)

# Funktion, um Materialdaten abzurufen
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
    coords = [np.ravel(grids[i]) for i in range(dim)]

    N = coords[0].size

    vx = np.full(N, velocity[0])
    vy = np.full(N, velocity[1]) if dim >= 2 else np.zeros(N)
    vz = np.full(N, velocity[2]) if dim == 3 else np.zeros(N)

    m = np.full(N, mass)
    material_ids = np.full(N, material_id)
    rho = np.full(N, density)

    # Fill missing coords with zeros depending on dim
    if dim == 1:
        y = np.zeros(N)
        z = np.zeros(N)
        particles = np.vstack([coords[0], y, z, vx, vy, vz, m, material_ids, rho]).T
    elif dim == 2:
        z = np.zeros(N)
        particles = np.vstack([coords[0], coords[1], z, vx, vy, vz, m, material_ids, rho]).T
    else:
        particles = np.vstack([coords[0], coords[1], coords[2], vx, vy, vz, m, material_ids, rho]).T

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

    vx = np.full(N, velocity[0])
    vy = np.full(N, velocity[1]) if dim >= 2 else np.zeros(N)
    vz = np.full(N, velocity[2]) if dim == 3 else np.zeros(N)

    m = np.full(N, mass)
    material_ids = np.full(N, material_id)
    rho = np.full(N, density)

    particles = np.vstack([X_valid, Y_valid, Z_valid, vx, vy, vz, m, material_ids, rho]).T
    return particles

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
    # ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Density (ρ) in [kg/m³]")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
def plot_2d_slice(x_proj, y_proj, rho_proj, axis_labels, title, filename, output_dir, cmap="viridis", dpi=150):
    """
    Erzeugt einen 2D-Scatter-Plot eines Slices (bzw. beliebiger 2D-Daten).

    Args:
        x_proj (np.ndarray): X-Koordinaten der Teilmenge
        y_proj (np.ndarray): Y-Koordinaten der Teilmenge
        rho_proj (np.ndarray): Farbwerte (z.B. Dichte) für die Teilmenge
        axis_labels (tuple): Achsenbeschriftungen (xlabel, ylabel)
        title (str): Titel des Plots
        filename (str): Dateiname zum Speichern
        output_dir (str): Verzeichnis zum Speichern
        cmap (str): Colormap (Standard: 'viridis')
        dpi (int): Auflösung (Standard: 150)
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    sc = ax.scatter(x_proj, y_proj, c=rho_proj, cmap=cmap, s=1.0)
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal", adjustable="box")
    # ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Density (ρ) in [kg/m³]")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def main(dim, verbose, outDir, delta, dry, material_key):
    material = get_material_properties(material_key)
    logging.info(f"Gewähltes Material: {material['name']} mit Dichte {material['density']} kg/m³")

    impact_radius = IMPACT_RADIUS  # m
    impact_speed = IMPACT_SPEED  # m/s
    # impact_speed = -7000  # m/s
    cube_length = CUBE_LENGTH  # half edge length in meters

    mass = material["density"] * delta ** 3
    density = material["density"]
    material_id_target = 0
    velocity_target = [0, 0, 0]

    logging.info("=== Simulation Parameters ===")
    logging.info(f"Dimensions: {dim}D")
    logging.debug(f"Output directory: {outDir}")
    logging.debug(f"Particle spacing (delta): {delta:.2e} m")
    logging.debug(f"Material density: {density:.1f} kg/m³")
    logging.debug(f"Cube half-length: {cube_length:.3e} m")
    logging.debug(f"Impactor radius: {impact_radius:.3e} m")
    logging.debug(f"Impactor speed: {impact_speed:.3f} m/s")

    target_particles = generate_cube_particles(cube_length, velocity_target, mass, material_id_target, density, delta=delta, dim=dim)

    if dim == 2:
        center_impact = np.array([0, cube_length + 4 * delta + impact_radius, 0])
        velocity_impact = [0, impact_speed, 0]
    elif dim == 3:
        center_impact = np.array([0, 0, cube_length + 4 * delta + impact_radius])
        velocity_impact = [0, 0, impact_speed]
    else:
        center_impact = np.array([cube_length + 4 * delta + impact_radius, 0, 0])
        velocity_impact = [impact_speed, 0, 0]

    material_id_impact = 0
    # material_id_impact = 1
    impactor_particles = generate_sphere_particles(impact_radius, velocity_impact, mass, material_id_impact, density, center=center_impact, delta=delta, dim=dim)

    logging.debug(f"Generated {len(target_particles):,} target particles.")
    logging.debug(f"Generated {len(impactor_particles):,} impactor particles.")
    volume_cube = (2 * cube_length) ** dim
    volume_sphere = (4/3 * np.pi * impact_radius**3) if dim == 3 else (np.pi * impact_radius**2) if dim == 2 else (2 * impact_radius)
    logging.debug(f"Cube volume: {volume_cube:.4e} m³")
    logging.debug(f"Sphere volume: {volume_sphere:.4e} m³")
    logging.debug(f"Particles per m³ (cube): {len(target_particles)/volume_cube:.2e}")
    logging.debug(f"Particles per m³ (sphere): {len(impactor_particles)/volume_sphere:.2e}")


    # Visualization
    fig = plt.figure(figsize=(10, 7))

    skip_target = max(1, int(len(target_particles) * 0.8))
    skip_impact = max(1, int(len(impactor_particles) * 0.2))

    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(target_particles[:, 0], target_particles[:, 1], target_particles[:, 2], color='b', s=1, label='Al6061 Cube')
        ax.scatter(impactor_particles[:, 0], impactor_particles[:, 1], impactor_particles[:, 2], color='r', s=1, label='Impactor Sphere')

        quiver_length_scale = 1.0  # 1 m/s = 1 m

        ax.quiver(target_particles[::skip_target, 0], target_particles[::skip_target, 1], target_particles[::skip_target, 2],
                  target_particles[::skip_target, 3]* quiver_length_scale, target_particles[::skip_target, 4]* quiver_length_scale, target_particles[::skip_target, 5]* quiver_length_scale,
                  color='c', length=1e-3, normalize=False, width=3.0e-3)

        ax.quiver(impactor_particles[::skip_impact, 0], impactor_particles[::skip_impact, 1], impactor_particles[::skip_impact, 2],
                  impactor_particles[::skip_impact, 3]* quiver_length_scale, impactor_particles[::skip_impact, 4]* quiver_length_scale, impactor_particles[::skip_impact, 5]* quiver_length_scale,
                  color='orange', length=1e-2, normalize=False, width=3.0e-3)

        ax.set_xlabel('x in [m]')
        ax.set_ylabel('y in [m]')
        ax.set_zlabel('z in [m]')
        ax.legend()
        ax.view_init(elev=0, azim=90)

    elif dim == 2:
        ax = fig.add_subplot(111)
        ax.scatter(target_particles[:, 0], target_particles[:, 1], color='b', s=1, label='Al6061 Cube')
        ax.scatter(impactor_particles[:, 0], impactor_particles[:, 1], color='r', s=1, label='Impactor Sphere')

        ax.quiver(target_particles[::skip_target, 0], target_particles[::skip_target, 1],
                  target_particles[::skip_target, 3], target_particles[::skip_target, 4],
                  color='b', scale=1.0, width=3.0e-3)

        ax.quiver(impactor_particles[::skip_impact, 0], impactor_particles[::skip_impact, 1],
                  impactor_particles[::skip_impact, 3], impactor_particles[::skip_impact, 4],
                  color='orange', scale=1.0, width=3.0e-3)

        ax.set_xlabel('x in [m]')
        ax.set_ylabel('y in [m]')
        ax.legend()

    else:  # dim == 1
        ax = fig.add_subplot(111)
        ax.scatter(target_particles[:, 0], np.zeros_like(target_particles[:, 0]), color='b', s=1, label='Al6061 Cube')
        ax.scatter(impactor_particles[:, 0], np.zeros_like(impactor_particles[:, 0]), color='r', s=1, label='Impactor Sphere')

        ax.quiver(target_particles[::skip_target, 0], np.zeros_like(target_particles[::skip_target, 0]),
                  target_particles[::skip_target, 3], np.zeros_like(target_particles[::skip_target, 0]),
                  color='b', scale=1.0, linewidth=3.0e-3)

        ax.quiver(impactor_particles[::skip_impact, 0], np.zeros_like(impactor_particles[::skip_impact, 0]),
                  impactor_particles[::skip_impact, 3], np.zeros_like(impactor_particles[::skip_impact, 0]),
                  color='orange', scale=1.0, linewidth=3.0e-3)

        ax.set_xlabel('x in [m]')
        ax.set_yticks([])  # hide y-axis ticks in 1D
        ax.legend()

    # Combine all particles for saving
    total_particles = np.concatenate((target_particles, impactor_particles))

    total_mass = np.sum(total_particles[:, 6])
    logging.debug(f"Total number of particles: {len(total_particles):,}")
    logging.debug(f"Total mass in system: {total_mass:.4e} kg")
    logging.debug(f"Mean density: {np.mean(total_particles[:, 8]):.2f} kg/m³")
    min_pos = np.min(total_particles[:, :dim], axis=0)
    max_pos = np.max(total_particles[:, :dim], axis=0)
    logging.debug("Domain bounding box:")
    for i, axis in enumerate("xyz"[:dim]):
        length = max_pos[i] - min_pos[i]
        logging.debug(f"  {axis}-range: [{min_pos[i]:.4f} m, {max_pos[i]:.4f} m] = {length:.4f} m")
    bbox_volume = np.prod(max_pos - min_pos)
    logging.debug(f"Bounding box volume: {bbox_volume:.4e} m³")
    logging.debug(f"Particle number density: {len(total_particles)/bbox_volume:.3e} particles/m³")



    positions = total_particles[:, :dim]

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


    logging.debug(f"Nearest-neighbor stats:")
    logging.debug(f"  min: {np.min(nearest_distances):.3e} m")
    logging.debug(f"  max: {np.max(nearest_distances):.3e} m")
    logging.debug(f"  mean: {np.mean(nearest_distances):.3e} m")
    logging.debug(f"  std: {np.std(nearest_distances):.3e} m")


    plt.tight_layout()

    # Format speed string for filename
    impact_speed_str = f"{impact_speed:.3f}".replace('.', 'p').replace('-', '')
    target_speed_str = f"{velocity_target[0]:.3f}".replace('.', 'p').replace('-', '')

    # Get current date as YYYYMMDD
    date_str = datetime.now().strftime("%Y%m%d")
    name = "al"
    basename = (
        f"{date_str}_{name}"
        f"_NT{len(target_particles)}_VT{target_speed_str}"
        f"_MT{material_id_target}"
        f"_NI{len(impactor_particles)}_VI{impact_speed_str}"
        f"_MI{material_id_impact}"
        f"_D{dim}"
    )
    if not dry:
        plt.savefig(os.path.join(outDir,f"{basename}.png"), dpi=300)
    plt.close()

    slice_eps = 0.005  # Toleranz für Ebenenschnitt
    if dim == 2 and not dry:
        x = total_particles[:, 0]
        y = total_particles[:, 1]
        rho = total_particles[:, 8]

        # 2D Projektion ist einfach Scatterplot in XY
        plot_2d_projection(x, y, rho,axis_labels=("x in [m]", "y in [m]"),title=f"(x-y projection)",filename=f"{basename}_proj_xy.png",output_dir=outDir)
        # Slice ist in 2D redundant, aber für Konsistenz:
        # XY-Slice (z ≈ 0)
        mask_xy = np.abs(np.zeros_like(x)) < slice_eps
        plot_2d_slice(x[mask_xy],y[mask_xy],rho[mask_xy],("x in [m]", "y in [m]"),f"Slice in x-y (|z|<{slice_eps} m)",f"{basename}_slice_xy.png",outDir)

    elif dim == 3 and not dry:
        x = total_particles[:, 0]
        y = total_particles[:, 1]
        z = total_particles[:, 2]
        rho = total_particles[:, 8]

        plot_2d_projection(x, y, rho, ("x in [m]", "y in [m]"),f"(x-y projection)",f"{basename}_proj_xy.png",outDir)
        plot_2d_projection(x, z, rho, ("x in [m]", "z in [m]"),f"(x-z projection)",f"{basename}_proj_xz.png",outDir)
        plot_2d_projection(y, z, rho, ("y in [m]", "z in [m]"),f"(y-z projection)",f"{basename}_proj_yz.png",outDir)

        # XY-Slice (z ≈ 0)
        mask_xy = np.abs(z) < slice_eps
        plot_2d_slice(x[mask_xy],y[mask_xy],rho[mask_xy],("x in [m]", "y  in [m]"),f"Slice in x-y (|z|<{slice_eps} m)",f"{basename}_slice_xy.png",outDir)

        # XZ-Slice (y ≈ 0)
        mask_xz = np.abs(y) < slice_eps
        plot_2d_slice(x[mask_xz],z[mask_xz],rho[mask_xz],("x  in [m]", "z  in [m]"),f"Slice in x-z (|y|<{slice_eps} m)",f"{basename}_slice_xz.png",outDir)

        # YZ-Slice (x ≈ 0)
        mask_yz = np.abs(x) < slice_eps
        plot_2d_slice(y[mask_yz],z[mask_yz],rho[mask_yz],("y  in [m]", "z  in [m]"),f"Slice in y-z (|x|<{slice_eps} m)",f"{basename}_slice_yz.png",outDir)

    if not dry:
        logging.info("Saving files as:")
        logging.info(f"  Plots: {basename}_*.png")
        try:
            with h5py.File(os.path.join(outDir, f"{basename}.h5"), "w") as h5f:
                h5f.create_dataset("x", data=total_particles[:, :dim]) # only spatial coordinates
                h5f.create_dataset("v", data=total_particles[:, 3:3+dim]) # velocity components
                h5f.create_dataset("m", data=total_particles[:, 6])  # mass
                h5f.create_dataset("materialId", data=total_particles[:, 7].astype(np.int32)) # material id (float64 to int32)
                h5f.create_dataset("rho", data=total_particles[:, 8]) # density

            logging.info(f"  HDF5: {basename}.h5")

        except Exception as e:
            logging.error(f"Fehler beim Schreiben der HDF5-Datei: {e}")

        # === SPH Smoothing Length Vorschlag ===
        eta = 1.3  # Sicherheitsfaktor eta ∈ [1.2, 2.0]
        smoothing_length = eta * average_distance
        logging.info(f"Empfohlene Smoothing Length h ≈ {smoothing_length:.6e} m (η = {eta}, average_distance = {average_distance:.6e})")

        if smoothing_length < delta:
            logging.warning("Vorgeschlagene smoothing length ist kleiner als delta! SPH-Ergebnisse können ungenau sein.")
        elif smoothing_length < 1.1 * delta:
            logging.warning("Smoothing length ist nur minimal größer als delta – eventuell zu wenig Nachbarn.")



        if verbose:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Particle simulation for cube and impactor")
    parser.add_argument("-d", "--dimensions", type=int, choices=[1, 2, 3], default=3,
                        help="Number of spatial dimensions (1, 2 or 3)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--output", "-o", type=str, default="./", help="Output directory")
    parser.add_argument("--delta", type=float, default=1e-3, help="Particle spacing (default: 1e-3m)")
    parser.add_argument("--material", type=str, default="AL6061", choices=MATERIALS.keys(),
                        help="Material key to use (default: AL6061)")
    parser.add_argument("--dry", action="store_true", help="Run the script without saving any files.")
    args = parser.parse_args()

    # Set logging level based on verbosity flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    if not os.path.exists(args.output):
        logging.error(f"Not existing output directory: {args.output}")
        sys.exit(1)

    if args.dry:
        logging.info("[Dry-run] Skipping file writes.")


    main(args.dimensions, args.verbose, args.output, args.delta, args.dry, args.material)


