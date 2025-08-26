#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import os
from datetime import datetime


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

def main(dim, verbose, outDir):
    # Material properties for Aluminum 6061
    rhoB = 2.7e3  # kg/m³

    impact_radius = 0.5 * 6.35e-3  # m
    impact_speed = -0.059  # m/s
    # impact_speed = -7000  # m/s
    cube_length = 5.0e-2  # half edge length in meters

    delta = 1e-3  # particle spacing

    mass = rhoB * delta ** 3
    density = rhoB
    material_id_target = 0
    velocity_target = [0, 0, 0]

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

    # Visualization
    fig = plt.figure(figsize=(10, 7))

    skip_target = max(1, int(len(target_particles) * 0.8))
    skip_impact = max(1, int(len(impactor_particles) * 0.2))

    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(target_particles[:, 0], target_particles[:, 1], target_particles[:, 2], color='b', s=1, label='Al6061 Cube')
        ax.scatter(impactor_particles[:, 0], impactor_particles[:, 1], impactor_particles[:, 2], color='r', s=1, label='Impactor Sphere')

        ax.quiver(target_particles[::skip_target, 0], target_particles[::skip_target, 1], target_particles[::skip_target, 2],
                  target_particles[::skip_target, 3], target_particles[::skip_target, 4], target_particles[::skip_target, 5],
                  color='c', length=1e-3, normalize=True)

        ax.quiver(impactor_particles[::skip_impact, 0], impactor_particles[::skip_impact, 1], impactor_particles[::skip_impact, 2],
                  impactor_particles[::skip_impact, 3], impactor_particles[::skip_impact, 4], impactor_particles[::skip_impact, 5],
                  color='orange', length=1e-2, normalize=True)

        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.set_zlabel('Z-Axis')
        ax.legend()
        ax.view_init(elev=0, azim=90)

    elif dim == 2:
        ax = fig.add_subplot(111)
        ax.scatter(target_particles[:, 0], target_particles[:, 1], color='b', s=1, label='Al6061 Cube')
        ax.scatter(impactor_particles[:, 0], impactor_particles[:, 1], color='r', s=1, label='Impactor Sphere')

        ax.quiver(target_particles[::skip_target, 0], target_particles[::skip_target, 1],
                  target_particles[::skip_target, 3], target_particles[::skip_target, 4],
                  color='b', scale=abs(impact_speed) * 2)

        ax.quiver(impactor_particles[::skip_impact, 0], impactor_particles[::skip_impact, 1],
                  impactor_particles[::skip_impact, 3], impactor_particles[::skip_impact, 4],
                  color='orange', scale=abs(impact_speed) * 2)

        ax.set_xlabel('X-Axis')
        ax.set_ylabel('Y-Axis')
        ax.legend()

    else:  # dim == 1
        ax = fig.add_subplot(111)
        ax.scatter(target_particles[:, 0], np.zeros_like(target_particles[:, 0]), color='b', s=1, label='Al6061 Cube')
        ax.scatter(impactor_particles[:, 0], np.zeros_like(impactor_particles[:, 0]), color='r', s=1, label='Impactor Sphere')

        ax.quiver(target_particles[::skip_target, 0], np.zeros_like(target_particles[::skip_target, 0]),
                  target_particles[::skip_target, 3], np.zeros_like(target_particles[::skip_target, 0]),
                  color='b', scale=abs(impact_speed) * 2)

        ax.quiver(impactor_particles[::skip_impact, 0], np.zeros_like(impactor_particles[::skip_impact, 0]),
                  impactor_particles[::skip_impact, 3], np.zeros_like(impactor_particles[::skip_impact, 0]),
                  color='orange', scale=abs(impact_speed) * 2)

        ax.set_xlabel('X-Axis')
        ax.set_yticks([])  # hide y-axis ticks in 1D
        ax.legend()

    # Combine all particles for saving
    total_particles = np.concatenate((target_particles, impactor_particles))

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
    plt.savefig(os.path.join(outDir,f"{basename}.png"), dpi=300)
    plt.close()

    if dim == 3:
        x = total_particles[:, 0]
        y = total_particles[:, 1]
        z = total_particles[:, 2]
        rho = total_particles[:, 8]

        plot_2d_projection(x, y, rho, ("x", "y"),f"{basename} (x-y projection)",f"{basename}_proj_xy.png",outDir)
        plot_2d_projection(x, z, rho, ("x", "z"),f"{basename} (x-z projection)",f"{basename}_proj_xz.png",outDir)
        plot_2d_projection(y, z, rho, ("y", "z"),f"{basename} (y-z projection)",f"{basename}_proj_yz.png",outDir)

        slice_eps = 0.005  # tolerance for being in plane

        # x, y, z, rho already loaded earlier
        plot_2d_slice(x, y, z, rho, plane="xy", eps=slice_eps, output_dir=outDir,filename=f"{basename}_slice_xy.png",title=f"Slice in x-y (|z|<{slice_eps})")
        plot_2d_slice(x, y, z, rho, plane="xz", eps=slice_eps,output_dir=outDir,filename=f"{basename}_slice_xz.png",title=f"Slice in x-z (|y|<{slice_eps})")
        plot_2d_slice(x, y, z, rho, plane="yz", eps=slice_eps,output_dir=outDir,filename=f"{basename}_slice_yz.png",title=f"Slice in y-z (|x|<{slice_eps})")

    # Save data to HDF5
    with h5py.File(os.path.join(outDir,f"{basename}.h5"), "w") as h5f:
        h5f.create_dataset("x", data=total_particles[:, :dim])  # only spatial coordinates
        h5f.create_dataset("v", data=total_particles[:, 3:3+dim])  # velocity components
        h5f.create_dataset("m", data=total_particles[:, 6])  # mass
        h5f.create_dataset("materialId", data=total_particles[:, 7])  # material id
        h5f.create_dataset("rho", data=total_particles[:, 8])  # density

    sys.stdout.write(f"\n\rSaving to {os.path.join(outDir,f"{basename}.h5")} ... done.\n")

    if verbose:
        with h5py.File(os.path.join(outDir,f"{basename}.h5"), "r") as f:
            sys.stdout.write("Datasets in file:\n")
            for key in f.keys():
                sys.stdout.write(f"{key}: shape = {f[key].shape}, dtype = {f[key].dtype}\n")
                sys.stdout.write(f"{f[key][:]}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Particle simulation for cube and impactor")
    parser.add_argument("-d", "--dimensions", type=int, choices=[1, 2, 3], default=3,
                        help="Number of spatial dimensions (1, 2 or 3)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--output", "-o", type=str, default="./", help="Output directory")
    args = parser.parse_args()


    main(args.dimensions, args.verbose, args.output)
