import argparse
import numpy as np
import h5py
import os
import logging
from datetime import datetime
import seagen
import matplotlib.pyplot as plt
logging.getLogger('matplotlib').setLevel(logging.WARNING)


from mpl_toolkits.mplot3d import Axes3D  # Für 3D-Plot (benötigt kein explizites Verwenden mehr)

def plot_particle_distribution(pos, dim, title, filename, output_dir, dpi=150):
    """
    Plottet die initiale Partikelverteilung:
    - 2D: einfache xy-Scatter-Darstellung
    - 3D: Scatter in 3D
    """
    if dim == 2:
        x, y = pos[:, 0], pos[:, 1]
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.scatter(x, y, color='black', s=0.5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
    elif dim == 3:
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        fig = plt.figure(figsize=(6, 6), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, color='black', s=0.1)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        raise ValueError("Dimension must be 2 or 3.")

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)
def visualize(pos, rho, dim, base_filename, output_dir):
    if dim == 2:
        x = pos[:, 0]
        y = pos[:, 1]

        plot_2d_projection(
            x, y, rho,
            axis_labels=("x", "y"),
            title="Sedov Blast - 2D Projection",
            filename=f"{base_filename}_proj_xy.png",
            output_dir=output_dir
        )

        # Slice (technisch in 2D redundant, aber zur Einheitlichkeit)
        slice_eps = 1e-4
        mask = np.abs(np.zeros_like(x)) < slice_eps
        plot_2d_slice(
            x[mask], y[mask], rho[mask],
            axis_labels=("x", "y"),
            title=f"Slice in x-y (|z|<{slice_eps})",
            filename=f"{base_filename}_slice_xy.png",
            output_dir=output_dir
        )

    elif dim == 3:
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]

        plot_2d_projection(x, y, rho, ("x", "y"), "xy-Projection", f"{base_filename}_proj_xy.png", output_dir)
        plot_2d_projection(x, z, rho, ("x", "z"), "xz-Projection", f"{base_filename}_proj_xz.png", output_dir)
        plot_2d_projection(y, z, rho, ("y", "z"), "yz-Projection", f"{base_filename}_proj_yz.png", output_dir)

        slice_eps = 1e-3
        mask = np.abs(z) < slice_eps
        plot_2d_slice(
            x[mask], y[mask], rho[mask],
            axis_labels=("x", "y"),
            title=f"Slice in x-y (|z|<{slice_eps})",
            filename=f"{base_filename}_slice_xy.png",
            output_dir=output_dir
        )
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


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


def plot_2d_slice(x_proj, y_proj, rho_proj, axis_labels, title, filename, output_dir, cmap="viridis", dpi=150):
    """
    Erzeugt einen 2D-Scatter-Plot eines Slices (bzw. beliebiger 2D-Daten).
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    sc = ax.scatter(x_proj, y_proj, c=rho_proj, cmap=cmap, s=1.0)
    ax.set_title(title)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Density (ρ)")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def get_sml_from_N(N_grid, dim):
    """Returns smoothing length for known grid sizes"""
    sml_table_3d = {
        61: 0.041833,
        81: 0.031375,
        101: 0.0251,
        126: 0.0200,
        171: 0.01476
    }
    sml_table_2d = {
        61: 0.0135,
        81: 0.0102,
        101: 0.0081,
        126: 0.0065,
        171: 0.0048
    }

    if dim == 3:
        table = sml_table_3d
    elif dim == 2:
        table = sml_table_2d
    else:
        raise ValueError("Only 2D or 3D allowed for --dim")

    if N_grid not in table:
        raise ValueError(f"No smoothing length defined for N = {N_grid} in {dim}D. "
                         f"Allowed values: {list(table.keys())}")

    return table[N_grid]


def cubicSpline(dx_vec, sml):
    r = np.linalg.norm(dx_vec)
    q = r / sml
    f = 8. / np.pi / (sml ** 3)

    if q > 1:
        return 0
    elif q > 0.5:
        return 2. * f * (1. - q) ** 3
    else:
        return f * (6. * q ** 3 - 6. * q ** 2 + 1.)

def generate_2d_particles_from_3d(N_total, radii, densities):
    particles_3d = seagen.GenSphere(N_total, radii, densities)

    class Particles2D:
        def __init__(self, p3d):
            self.x = p3d.x
            self.y = p3d.y
            self.z = np.zeros_like(p3d.x)  # Setze z auf 0 für 2D-Projektion
            self.m = p3d.m

    return Particles2D(particles_3d)

def generate_sedov(N_grid, sml, dim):
    N_total = N_grid ** dim
    r_smooth = 2 * sml

    radii = np.arange(0.001, 0.5, 0.001)
    densities = np.ones_like(radii)

    if dim == 3:
        particles = seagen.GenSphere(N_total, radii, densities)
    elif dim == 2:
        particles = generate_2d_particles_from_3d(N_total, radii, densities)
    else:
        raise ValueError("Dimension must be 2 or 3.")

    pos = np.zeros((len(particles.x), dim))
    vel = np.zeros_like(pos)
    u = np.zeros(len(particles.x))
    mass = np.zeros(len(particles.x))
    materialId = np.zeros(len(particles.x), dtype=np.int8)

    efloor = 1e-6
    verify = 0.
    numParticles = 0

    for i in range(len(particles.x)):
        if dim == 3:
            dx_vec = np.array([particles.x[i], particles.y[i], particles.z[i]])
        else:
            dx_vec = np.array([particles.x[i], particles.y[i]])

        W = cubicSpline(dx_vec, r_smooth)
        e = max(W, efloor)
        verify += W * particles.m[i]

        pos[numParticles] = dx_vec
        vel[numParticles] = 0
        mass[numParticles] = particles.m[i]
        u[numParticles] = e
        numParticles += 1

    return pos[:numParticles], vel[:numParticles], mass[:numParticles], u[:numParticles], materialId[:numParticles]

from scipy.spatial import cKDTree

def compute_density(pos, mass, h):
    """
    Berechnet SPH-Dichte für alle Partikel über die Summenformel:
    ρ_i = ∑_j m_j * W(|r_i - r_j|, h)

    Args:
        pos: Partikelpositionen (N, dim)
        mass: Partikelmassen (N,)
        h: smoothing length (sml)

    Returns:
        rho: Array mit Dichten (N,)
    """
    N = pos.shape[0]
    rho = np.zeros(N)
    tree = cKDTree(pos)
    neighbors_list = tree.query_ball_tree(tree, 2*h)  # Radius-Suche

    for i in range(N):
        neighbors = neighbors_list[i]
        for j in neighbors:
            dx = pos[i] - pos[j]
            W = cubicSpline(dx, h)
            rho[i] += mass[j] * W

    return rho


def save_hdf5(filepath, pos, vel, mass, u, matId):
    with h5py.File(filepath, "w") as f:
        f.create_dataset("x", data=pos)
        f.create_dataset("v", data=vel)
        f.create_dataset("m", data=mass)
        f.create_dataset("u", data=u)
        f.create_dataset("materialId", data=matId)
    logging.info(f"HDF5 file saved to: {filepath}")


def main(N_grid, dim, outdir, verbose, dry_run):
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format='[%(levelname)s] %(message)s')

    if not os.path.exists(outdir):
        os.makedirs(outdir)
        logging.info(f"Created output directory: {outdir}")

    try:
        sml = get_sml_from_N(N_grid, dim)
    except ValueError as e:
        logging.error(str(e))
        return

    logging.info(f"Using smoothing length for N = {N_grid} in {dim}D: sml = {sml:.5f}")
    pos, vel, mass, u, matId = generate_sedov(N_grid, sml, dim)

    total_particles = len(pos)
    bbox_volume = 1.0
    particle_density = total_particles / bbox_volume
    avg_distance = bbox_volume ** (1 / dim) / N_grid
    eta = 1.3
    h_suggested = eta * avg_distance

    logging.info(f"Total particles: {total_particles}")
    logging.info(f"Mean particle density: {particle_density:.3e} particles/m^{dim}")
    logging.info(f"Mean particle spacing: {avg_distance:.3e}")
    logging.info(f"Suggested smoothing length (η = {eta}): h = {h_suggested:.3e}")

    # Dynamischer Name
    date = datetime.now().strftime("%Y%m%d")
    base_filename = f"{date}_sedov_N{total_particles}_sml{sml:.5f}_{dim}D"

    logging.info("Computing SPH density...")
    rho = compute_density(pos, mass, sml)

    visualize(pos, rho, dim, base_filename, outdir)

    plot_particle_distribution(pos=pos,dim=dim,title="Initial Particle Distribution",filename=f"{base_filename}_distribution.png",output_dir=outdir)

    if not dry_run:
        save_path = os.path.join(outdir, f"{base_filename}.h5")
        save_hdf5(save_path, pos, vel, mass, u, matId)

    logging.info("Sedov initialization completed.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Sedov initial conditions in 2D or 3D")
    parser.add_argument("-N", "--gridSize", type=int, required=True, help="Grid size (e.g. 61, 81, 101, 126, 171)")
    parser.add_argument("-d", "--dim", type=int, choices=[2, 3], default=3, help="Dimension (2 or 3)")
    parser.add_argument("-o", "--output", type=str, default="./", help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--dry", action="store_true", help="Run without writing file")
    args = parser.parse_args()

    main(args.gridSize, args.dim, args.output, args.verbose, args.dry)