import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
def generate_cube_particles(edge_length, *arr, center=[0, 0, 0], delta=1e-3, dim=3):
    # Erstelle eine regelmäßige Gitteranordnung der Teilchen, abhängig von dim
    ranges = [np.arange(center[i] - edge_length, center[i] + edge_length + delta, delta) for i in range(dim)]

    # Erstelle ein Gitter der Teilchenpositionen (DIMxDIM Meshgrid)
    grids = np.meshgrid(*ranges)

    # Reshape und flache das Gitter, um die Teilchenkoordinaten zu bekommen
    X_flat = np.ravel(grids[0])  # Alle X-Koordinaten
    Y_flat = np.ravel(grids[1]) if dim >= 2 else np.zeros_like(X_flat) # Alle Y-Koordinaten
    Z_flat = np.ravel(grids[2]) if dim == 3 else np.zeros_like(X_flat)  # Z-Koordinaten, falls 3D

    N = X_flat.size
    vx = np.full(N, arr[0][0])
    vy = np.full(N, arr[0][1]) if dim >= 2 else np.zeros_like(X_flat)  # Geschwindigkeit in Y für 2D und 3D
    vz = np.full(N, arr[0][2]) if dim == 3 else np.zeros_like(X_flat)  # Geschwindigkeit in Z für 3D

    m = np.full(N, arr[1])  # Masse
    id = np.full(N, arr[2])  # Material-ID
    rho = np.full(N, arr[3])  # Dichte

    # Zusammenführen der Daten in einem Array
    particles = np.vstack([X_flat, Y_flat, Z_flat, vx, vy, vz, m, id, rho]).T

    return particles

def generate_sphere_particles(radius, *arr, center=[0, 0, 0], delta=1e-3, dim=3):
    # Erzeuge ein Gitter, das die Kugel vollständig enthält
    x = np.arange(center[0] - radius, center[0] + radius + delta, delta)
    y = np.arange(center[1] - radius, center[1] + radius + delta, delta) if dim >= 2  else [center[1]]
    z = np.arange(center[2] - radius, center[2] + radius + delta, delta) if dim == 3 else [center[2]]

    X, Y, Z = np.meshgrid(x, y, z)
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()

    # Filtere nur die Punkte, die innerhalb des Kugelradius liegen
    distances = np.sqrt((X_flat - center[0])**2 + (Y_flat - center[1])**2 + (Z_flat - center[2])**2)
    mask = distances <= radius

    # Wende die Maske an, um nur die gültigen Positionen zu behalten
    X_valid = X_flat[mask]
    Y_valid = Y_flat[mask]
    Z_valid = Z_flat[mask]

    N = len(X_valid)

    # Zusatzdaten pro Partikel
    vx = np.full(N, arr[0][0])
    vy = np.full(N, arr[0][1])
    vz = np.full(N, arr[0][2])
    m = np.full(N, arr[1])
    id = np.full(N, arr[2])
    rho = np.full(N, arr[3])

    particles = np.vstack([X_valid, Y_valid, Z_valid, vx, vy, vz, m, id, rho]).T

    return particles
def main(dim, ver):
    # SI-Einheiten
    # Material ist Aluminiumlegierung 6061
    rhoB = 2.7 * 1e3  # kg/m³

    impact_radius = 0.5 * 6.35e-3 # m
    impact_speed = -7000  # m/s
    cube_length = 5.0e-2  # halbe Kantenlänge

    # Abstand zwischen den Teilchen
    delta = 1e-3

    # Parameter für Aluminiumteilchen
    mass = rhoB * delta ** 3
    density = rhoB
    materialtype_al = 0
    materialId = 2
    velocity_al = [0, 0, 0]

    cube_particles = generate_cube_particles(cube_length, velocity_al, mass, materialId, density, delta=delta, dim=dim)

    # Parameter für Impaktorpartikel
    if dim == 2:
        center_impact  = np.array([0, cube_length + 4 * delta + impact_radius, 0])
        velocity_impact  = [0, impact_speed, 0]
    elif dim == 3:
        center_impact  = np.array([0, 0, cube_length + 4 * delta + impact_radius])
        velocity_impact  = [0, 0, impact_speed]
    else:
        center_impact  = np.array([cube_length + 4 * delta + impact_radius, 0, 0])
        velocity_impact  = [impact_speed,0,0]
    materialtype_impact = 0
    materialId = 2

    impactor_particles = generate_sphere_particles(impact_radius, velocity_impact, mass, materialId, density, center=center_impact, delta=delta, dim=dim)

    # Visualisierung der Teilchen
    fig = plt.figure(figsize=(10, 7))

    # Je nach Dimension (1D, 2D oder 3D) wird die Visualisierung angepasst
    if dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(cube_particles[:, 0], cube_particles[:, 1], cube_particles[:, 2], color='b', s=1, label='Al6061 Cube')
        ax.scatter(impactor_particles[:, 0], impactor_particles[:, 1], impactor_particles[:, 2], color='r', s=1, label='Impactor Sphere')

        skip = max(1, int(len(cube_particles)*0.8))
        # # Cube-Geschwindigkeitspfeile
        ax.quiver(cube_particles[::skip, 0], cube_particles[::skip, 1], cube_particles[::skip, 2],
                  cube_particles[::skip, 3], cube_particles[::skip, 4], cube_particles[::skip, 5],
                  color='c', length=1e-3, normalize=True)

        skip = max(1, int(len(impactor_particles)*0.2))
        # Impaktor-Geschwindigkeitspfeile
        ax.quiver(impactor_particles[::skip, 0], impactor_particles[::skip, 1], impactor_particles[::skip, 2],
                  impactor_particles[::skip, 3], impactor_particles[::skip, 4], impactor_particles[::skip, 5]
                  ,color='orange', length=1e-2 ,normalize=True)


        ax.set_xlabel('X-Achse')
        ax.set_ylabel('Y-Achse')
        ax.set_zlabel('Z-Achse')
        ax.legend()
        ax.view_init(elev=0, azim=90)
    elif dim == 2:
        ax = fig.add_subplot(111)
        ax.scatter(cube_particles[:, 0], cube_particles[:, 1], color='b', s=1, label='Al6061 Cube')
        ax.scatter(impactor_particles[:, 0], impactor_particles[:, 1], color='r', s=1, label='Impactor Sphere')
        # Quiver-Plot (Geschwindigkeiten als Pfeile)
        skip = max(1, int(len(cube_particles)*1.0))
        ax.quiver(cube_particles[::skip, 0], cube_particles[::skip, 1], cube_particles[::skip, 3], cube_particles[::skip, 4], color='b', scale=impact_speed*2)
        skip = max(1, int(len(impactor_particles)*0.2))
        ax.quiver(impactor_particles[::skip, 0], impactor_particles[::skip, 1], impactor_particles[::skip, 3], impactor_particles[::skip, 4], color='orange', scale=impact_speed*2)
        ax.set_xlabel('X-Achse')
        ax.set_ylabel('Y-Achse')
        ax.legend()
    elif dim == 1:
        ax = fig.add_subplot(111)
        ax.scatter(cube_particles[:, 0], np.zeros_like(cube_particles[:, 0]), color='b', s=1, label='Al6061 Cube')
        ax.scatter(impactor_particles[:, 0], np.zeros_like(impactor_particles[:, 0]), color='r', s=1, label='Impactor Sphere')
        # Quiver-Plot (X-Geschwindigkeitspfeile)
        skip = max(1, int(len(cube_particles)*1.0))
        ax.quiver(cube_particles[::skip, 0], np.zeros_like(cube_particles[::skip, 0]),
                  cube_particles[::skip, 3], np.zeros_like(cube_particles[::skip, 0]),
                  color='b', scale=impact_speed*2)
        skip =  max(1, int(len(impactor_particles)*0.2))
        ax.quiver(impactor_particles[::skip, 0], np.zeros_like(impactor_particles[::skip, 0]),
                  impactor_particles[::skip, 3], np.zeros_like(impactor_particles[::skip, 0]),
                  color='orange', scale=impact_speed*2)
        ax.set_xlabel('X-Achse')
        ax.set_yticks([])  # Y-Achse ausblenden, da nur 1D
        ax.legend()


    # Zusammenführen der Teilchen
    totalParticles = np.concatenate((cube_particles, impactor_particles))
    totalNum_particles = len(totalParticles)

    plt.tight_layout()
    name = f"al_NC{len(cube_particles)}_NI{len(impactor_particles)}_D{dim}"
    plt.savefig(f"{name}.png", dpi=300)
    # plt.show()
    plt.close()
    h5f = h5py.File(f"{name}.h5", "w")
    sys.stdout.write(f"\n\rSaving to {name} ...")

    # Daten in HDF5-Datei speichern
    h5f.create_dataset("x", data=totalParticles[:, :dim])  # Nur dim-Koordinaten
    h5f.create_dataset("v", data=totalParticles[:, 3:3+dim])  # Geschwindigkeit für die entsprechenden Dimensionen
    h5f.create_dataset("m", data=totalParticles[:, 6])
    h5f.create_dataset("materialId", data=totalParticles[:, 7])
    h5f.create_dataset("rho", data=totalParticles[:, 8])

    h5f.close()
    sys.stdout.write(" done.\n")

    if ver:
        # Datei öffnen
        with h5py.File(f"{name}.h5", "r") as f:
            # Zeige alle enthaltenen Datensätze
            sys.stdout.write("Datasets in file:")
            for key in f.keys():
                sys.stdout.write(f"\r{key}: shape = {f[key].shape}, dtype = {f[key].dtype}\n")
                sys.stdout.write(f"\r{f[key][:]}\n")

if __name__ == '__main__':
    # Argumentparser für Kommandozeilenargumente
    parser = argparse.ArgumentParser(description="Teilchensimulation für Cube und Impaktor")
    parser.add_argument("-d", "--dimensions", type=int, choices=[1, 2, 3], default=3,
                        help="Anzahl der Dimensionen (2 oder 3)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Aktiviere ausführliche Ausgabe")
    args = parser.parse_args()

    main(args.dimensions, args.verbose)



