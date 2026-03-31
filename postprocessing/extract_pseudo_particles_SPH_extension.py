#!/usr/bin/env python3
"""
Extrahiere Pseudo-Partikel aus allen miluphpc HDF5-Dumps in einem Ordner,
bestimmt Pseudo-Partikel anhand von Smoothing Length aus einem Materialfile,
und optional werden Achsenlimits aus einem Referenz-File übernommen.
"""

import h5py
import numpy as np
from scipy.spatial import cKDTree
import argparse
import os
import re

# ======================================================================================
# Argumente
# ======================================================================================
parser = argparse.ArgumentParser(
    description="Extract pseudo particles from all tsXXXXXX.h5 files in a folder"
)
parser.add_argument(
    "--input-dir", required=True,
    help="Input folder containing HDF5 files (tsXXXXXX.h5)"
)
parser.add_argument(
    "--material-file", required=True,
    help="Material file to read smoothing lengths (sml)"
)
parser.add_argument(
    "--ref-file", default=None,
    help="Reference HDF5 file to determine axis limits"
)
args = parser.parse_args()

input_dir = os.path.abspath(args.input_dir)
material_file = os.path.abspath(args.material_file)
ref_file = os.path.abspath(args.ref_file) if args.ref_file else None

# Ausgabeordner
output_dir = os.path.join(input_dir, "pseudo")
os.makedirs(output_dir, exist_ok=True)

print(f"Input folder : {input_dir}")
print(f"Output folder: {output_dir}")
print(f"Material file: {material_file}")
if ref_file:
    print(f"Reference file for axis limits: {ref_file}")

# ======================================================================================
# Materialfile parsen
# ======================================================================================
def parse_material_file(file_path):
    mat_sml = {}
    current_id = None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("ID"):
                current_id = int(line.split("=")[1].replace(";", "").strip())
            elif line.startswith("sml") and current_id is not None:
                sml_value = float(line.split("=")[1].split("#")[0].replace(";", "").strip())
                mat_sml[current_id] = sml_value
                current_id = None
    return mat_sml

mat_sml = parse_material_file(material_file)
print(f"Read smoothing lengths for {len(mat_sml)} materials.")

# ======================================================================================
# Dateien finden
# ======================================================================================
pattern = re.compile(r"ts\d{6}\.h5$")
all_files = [f for f in os.listdir(input_dir) if pattern.match(f)]

if not all_files:
    print("Keine tsXXXXXX.h5 Dateien gefunden!")
    exit(1)

print(f"Found {len(all_files)} files to process.")

# ======================================================================================
# Achsenlimits aus Referenzfile
# ======================================================================================
if ref_file:
    with h5py.File(ref_file, "r") as f:
        x_ref = f["x"][:]
        x_min = np.min(x_ref, axis=0)
        x_max = np.max(x_ref, axis=0)
    print(f"Axis limits from reference file: min={x_min}, max={x_max}")
else:
    x_min = x_max = None  # werden pro Datei berechnet, falls nötig

# ======================================================================================
# Verarbeitung jeder Datei
# ======================================================================================
for file_name in all_files:
    input_file = os.path.join(input_dir, file_name)
    output_file = os.path.join(output_dir, file_name.replace(".h5", "_pseudo.h5"))

    print(f"\nProcessing {file_name} ...")

    # HDF5 laden
    with h5py.File(input_file, "r") as f:
        x = f["x"][:]
        proc = f["proc"][:]
        matId = f["matId"][:] if "matId" in f else np.zeros(len(x), dtype=int)

        coords = x.copy()

        # h pro Partikel aus Materialfile
        h_array = np.array([mat_sml[mid] for mid in matId])

        # Optionale Felder
        optional_fields = {}
        for name in ["v", "m", "rho", "e", "matId", "noi"]:
            if name in f:
                optional_fields[name] = f[name][:]

    num_particles = coords.shape[0]
    print(f"Loaded {num_particles} particles")

    # Bounding box für diese Datei, falls kein Referenzfile
    if x_min is None or x_max is None:
        x_min_file = np.min(coords, axis=0)
        x_max_file = np.max(coords, axis=0)
    else:
        x_min_file = x_min
        x_max_file = x_max

    print(f"Bounding box: min={x_min_file}, max={x_max_file}")

    # KD-Tree
    tree = cKDTree(coords)

    # Pseudo-Partikel
    is_pseudo = np.zeros(num_particles, dtype=bool)
    for i in range(num_particles):
        neighbors = tree.query_ball_point(coords[i], h_array[i])
        for j in neighbors:
            if proc[j] != proc[i]:
                is_pseudo[i] = True
                break

    pseudo_idx = np.where(is_pseudo)[0]
    num_pseudo = len(pseudo_idx)
    print(f"Found {num_pseudo} pseudo particles")

    # Extrahieren
    pseudo_fields = {
        "x": coords[pseudo_idx],
        "proc": proc[pseudo_idx],
    }
    for name, data in optional_fields.items():
        pseudo_fields[name] = data[pseudo_idx]

    # Speichern
    with h5py.File(output_file, "w") as f:
        for name, data in pseudo_fields.items():
            f.create_dataset(name, data=data)

    print(f"Saved pseudo particles to {output_file}")

print("\nAll done.")
