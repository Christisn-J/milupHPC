import os
import glob
import re

RESOURCE_REGEX = re.compile(r"^GPU\d+[a-zA-Z]\d+_NP\d+_T\d{2}H\d{2}M\d{2}S\.res$")

from constants import si_prefixes, FIELD_META

def load_indices_from_file(filename):
    indices = []
    with open(filename, "r") as f:
        for line in f:
            if "NAN for index:" in line:
                parts = line.split()
                index_str = parts[3]        # z.B. "a_2392"
                number = index_str.split("_")[1]  # nur "2392"
                indices.append(int(number))
    # Duplikate entfernen und sortieren
    unique_indices = sorted(set(indices))
    return unique_indices

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

def find_resource_files(pattern, regex=RESOURCE_REGEX):
    """
    Returns glob pattern if at least one matching resource file exists,
    otherwise None.
    """
    files = glob.glob(pattern)

    valid = [
        f for f in files
        if regex.match(os.path.basename(f))
    ]

    return valid

def read_material(material_cfg):
    """
    Extract materials from material.cfg by explicitly searching for
    ID and name entries, independent of nested {} blocks.
    """

    materials = []
    current = {}

    id_pattern = re.compile(r'\bID\s*=\s*([0-9]+)\s*;')
    name_pattern = re.compile(r'\bname\s*=\s*"([^"]+)"\s*;')

    with open(material_cfg, "r") as f:
        for raw_line in f:
            # remove inline comments
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue

            id_match = id_pattern.search(line)
            if id_match:
                # if we already collected a material, store it
                if "ID" in current and "name" in current:
                    materials.append(current)
                    current = {}

                current["ID"] = id_match.group(1)
                continue

            name_match = name_pattern.search(line)
            if name_match:
                current["name"] = name_match.group(1)
                continue

    # append last material
    if "ID" in current and "name" in current:
        materials.append(current)

    return materials

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def truncate_cmap(cmap_name, minval=0.0, maxval=0.85, n=256):
    cmap = plt.get_cmap(cmap_name)
    new_colors = cmap(np.linspace(minval, maxval, n))
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc", new_colors
    )

