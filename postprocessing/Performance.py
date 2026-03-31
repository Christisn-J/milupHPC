#!/usr/bin/env python3

from statistics import mean
import numpy as np
import h5py
import argparse
import sys
import csv
import logging
import os
from pathlib import Path
import utility

from constants import si_prefixes, FIELD_META
utility.setup_dynamic_plotStyle(factor=2)
utility.setup_global_latex()

from help import si_prefixes

scale = {"t": "milli"}

def get_dataset_keys(f):
    keys = []
    f.visit(lambda key: keys.append(key) if isinstance(f[key], h5py.Dataset) else None)
    return keys


class H5entry(object):

    def __init__(self, name, data, extend=False, extend_length=0):
        self.name = name
        self.data = data
        self.numProcesses = len(data[0])
        if extend:
            self.extend_data(extend_length)

    def extend_data(self, extend_length):
        to_be_extended = self.data  # np.array(self.data)
        length = len(to_be_extended)
        append_length = int(extend_length / length)
        # print("shape: {}".format(to_be_extended.shape))
        # print("to_be_extended[0] = {}".format(to_be_extended[0]))
        # print("to_be_extended[1] = {}".format(to_be_extended[1]))
        self.data = []
        for i_step in range(length):
            value = []
            for i_proc in range(self.numProcesses):
                value.append(to_be_extended[i_step][i_proc])
            # print("append: {}, shape: {}".format(value, np.array(value).shape))
            self.data.append(value)
            for i in range(append_length - 1):
                self.data.append(np.zeros(np.array(value).shape))
                # print("append: {}".format(np.zeros(np.array(value).shape)))
        self.data = np.array(self.data)
        # print("new shape: {}".format(self.data.shape))

    @staticmethod
    def get_percentage(data, reference):
        return data / reference

    def get_proc_data(self, proc=0):
        if 0 <= proc < self.numProcesses:
            return [elem[proc] for elem in self.data]
        else:
            logging.warning(f"Only {self.numProcesses} processes available!")

    def get_averages(self):
        return [mean(elem) for elem in self.data]

    def get_average_mean(self):
        return mean(self.get_averages())

    def get_maxima(self):
        return [max(elem) for elem in self.data]

    def get_maxima_mean(self):
        maxima = self.get_maxima()
        zeros = maxima.count(0)

        if len(maxima) - zeros == 0:
            return np.nan, np.nan, zeros

        return mean(maxima), sum(maxima) / (len(maxima) - zeros), zeros

    def get_minima(self):
        return [min(elem) for elem in self.data]

    def get_minima_mean(self):
        return mean(self.get_minima())

    def get_average_per_proc(self):
        averages = []
        for proc in range(self.numProcesses):
            averages.append(mean(self.get_proc_data(proc)))
        return averages

    def get_sum_per_step(self):
        sums = []
        for i_step in range(len(self.data)):
            sum_per_proc = []
            for i_proc in range(self.numProcesses):
                sum_per_proc.append(sum(self.data[i_step][i_proc]))
            sums.append(sum_per_proc)
        return sums


class TimeEntry:

    def __init__(self, key, name, color="grey"):
        self.key = key
        self.name = name
        self.color = color
        self.mean = 0
        self.real_mean = 0
        self.unit = f"{si_prefixes[scale['t']]['abbr']}s"

    def __repr__(self):
        return f"TimeEntry(key='{self.key}', name='{self.name}')"


class TimeEvaluator(object):
    reference_entry = [
        TimeEntry("rhsElapsed", "rhsElapsed"),
        TimeEntry("rhs", "rhs"),
    ]

    preprocessing_entries = \
        [
            TimeEntry("removingParticles", "remove particles"),
            TimeEntry("loadBalancing", "load balancing")
        ]

    postprocessing_entries = \
        [
            TimeEntry("IO", "IO"),
            TimeEntry("integrate", "integration")
        ]

    gravity_sim_entries = \
        [
            TimeEntry("reset", "reset"),
            TimeEntry("assignParticles", "assign particles"),
            TimeEntry("boundingBox", "bounding box"),
            TimeEntry("tree", "build tree"),
            TimeEntry("pseudoParticle", "calculate pseudo-particle"),
            TimeEntry("gravity", "gravitational force")
        ]

    sph_sim_entries = \
        [
            TimeEntry("reset", "reset"),
            TimeEntry("assignParticles", "assign particles"),
            TimeEntry("boundingBox", "bounding box"),
            TimeEntry("tree", "build tree"),
            TimeEntry("pseudoParticle", "calculate pseudo-particle"),
            TimeEntry("sph", "SPH"),
        ]

    gravity_sph_sim_entries = \
        [
            TimeEntry("reset", "reset"),
            TimeEntry("assignParticles", "assign particles"),
            TimeEntry("boundingBox", "bounding box"),
            TimeEntry("tree", "build tree"),
            TimeEntry("pseudoParticle", "calculate pseudo-particle"),
            TimeEntry("gravity", "gravitational force"),
            TimeEntry("sph", "SPH")
        ]

    details_tree_entries = \
        [
            TimeEntry("tree_createDomainList", "create domain list"),
            TimeEntry("tree_buildTree", "build tree from particles"),
            TimeEntry("tree_buildDomainTree", "assign and add domain list nodes")
        ]

    details_gravity_entries = \
        [
            TimeEntry("gravity_compTheta", "determine relevant domain list nodes"),
            TimeEntry("gravity_symbolicForce", "determine (pseudo-) particles to be sent"),
            TimeEntry("gravity_gravitySendingParticles", "send (pseudo-) particles"),
            TimeEntry("gravity_insertReceivedPseudoParticles", "insert received pseudo-particles"),
            TimeEntry("gravity_insertReceivedParticles", "insert received particles"),
            TimeEntry("gravity_force", "gravitational force calculation"),
            TimeEntry("gravity_repairTree", "repair tree (delete received particles)")
        ]

    details_sph_entries = \
        [
            TimeEntry("sph_compTheta", "determine relevant domain list nodes"),
            TimeEntry("sph_determineSearchRadii", "calculate search radius"),
            TimeEntry("sph_symbolicForce", "determine particles to be sent"),
            TimeEntry("sph_sendingParticles", "send particles"),
            TimeEntry("sph_insertReceivedParticles", "insert received particles"),
            TimeEntry("sph_fixedRadiusNN", "FRNN search"),
            TimeEntry("sph_density", "calculate density"),
            TimeEntry("sph_pressure", "calculate pressure"),
            TimeEntry("sph_soundSpeed", "calculate speed of sound"),
            TimeEntry("sph_resendingParticles", "resend relevant entries"),
            TimeEntry("sph_internalForces", "calculate internal forces"),
            TimeEntry("sph_repairTree", "repair tree (delete received particles)")
        ]

    def __init__(self, input_file, data_dic, sim_type, unit="{si_prefixes[scale['t']]['abbr']}s"):
        logging.info("Time evaluation ...")
        self.input_file = input_file
        self.data_dic = data_dic
        # 0: gravity, 1: sph, 2: gravity + sph
        self.sim_type = sim_type
        # print("sim_type: {}".format(self.sim_type))
        self.unit = unit
        self.relevant_entries = None
        self.numProcesses = self.data_dic["tree"].numProcesses
        self.get_relevant_entries()

    def get_relevant_entries(self):
        if self.sim_type == 1:
            self.relevant_entries = self.gravity_sim_entries
        elif self.sim_type == 2:
            self.relevant_entries = self.sph_sim_entries
        elif self.sim_type == 3:
            self.relevant_entries = self.gravity_sph_sim_entries
        elif self.sim_type == 0:
            self.relevant_entries = []  # none
        else:
            logging.error(f"sim_type {self.sim_type} not available!")
            sys.exit(1)

    def summarize(self, filename, preprocessing=False, postprocessing=False):

        summarize_entries = []
        if preprocessing:
            summarize_entries.extend(self.preprocessing_entries)
        summarize_entries.extend(self.relevant_entries)
        if postprocessing:
            summarize_entries.extend(self.postprocessing_entries)

        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
            header = ["key", "name", "total_average", "real_average"]
            csv_writer.writerow(header)
            for relevant_entry in summarize_entries:
                if relevant_entry.key not in self.data_dic:
                    logging.warning(f"Missing dataset: {relevant_entry.key}")
                    continue

                # print(relevant_entry)
                relevant_entry.mean, self.real_mean, zeros = self.data_dic[relevant_entry.key].get_maxima_mean()
                # print("{}: {} {} | {} ({})".format(relevant_entry.name, relevant_entry.mean, relevant_entry.unit,
                #                                self.real_mean, zeros))
                csv_writer.writerow([relevant_entry.key, relevant_entry.name, relevant_entry.mean, self.real_mean])

    def summarize_reference(self, filename):
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
            header = ["key", "name", "total_average", "real_average"]
            csv_writer.writerow(header)

            for entry in self.reference_entry:
                if entry.key not in self.data_dic:
                    continue

                mean_val, real_mean, zeros = self.data_dic[entry.key].get_maxima_mean()
                csv_writer.writerow([
                    entry.key,
                    entry.name,
                    mean_val,
                    real_mean
                ])

    def summarize_tree(self, filename):
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
            header = ["key", "name", "total_average", "real_average"]
            csv_writer.writerow(header)
            for relevant_entry in self.details_tree_entries:
                relevant_entry.mean, self.real_mean, zeros = self.data_dic[relevant_entry.key].get_maxima_mean()
                # print("{}: {} {} | {} ({})".format(relevant_entry.name, relevant_entry.mean, relevant_entry.unit,
                #                                    self.real_mean, zeros))
                csv_writer.writerow([relevant_entry.key, relevant_entry.name, relevant_entry.mean, self.real_mean])

    def summarize_gravity(self, filename):
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
            header = ["key", "name", "total_average", "real_average"]
            csv_writer.writerow(header)
            for relevant_entry in self.details_gravity_entries:
                relevant_entry.mean, self.real_mean, zeros = self.data_dic[relevant_entry.key].get_maxima_mean()
                # print("{}: {} {} | {} ({})".format(relevant_entry.name, relevant_entry.mean, relevant_entry.unit,
                #                                    self.real_mean, zeros))
                csv_writer.writerow([relevant_entry.key, relevant_entry.name, relevant_entry.mean, self.real_mean])

    def summarize_sph(self, filename):
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
            header = ["key", "name", "total_average", "real_average"]
            csv_writer.writerow(header)
            for relevant_entry in self.details_sph_entries:
                relevant_entry.mean, self.real_mean, zeros = self.data_dic[relevant_entry.key].get_maxima_mean()
                # print("{}: {} {} | {} ({})".format(relevant_entry.name, relevant_entry.mean, relevant_entry.unit,
                #                                    self.real_mean, zeros))
                csv_writer.writerow([relevant_entry.key, relevant_entry.name, relevant_entry.mean, self.real_mean])


class ParticleEvaluator(object):

    # entries = ["numParticles", "numParticlesLocal", "ranges"]

    def __init__(self, input_file, data_dic, sim_type):
        logging.info("Particle evaluation ...")
        self.input_file = input_file
        self.data_dic = data_dic
        self.sim_type = sim_type
        self.numProcesses = self.data_dic["numParticles"].numProcesses

    def summarize(self, filename):
        start_number_particles = self.data_dic["numParticles"].data[0][0]
        with open(filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])

            # Header dynamisch erstellen
            header = ["numParticles"]
            header.extend([f"numParticlesLocal_proc{proc}" for proc in range(self.numProcesses)])
            header.extend([f"numParticlesPercentage_proc{proc}" for proc in range(self.numProcesses)])
            header.extend(["loss", "lossPercentage"])
            csv_writer.writerow(header)

            # Daten schreiben
            for i_step in range(len(self.data_dic["numParticles"].data)):
                num_particles = self.data_dic["numParticles"].data[i_step][0]
                num_particles_local = self.data_dic["numParticlesLocal"].data[i_step]
                num_particles_percentage = 100 * (self.data_dic["numParticlesLocal"].data[i_step] /
                                                  self.data_dic["numParticles"].data[i_step][0])

                row = [num_particles]
                row.extend([num_particles_local[proc] for proc in range(self.numProcesses)])
                row.extend([num_particles_percentage[proc] for proc in range(self.numProcesses)])
                row.append(start_number_particles - num_particles)
                row.append((start_number_particles - num_particles) / num_particles)
                csv_writer.writerow(row)


class CommunicationEvaluator(object):

    # entries = ["receiving/gravityParticles", "receiving/gravityPseudoParticles", "receiving/sph"]
    # entries = ["sending/gravityParticles", "sending/gravityPseudoParticles", "sending/sph"]

    def __init__(self, input_file, data_dic, sim_type):
        logging.info("Communication evaluation ...")
        self.input_file = input_file
        self.data_dic = data_dic
        # 0: gravity, 1: sph, 2: gravity + sph
        self.sim_type = sim_type
        # print("sim_type: {}".format(self.sim_type))
        self.numProcesses = 0
        try:
            self.numProcesses = self.data_dic["gravityParticles"].numProcesses
        except:
            self.numProcesses = self.data_dic["sph"].numProcesses
        self.gravity_particle_sums = None
        self.gravity_pseudo_particle_sums = None
        self.gravity_sums = None
        self.sph_sums = None
        self.total_communication_per_step_per_proc()

        self.gravity_total_sums = None
        self.gravity_total_particle_sums = None
        self.gravity_total_pseudo_particle_sums = None
        self.sph_total_sums = None
        self.total_sums = None
        self.total_communication_per_step()

    def total_communication_per_step_per_proc(self):
        if self.sim_type == 1:
            self.gravity_particle_sums = self.data_dic["gravityParticles"].get_sum_per_step()
            self.gravity_pseudo_particle_sums = self.data_dic["gravityPseudoParticles"].get_sum_per_step()
            self.gravity_sums = [self.gravity_particle_sums[i] + self.gravity_pseudo_particle_sums[i] for i in range(len(self.gravity_particle_sums))]
            self.sph_sums = [0 for i in range(len(self.gravity_particle_sums))]
        elif self.sim_type == 2:
            self.sph_sums = self.data_dic["sph"].get_sum_per_step()
            self.gravity_particle_sums = [0 for i in range(len(self.sph_sums))]
            self.gravity_pseudo_particle_sums = [0 for i in range(len(self.sph_sums))]
            self.gravity_sums = [0 for i in range(len(self.sph_sums))]
        elif self.sim_type == 3:
            self.gravity_particle_sums = self.data_dic["gravityParticles"].get_sum_per_step()
            self.gravity_pseudo_particle_sums = self.data_dic["gravityPseudoParticles"].get_sum_per_step()
            self.gravity_sums = [self.gravity_particle_sums[i] + self.gravity_pseudo_particle_sums[i] for i in range(len(self.gravity_particle_sums))]
            self.sph_sums = self.data_dic["sph"].get_sum_per_step()
        else:
            print("sim_type {} not available! exiting...".format(self.sim_type))
            sys.exit(1)

    def total_communication_per_step(self):
        if self.sim_type == 1:
            self.gravity_total_sums = [sum(elem) for elem in self.gravity_sums]
            self.gravity_total_particle_sums = [sum(elem) for elem in self.gravity_particle_sums]
            self.gravity_total_pseudo_particle_sums = [sum(elem) for elem in self.gravity_pseudo_particle_sums]
            self.sph_total_sums = [0 for i in range(len(self.sph_sums))]
            self.total_sums = self.total_sums
        elif self.sim_type == 2:
            self.gravity_total_sums = [0 for i in range(len(self.sph_sums))]
            self.gravity_total_particle_sums = [0 for i in range(len(self.sph_sums))]
            self.gravity_total_pseudo_particle_sums = [0 for i in range(len(self.sph_sums))]
            self.sph_total_sums = [sum(elem) for elem in self.sph_sums]
            self.total_sums = self.sph_total_sums
        elif self.sim_type == 3:
            self.gravity_total_sums = [sum(elem) for elem in self.gravity_sums]
            self.gravity_total_particle_sums = [sum(elem) for elem in self.gravity_particle_sums]
            self.gravity_total_pseudo_particle_sums = [sum(elem) for elem in self.gravity_pseudo_particle_sums]
            self.sph_total_sums = [sum(elem) for elem in self.sph_sums]
            self.total_sums = [self.gravity_total_sums[i] + self.sph_total_sums[i] for i in range(len(self.gravity_total_sums))]
        else:
            print("sim_type {} not available! exiting...".format(self.sim_type))
            sys.exit(1)

    def summarize(self, filename):

        # self.gravity_particle_sums = None
        # self.gravity_pseudo_particle_sums = None
        # self.gravity_sums = None
        # self.sph_sums = None
        # self.total_communication_per_step_per_proc()
        # self.gravity_total_sums = None
        # self.sph_total_sums = None
        # self.total_sums = None

        # gravity all, gravity particle all, gravity pseudo-particle all, sph all, gravity per proc,
        # gravity particle per proc, gravity pseudo-particle per proc, sph per proc

        with open(filename, 'w', newline='') as csv_file:
            if self.sim_type == 1:
                csv_writer = csv.writer(csv_file, delimiter=";")
                csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
                header = ["total_gravity", "total_gravity_particle", "total_gravity_pseudo_particle"]
                [header.append("gravity_proc{}".format(proc)) for proc in range(self.numProcesses)]
                [header.append("gravity_particle_proc{}".format(proc)) for proc in range(self.numProcesses)]
                [header.append("gravity_pseudo_particle_proc{}".format(proc)) for proc in range(self.numProcesses)]
                csv_writer.writerow(header)

                for i in range(len(self.gravity_sums)):
                    row = [self.gravity_total_sums[i],
                           self.gravity_total_particle_sums[i],
                           self.gravity_total_pseudo_particle_sums[i]]
                    temp = [self.gravity_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)
                    temp = [self.gravity_particle_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)
                    temp = [self.gravity_pseudo_particle_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)

                    csv_writer.writerow(row)
            elif self.sim_type == 2:
                csv_writer = csv.writer(csv_file, delimiter=";")
                csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
                header = ["total_sph"]
                [header.append("sph_proc{}".format(proc)) for proc in range(self.numProcesses)]
                csv_writer.writerow(header)

                for i in range(len(self.gravity_sums)):
                    row = [self.sph_total_sums[i]]
                    temp = [self.sph_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)

                    csv_writer.writerow(row)
            elif self.sim_type == 3:
                csv_writer = csv.writer(csv_file, delimiter=";")
                csv_writer.writerow([self.input_file, self.sim_type, self.numProcesses])
                header = ["total_gravity", "total_gravity_particle", "total_gravity_pseudo_particle", "total_sph"]
                [header.append("gravity_proc{}".format(proc)) for proc in range(self.numProcesses)]
                [header.append("gravity_particle_proc{}".format(proc)) for proc in range(self.numProcesses)]
                [header.append("gravity_pseudo_particle_proc{}".format(proc)) for proc in range(self.numProcesses)]
                [header.append("sph_proc{}".format(proc)) for proc in range(self.numProcesses)]
                csv_writer.writerow(header)

                for i in range(len(self.gravity_sums)):
                    row = [self.gravity_total_sums[i],
                           self.gravity_total_particle_sums[i],
                           self.gravity_total_pseudo_particle_sums[i],
                           self.sph_total_sums[i]]
                    temp = [self.gravity_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)
                    temp = [self.gravity_particle_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)
                    temp = [self.gravity_pseudo_particle_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)
                    temp = [self.sph_sums[i][proc] for proc in range(self.numProcesses)]
                    row.extend(temp)

                    csv_writer.writerow(row)
            else:
                print("sim type {} not available!".format(self.sim_type))
                sys.exit(1)


class Dic2H5(object):

    def __init__(self, data_dic):
        self.data_dic = data_dic

    def write_to_h5(self, filename="summary.h5"):
        hf = h5py.File(filename, "w")
        for data_key in self.data_dic:
            hf.create_dataset(data_key, data=self.data_dic[data_key])

        hf.close()


import pandas as pd
import matplotlib.pyplot as plt
import re


def plot_csv_line(csv_path, output_dir, column=None, separate=False, dpi=300, extension=None):
    """
    Reads a CSV file and creates line plots.

    - If `column` is specified, only that base column (including _proc*) is plotted.
    - If `column` is None, all base columns are detected and grouped by suffix (_proc0, _proc1, etc.).
      Each group is plotted in one figure with different lines for each process.

    Parameters:
        csv_path (str or Path): Path to the CSV file.
        output_dir (str or Path): Directory to save the plots.
        column (str, optional): Specific base column to plot (without _procX). If None, all groups are plotted.
        dpi (int): Resolution of the saved figure.
        extension (str): File format for saving (png, pdf, etc.).
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path, sep=';', skiprows=1)
    except Exception as e:
        logging.error(f"Error reading file {csv_path}: {e}")
        return

    # --- Identify column groups based on _proc suffix ---
    col_pattern = re.compile(r"^(.*)(_proc\d+)$")
    groups = {}
    for col in df.columns:
        match = col_pattern.match(col)
        if match:
            base = match.group(1)
            if base not in groups:
                groups[base] = []
            groups[base].append(col)
        else:
            # columns without _proc suffix
            groups[col] = [col]

    # --- Decide which groups to plot ---
    if column is not None:
        if column not in groups:
            logging.warning(f"Column group '{column}' not found in {csv_path}. Skipping plot.")
            return
        groups_to_plot = {column: groups[column]}
    else:
        groups_to_plot = groups

    # --- Generate plots ---
    for base_name, cols in groups_to_plot.items():
        # --- Combined plot ---
        plt.figure(figsize=(8, 4.5))
        for col in cols:
            plt.plot(df[col], marker='o', linestyle='-', markersize=3, label=col)
        plt.title(f"{base_name} over Time Step / Processes")
        plt.xlabel("Time Step")
        plt.ylabel(base_name)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()

        for e in args.extension:
            plt.savefig(os.path.join(output_dir, f"{csv_path.stem}_{base_name}_line.{e}"), dpi=dpi, bbox_inches='tight')
        plt.close()
        logging.debug(f"Combined line plot saved: '{csv_path.stem}_{base_name}_line'")

        # --- Individual per-column plots, only if flag is True ---
        if separate:
            for col in cols:
                plt.figure(figsize=(8, 4.5), dpi=dpi)
                plt.plot(df[col], marker='o', linestyle='-', markersize=3)
                plt.title(f"{col} over Time Step")
                plt.xlabel("Time Step")
                plt.ylabel(col)
                plt.grid(True, which='both', linestyle='--', alpha=0.5)
                plt.tight_layout()

                for e in args.extension:
                    plt.savefig(os.path.join(output_dir, f"{csv_path.stem}_{col}_line.{e}"), dpi=dpi, bbox_inches='tight')
                plt.close()
                logging.debug(f"Individual line plot saved: '{csv_path.stem}_{col}_line.png'")


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

def plot_csv_bar(csv_path, output_dir, column=None, extension=None, flag_relativ=True, dpi=300):
    """
    Reads a CSV file and creates a bar chart.
    x = name, y = column (e.g., total_average)
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(csv_path, sep=';', skiprows=1)
    except Exception as e:
        logging.error(f"Error reading file {csv_path}: {e}")
        return

    if column not in df.columns or 'name' not in df.columns:
        logging.warning(f"Columns 'name' or '{column}' not found in {csv_path}.")
        return

    df_clean = df.dropna(subset=[column, 'name'])

    # --- Summe aller Werte berechnen ---
    total_value = df_clean[column].sum()

    plt.figure(figsize=(10, 5), dpi=dpi)
    bars = plt.bar(df_clean['name'], df_clean[column], color='steelblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')

    plt.xlabel("Performed Operation", fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    plt.ylabel(rf"$\langle t_{{\mathrm{{real}}}} \rangle$ [${si_prefixes[scale['t']]['abbr']}{FIELD_META['t']['unit'].strip('$')}$]", fontsize=plt.rcParams["axes.labelsize"], labelpad=15)

    # --- Titel mit Summe ---
    plt.title(rf"Average run time per function call ($\langle t_{{\mathrm{{real, tot}}}} \rangle={total_value:.2f}$ ${si_prefixes[scale['t']]['abbr']}{FIELD_META['t']['unit'].strip('$')}$)", fontsize=plt.rcParams["figure.titlesize"], y=1.01)

    if flag_relativ:
        # --- Prozentzahlen auf die Balken schreiben ---
        for bar, value in zip(bars, df_clean[column]):
            percent = (value / total_value) * 100
            plt.text(
                bar.get_x() + bar.get_width() / 2,  # Mitte des Balkens
                value,                               # oberhalb des Balkens
                f"{percent:.1f}%",
                ha='center', va='bottom', fontsize=11
            )

        # --- Y-Achse anpassen, damit Text nicht abgeschnitten wird ---
        y_max = df_clean[column].max()
        plt.ylim(0, y_max * 1.15)  # 15% Puffer nach oben
    plt.tight_layout()

    max_label_len = df_clean['name'].astype(str).str.len().max()
    bottom_margin = min(0.5, 0.15 + max_label_len * 0.01)
    plt.subplots_adjust(bottom=bottom_margin)

    for e in args.extension:
        plt.savefig(os.path.join(output_dir, f"{csv_path.stem}_{column}_bar.{e}"), dpi=dpi, bbox_inches='tight')
    plt.close()
    logging.debug(f"Bar chart saved: '{csv_path.stem}_{column}_line'")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Performance postprocessing, generating csv (summary) files.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--file", "-f", type=Path, required=True, help="Path to the h5 profiling data file")
    parser.add_argument("--output", "-o", type=Path, default="./output", help="Output directory")
    parser.add_argument("--sim_type", type=int, nargs="+", choices=[0, 1, 2], default=None, required=True, help="Simulation types (0=None, 1=gravity, 2=sph). Multiple allowed.")
    parser.add_argument("--scaling", default="none",choices=["strong", "weak", "none"],help="Scaling analysis type")
    parser.add_argument("--details", action="store_true", help="Enable all detailed evaluations")
    parser.add_argument("--verbose", "-v", type=int, choices=[1, 2, 3], default=3, help="Enable verbose output")
    parser.add_argument("--config", "-c", type=str, default=None, help="Config file")
    parser.add_argument("--resource", "-r", type=str, default=None, help="Resource file")
    parser.add_argument("--material", "-m", type=str, default=None, help="Material file")
    parser.add_argument("--dpi", type=int, default=300, help="Set DPI for all output plots (default: 300).")
    parser.add_argument("--extension", "-e",nargs="+",default=["png"],choices=["png", "pdf", "svg", "jpg"],help="Output file formats (default: png). Example: -e png pdf svg")

    args = parser.parse_args()

    # Set logging level based on verbosity flag
    if args.verbose >= 3:
        log_level = logging.DEBUG
    elif args.verbose == 2:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    utility.setup_logging(level=log_level)

    base = os.path.abspath(os.path.dirname(args.file))
    configured_path = os.path.join(base, ".." ,"configured")

    # --- config ---
    if args.config is None:
        args.config = sorted([os.path.join(configured_path, f) for f in os.listdir(configured_path) if f.endswith(".info")])[0]

    # --- material ---
    if args.material is None:
        args.material = sorted([os.path.join(configured_path, f) for f in os.listdir(configured_path) if f.endswith(".cfg")])[0]

    # --- resource ---
    if args.resource is None:
        args.resource = sorted([os.path.join(configured_path, f) for f in os.listdir(configured_path) if f.endswith(".res")])[0]

    # --------------------------------------------------
    # logging
    # --------------------------------------------------
    logging.info(f"Using performance:   {args.file or 'None'}")
    logging.info(f"Using config:        {args.config or 'None'}")
    logging.info(f"Using material:      {args.material or 'None'}")
    logging.info(f"Using resource:      {args.resource or 'None'}")

    particle_evaluation_file = args.output / "particle"
    time_evaluation_file = args.output / "time"
    communication_evaluation_file = args.output / "communication"

    particle_evaluation = True
    communication_evaluation = True
    time_evaluation = True

    if not args.file.exists():
        logging.error(f"Input file not found: {args.file}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    f = h5py.File(args.file, 'r')
    keys = get_dataset_keys(f)

    # --------------------------------------------------
    # Extract NPROCs from .res file
    # --------------------------------------------------

    num_procs = "unknown"
    try:
        with open(args.resource, "r") as res_file:
            for line in res_file:
                line = line.strip()

                # überspringe Kommentare
                if line.startswith("#"):
                    continue

                if line.startswith("NPROCs"):
                    # NPROCs=1 → ["NPROCs", "1"]
                    num_procs = line.split("=")[1].strip()
                    break

    except Exception as e:
        logging.warning(f"Could not read NPROCs from resource file: {e}")

    # --- Number of particles from HDF5 ---
    try:
        num_particles_start = int(f["general/numParticles"][0][0])
    except Exception:
        num_particles_start = "unknown"

    # --- filename suffix ---
    file_suffix = f"{args.scaling}_P{num_procs}_N{num_particles_start}"

    max_length = 0
    for key in keys:
        if len(f[key][:]) > max_length:
            max_length = len(f[key][:])

    # --- Automatisch sim_type = 3 setzen, falls sowohl 1 (gravity) als 2 (sph) angegeben wurde ---
    if isinstance(args.sim_type, list):
        if 1 in args.sim_type and 2 in args.sim_type:
            # Remove 1 and 2, keep 0 if present
            args.sim_type = [t for t in args.sim_type if t not in (1, 2)]
            # Add combined type 3
            args.sim_type.append(3) # kombiniere gravity + sph
        else:
            # keine Änderung, einzelne SimTypes bleiben
            pass

    ##########################################################################
    if particle_evaluation:
        particle_dic = {}
        for key in keys:
            if key.startswith("general/"):
                # print(key)
                name = key.replace("general/", "")
                particle_dic[name] = H5entry(name, f[key][:])

        for sim in args.sim_type:
            particle_evaluator = ParticleEvaluator(args.file, particle_dic, sim)
            csv_file = Path(f"{particle_evaluation_file}_{file_suffix}_sim{sim}.csv")
            particle_evaluator.summarize(csv_file)
            plot_csv_line(csv_file, args.output, column=None, separate=True, dpi=args.dpi, extension=args.extension)

    ##########################################################################
    if time_evaluation:
        time_dic = {}
        for key in keys:
            if key.startswith("time/"):
                logging.debug(f"Found time dataset: {key}")
                name = key.replace("time/", "")
                if len(f[key][:]) < max_length:
                    time_dic[name] = H5entry(name, f[key][:], True, max_length)
                else:
                    time_dic[name] = H5entry(name, f[key][:])

        for sim in args.sim_type:

            time_evaluator = TimeEvaluator(args.file, time_dic, sim)

            base_csv = Path(f"{time_evaluation_file}_{file_suffix}_sim{sim}.csv")
            time_evaluator.summarize(base_csv, False, True)
            plot_csv_bar(base_csv, args.output, column="total_average", dpi=args.dpi, extension=args.extension)

            if args.details:
                tree_csv = Path(f"{time_evaluation_file}_{file_suffix}_tree_sim{sim}.csv")
                time_evaluator.summarize_tree(tree_csv)
                plot_csv_bar(tree_csv, args.output, column="total_average", dpi=args.dpi, extension=args.extension)

                rhs_csv = Path(f"{time_evaluation_file}_{file_suffix}_reference_sim{sim}.csv")
                time_evaluator.summarize_reference(rhs_csv)
                plot_csv_bar(rhs_csv, args.output, column="total_average", dpi=args.dpi, extension=args.extension)

                if sim == 1:
                    grav_csv = Path(f"{time_evaluation_file}_{file_suffix}_gravity_sim{sim}.csv")
                    time_evaluator.summarize_gravity(grav_csv)
                    plot_csv_bar(grav_csv, args.output, column="total_average", dpi=args.dpi, extension=args.extension)

                if sim == 2:
                    sph_csv = Path(f"{time_evaluation_file}_{file_suffix}_sph_sim{sim}.csv")
                    time_evaluator.summarize_sph(sph_csv)
                    plot_csv_bar(sph_csv, args.output, column="total_average", dpi=args.dpi, extension=args.extension)

    ##########################################################################
    if communication_evaluation:
        communication_dic = {}
        for key in keys:
            if key.startswith("sending/"):
                name = key.replace("sending/", "")
                # print("key: {}, name: {}".format(key, name))
                if len(f[key][:]) < max_length:
                    communication_dic[name] = H5entry(name, f[key][:], True, max_length)
                else:
                    communication_dic[name] = H5entry(name, f[key][:])

        for sim in args.sim_type:
            communication_evaluator = CommunicationEvaluator(args.file, communication_dic, sim)

            csv_file = Path(f"{communication_evaluation_file}_{file_suffix}_sim{sim}.csv")
            communication_evaluator.summarize(csv_file)
            plot_csv_line(csv_file, args.output, column=None, separate=True, dpi=args.dpi, extension=args.extension)

    f.close()
