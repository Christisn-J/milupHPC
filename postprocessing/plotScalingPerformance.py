#!/usr/bin/env python3
"""
Unified Scaling Analysis for HPC Simulations

This script provides a framework to analyze scaling behavior in HPC applications.
It supports:

1. Raw time plots for different metrics (total/real time)
2. Strong scaling analysis (speedup + efficiency)
3. Weak scaling analysis (efficiency)

Key Concepts:
- Strong scaling: measures how the solution time decreases as more processors are used
  while the total problem size remains constant. Governed by Amdahl's Law.
- Weak scaling: measures how the solution time behaves as the Processors and
  the problem size both increase, keeping workload per process constant. Governed by
  Gustafson's Law.
- Baseline: the reference process count used for computing speedup/efficiency, ideally N=1.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import utility
import logging

from constants import si_prefixes, FIELD_META

scale = {"t": "milli"}

FACTOR=1.5
utility.setup_dynamic_plotStyle(factor=FACTOR)  # global plotting style (colors, fonts, etc.)
utility.setup_global_latex()


# ---------------------------------------------------------------
# Basic time plot function (raw scaling)
# ---------------------------------------------------------------
def create_time_plot(df, value, mode, output_dir, scaling=None, dim=None, dpi=300, extension=None):
    """
    Create a raw time plot for a given metric.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing HPC timing data.
    value : str
        Column prefix for the metric (e.g., 'runtime').
    mode : str
        Metric type ('total' or 'real').
    output_dir : str
        Directory to save the output plot.
    """
    column = f"{value}_{mode}"
    if column not in df.columns:
        logging.warning(f"Column {column} not found. Skipping raw scaling plot.")
        return

    df = df.sort_values("N_process")
    N = df["N_process"]
    t = df[column]

    plt.figure(figsize=plt.rcParams["figure.figsize"], dpi=dpi)
    # plt.plot(N, t, marker="o")
    plt.scatter(N, t, marker="x", s=50)
    plt.xlabel(utility.format_math_text(r"$N_{op}\,/\,N_{GPU}$ [$-$]"), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    plt.ylabel(utility.format_math_text(rf"$\langle t_{{\mathrm{{{mode}}}}} \rangle$ [${si_prefixes[scale['t']]['abbr']}s$]"), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    plt.title(utility.format_math_text(rf"{scaling.lower().capitalize()} scaling ${dim}D$: Averaged $t_{{rhs()}}$ vs. $N_{{op}}$"), fontsize=plt.rcParams["figure.titlesize"])
    plt.xticks(N)
    plt.tight_layout()
    for e in extension:
        plt.savefig(os.path.join(output_dir, f"{scaling}_{dim}D_{value}_{mode}_scaling.{e}"), dpi=dpi, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------
# Strong scaling analysis
# ---------------------------------------------------------------
def strong_scaling(df, value, output_dir, scaling=None, dim=None, dpi=300, extension=None):
    """
    Compute and plot strong scaling metrics: speedup and efficiency.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing timing data.
    value : str
        Column prefix for the metric (e.g., 'runtime').
    output_dir : str
        Directory to save output plots.
    """
    FACTOR=2
    utility.setup_dynamic_plotStyle(factor=FACTOR)


    df = df.sort_values("N_process")
    N = df["N_process"]
    t = df[f"{value}_real"]

    logging.info(N)

    # -----------------------------------------------------------
    # Check if baseline N=1 exists. If not, warn the user.
    # -----------------------------------------------------------
    if 1 not in N.values:
        logging.warning(
            "Strong scaling baseline N=1 not found. "
            "Speedup and efficiency will be computed relative to the smallest available process count."
            "Results are not strictly guideline-compliant."
        )

    # -----------------------------------------------------------
    # Set baseline for speedup/efficiency computations
    # -----------------------------------------------------------
    N_base = N.iloc[0]  # smallest available process count
    t_base = t.iloc[0]

    logging.info(f"Strong scaling baseline set to N={N_base}")

    # -----------------------------------------------------------
    # Compute speedup and efficiency
    #   speedup = t_base / t(N)
    #   efficiency = speedup / (N / N_base) = (t_base * N_base) / (t * N)
    # -----------------------------------------------------------
    speedup = t_base / t
    efficiency = (t_base * N_base) / (t * N)

    # -----------------------------------------------------------
    # Ideal strong scaling curve
    # Ideal speedup relative to baseline:
    #   N_base = 1 -> ideal = N
    #   N_base = 2 -> ideal = N / 2
    #   N_base = 4 -> ideal = N / 4
    # General formula:
    #   ideal_speedup = N / N_base
    # -----------------------------------------------------------
    ideal_speedup = N / N_base

    # --- Plot speedup ---
    plt.figure(figsize=plt.rcParams["figure.figsize"], dpi=dpi)
    # plt.plot(N, speedup, marker="o", label=utility.format_math_text(r"$a_{real}$"))
    plt.scatter(N, speedup, marker="x", s=int(50*FACTOR), label=utility.format_math_text(r"$a_{real}$"))
    N_extended = np.insert(np.sort(N.values), 0, 0)  # füge N=0 hinzu
    ideal_speedup_extended = np.insert((N / N_base).values, 0, 0)  # Start bei 0
    plt.plot(N_extended, ideal_speedup_extended, linestyle="--", color="orange", label=utility.format_math_text(rf"$a_{{ideal}}=\frac{{N_{{op}}}}{{{N_base}}}$"))
    plt.xlabel(utility.format_math_text(r"$N_{op} \,/\, N_{GPU}$ [$-$]"), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    plt.ylabel(utility.format_math_text(rf"$a_{scaling.lower()[0]}$ [$-$]"), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    plt.title(utility.format_math_text(rf"{scaling.lower().capitalize()} scaling ${dim}D$: Parallel Speedup vs. $N_{{op}}$"), fontsize=plt.rcParams["figure.titlesize"])
    plt.xticks(N_extended)  # nur gewünschte x-Ticks
    plt.ylim(0, max(speedup.max(), ideal_speedup.max()) * 1.1)  # y startet bei 0
    plt.legend(fontsize=int(plt.rcParams["legend.fontsize"]*FACTOR))
    plt.tight_layout()
    for e in extension:
        plt.savefig(os.path.join(output_dir, f"{scaling}_{dim}D_speedup.{e}"), dpi=dpi, bbox_inches='tight')
    plt.close()

    # --- Plot efficiency ---
    plt.figure(figsize=plt.rcParams["figure.figsize"], dpi=dpi)
    # plt.plot(N, efficiency, marker="o", label=utility.format_math_text(r"$\eta_{real}$"))
    plt.scatter(N, efficiency, marker="x", s=int(50*FACTOR), label=utility.format_math_text(r"$\eta_{real}$"))
    plt.axhline(1.0, linestyle="--", color="orange", label=utility.format_math_text(r"$\eta_{ideal}=1$"))
    plt.xlabel(utility.format_math_text(r"$N_{op} \,/\, N_{GPUs}$ [$-$]"), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    plt.ylabel(utility.format_math_text(rf"$\eta_{scaling.lower()[0]}$ [$-$]"), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    plt.title(utility.format_math_text(rf"{scaling.lower().capitalize()} scaling ${dim}D$: Parallel Efficiency vs. $N_{{op}}$"), fontsize=plt.rcParams["figure.titlesize"])
    plt.xticks(np.sort(N.values))  # nur gewünschte x-Ticks
    plt.ylim(0, max(efficiency.max(), 1.0) * 1.1)  # y startet bei 0
    plt.legend(fontsize=int(plt.rcParams["legend.fontsize"]*FACTOR))
    plt.tight_layout()
    for e in extension:
        plt.savefig(os.path.join(output_dir, f"{scaling}_{dim}D_efficiency.{e}"), dpi=dpi, bbox_inches='tight')
    plt.close()

    FACTOR=1.5
    utility.setup_dynamic_plotStyle(factor=FACTOR)


# ---------------------------------------------------------------
# Weak scaling analysis
# ---------------------------------------------------------------
def weak_scaling(df, value, output_dir, scaling=None, dim=None, dpi=300, extension=None):
    """
    Compute and plot weak scaling efficiency.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing timing data.
    value : str
        Column prefix for the metric (e.g., 'runtime').
    output_dir : str
        Directory to save output plots.
    """
    df = df.sort_values("N_process")
    N = df["N_process"]
    t = df[f"{value}_real"]

    logging.info(N)

    # -----------------------------------------------------------
    # Check if baseline N=1 exists. Warn if not.
    # -----------------------------------------------------------
    if 1 not in N.values:
        logging.warning(
            "Weak scaling baseline N=1 not found. "
            "Efficiency will be computed relative to the smallest available process count."
        )

    # -----------------------------------------------------------
    # Set baseline
    # In weak scaling, ideal runtime is constant (efficiency = 1)
    # -----------------------------------------------------------
    N_base = N.iloc[0]
    t_base = t.iloc[0]
    efficiency = t_base / t

    # --- Plot efficiency ---
    plt.figure(figsize=plt.rcParams["figure.figsize"], dpi=dpi)
    # plt.plot(N, efficiency, marker="o", label=utility.format_math_text(r"$\eta_{real}$"))
    plt.scatter(N, efficiency, marker="x", s=50, label=utility.format_math_text(r"$\eta_{real}$"))
    plt.axhline(1.0, linestyle="--", color="orange", label=utility.format_math_text(r"$\eta_{ideal}=1$"))
    plt.xlabel(utility.format_math_text(r"$N_{op} \,/\, N_{GPUs}$ [$-$]"), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    plt.ylabel(utility.format_math_text(rf"$\eta_{scaling.lower()[0]}$ [$-$]"), fontsize=plt.rcParams["axes.labelsize"], labelpad=15)
    plt.title(utility.format_math_text(f"Weak Scaling ${dim}D$: Parallel Efficiency vs. $N_{{op}}$"), fontsize=plt.rcParams["figure.titlesize"])
    plt.xticks(np.sort(N.values))  # nur gewünschte x-Ticks
    plt.ylim(0, max(efficiency.max(), 1, 0) * 1.1)  # y startet bei 0
    plt.legend(loc='upper right', fontsize=int(plt.rcParams["legend.fontsize"]*FACTOR))
    plt.tight_layout()
    for e in extension:
        plt.savefig(os.path.join(output_dir, f"{scaling}_{dim}D_efficiency.{e}"), dpi=dpi, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HPC Scaling Analysis Tool")

    parser.add_argument("-i", "--input", required=True, help="Path to CSV file containing timing data")
    parser.add_argument("-V", "--value", required=True, help="Metric prefix (e.g., 'runtime')")
    parser.add_argument("-m", "--mode", default="none", choices=["total", "real", "all", "none"], help="Raw scaling plots mode")
    parser.add_argument("-s", "--scaling", default="none", choices=["strong", "weak", "none"], help="Scaling analysis type")
    parser.add_argument("--dim", "-d", type=int, choices=[2, 3], default=3, help="Problem dimensionality (default: 3).")
    parser.add_argument("-o", "--output", default=None, help="Output directory for plots")
    parser.add_argument("-v", "--verbose", type=int, choices=[1, 2, 3], default=3, help="Verbosity level (1=WARNING, 2=INFO, 3=DEBUG)")
    parser.add_argument("--dpi", type=int, default=300, help="Set DPI for all output plots (default: 300).")
    parser.add_argument("--extension", nargs="+", default=["png"], choices=["png", "pdf", "svg", "jpg"], help="Output file formats (default: png). Example: -e png pdf svg")

    args = parser.parse_args()

    args.extension=["png", "pdf"]

    # --- Setup logging ---
    utility.setup_logging()
    if args.verbose >= 3:
        log_level = logging.DEBUG
    elif args.verbose == 2:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    logging.getLogger().setLevel(log_level)

    # --- Load data ---
    df = pd.read_csv(args.input)
    output_dir = args.output if args.output else os.path.dirname(args.input)
    os.makedirs(output_dir, exist_ok=True)

    # --- Raw scaling plots ---
    if args.mode == "all":
        create_time_plot(df, args.value, "total", output_dir, args.scaling, args.dim, dpi=args.dpi, extension=args.extension)
        create_time_plot(df, args.value, "real", output_dir, args.scaling, args.dim, dpi=args.dpi, extension=args.extension)
    elif args.mode in ["total", "real"]:
        create_time_plot(df, args.value, args.mode, output_dir, args.scaling, args.dim, dpi=args.dpi, extension=args.extension)

    # --- Scaling analysis ---
    if args.scaling == "strong":
        strong_scaling(df, args.value, output_dir, args.scaling, args.dim, dpi=args.dpi, extension=args.extension)
    elif args.scaling == "weak":
        weak_scaling(df, args.value, output_dir, args.scaling, args.dim, dpi=args.dpi, extension=args.extension)


if __name__ == "__main__":
    main()
