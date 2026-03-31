#!/usr/bin/env python3
"""
Plot all columns from a Δ=N CSV table vs delta_particles
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import ScalarFormatter
import logging
import utility
from datetime import datetime

from constants import si_prefixes, FIELD_META

utility.setup_dynamic_plotStyle(factor=1.5)  # global plotting style (colors, fonts, etc.)
utility.setup_global_latex()


def plot_sml_combined(df, output_dir, prefix, dpi=300, extension=None):
    required = [
        "SML_estimate",
        "SML_optimize",
        "delta_gap",
        "avg_neighbors_optimized"
    ]

    if not all(c in df.columns for c in required):
        logging.warning("Skipping SML combined plot (missing columns)")
        return

    x = df["delta_particles"]
    avg_neighbors = df["avg_neighbors_optimized"].mean()

    # mittlere Steigungen berechnen
    slope_est = np.mean(np.gradient(df["SML_estimate"], x))
    slope_opt = np.mean(np.gradient(df["SML_optimize"], x))
    slope_gap = np.mean(np.gradient(df["delta_gap"], x))

    plt.figure(figsize=(7, 4.5), dpi=dpi)

    plt.plot(x, df["SML_estimate"],
             marker='o', markersize=4,
             linewidth=1.8,
             label=utility.format_math_text(rf"$h_{{\mathrm{{est}}}}$ (avg. slope=$\num{{{slope_est:.3f}}}$)"))

    plt.plot(x, df["SML_optimize"],
             marker='s', markersize=4,
             linewidth=1.8,
             label=utility.format_math_text(rf"$h_{{\mathrm{{opt}}}}$ (avg. slope=$\num{{{slope_opt:.3f}}}$)"))

    plt.plot(x, df["delta_gap"],
             marker='^', markersize=4,
             linewidth=1.8,
             label=utility.format_math_text(rf"$\Delta_{{\mathrm{{gap}}}}$ (avg. slope=$\num{{{slope_gap:.3f}}}$)"))

    plt.xlabel(utility.format_math_text(rf"$\Delta_{{\mathrm{{par}}}}$ [${FIELD_META['x']['unit'].strip('$')}$]"), fontsize=plt.rcParams["axes.labelsize"])
    plt.ylabel(utility.format_math_text(rf"$\ell$ [${FIELD_META['x']['unit'].strip('$')}$]"), fontsize=plt.rcParams["axes.labelsize"])

    plt.title(utility.format_math_text(rf"Smoothing length convergence ($\langle N_{{nn}}\rangle = \num{{{avg_neighbors:.1f}}}$)"), fontsize=plt.rcParams["figure.titlesize"])

    plt.grid(True, which="both", linestyle="--")
    plt.legend(frameon=False, fontsize=plt.rcParams["legend.fontsize"])

    plt.gca().invert_xaxis()
    plt.minorticks_on()

    ax = plt.gca()

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-5, 5))

    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    ax.ticklabel_format(style='sci', axis='both', scilimits=(-3, 3))
    plt.tight_layout()

    plot_file = os.path.join(output_dir, f"{prefix}_SML_combined")
    for e in extension:
        plt.savefig(f"{plot_file}.{e}", dpi=dpi, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved combined SML plot: {plot_file}")


def plot_N_combined(df, output_dir, prefix, dpi=300, extension=None):
    required = ["N_tot", "N_target", "N_projectile"]

    if not all(c in df.columns for c in required):
        logging.warning("Skipping particle-number plot (missing columns)")
        return

    x = df["delta_particles"]

    # mittlere Verhältnisse berechnen
    ratio_target = (df["N_target"] / df["N_tot"]).mean()
    ratio_proj = (df["N_projectile"] / df["N_tot"]).mean()

    plt.figure(figsize=(7, 4.5), dpi=dpi)

    plt.plot(x, df["N_tot"],
             marker='o', markersize=4,
             linewidth=1.8,
             label=utility.format_math_text(rf"$N_{{\mathrm{{par, tot}}}}$"))

    plt.plot(x, df["N_target"],
             marker='s', markersize=4,
             linewidth=1.8,
             label=utility.format_math_text(rf"$N_{{\mathrm{{targ}}}}$ (avg. ratio=$\num{{{ratio_target:.2e}}}$)"))

    plt.plot(x, df["N_projectile"],
             marker='^', markersize=4,
             linewidth=1.8,
             label=utility.format_math_text(rf"$N_{{\mathrm{{proj}}}}$ (avg. ratio=$\num{{{ratio_proj:.2e}}}$)"))

    plt.xlabel(utility.format_math_text(rf"$\Delta_{{\mathrm{{par}}}}$ [${FIELD_META['x']['unit'].strip('$')}$]"))
    plt.ylabel(utility.format_math_text(rf"$N_{{\mathrm{{par}}}}$"))

    plt.title(utility.format_math_text("Particle number scaling"), fontsize=plt.rcParams["figure.titlesize"])

    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend(frameon=False, fontsize=plt.rcParams["legend.fontsize"])
    plt.minorticks_on()

    plt.tight_layout()

    plot_file = os.path.join(output_dir, f"{prefix}_N_combined")
    for e in extension:
        plt.savefig(f"{plot_file}.{e}", dpi=dpi, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved particle-number plot: {plot_file}")


def plot_single_column(df, column, output_dir, prefix, dpi=300, extension=None):
    x = df["delta_particles"]
    y = df[column]

    plt.figure(figsize=(7, 4.5), dpi=dpi)

    plt.plot(
        x, y,
        marker='o',
        markersize=4,
        linewidth=1.8
    )

    # log scale bei großen Werten
    if np.max(y) > 1e4:
        plt.yscale("log")

    # Achsenbeschriftung
    plt.xlabel(utility.format_math_text(rf"$\Delta_{{\mathrm{{par}}}}$ [${FIELD_META['x']['unit'].strip('$')}$]"))
    plt.ylabel(utility.format_math_text(column.replace("_", " ")))

    plt.title(utility.format_math_text(rf"{column.replace('_', ' ')} vs $\Delta_{{\mathrm{{par}}}}$"), fontsize=plt.rcParams["figure.titlesize"])

    plt.gca().invert_xaxis()

    plt.tight_layout()

    safe_col = column.replace(" ", "_")
    plot_file = os.path.join(
        output_dir,
        f"{prefix}_{safe_col}_vs_delta"
    )
    for e in extension:
        plt.savefig(f"{plot_file}.{e}", dpi=dpi, bbox_inches='tight')

    plt.close()

    logging.info(f"Saved plot as {plot_file}")


def main(args):
    # --- Read CSV, skip comment lines ---
    df = pd.read_csv(args.csv_file, comment="#")
    if "delta_particles" not in df.columns:
        raise ValueError("CSV must contain a 'delta_particles' column")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Prepare name prefix based on date, mode, delta_mode ---
    date_tag = datetime.now().strftime("%Y%m%d")
    prefix = f"{date_tag}_{args.dim}D_{args.delta_mode}_{args.delta_mode}"

    # --- Plot each column ---
    for col in df.columns:
        if col == "delta_particles":
            continue
        plot_single_column(df, col, args.output_dir, prefix, dpi=args.dpi, extension=args.extension)

    # --- Combined plots ---
    plot_sml_combined(df, args.output_dir, prefix, dpi=args.dpi, extension=args.extension)
    plot_N_combined(df, args.output_dir, prefix, dpi=args.dpi, extension=args.extension)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot all CSV columns vs delta_particles")
    parser.add_argument("csv_file", help="Path to CSV file")
    parser.add_argument("--output_dir", "-o", default="./", help="Directory to save plots")
    parser.add_argument("--test_mode", required=True, help="Mode / testcase name")
    parser.add_argument("--delta_mode", default="linear", help="Delta mode used (linear, log, relative, mantissa)")
    parser.add_argument("--dim", "-d", type=int, choices=[2, 3], default=3, help="Problem dimensionality (default: 3).")
    parser.add_argument("--dpi", type=int, default=300, help="Set DPI for all output plots (default: 300).")
    parser.add_argument("--extension", nargs="+", default=["png"], choices=["png", "pdf", "svg", "jpg"], help="Output file formats (default: png). Example: -e png pdf svg")

    args = parser.parse_args()
    args.extension=["png", "pdf"]

    utility.setup_logging(time=False, level=logging.INFO)
    main(args)
