#!/usr/bin/env python3
"""
Script 4: Quantitative Results Bar Chart

Grouped bar chart showing success rate across terrain types for all comparison
methods.  Value labels are placed on top of each bar; error bars are drawn when
standard deviations are provided.

Usage:
    python plot_bar_comparison.py                     # uses built-in data dict
    python plot_bar_comparison.py --csv results.csv   # or load from CSV

CSV format (optional):
    method,alternating_slopes,stair,gap
    ours,92.3,85.1,78.5
    depth_baseline,88.0,79.2,70.1
    attn_baseline,89.5,81.3,72.4
"""

import argparse
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import (
    apply_style, COLORS, METHOD_NAMES, TERRAIN_NAMES_DISPLAY,
    save_figure,
)


OUTPUT_DIR = './figures'
OUTPUT_STEM = 'bar_comparison'

# ============================================================================
# DATA INPUT — mean ± std (std is optional, set to None if single seed)
# Edit these values with your actual experimental results.
# ============================================================================
RESULTS_MEAN = {
    # method_key: [alternating_slopes, stair, gap]
    'ours':           [0.0, 0.0, 0.0],
    'depth_baseline': [0.0, 0.0, 0.0],
    'attn_baseline':  [0.0, 0.0, 0.0],
}

RESULTS_STD = {
    # Set to None or omit the key if single-seed
    # 'ours':           [1.2, 1.5, 2.0],
}

TERRAIN_ORDER = ['alternating_slopes', 'stair', 'gap']

# ============================================================================


def plot_bar_comparison(
    results_mean: dict,
    results_std: dict = None,
    terrain_order: list = None,
):
    if terrain_order is None:
        terrain_order = TERRAIN_ORDER
    if results_std is None:
        results_std = {}

    method_keys = [k for k in results_mean if k in COLORS]
    n_methods = len(method_keys)
    n_terrains = len(terrain_order)

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    bar_width = 0.7 / max(n_methods, 1)
    x = np.arange(n_terrains)

    for i, mk in enumerate(method_keys):
        vals = np.array(results_mean[mk])
        offset = (i - n_methods / 2 + 0.5) * bar_width
        yerr = np.array(results_std[mk]) if mk in results_std else None

        bars = ax.bar(
            x + offset, vals, bar_width,
            label=METHOD_NAMES.get(mk, mk),
            color=COLORS[mk],
            edgecolor='white', linewidth=0.5,
            yerr=yerr,
            capsize=3 if yerr is not None else 0,
            error_kw={'linewidth': 0.8},
        )

        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 0:
                y_top = bar.get_height()
                if yerr is not None:
                    y_top += yerr[j]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_top + 0.8,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontsize=8,
                )

    terrain_labels = [TERRAIN_NAMES_DISPLAY.get(t, t) for t in terrain_order]
    ax.set_xticks(x)
    ax.set_xticklabels(terrain_labels)
    ax.set_ylabel('Success Rate (%)')
    ax.set_ylim(bottom=0)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    return fig


# ============================================================================
# CSV loader
# ============================================================================

def load_csv(csv_path: str):
    means = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        terrain_keys = [c for c in reader.fieldnames if c != 'method']
        for row in reader:
            mk = row['method']
            means[mk] = [float(row[t]) for t in terrain_keys]
    return means, terrain_keys


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Grouped bar chart of success rates.')
    p.add_argument('--csv', type=str, default=None, help='Path to CSV with results')
    p.add_argument('--output', type=str, default=None)
    return p.parse_args()


def main():
    apply_style()
    args = parse_args()

    if args.csv:
        results_mean, terrain_order = load_csv(args.csv)
        results_std = None
    else:
        results_mean = RESULTS_MEAN
        results_std = RESULTS_STD if RESULTS_STD else None
        terrain_order = TERRAIN_ORDER

    fig = plot_bar_comparison(results_mean, results_std, terrain_order)
    save_figure(fig, args.output or OUTPUT_STEM, OUTPUT_DIR)
    plt.close(fig)


if __name__ == '__main__':
    main()
