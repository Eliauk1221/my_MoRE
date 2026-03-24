#!/usr/bin/env python3
"""
Script 5: Ablation Results Comparison

Two side-by-side sub-figures:
  A (left, wider)  — Prior component ablation:
      3 terrain types × 3 configs: Ours, w/o S_support, w/o S_margin
  B (right, narrow) — Depth feature position:
      3 terrain types × 2 configs: Depth in Query (Ours) vs Depth in Actor

Usage:
    python plot_ablation.py                   # uses built-in data dict
    python plot_ablation.py --csv_a a.csv --csv_b b.csv   # from CSV
"""

import argparse
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import (
    apply_style, COLORS, METHOD_NAMES, TERRAIN_NAMES_DISPLAY,
    save_figure, add_panel_label,
)


OUTPUT_DIR = './figures'
OUTPUT_STEM = 'ablation'

TERRAIN_ORDER = ['alternating_slopes', 'stair', 'gap']

# ============================================================================
# DATA INPUT — Sub-figure A: prior component ablation
# ============================================================================
ABLATION_A_MEAN = {
    # method_key: [alternating_slopes, stair, gap]
    'ours':                [0.0, 0.0, 0.0],
    'ablation_no_support': [0.0, 0.0, 0.0],
    'ablation_no_margin':  [0.0, 0.0, 0.0],
}

ABLATION_A_STD = {
    # Optional — uncomment and fill if multi-seed
    # 'ours':                [1.0, 1.0, 1.0],
}

# ============================================================================
# DATA INPUT — Sub-figure B: depth feature position
# ============================================================================
ABLATION_B_MEAN = {
    'ours':                 [0.0, 0.0, 0.0],
    'ablation_depth_actor': [0.0, 0.0, 0.0],
}

ABLATION_B_STD = {}


# ============================================================================
# Bar-chart builder (reusable for both panels)
# ============================================================================

def _draw_grouped_bars(
    ax,
    results_mean: dict,
    results_std: dict,
    terrain_order: list,
):
    method_keys = [k for k in results_mean if k in COLORS]
    n_methods = len(method_keys)
    n_terrains = len(terrain_order)

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
                    ha='center', va='bottom', fontsize=7,
                )

    terrain_labels = [TERRAIN_NAMES_DISPLAY.get(t, t) for t in terrain_order]
    ax.set_xticks(x)
    ax.set_xticklabels(terrain_labels, fontsize=9)
    ax.set_ylim(bottom=0)


# ============================================================================
# Main figure
# ============================================================================

def plot_ablation(
    abl_a_mean, abl_a_std,
    abl_b_mean, abl_b_std,
    terrain_order=None,
):
    if terrain_order is None:
        terrain_order = TERRAIN_ORDER

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5.5, 2.8),
        gridspec_kw={'width_ratios': [3, 2]},
    )

    # --- Panel A ---
    _draw_grouped_bars(ax1, abl_a_mean, abl_a_std or {}, terrain_order)
    ax1.set_ylabel('Success Rate (%)')
    ax1.legend(loc='upper right', fontsize=7)
    add_panel_label(ax1, '(a)')

    # --- Panel B ---
    _draw_grouped_bars(ax2, abl_b_mean, abl_b_std or {}, terrain_order)
    ax2.legend(loc='upper right', fontsize=7)
    add_panel_label(ax2, '(b)')

    fig.tight_layout()
    return fig


# ============================================================================
# CSV loader (reused from Script 4 pattern)
# ============================================================================

def _load_csv(path: str):
    means = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        t_keys = [c for c in reader.fieldnames if c != 'method']
        for row in reader:
            means[row['method']] = [float(row[t]) for t in t_keys]
    return means, t_keys


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Ablation results figure.')
    p.add_argument('--csv_a', type=str, default=None, help='CSV for sub-fig A')
    p.add_argument('--csv_b', type=str, default=None, help='CSV for sub-fig B')
    p.add_argument('--output', type=str, default=None)
    return p.parse_args()


def main():
    apply_style()
    args = parse_args()

    if args.csv_a:
        abl_a_mean, _ = _load_csv(args.csv_a)
        abl_a_std = None
    else:
        abl_a_mean = ABLATION_A_MEAN
        abl_a_std = ABLATION_A_STD if ABLATION_A_STD else None

    if args.csv_b:
        abl_b_mean, _ = _load_csv(args.csv_b)
        abl_b_std = None
    else:
        abl_b_mean = ABLATION_B_MEAN
        abl_b_std = ABLATION_B_STD if ABLATION_B_STD else None

    fig = plot_ablation(abl_a_mean, abl_a_std, abl_b_mean, abl_b_std)
    save_figure(fig, args.output or OUTPUT_STEM, OUTPUT_DIR)
    plt.close(fig)


if __name__ == '__main__':
    main()
