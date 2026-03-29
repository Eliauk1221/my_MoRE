#!/usr/bin/env python3
"""
Script 5: Ablation Results Comparison

Three side-by-side sub-figures:
  A — KL prior ablation: Ours vs w/o KL (D1)
  B — Prior component ablation: Ours vs w/o forward (A2) vs w/o edge (A3)
  C — Depth feature position: Depth in Query (Ours) vs Depth in Actor (B2)

Usage:
    python plot_ablation.py --csv_a ablation_a.csv --csv_b ablation_b.csv --csv_c ablation_c.csv
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import (
    apply_style, COLORS, METHOD_NAMES, TERRAIN_NAMES_DISPLAY, TERRAIN_ORDER,
    save_figure, add_panel_label,
)


OUTPUT_DIR = './figures'
OUTPUT_STEM = 'ablation'


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
    ax.set_xticklabels(terrain_labels, fontsize=8, rotation=15, ha='right')
    ax.set_ylim(bottom=0)


def plot_ablation(
    abl_a_mean, abl_a_std,
    abl_b_mean, abl_b_std,
    abl_c_mean, abl_c_std,
    terrain_order=None,
):
    if terrain_order is None:
        terrain_order = TERRAIN_ORDER

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(8.0, 2.8),
        gridspec_kw={'width_ratios': [2, 3, 2]},
    )

    _draw_grouped_bars(ax1, abl_a_mean, abl_a_std or {}, terrain_order)
    ax1.set_ylabel('Success Rate (%)')
    ax1.legend(loc='upper right', fontsize=6)
    add_panel_label(ax1, '(a)')

    _draw_grouped_bars(ax2, abl_b_mean, abl_b_std or {}, terrain_order)
    ax2.legend(loc='upper right', fontsize=6)
    add_panel_label(ax2, '(b)')

    _draw_grouped_bars(ax3, abl_c_mean, abl_c_std or {}, terrain_order)
    ax3.legend(loc='upper right', fontsize=6)
    add_panel_label(ax3, '(c)')

    fig.tight_layout()
    return fig


def _load_csv(path: str):
    means = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        t_keys = [c for c in reader.fieldnames if c != 'method']
        for row in reader:
            means[row['method']] = [float(row[t]) for t in t_keys]
    return means, t_keys


def parse_args():
    p = argparse.ArgumentParser(description='Ablation results figure.')
    p.add_argument('--csv_a', type=str, required=True, help='CSV for KL ablation')
    p.add_argument('--csv_b', type=str, required=True, help='CSV for prior component ablation')
    p.add_argument('--csv_c', type=str, required=True, help='CSV for depth position ablation')
    p.add_argument('--output', type=str, default=None)
    return p.parse_args()


def main():
    apply_style()
    args = parse_args()

    abl_a_mean, _ = _load_csv(args.csv_a)
    abl_b_mean, _ = _load_csv(args.csv_b)
    abl_c_mean, _ = _load_csv(args.csv_c)

    fig = plot_ablation(abl_a_mean, None, abl_b_mean, None, abl_c_mean, None)
    save_figure(fig, args.output or OUTPUT_STEM, OUTPUT_DIR)
    plt.close(fig)


if __name__ == '__main__':
    main()
