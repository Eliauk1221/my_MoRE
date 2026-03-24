#!/usr/bin/env python3
"""
Script 1: Training Curves Comparison

Plot success rate (and optionally traverse rate) vs. training iterations for
multiple methods on the same axes.  Supports:
  - Layout A: single metric, single plot
  - Layout B: per-terrain breakdown (one subplot per terrain type)

Data source: TensorBoard event files.

Usage:
    python plot_training_curves.py \
        --logs ours=/path/to/M1/logs \
               depth_baseline=/path/to/M2/logs \
               attn_baseline=/path/to/M3/logs \
        --metric success_rate \
        --layout A \
        --output figures/training_curves
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import (
    apply_style, COLORS, METHOD_NAMES, LINE_STYLES,
    TERRAIN_NAMES_DISPLAY, smooth, save_figure, load_tb_scalar,
)

# ============================================================================
# DATA INPUT — edit these directly or use CLI args
# ============================================================================
LOG_DIRS = {
    # 'ours':           '/path/to/M1/tensorboard_logs',
    # 'depth_baseline': '/path/to/M2/tensorboard_logs',
    # 'attn_baseline':  '/path/to/M3/tensorboard_logs',
}

METRIC = 'success_rate'          # success_rate | traverse_rate | survival_rate
TERRAIN_KEYS = ['alternating_slopes', 'stair', 'gap']
SMOOTH_ALPHA = 0.9
OUTPUT_DIR = './figures'
OUTPUT_STEM = 'training_curves'

# ============================================================================


def _tb_tag(metric: str, terrain_key: str = None) -> str:
    """Build the TensorBoard scalar tag for *metric* on *terrain_key*."""
    scope = terrain_key if terrain_key else 'overall'
    return f'Terrain/{scope}/{metric}'


def _load_curve(log_dir: str, tag: str):
    """Return (steps, raw, smoothed) or None if tag is missing."""
    try:
        steps, values = load_tb_scalar(log_dir, tag)
    except (KeyError, FileNotFoundError) as exc:
        print(f"  [WARN] {exc}", file=sys.stderr)
        return None
    if METRIC in ('success_rate', 'traverse_rate', 'survival_rate'):
        values = values * 100.0  # fraction → percentage
    smoothed = smooth(values, alpha=SMOOTH_ALPHA)
    return steps, values, smoothed


# ---------------------------------------------------------------------------
# Layout A — single metric, single plot
# ---------------------------------------------------------------------------
def plot_layout_a(log_dirs: dict, metric: str):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    tag = _tb_tag(metric)
    for method_key, log_dir in log_dirs.items():
        result = _load_curve(log_dir, tag)
        if result is None:
            continue
        steps, raw, smoothed = result
        color = COLORS.get(method_key, '#333333')
        ls    = LINE_STYLES.get(method_key, '-')
        name  = METHOD_NAMES.get(method_key, method_key)

        ax.plot(steps, raw, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(steps, smoothed, color=color, linestyle=ls, label=name)

    metric_label = metric.replace('_', ' ').title()
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel(f'{metric_label} (%)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Layout B — per-terrain breakdown (1 × N subplots)
# ---------------------------------------------------------------------------
def plot_layout_b(log_dirs: dict, metric: str, terrain_keys: list):
    n_cols = len(terrain_keys)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5, 2.5), sharey=True)
    if n_cols == 1:
        axes = [axes]

    handles, labels = [], []

    for col_idx, t_key in enumerate(terrain_keys):
        ax = axes[col_idx]
        tag = _tb_tag(metric, t_key)
        display = TERRAIN_NAMES_DISPLAY.get(t_key, t_key)
        ax.set_title(display)

        for method_key, log_dir in log_dirs.items():
            result = _load_curve(log_dir, tag)
            if result is None:
                continue
            steps, raw, smoothed = result
            color = COLORS.get(method_key, '#333333')
            ls    = LINE_STYLES.get(method_key, '-')
            name  = METHOD_NAMES.get(method_key, method_key)

            ax.plot(steps, raw, color=color, alpha=0.15, linewidth=0.8)
            line, = ax.plot(steps, smoothed, color=color, linestyle=ls, label=name)
            if col_idx == 0:
                handles.append(line)
                labels.append(name)

        ax.set_xlabel('Training Iterations')
        if col_idx == 0:
            metric_label = metric.replace('_', ' ').title()
            ax.set_ylabel(f'{metric_label} (%)')

    fig.legend(
        handles, labels,
        loc='lower center',
        ncol=min(len(handles), 4),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Plot training curves from TensorBoard logs.')
    parser.add_argument(
        '--logs', nargs='+', metavar='KEY=DIR',
        help='Method=LogDir pairs, e.g. ours=/path/to/logs',
    )
    parser.add_argument('--metric', type=str, default=None)
    parser.add_argument('--layout', choices=['A', 'B'], default='A')
    parser.add_argument('--terrains', nargs='+', default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--smooth', type=float, default=None)
    return parser.parse_args()


def main():
    apply_style()
    args = parse_args()

    global SMOOTH_ALPHA
    if args.smooth is not None:
        SMOOTH_ALPHA = args.smooth

    log_dirs = dict(LOG_DIRS)
    if args.logs:
        for item in args.logs:
            key, path = item.split('=', 1)
            log_dirs[key] = path

    if not log_dirs:
        print(
            "No log directories specified.\n"
            "Either edit LOG_DIRS at the top of this script or use --logs KEY=DIR ...",
            file=sys.stderr,
        )
        sys.exit(1)

    metric = args.metric or METRIC
    terrain_keys = args.terrains or TERRAIN_KEYS
    output_stem = args.output or OUTPUT_STEM

    if args.layout == 'A':
        fig = plot_layout_a(log_dirs, metric)
    else:
        fig = plot_layout_b(log_dirs, metric, terrain_keys)

    save_figure(fig, output_stem, OUTPUT_DIR)
    plt.close(fig)


if __name__ == '__main__':
    main()
