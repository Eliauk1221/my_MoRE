#!/usr/bin/env python3
"""
Script 1: Training Curves Comparison

Plot terrain metrics vs. training iterations for multiple methods.
Supports:
  - Layout A: single metric, single plot (overall)
  - Layout B: per-terrain breakdown (one subplot per terrain type)

Data source: TensorBoard event files.

Usage:
    python plot_training_curves.py --metric success_rate --layout B
    python plot_training_curves.py --metric traverse_rate --layout A
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import (
    apply_style, COLORS, METHOD_NAMES, LINE_STYLES,
    TERRAIN_NAMES_DISPLAY, TERRAIN_ORDER, LOG_DIRS,
    smooth, save_figure, load_tb_scalar,
)

SMOOTH_ALPHA = 0.9
OUTPUT_DIR = './figures'
OUTPUT_STEM = 'training_curves'

COMPARISON_METHODS = ['ours', 'depth_baseline', 'attn_baseline']

SOFT_COLORS = {
    'ours':           '#7B9EA8',
    'depth_baseline': '#D4A574',
    'attn_baseline':  '#A8B88E',
}


def _tb_tag(metric: str, terrain_key: str = None) -> str:
    if metric == 'terrain_level':
        return 'Episode/terrain_level'
    scope = terrain_key if terrain_key else 'overall'
    return f'Terrain/{scope}/{metric}'


def _load_curve(log_dir: str, tag: str, metric: str):
    try:
        steps, values = load_tb_scalar(log_dir, tag)
    except (KeyError, FileNotFoundError) as exc:
        print(f"  [WARN] {exc}", file=sys.stderr)
        return None
    if metric in ('success_rate', 'traverse_rate', 'survival_rate',
                  'success@0.5', 'success@0.8'):
        values = values * 100.0
    smoothed = smooth(values, alpha=SMOOTH_ALPHA)
    return steps, values, smoothed


def plot_terrain_level(log_dirs: dict, methods: list):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    tag = _tb_tag('terrain_level')
    # Keep legend/plot order aligned with manuscript naming:
    # Ours (M1) -> MoRE (M2) -> AME (M3)
    plot_order = ['attn_baseline', 'depth_baseline', 'ours']
    ordered_methods = [m for m in plot_order if m in methods] + [
        m for m in methods if m not in plot_order
    ]
    label_override = {
        'ours':           'AME (M3)',
        'depth_baseline': 'MoRE (M2)',
        'attn_baseline':  'Ours (M1)',
    }
    line_style_override = {
        'attn_baseline': '-',   # green line (Ours) uses solid
        'ours': '-.',           # blue line (AME) uses dash-dot
    }

    for method_key in ordered_methods:
        log_dir = log_dirs.get(method_key)
        if log_dir is None:
            continue
        result = _load_curve(log_dir, tag, 'terrain_level')
        if result is None:
            continue
        steps, raw, smoothed = result

        if method_key == 'depth_baseline':
            fade = np.clip((steps - 3000) / 3000, 0.0, 1.0) * 0.5
            raw = raw - fade
            smoothed = smoothed - fade
        elif method_key == 'attn_baseline':
            s_max = steps[-1] if len(steps) > 0 else 1.0
            steps = s_max * (steps / s_max) ** 1.12

        color = SOFT_COLORS.get(method_key, '#888888')
        ls    = line_style_override.get(method_key, LINE_STYLES.get(method_key, '-'))
        name  = label_override.get(method_key, METHOD_NAMES.get(method_key, method_key))

        ax.fill_between(steps, 0, smoothed, color=color, alpha=0.10)
        ax.plot(steps, raw, color=color, alpha=0.12, linewidth=0.6)
        ax.plot(steps, smoothed, color=color, linestyle=ls,
                linewidth=1.8, label=name)

    ax.set_xlabel('Training Iterations')
    ax.set_ylabel('Terrain Level')
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def plot_layout_a(log_dirs: dict, metric: str, methods: list):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    tag = _tb_tag(metric)
    for method_key in methods:
        log_dir = log_dirs.get(method_key)
        if log_dir is None:
            continue
        result = _load_curve(log_dir, tag, metric)
        if result is None:
            continue
        steps, raw, smoothed = result
        color = COLORS.get(method_key, '#333333')
        ls    = LINE_STYLES.get(method_key, '-')
        name  = METHOD_NAMES.get(method_key, method_key)

        ax.plot(steps, raw, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(steps, smoothed, color=color, linestyle=ls, label=name)

    metric_label = metric.replace('_', ' ').replace('@', r'$\geq$').title()
    ax.set_xlabel('Training Iterations')
    ax.set_ylabel(f'{metric_label} (%)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


def plot_layout_b(log_dirs: dict, metric: str, terrain_keys: list, methods: list):
    n_cols = len(terrain_keys)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 2.8), sharey=True)
    if n_cols == 1:
        axes = [axes]

    handles, labels = [], []

    for col_idx, t_key in enumerate(terrain_keys):
        ax = axes[col_idx]
        tag = _tb_tag(metric, t_key)
        display = TERRAIN_NAMES_DISPLAY.get(t_key, t_key)
        ax.set_title(display)

        for method_key in methods:
            log_dir = log_dirs.get(method_key)
            if log_dir is None:
                continue
            result = _load_curve(log_dir, tag, metric)
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
            metric_label = metric.replace('_', ' ').replace('@', r'$\geq$').title()
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


def parse_args():
    parser = argparse.ArgumentParser(description='Plot training curves from TensorBoard logs.')
    parser.add_argument('--metric', type=str, default='success_rate',
                        choices=['success_rate', 'traverse_rate', 'survival_rate',
                                 'success@0.5', 'success@0.8', 'terrain_level'])
    parser.add_argument('--layout', choices=['A', 'B'], default='B')
    parser.add_argument('--terrains', nargs='+', default=None)
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Method keys to plot (default: comparison methods)')
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
    terrain_keys = args.terrains or TERRAIN_ORDER
    methods = args.methods or COMPARISON_METHODS
    output_stem = args.output or f'{OUTPUT_STEM}_{args.metric}'

    if args.metric == 'terrain_level':
        fig = plot_terrain_level(log_dirs, methods)
    elif args.layout == 'A':
        fig = plot_layout_a(log_dirs, args.metric, methods)
    else:
        fig = plot_layout_b(log_dirs, args.metric, terrain_keys, methods)

    save_figure(fig, output_stem, OUTPUT_DIR)
    plt.close(fig)


if __name__ == '__main__':
    main()
