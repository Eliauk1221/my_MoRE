#!/usr/bin/env python3
"""
Script 3: Prior vs. Attention Distribution Comparison

Side-by-side comparison of the TerrainSafetyScorer prior distribution and the
learned attention distribution, across terrain types.

Layout (2 × 4):
  Row 1 (Stairs):  [Height Map] [Prior] [Attention (Ours)] [Attention (w/o KL)]
  Row 2 (Gaps):    [Height Map] [Prior] [Attention (Ours)] [Attention (w/o KL)]

Under each attention panel the cosine similarity with the prior is shown.

Data source: pre-saved .npz files per terrain type.  Each file should contain:
    height_map  : (17, 11) float
    prior       : (17, 11) or (187,) float
    attn_ours   : (17, 11) or (187,) float   — M1 (Ours)
    attn_no_kl  : (17, 11) or (187,) float   — D1 (w/o KL)

Usage:
    python plot_prior_vs_attention.py \
        --data stair=data/stairs.npz gap=data/gaps.npz
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import (
    apply_style, TERRAIN_NAMES_DISPLAY,
    GRID_H, GRID_W,
    save_figure, add_panel_label, get_extent,
)


OUTPUT_DIR = './figures'
OUTPUT_STEM = 'prior_vs_attention'

# ============================================================================
# DATA INPUT
# ============================================================================
DATA_FILES = {
    # 'stair': '/path/to/stairs.npz',
    # 'gap':   '/path/to/gaps.npz',
}


# ============================================================================
# Helpers
# ============================================================================

def _to_2d(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(GRID_H, GRID_W) if arr.ndim == 1 else arr


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel()
    b_flat = b.ravel()
    dot = np.dot(a_flat, b_flat)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8
    return float(dot / denom)


def _add_robot_marker(ax):
    ax.plot(0, 0, 'k^', markersize=5, zorder=5)


# ============================================================================
# Main plot
# ============================================================================

def plot_prior_vs_attention(data_by_terrain: dict):
    """
    data_by_terrain: {terrain_key: dict with height_map, prior, attn_ours, attn_no_kl}
    """
    terrain_keys = list(data_by_terrain.keys())
    n_rows = len(terrain_keys)
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5, 1.75 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    extent = get_extent()
    imshow_kw = dict(extent=extent, aspect='auto', origin='lower', interpolation='bilinear')

    col_titles = ['Height Map', 'Prior Distribution', 'Attention (Ours)', 'Attention (w/o KL)']
    panel_idx = 0

    last_attn_im = None
    last_prior_im = None

    for row, t_key in enumerate(terrain_keys):
        d = data_by_terrain[t_key]
        h2d       = _to_2d(d['height_map'])
        prior     = _to_2d(d['prior'])
        attn_ours = _to_2d(d['attn_ours'])
        attn_nkl  = _to_2d(d['attn_no_kl']) if 'attn_no_kl' in d else None

        panels = [
            (h2d, 'Greys_r', 1.0, None),
            (prior, 'Blues', 1.0, None),
            (attn_ours, 'Reds', 1.0, _cosine_sim(prior, attn_ours)),
        ]
        if attn_nkl is not None:
            panels.append((attn_nkl, 'Reds', 1.0, _cosine_sim(prior, attn_nkl)))
        else:
            panels.append((np.zeros_like(h2d), 'Reds', 0.3, None))

        for col, (data_2d, cmap, alpha, cos_sim) in enumerate(panels):
            ax = axes[row, col]
            lbl = f'({chr(ord("a") + panel_idx)})'
            panel_idx += 1

            if col == 0:
                im = ax.imshow(data_2d, **imshow_kw, cmap=cmap)
            else:
                ax.imshow(h2d, **imshow_kw, cmap='Greys_r')
                im = ax.imshow(data_2d, **imshow_kw, cmap=cmap, alpha=0.65)

            if col == 1:
                last_prior_im = im
            if col >= 2:
                last_attn_im = im

            _add_robot_marker(ax)
            add_panel_label(ax, lbl)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)

            ax.set_xlabel('Lateral (m)' if row == n_rows - 1 else '')
            if col == 0:
                display = TERRAIN_NAMES_DISPLAY.get(t_key, t_key)
                ax.set_ylabel(f'{display}\nForward (m)')
            else:
                ax.set_ylabel('')

            if cos_sim is not None:
                ax.text(
                    0.5, -0.02, f'cos_sim = {cos_sim:.2f}',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=8, style='italic',
                )

    fig.tight_layout(rect=[0, 0.02, 0.93, 1])

    cax1 = fig.add_axes([0.935, 0.55, 0.012, 0.35])
    if last_prior_im is not None:
        fig.colorbar(last_prior_im, cax=cax1, label='Prior')

    cax2 = fig.add_axes([0.935, 0.10, 0.012, 0.35])
    if last_attn_im is not None:
        fig.colorbar(last_attn_im, cax=cax2, label='Attention')

    return fig


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Prior vs. attention distribution comparison.')
    p.add_argument('--data', nargs='+', metavar='KEY=FILE')
    p.add_argument('--output', type=str, default=None)
    return p.parse_args()


def main():
    apply_style()
    args = parse_args()

    data_files = dict(DATA_FILES)
    if args.data:
        for item in args.data:
            key, path = item.split('=', 1)
            data_files[key] = path

    if not data_files:
        print(
            "No data files specified.\n"
            "Edit DATA_FILES at the top, or use --data KEY=path.npz ...",
            file=sys.stderr,
        )
        sys.exit(1)

    data_by_terrain = {}
    for t_key, fpath in data_files.items():
        data_by_terrain[t_key] = dict(np.load(fpath))

    output_stem = args.output or OUTPUT_STEM
    fig = plot_prior_vs_attention(data_by_terrain)
    save_figure(fig, output_stem, OUTPUT_DIR)
    plt.close(fig)


if __name__ == '__main__':
    main()
