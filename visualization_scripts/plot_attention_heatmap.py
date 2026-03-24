#!/usr/bin/env python3
"""
Script 2: Attention Heatmap Visualization

Visualize 17×11 attention weights overlaid on the terrain height map.
Supports two data sources:
  1. Pre-saved .npz files  (offline, no GPU / IsaacGym required)
  2. Live collection from a checkpoint  (requires IsaacGym environment)

Layout options:
  A — 1×4 side-by-side: [Height Map] [Method A] [Method B] [Prior]
  B — 2×3 for 2 terrain types × 3 views

Usage:
    # From pre-saved data
    python plot_attention_heatmap.py --data stairs=data/stairs.npz gaps=data/gaps.npz

    # Pre-saved .npz should contain:
    #   height_map : (17, 11) float
    #   attn_ours  : (17, 11) or (187,) float — attention weights (method A)
    #   attn_base  : (17, 11) or (187,) float — attention weights (method B)  [optional]
    #   prior      : (17, 11) or (187,) float — prior distribution           [optional]
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from plot_utils import (
    apply_style, COLORS, TERRAIN_NAMES_DISPLAY,
    MEASURED_POINTS_X, MEASURED_POINTS_Y, GRID_H, GRID_W,
    save_figure, add_panel_label, get_extent, ensure_dir,
)


OUTPUT_DIR = './figures'
OUTPUT_STEM = 'attention_heatmap'

# ============================================================================
# DATA INPUT — edit these directly or use CLI args
# ============================================================================
DATA_FILES = {
    # 'stairs': '/path/to/stairs.npz',
    # 'gaps':   '/path/to/gaps.npz',
}


# ============================================================================
# Helpers
# ============================================================================

def _to_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(GRID_H, GRID_W)
    return arr


def _add_robot_marker(ax):
    ax.plot(0, 0, 'k^', markersize=6, zorder=5)


def _plot_height(ax, h2d, extent, label='(a)'):
    """Gray-scale terrain height map (background layer)."""
    im = ax.imshow(
        h2d, extent=extent, aspect='auto', origin='lower',
        cmap='Greys_r', interpolation='bilinear',
    )
    _add_robot_marker(ax)
    add_panel_label(ax, label)
    return im


def _plot_overlay(ax, h2d, w2d, extent, cmap='Reds', alpha=0.6, label='(b)'):
    """Height map background + semi-transparent attention/prior overlay."""
    ax.imshow(
        h2d, extent=extent, aspect='auto', origin='lower',
        cmap='Greys_r', interpolation='bilinear',
    )
    im = ax.imshow(
        w2d, extent=extent, aspect='auto', origin='lower',
        cmap=cmap, alpha=alpha, interpolation='bilinear',
    )
    _add_robot_marker(ax)
    add_panel_label(ax, label)
    return im


def _set_axes_labels(ax, show_ylabel=True):
    ax.set_xlabel('Lateral (m)')
    if show_ylabel:
        ax.set_ylabel('Forward (m)')


# ============================================================================
# Layout A — 1 × 4 side-by-side for a single terrain snapshot
# ============================================================================

def plot_layout_a(data: dict, terrain_key: str = None):
    """
    data keys: height_map, attn_ours, [attn_base], [prior]
    All shapes (17, 11) or (187,).
    """
    h2d = _to_2d(data['height_map'])
    attn_ours = _to_2d(data['attn_ours'])
    attn_base = _to_2d(data['attn_base']) if 'attn_base' in data else None
    prior     = _to_2d(data['prior'])      if 'prior'     in data else None

    panels = [('Height Map', None, None)]
    panels.append(('Ours', attn_ours, 'Reds'))
    if attn_base is not None:
        panels.append(('Baseline', attn_base, 'Reds'))
    if prior is not None:
        panels.append(('Prior', prior, 'Blues'))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5, 2.0))
    if n == 1:
        axes = [axes]

    extent = get_extent()
    labels_iter = iter('(a) (b) (c) (d) (e)'.split())

    for ax, (title, w2d, cmap) in zip(axes, panels):
        lbl = next(labels_iter)
        if w2d is None:
            im = _plot_height(ax, h2d, extent, label=lbl)
        else:
            im = _plot_overlay(ax, h2d, w2d, extent, cmap=cmap, label=lbl)
        ax.set_title(title, fontsize=10)
        _set_axes_labels(ax, show_ylabel=(ax is axes[0]))

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Weight')
    fig.subplots_adjust(right=0.90, wspace=0.35)
    return fig


# ============================================================================
# Layout B — 2 × 3: rows = terrain types, cols = Height / Attention / Prior
# ============================================================================

def plot_layout_b(data_by_terrain: dict):
    """
    data_by_terrain: {terrain_key: {height_map, attn_ours, prior}, …}
    """
    terrain_keys = list(data_by_terrain.keys())
    n_rows = len(terrain_keys)
    n_cols = 3  # Height, Attention, Prior
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5, 2.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    extent = get_extent()
    col_titles = ['Height Map', 'Attention (Ours)', 'Prior Distribution']
    panel_idx = 0

    for row, t_key in enumerate(terrain_keys):
        d = data_by_terrain[t_key]
        h2d = _to_2d(d['height_map'])
        attn = _to_2d(d['attn_ours'])
        prior = _to_2d(d['prior']) if 'prior' in d else np.zeros_like(h2d)

        panels_data = [
            (h2d, None, None),
            (h2d, attn, 'Reds'),
            (h2d, prior, 'Blues'),
        ]

        for col, (bg, overlay, cmap) in enumerate(panels_data):
            ax = axes[row, col]
            lbl = f'({chr(ord("a") + panel_idx)})'
            panel_idx += 1

            if overlay is None:
                _plot_height(ax, bg, extent, label=lbl)
            else:
                im = _plot_overlay(ax, bg, overlay, extent, cmap=cmap, label=lbl)

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            _set_axes_labels(ax, show_ylabel=(col == 0))

            display = TERRAIN_NAMES_DISPLAY.get(t_key, t_key)
            if col == 0:
                ax.set_ylabel(f'{display}\nForward (m)')

    fig.tight_layout(rect=[0, 0, 0.92, 1])

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Weight')
    return fig


# ============================================================================
# Data collection helper (requires IsaacGym / checkpoint)
# ============================================================================

def collect_and_save(
    checkpoint_path: str,
    output_npz: str,
    env_cfg_override: dict = None,
    num_steps: int = 50,
):
    """
    Load a checkpoint, run the environment for *num_steps*, and save
    height_map, attn_ours, prior to an .npz file.

    This function requires IsaacGym and the training codebase on PYTHONPATH.
    """
    import torch
    sys.path.insert(0, '../MoRE/legged_gym')
    sys.path.insert(0, '../MoRE/rsl_rl')

    from legged_gym.envs import task_registry

    env_cfg, train_cfg = task_registry.get_cfgs(name='g1_16dof_loco')
    if env_cfg_override:
        for k, v in env_cfg_override.items():
            setattr(env_cfg, k, v)

    env, _ = task_registry.make_env(name='g1_16dof_loco', args=None, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = checkpoint_path

    ppo_runner, _ = task_registry.make_alg_runner(
        env=env, name='g1_16dof_loco', args=None, train_cfg=train_cfg,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs = env.get_observations()
    for _ in range(num_steps):
        actions = policy(obs)
        obs, _, _, _, infos = env.step(actions)

    height_map = env.height_map[0].detach().cpu().numpy()
    attn_ours = ppo_runner.alg.actor_critic.terrain_attention.last_attention_weights[0].detach().cpu().numpy()

    save_dict = {'height_map': height_map, 'attn_ours': attn_ours}

    scorer = getattr(ppo_runner.alg.actor_critic, 'terrain_safety_scorer', None)
    if scorer is not None:
        with torch.no_grad():
            prior = scorer(env.height_map, env.base_lin_vel)
        save_dict['prior'] = prior[0].detach().cpu().numpy()

    ensure_dir(os.path.dirname(output_npz) or '.')
    np.savez(output_npz, **save_dict)
    print(f'Saved snapshot → {output_npz}')


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Attention heatmap visualization.')
    p.add_argument(
        '--data', nargs='+', metavar='KEY=FILE',
        help='terrain_key=path/to/data.npz pairs',
    )
    p.add_argument('--layout', choices=['A', 'B'], default='A')
    p.add_argument('--output', type=str, default=None)
    return p.parse_args()


def main():
    import os
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
        d = dict(np.load(fpath))
        data_by_terrain[t_key] = d

    output_stem = args.output or OUTPUT_STEM

    if args.layout == 'A':
        first_key = list(data_by_terrain.keys())[0]
        fig = plot_layout_a(data_by_terrain[first_key], terrain_key=first_key)
    else:
        fig = plot_layout_b(data_by_terrain)

    save_figure(fig, output_stem, OUTPUT_DIR)
    plt.close(fig)


if __name__ == '__main__':
    main()
