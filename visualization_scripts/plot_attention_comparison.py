#!/usr/bin/env python3
"""
Figure: Attention Heatmap Comparison (CVPR style)

Compare attention distributions between M1 (with KL prior) and another method
(e.g., M3 w/o KL prior guidance) on two terrain types, showing how the prior
guides attention to focus on safety-relevant regions.

Layout:  2 rows (terrains) × 4 columns
  [Height Map] [Prior Distribution] [Attention (M1, with KL)] [Attention (Reference)]

Data source options:
  1. Pre-collected .npz files (from collect_attention_data.py)
  2. Synthetic demo (generated from terrain scorer, for layout preview)

Usage:
    # With real data
    python plot_attention_comparison.py \
        --data_m1 figures/attn_data_M1.npz \
        --data_ref figures/attn_data_M3.npz \
        --label_ref "M3 (w/o KL prior)"

    # Synthetic demo (no GPU needed)
    python plot_attention_comparison.py --demo
"""

import sys
import os
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', 'MoRE')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'rsl_rl'))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plot_utils import (
    apply_style, save_figure, MEASURED_POINTS_X, MEASURED_POINTS_Y,
    GRID_H, GRID_W,
)

OUTPUT_DIR = './figures'

# ── Soft colormaps ───────────────────────────────────────────────────────

def _cmap(name, colors):
    return LinearSegmentedColormap.from_list(name, colors, N=256)

MORANDI_BLUE = '#7B9EA8'
MORANDI_ORANGE = '#D4A574'
MORANDI_GREEN = '#A8B88E'
MORANDI_BG = '#F8F7F4'

CMAP_HEIGHT = _cmap('c_ht', [MORANDI_BG, '#E4EBDD', '#CBD7BB', MORANDI_GREEN, '#8F9F78'])
CMAP_PRIOR  = _cmap('c_pr', [MORANDI_BG, '#E4ECEE', '#C1D2D6', MORANDI_BLUE, '#607B84'])
CMAP_ATTN_M1 = _cmap('c_a1', [MORANDI_BG, '#EDF2E9', '#CFDBC0', MORANDI_GREEN, '#859A6D'])
CMAP_ATTN_D1 = _cmap('c_d1', [MORANDI_BG, '#F3EADF', '#E6C7A3', MORANDI_ORANGE, '#B8875F'])


# ── Terrain helpers (same as plot_prior_decomposition.py) ────────────────

HORIZONTAL_SCALE = 0.1
VERTICAL_SCALE = 0.005
TERRAIN_LENGTH = 14.0
TERRAIN_WIDTH = 4.0
LENGTH_PER_ENV_PX = int(TERRAIN_LENGTH / HORIZONTAL_SCALE)
WIDTH_PER_ENV_PX = int(TERRAIN_WIDTH / HORIZONTAL_SCALE)
NORMALIZATION_BASE_HEIGHT = 0.8
NOMINAL_STANDING_HEIGHT = 0.75


class MockSubTerrain:
    def __init__(self):
        self.width = LENGTH_PER_ENV_PX
        self.length = WIDTH_PER_ENV_PX
        self.vertical_scale = VERTICAL_SCALE
        self.horizontal_scale = HORIZONTAL_SCALE
        self.height_field_raw = np.zeros((LENGTH_PER_ENV_PX, WIDTH_PER_ENV_PX), dtype=np.int16)


def _get_terrain_z(hf, px, py):
    ix = int(np.clip(px, 0, hf.shape[0] - 2))
    iy = int(np.clip(py, 0, hf.shape[1] - 2))
    return min(hf[ix, iy], hf[ix+1, iy], hf[ix, iy+1]) * VERTICAL_SCALE


def sample_height_map(hf, rx, ry):
    rpx = rx / HORIZONTAL_SCALE
    rpy = ry / HORIZONTAL_SCALE
    rz = _get_terrain_z(hf, rpx, rpy) + NOMINAL_STANDING_HEIGHT
    hm = torch.zeros(1, GRID_H, GRID_W)
    for i, dx in enumerate(MEASURED_POINTS_X):
        for j, dy in enumerate(MEASURED_POINTS_Y):
            px = (rx + dx) / HORIZONTAL_SCALE
            py = (ry + dy) / HORIZONTAL_SCALE
            hm[0, i, j] = rz - NORMALIZATION_BASE_HEIGHT - _get_terrain_z(hf, px, py)
    return torch.clip(hm, -1.0, 1.0)


def make_stepping_stones():
    t = MockSubTerrain()
    t.height_field_raw[:] = -round(1.0 / VERTICAL_SCALE)
    stone_size_px = round(0.39 / HORIZONTAL_SCALE)
    pitch_px = round(0.50 / HORIZONTAL_SCALE)
    lane_px = round(0.18 / HORIZONTAL_SCALE)
    plat_px = round(2.5 / HORIZONTAL_SCALE)
    mid_y = t.length // 2
    t.height_field_raw[:plat_px, :] = 0
    centers = []
    for col in range(10):
        cx = plat_px + col * pitch_px + stone_size_px // 2
        cy = mid_y + ((-1)**col) * lane_px
        xs, xe = cx - stone_size_px // 2, cx + stone_size_px // 2
        ys, ye = cy - stone_size_px // 2, cy + stone_size_px // 2
        t.height_field_raw[xs:xe, ys:ye] = 0
        centers.append((cx, cy))
    t.height_field_raw[plat_px + 10 * pitch_px:, :] = 0
    rx = centers[3][0] * HORIZONTAL_SCALE
    ry = centers[3][1] * HORIZONTAL_SCALE
    return t, rx, ry


def make_stairs():
    t = MockSubTerrain()
    step_h_px = round(0.175 / VERTICAL_SCALE)
    x_range_px = round(0.31 / HORIZONTAL_SCALE)
    plat_px = round(1.5 / HORIZONTAL_SCALE)
    mid_y = t.length // 2
    hvw_px = round(1.5 / HORIZONTAL_SCALE)
    t.height_field_raw[:plat_px, :] = 0
    dx = plat_px
    sh = 0
    edges = []
    for i in range(5):
        sh += step_h_px
        t.height_field_raw[dx:dx + x_range_px, :] = sh
        t.height_field_raw[dx:dx + x_range_px, :mid_y - hvw_px] = 0
        t.height_field_raw[dx:dx + x_range_px, mid_y + hvw_px:] = 0
        dx += x_range_px
        edges.append(dx)
    mid_plat_px = round(1.5 / HORIZONTAL_SCALE)
    t.height_field_raw[dx:dx + mid_plat_px, mid_y - hvw_px:mid_y + hvw_px] = sh
    dx += mid_plat_px
    for i in range(5):
        sh -= step_h_px
        t.height_field_raw[dx:dx + x_range_px, :] = sh
        t.height_field_raw[dx:dx + x_range_px, :mid_y - hvw_px] = 0
        t.height_field_raw[dx:dx + x_range_px, mid_y + hvw_px:] = 0
        dx += x_range_px
        edges.append(dx)
    rx = edges[2] * HORIZONTAL_SCALE
    ry = mid_y * HORIZONTAL_SCALE
    return t, rx, ry


# ── Synthetic attention demo ─────────────────────────────────────────────

def generate_demo_data():
    """Generate synthetic data for layout preview without GPU."""
    from rsl_rl.modules.terrain_safety_scorer import TerrainSafetyScorer

    np.random.seed(42)
    scorer = TerrainSafetyScorer(
        grid_h=GRID_H, grid_w=GRID_W,
        measured_points_x=MEASURED_POINTS_X,
        measured_points_y=MEASURED_POINTS_Y,
    )
    vel = torch.tensor([[0.5, 0.0, 0.0]])

    demos = []
    for name, make_fn in [('Stepping Stones', make_stepping_stones),
                           ('Stair', make_stairs)]:
        t, rx, ry = make_fn()
        hm = sample_height_map(t.height_field_raw, rx, ry)
        prior, dbg = scorer(hm, vel, return_debug_info=True)
        prior_2d = prior[0].numpy().reshape(GRID_H, GRID_W)

        attn_m1 = prior_2d + np.random.normal(0, 0.0003, prior_2d.shape)
        attn_m1 = np.maximum(attn_m1, 0)
        attn_m1 /= attn_m1.sum()

        noise = np.random.exponential(0.002, prior_2d.shape)
        attn_ref = 0.3 * prior_2d + 0.7 * noise
        attn_ref = np.maximum(attn_ref, 0)
        attn_ref /= attn_ref.sum()

        demos.append({
            'terrain_name': name,
            'height_map': hm[0].numpy(),
            'prior': prior_2d,
            'attn_m1': attn_m1,
            'attn_ref': attn_ref,
        })

    return demos


def load_real_data(path_m1, path_ref, terrain_indices=None):
    """Load real attention data from .npz files."""
    data_m1 = np.load(path_m1, allow_pickle=True)
    data_ref = np.load(path_ref, allow_pickle=True)

    terrain_names_map = {0: 'Stepping Stones', 1: 'Tilted Ramp',
                         2: 'Pits', 3: 'Gap', 4: 'Stair'}

    if terrain_indices is None:
        terrain_indices = [0, 4]

    from rsl_rl.modules.terrain_safety_scorer import TerrainSafetyScorer
    scorer = TerrainSafetyScorer(
        grid_h=GRID_H, grid_w=GRID_W,
        measured_points_x=MEASURED_POINTS_X,
        measured_points_y=MEASURED_POINTS_Y,
    )
    vel = torch.tensor([[0.5, 0.0, 0.0]])

    demos = []
    m1_type_raw = data_m1['terrain_types']
    ref_type_raw = data_ref['terrain_types']

    # 有些环境记录的是“子地形编号”(例如 0~29)，这里折叠为 terrain class (0~4)
    m1_class = m1_type_raw % 5 if np.max(m1_type_raw) > 4 else m1_type_raw
    ref_class = ref_type_raw % 5 if np.max(ref_type_raw) > 4 else ref_type_raw

    for tidx in terrain_indices:
        t_name = terrain_names_map.get(tidx, f'Terrain {tidx}')

        m1_mask = m1_class == tidx
        ref_mask = ref_class == tidx

        if not m1_mask.any() or not ref_mask.any():
            print(f"  [WARN] No data for terrain {tidx} ({t_name}), skipping")
            continue

        m1_indices = np.where(m1_mask)[0]
        ref_indices = np.where(ref_mask)[0]

        # 选择地形起伏最明显的样本，避免抽到近似平地帧
        m1_var = np.var(data_m1['height_maps'][m1_indices], axis=(1, 2))
        ref_var = np.var(data_ref['height_maps'][ref_indices], axis=(1, 2))
        m1_idx = m1_indices[np.argmax(m1_var)]
        ref_idx = ref_indices[np.argmax(ref_var)]

        hm_m1 = data_m1['height_maps'][m1_idx]
        attn_m1 = data_m1['attn_weights'][m1_idx].reshape(GRID_H, GRID_W)

        attn_ref = data_ref['attn_weights'][ref_idx].reshape(GRID_H, GRID_W)

        hm_t = torch.from_numpy(hm_m1).unsqueeze(0).float()
        prior = scorer(hm_t, vel)[0].numpy().reshape(GRID_H, GRID_W)

        demos.append({
            'terrain_name': t_name,
            'height_map': hm_m1,
            'prior': prior,
            'attn_m1': attn_m1,
            'attn_ref': attn_ref,
        })

    return demos


# ── Main plotting ────────────────────────────────────────────────────────

def plot_attention_comparison(demos, label_ref='M3 (w/o KL prior)'):
    apply_style()
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
    })

    n_rows = len(demos)
    n_cols = 4
    fig = plt.figure(figsize=(8.5, 2.4 * n_rows + 0.5))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           wspace=0.08, hspace=0.25,
                           left=0.06, right=0.97, top=0.91, bottom=0.06)

    extent = [MEASURED_POINTS_Y[0], MEASURED_POINTS_Y[-1],
              MEASURED_POINTS_X[0], MEASURED_POINTS_X[-1]]
    imkw = dict(extent=extent, aspect='auto', origin='lower', interpolation='bilinear')

    col_titles = ['Height Map', 'Prior Distribution',
                  'Attention (Ours)', f'Attention ({label_ref})']
    cmaps = [CMAP_HEIGHT, CMAP_PRIOR, CMAP_ATTN_M1, CMAP_ATTN_D1]
    panel_labels = 'abcdefghijkl'

    for row, demo in enumerate(demos):
        arrays = [demo['height_map'], demo['prior'],
                  demo['attn_m1'], demo['attn_ref']]

        prior_flat = demo['prior'].ravel()
        cos_m1 = _cosine_sim(prior_flat, demo['attn_m1'].ravel())
        cos_ref = _cosine_sim(prior_flat, demo['attn_ref'].ravel())

        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col])
            arr = arrays[col]

            if col == 0:
                im = ax.imshow(arr, **imkw, cmap=cmaps[col])
            else:
                ax.imshow(demo['height_map'], **imkw, cmap=CMAP_HEIGHT, alpha=0.3)
                im = ax.imshow(arr, **imkw, cmap=cmaps[col], alpha=0.85)

            ax.plot(0, 0, marker='^', color='#333333', markersize=4,
                    zorder=5, markeredgecolor='white', markeredgewidth=0.5)

            pidx = row * n_cols + col
            ax.text(0.03, 0.95, f'({panel_labels[pidx]})',
                    transform=ax.transAxes, fontsize=8, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              alpha=0.7, ec='none'))

            if col >= 2:
                cos_val = cos_m1 if col == 2 else cos_ref
                ax.text(0.5, -0.01, f'cos_sim = {cos_val:.3f}',
                        transform=ax.transAxes, ha='center', va='top',
                        fontsize=7, fontstyle='italic', color='#555555')

            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, pad=4)

            if col == 0:
                ax.set_ylabel(demo['terrain_name'], fontsize=9, fontweight='bold')
            else:
                ax.set_yticklabels([])

            if row < n_rows - 1:
                ax.set_xticklabels([])

            if row == n_rows - 1 and col == 1:
                ax.set_xlabel('Lateral (m)', fontsize=8)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.03)
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6)

    return fig


def _cosine_sim(a, b):
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
    return float(dot / norm)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_m1', type=str, default=None)
    p.add_argument('--data_ref', type=str, default=None,
                   help='Path to reference method .npz (e.g., M3)')
    p.add_argument('--data_d1', type=str, default=None,
                   help='Backward-compatible alias of --data_ref')
    p.add_argument('--label_ref', type=str, default='M3 (w/o KL prior)',
                   help='Legend/title label for the reference method')
    p.add_argument('--demo', action='store_true',
                   help='Generate synthetic demo (no GPU needed)')
    p.add_argument('--output', type=str, default='attention_comparison')
    return p.parse_args()


def main():
    args = parse_args()
    data_ref = args.data_ref or args.data_d1

    if args.data_m1 and data_ref:
        print("Loading real attention data...")
        demos = load_real_data(args.data_m1, data_ref)
    else:
        if not args.demo:
            print("No data files provided. Use --demo for synthetic preview,")
            print("or --data_m1/--data_ref with .npz files from collect_attention_data.py")
            args.demo = True
        print("Generating synthetic demo data...")
        demos = generate_demo_data()

    fig = plot_attention_comparison(demos, label_ref=args.label_ref)
    save_figure(fig, args.output, OUTPUT_DIR)
    plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
