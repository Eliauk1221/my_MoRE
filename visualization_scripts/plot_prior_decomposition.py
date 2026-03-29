#!/usr/bin/env python3
"""
Figure: Physical Prior Decomposition Visualization (CVPR style)

For two terrain types (stepping stones & stairs), show the decomposition of
the TerrainSafetyScorer prior into three semantic components:
  - Steppability  (S_support * ~is_pit  +  S_margin)
  - Informativeness  (edge_bonus)
  - Spatial Relevance  (forward_bias + behind_penalty)
  - Combined Prior  (final softmax distribution)

Layout:  2 rows (terrains) × 5 columns (height | steppability | info | spatial | combined)
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', 'MoRE')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'rsl_rl'))

import torch
import numpy as np
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
from rsl_rl.modules.terrain_safety_scorer import TerrainSafetyScorer

HORIZONTAL_SCALE = 0.1
VERTICAL_SCALE = 0.005
TERRAIN_LENGTH = 14.0
TERRAIN_WIDTH = 4.0
LENGTH_PER_ENV_PX = int(TERRAIN_LENGTH / HORIZONTAL_SCALE)
WIDTH_PER_ENV_PX = int(TERRAIN_WIDTH / HORIZONTAL_SCALE)
NORMALIZATION_BASE_HEIGHT = 0.8
NOMINAL_STANDING_HEIGHT = 0.75

OUTPUT_DIR = './figures'


# ── Soft colormaps for CVPR ──────────────────────────────────────────────

def _make_soft_cmap(name, colors):
    return LinearSegmentedColormap.from_list(name, colors, N=256)

MORANDI_BLUE = '#7B9EA8'
MORANDI_ORANGE = '#D4A574'
MORANDI_GREEN = '#A8B88E'
MORANDI_BG = '#F8F7F4'

CMAP_HEIGHT = _make_soft_cmap('soft_terrain',
    [MORANDI_BG, '#E4EBDD', '#CAD6B9', MORANDI_GREEN, '#8F9F77'])
CMAP_STEP = _make_soft_cmap('soft_step',
    [MORANDI_BG, '#EEF2E8', '#D7E0C9', MORANDI_GREEN, '#8A9C72'])
CMAP_INFO = _make_soft_cmap('soft_info',
    [MORANDI_BG, '#F4EBDD', '#E8CCAB', MORANDI_ORANGE, '#B9875D'])
CMAP_SPATIAL = _make_soft_cmap('soft_spatial',
    [MORANDI_BG, '#DEE8EA', '#B8CCD1', MORANDI_BLUE, '#5E7980'])
CMAP_PRIOR = _make_soft_cmap('soft_prior',
    [MORANDI_BG, '#E4ECEE', '#C1D2D6', MORANDI_BLUE, '#607C84'])


# ── Terrain generation (from test_terrain_scorer.py) ─────────────────────

class MockSubTerrain:
    def __init__(self, width, length):
        self.width = width
        self.length = length
        self.vertical_scale = VERTICAL_SCALE
        self.horizontal_scale = HORIZONTAL_SCALE
        self.height_field_raw = np.zeros((width, length), dtype=np.int16)


def _create_terrain():
    return MockSubTerrain(LENGTH_PER_ENV_PX, WIDTH_PER_ENV_PX)


def make_stepping_stones(difficulty=0.5):
    t = _create_terrain()
    pit_depth = 1.0
    t.height_field_raw[:] = -round(pit_depth / VERTICAL_SCALE)

    stone_size = 0.42 - 0.06 * difficulty
    stone_size_px = round(stone_size / HORIZONTAL_SCALE)
    pitch_x_px = round(0.50 / HORIZONTAL_SCALE)
    lane_offset_px = round(0.18 / HORIZONTAL_SCALE)
    platform_len_px = round(2.5 / HORIZONTAL_SCALE)
    mid_y = t.length // 2

    t.height_field_raw[0:platform_len_px, :] = 0

    centers = []
    for col in range(10):
        cx = platform_len_px + col * pitch_x_px + stone_size_px // 2
        cy = mid_y + ((-1)**col) * lane_offset_px
        xs = cx - stone_size_px // 2
        xe = xs + stone_size_px
        ys = cy - stone_size_px // 2
        ye = ys + stone_size_px
        t.height_field_raw[xs:xe, ys:ye] = 0
        centers.append((cx, cy))

    end_x = platform_len_px + 10 * pitch_x_px
    t.height_field_raw[end_x:, :] = 0

    rx = centers[3][0] * HORIZONTAL_SCALE
    ry = centers[3][1] * HORIZONTAL_SCALE
    return t, rx, ry


def make_stairs(difficulty=0.5):
    t = _create_terrain()
    step_height = 0.10 + 0.15 * difficulty
    step_height_px = round(step_height / VERTICAL_SCALE)
    x_range_px = round(0.31 / HORIZONTAL_SCALE)
    platform_len_px = round(1.5 / HORIZONTAL_SCALE)
    mid_y = t.length // 2
    half_valid_px = round(1.5 / HORIZONTAL_SCALE)

    t.height_field_raw[0:platform_len_px, :] = 0

    dis_x = platform_len_px
    stair_h = 0
    edges = []
    for i in range(5):
        stair_h += step_height_px
        t.height_field_raw[dis_x:dis_x + x_range_px, :] = stair_h
        t.height_field_raw[dis_x:dis_x + x_range_px, :mid_y - half_valid_px] = 0
        t.height_field_raw[dis_x:dis_x + x_range_px, mid_y + half_valid_px:] = 0
        dis_x += x_range_px
        edges.append(dis_x)

    mid_plat_px = round(1.5 / HORIZONTAL_SCALE)
    t.height_field_raw[dis_x:dis_x + mid_plat_px,
                       mid_y - half_valid_px:mid_y + half_valid_px] = stair_h
    dis_x += mid_plat_px

    for i in range(5):
        stair_h -= step_height_px
        t.height_field_raw[dis_x:dis_x + x_range_px, :] = stair_h
        t.height_field_raw[dis_x:dis_x + x_range_px, :mid_y - half_valid_px] = 0
        t.height_field_raw[dis_x:dis_x + x_range_px, mid_y + half_valid_px:] = 0
        dis_x += x_range_px
        edges.append(dis_x)

    rx = edges[2] * HORIZONTAL_SCALE
    ry = mid_y * HORIZONTAL_SCALE
    return t, rx, ry


# ── Height map sampling ──────────────────────────────────────────────────

def _get_terrain_z(hf, px, py):
    ix = int(np.clip(px, 0, hf.shape[0] - 2))
    iy = int(np.clip(py, 0, hf.shape[1] - 2))
    return min(hf[ix, iy], hf[ix+1, iy], hf[ix, iy+1]) * VERTICAL_SCALE


def sample_height_map(hf, robot_x, robot_y):
    robot_px = robot_x / HORIZONTAL_SCALE
    robot_py = robot_y / HORIZONTAL_SCALE
    robot_z = _get_terrain_z(hf, robot_px, robot_py) + NOMINAL_STANDING_HEIGHT

    hm = torch.zeros(1, GRID_H, GRID_W)
    for i, dx in enumerate(MEASURED_POINTS_X):
        for j, dy in enumerate(MEASURED_POINTS_Y):
            px = (robot_x + dx) / HORIZONTAL_SCALE
            py = (robot_y + dy) / HORIZONTAL_SCALE
            tz = _get_terrain_z(hf, px, py)
            hm[0, i, j] = robot_z - NORMALIZATION_BASE_HEIGHT - tz
    return torch.clip(hm, -1.0, 1.0)


# ── Compute decomposed components ───────────────────────────────────────

def compute_decomposition(scorer, height_map, vel):
    """Return the three semantic components + combined prior as 2D arrays."""
    _, dbg = scorer(height_map, vel, return_debug_info=True)

    S_support = dbg['S_support'][0].numpy()
    S_margin = dbg['S_margin'][0].numpy()
    is_pit = dbg['is_pit'][0].numpy().astype(float)
    is_edge = dbg['is_edge'][0].numpy().astype(float)
    edge_sal = dbg['edge_salience'][0].numpy()
    behind = dbg['behind_mask'][0].numpy().astype(float)
    prior = dbg['prior_dist'][0].numpy().reshape(GRID_H, GRID_W)

    steppability = (scorer.w_support * S_support * (1.0 - is_pit)
                    + scorer.w_margin * S_margin)
    informativeness = scorer.w_edge * is_edge * edge_sal
    spatial = (scorer.w_forward * scorer.forward_bias.cpu().numpy()
               + behind * scorer.behind_penalty)

    hm_np = height_map[0].numpy()

    return {
        'height_map': hm_np,
        'steppability': steppability,
        'informativeness': informativeness,
        'spatial': spatial,
        'prior': prior,
    }


# ── Main plotting ────────────────────────────────────────────────────────

def plot_prior_decomposition(terrains_data, terrain_labels):
    apply_style()
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
    })

    n_rows = len(terrains_data)
    n_cols = 5
    fig = plt.figure(figsize=(10.0, 2.2 * n_rows + 0.6))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           wspace=0.08, hspace=0.25,
                           left=0.05, right=0.97, top=0.92, bottom=0.06)

    extent = [MEASURED_POINTS_Y[0], MEASURED_POINTS_Y[-1],
              MEASURED_POINTS_X[0], MEASURED_POINTS_X[-1]]
    imkw = dict(extent=extent, aspect='auto', origin='lower', interpolation='bilinear')

    col_titles = ['Height Map', 'Steppability', 'Informativeness',
                  'Spatial Relevance', 'Combined Prior']
    cmaps = [CMAP_HEIGHT, CMAP_STEP, CMAP_INFO, CMAP_SPATIAL, CMAP_PRIOR]
    panel_labels = 'abcdefghij'

    for row, (data, label) in enumerate(zip(terrains_data, terrain_labels)):
        arrays = [data['height_map'], data['steppability'],
                  data['informativeness'], data['spatial'], data['prior']]

        for col in range(n_cols):
            ax = fig.add_subplot(gs[row, col])
            arr = arrays[col]
            cmap = cmaps[col]

            if col == 0:
                im = ax.imshow(arr, **imkw, cmap=cmap)
            elif col == 3:
                vabs = max(abs(arr.min()), abs(arr.max()), 0.1)
                cmap_div = _make_soft_cmap('soft_div',
                    ['#C9DADF', MORANDI_BLUE, MORANDI_BG, MORANDI_ORANGE, '#B9875D'])
                im = ax.imshow(arr, **imkw, cmap=cmap_div, vmin=-vabs, vmax=vabs)
            else:
                im = ax.imshow(arr, **imkw, cmap=cmap, vmin=0)

            ax.plot(0, 0, marker='^', color='#333333', markersize=4, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.5)

            if col == 3:
                vx, vy = 0.5, 0.0
                ax.annotate('', xy=(vy * 0.25, vx * 0.25), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color='#333333',
                                            lw=1.2, shrinkA=0, shrinkB=0))

            pidx = row * n_cols + col
            ax.text(0.03, 0.95, f'({panel_labels[pidx]})',
                    transform=ax.transAxes, fontsize=8, fontweight='bold',
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7,
                              ec='none'))

            if row == 0:
                ax.set_title(col_titles[col], fontsize=9, pad=4)

            if col == 0:
                ax.set_ylabel(label, fontsize=9, fontweight='bold')
            else:
                ax.set_yticklabels([])

            if row < n_rows - 1:
                ax.set_xticklabels([])

            if row == n_rows - 1 and col == 2:
                ax.set_xlabel('Lateral (m)', fontsize=8)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.03)
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=6)

    return fig


def main():
    np.random.seed(42)

    scorer = TerrainSafetyScorer(
        grid_h=GRID_H, grid_w=GRID_W,
        measured_points_x=MEASURED_POINTS_X,
        measured_points_y=MEASURED_POINTS_Y,
    )
    vel = torch.tensor([[0.5, 0.0, 0.0]])

    terrains = []
    labels = []

    t_ss, rx_ss, ry_ss = make_stepping_stones(0.5)
    hm_ss = sample_height_map(t_ss.height_field_raw, rx_ss, ry_ss)
    terrains.append(compute_decomposition(scorer, hm_ss, vel))
    labels.append('Stepping\nStones')

    t_st, rx_st, ry_st = make_stairs(0.5)
    hm_st = sample_height_map(t_st.height_field_raw, rx_st, ry_st)
    terrains.append(compute_decomposition(scorer, hm_st, vel))
    labels.append('Stair')

    fig = plot_prior_decomposition(terrains, labels)
    save_figure(fig, 'prior_decomposition', OUTPUT_DIR)
    plt.close(fig)
    print("Done!")


if __name__ == '__main__':
    main()
