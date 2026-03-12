#!/usr/bin/env python3
"""
TerrainSafetyScorer v2 验证脚本

使用训练中真实的地形参数和尺寸生成测试地形，
验证 S_support（支撑面积分数）和 S_margin（边缘裕度分数）的准确性。

测试地形类型（与 terrain.py 一致）：
1. 平地 (flat) — 基线
2. 均匀斜坡 (slope) — 验证平面拟合对斜坡的容忍度
3. 梅花桩 (stepping_stones) — 验证石块中心高分、边缘低分
4. 台阶 (stair) — 验证踏面高分、边缘低分
5. 间隙 (gap) — 验证间隙区域低分、平台高分
6. 交替斜坡平台 (parkour) — 验证倾斜面上的平面拟合
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../..')  # MoRE/
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'rsl_rl'))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
from datetime import datetime

from rsl_rl.modules.terrain_safety_scorer import TerrainSafetyScorer


# ==================== 训练配置常量 (来自 g1_16dof_loco_config.py) ====================

HORIZONTAL_SCALE = 0.1    # [m/pixel]
VERTICAL_SCALE = 0.005    # [m/int16_unit]
TERRAIN_LENGTH = 14.0      # [m] 地形 x 方向长度（机器人前进方向）
TERRAIN_WIDTH = 4.0        # [m] 地形 y 方向宽度
LENGTH_PER_ENV_PX = int(TERRAIN_LENGTH / HORIZONTAL_SCALE)   # 140
WIDTH_PER_ENV_PX = int(TERRAIN_WIDTH / HORIZONTAL_SCALE)     # 40

MEASURED_POINTS_X = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                      0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MEASURED_POINTS_Y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

GRID_H = len(MEASURED_POINTS_X)  # 17
GRID_W = len(MEASURED_POINTS_Y)  # 11

NORMALIZATION_BASE_HEIGHT = 0.8  # [m] from legged_robot_config
NOMINAL_STANDING_HEIGHT = 0.75   # [m] G1 nominal CoM height


# ==================== Mock SubTerrain ====================

class MockSubTerrain:
    """简化版 SubTerrain，替代 isaacgym.terrain_utils.SubTerrain"""
    def __init__(self, name, width, length, vertical_scale, horizontal_scale):
        self.name = name
        self.width = width      # axis 0 大小 (= terrain_length 方向像素数)
        self.length = length    # axis 1 大小 (= terrain_width 方向像素数)
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.height_field_raw = np.zeros((width, length), dtype=np.int16)
        self.idx = 0


def create_sub_terrain():
    """创建与训练配置一致的 SubTerrain"""
    return MockSubTerrain(
        "terrain",
        width=LENGTH_PER_ENV_PX,
        length=WIDTH_PER_ENV_PX,
        vertical_scale=VERTICAL_SCALE,
        horizontal_scale=HORIZONTAL_SCALE,
    )


# ==================== 地形生成函数 (从 terrain.py 复制，与训练完全一致) ====================

def stepping_stones_terrain(terrain, stone_size, pitch_x=0.48,
                            lane_offset=0.18, num_cols=10,
                            platform_len=2.5, platform_height=0.,
                            pit_depth=1.0, pad_width=0.1, pad_height=0.5):
    if isinstance(pit_depth, (list, tuple)):
        pit_depth = np.random.uniform(*pit_depth)
    terrain.height_field_raw[:] = -round(pit_depth / terrain.vertical_scale)

    stone_size_px = round(stone_size / terrain.horizontal_scale)
    pitch_x_px = round(pitch_x / terrain.horizontal_scale)
    lane_offset_px = round(lane_offset / terrain.horizontal_scale)
    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_px = round(platform_height / terrain.vertical_scale)
    pad_width_px = int(pad_width // terrain.horizontal_scale)
    pad_height_px = int(pad_height // terrain.vertical_scale)

    mid_y = terrain.length // 2
    terrain.height_field_raw[0:platform_len_px, :] = platform_height_px

    stone_region_start = platform_len_px
    stone_centers = []

    for col in range(num_cols):
        center_x = stone_region_start + col * pitch_x_px + stone_size_px // 2
        if col % 2 == 0:
            center_y = mid_y - lane_offset_px
        else:
            center_y = mid_y + lane_offset_px

        x_start = center_x - stone_size_px // 2
        x_end = x_start + stone_size_px
        y_start = center_y - stone_size_px // 2
        y_end = y_start + stone_size_px
        terrain.height_field_raw[x_start:x_end, y_start:y_end] = platform_height_px
        stone_centers.append((center_x, center_y))

    stone_region_end = stone_region_start + num_cols * pitch_x_px
    terrain.height_field_raw[stone_region_end:, :] = platform_height_px

    terrain.height_field_raw[:, :pad_width_px] = pad_height_px
    terrain.height_field_raw[:, -pad_width_px:] = pad_height_px
    terrain.height_field_raw[:pad_width_px, :] = pad_height_px
    terrain.height_field_raw[-pad_width_px:, :] = pad_height_px

    return stone_centers


def parkour_stair_terrain(terrain, platform_len=2.5, platform_height=0.,
                          num_stones=10, x_range=0.13,
                          y_range=[-0.15, 0.15], half_valid_width=1,
                          step_height=0.15, pad_width=0.1, pad_height=0.5,
                          num_groups=2, middle_platform_len=3):
    mid_y = terrain.length // 2
    dis_x_min = round(x_range / terrain.horizontal_scale)
    step_height_px = round(step_height / terrain.vertical_scale)
    half_valid_width_px = round(half_valid_width / terrain.horizontal_scale)
    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_px = round(platform_height / terrain.vertical_scale)
    middle_platform_len_px = round(middle_platform_len / terrain.horizontal_scale)

    terrain.height_field_raw[0:platform_len_px, :] = platform_height_px

    dis_x = platform_len_px
    last_dis_x = dis_x
    step_edges = []

    for group in range(num_groups):
        stair_height = 0
        for i in range(num_stones // 2):
            rand_x = dis_x_min
            stair_height += step_height_px
            terrain.height_field_raw[dis_x:dis_x + rand_x, :] = stair_height
            dis_x += rand_x
            terrain.height_field_raw[last_dis_x:dis_x, :mid_y - half_valid_width_px] = 0
            terrain.height_field_raw[last_dis_x:dis_x, mid_y + half_valid_width_px:] = 0
            step_edges.append(dis_x)
            last_dis_x = dis_x

        terrain.height_field_raw[dis_x:dis_x + middle_platform_len_px,
                                 mid_y - half_valid_width_px:mid_y + half_valid_width_px] = stair_height
        dis_x += middle_platform_len_px
        last_dis_x = dis_x

        for i in range(num_stones // 2, num_stones):
            rand_x = dis_x_min
            stair_height -= step_height_px
            terrain.height_field_raw[dis_x:dis_x + rand_x, :] = stair_height
            dis_x += rand_x
            terrain.height_field_raw[last_dis_x:dis_x, :mid_y - half_valid_width_px] = 0
            terrain.height_field_raw[last_dis_x:dis_x, mid_y + half_valid_width_px:] = 0
            step_edges.append(dis_x)
            last_dis_x = dis_x

        if group < num_groups - 1:
            interval = dis_x_min * 10
            dis_x += interval
            last_dis_x = dis_x

    pad_width_px = int(pad_width // terrain.horizontal_scale)
    pad_height_px = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width_px] = pad_height_px
    terrain.height_field_raw[:, -pad_width_px:] = pad_height_px
    terrain.height_field_raw[:pad_width_px, :] = pad_height_px
    terrain.height_field_raw[-pad_width_px:, :] = pad_height_px

    return step_edges


def parkour_gap_terrain(terrain, platform_len=2.5, platform_height=0.,
                        num_gaps=8, gap_size=0.3,
                        x_range=[1.6, 2.4], y_range=[-1.2, 1.2],
                        half_valid_width=1, gap_depth=-200,
                        pad_width=0.1, pad_height=0.5, flat=False):
    mid_y = terrain.length // 2
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = round(y_range[1] / terrain.horizontal_scale)
    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_px = round(platform_height / terrain.vertical_scale)

    if isinstance(gap_depth, (list, tuple)):
        gap_depth_px = -round(np.random.uniform(gap_depth[0], gap_depth[1]) / terrain.vertical_scale)
    else:
        gap_depth_px = -round(abs(gap_depth) / terrain.vertical_scale) if gap_depth < 0 else round(gap_depth / terrain.vertical_scale)

    half_valid_width_px = round(half_valid_width / terrain.horizontal_scale)
    terrain.height_field_raw[0:platform_len_px, :] = platform_height_px

    gap_size_px = round(gap_size / terrain.horizontal_scale)
    dis_x_min = round(x_range[0] / terrain.horizontal_scale) + gap_size_px
    dis_x_max = round(x_range[1] / terrain.horizontal_scale) + gap_size_px

    dis_x = platform_len_px
    last_dis_x = dis_x
    gap_positions = []

    for i in range(num_gaps):
        rand_x = np.random.randint(dis_x_min, dis_x_max)
        dis_x += rand_x
        rand_y = np.random.randint(dis_y_min, max(dis_y_min + 1, dis_y_max))
        if not flat:
            terrain.height_field_raw[dis_x - gap_size_px // 2:dis_x + gap_size_px // 2, :] = gap_depth_px
        terrain.height_field_raw[last_dis_x:dis_x, :mid_y + rand_y - half_valid_width_px] = gap_depth_px
        terrain.height_field_raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width_px:] = gap_depth_px
        gap_positions.append(dis_x)
        last_dis_x = dis_x

    pad_width_px = int(pad_width // terrain.horizontal_scale)
    pad_height_px = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width_px] = pad_height_px
    terrain.height_field_raw[:, -pad_width_px:] = pad_height_px
    terrain.height_field_raw[:pad_width_px, :] = pad_height_px
    terrain.height_field_raw[-pad_width_px:, :] = pad_height_px

    return gap_positions


def parkour_terrain(terrain, platform_len=2.5, platform_height=0.,
                    num_stones=8, x_range=[1.8, 1.9], y_range=[0., 0.1],
                    z_range=[-0.2, 0.2], stone_len=1.0, stone_width=0.6,
                    pad_width=0.1, pad_height=0.5, incline_height=0.1,
                    last_incline_height=0.6, last_stone_len=1.6,
                    pit_depth=[0.5, 1.]):
    terrain.height_field_raw[:] = -round(np.random.uniform(pit_depth[0], pit_depth[1]) / terrain.vertical_scale)
    mid_y = terrain.length // 2

    if isinstance(stone_len, (list, tuple)):
        stone_len = np.random.uniform(*stone_len)
    stone_len = 2 * round(stone_len / 2.0, 1)
    stone_len_px = round(stone_len / terrain.horizontal_scale)

    dis_x_min = stone_len_px + round(x_range[0] / terrain.horizontal_scale)
    dis_x_max = stone_len_px + round(x_range[1] / terrain.horizontal_scale)
    dis_y_min = round(y_range[0] / terrain.horizontal_scale)
    dis_y_max = max(dis_y_min + 1, round(y_range[1] / terrain.horizontal_scale))

    platform_len_px = round(platform_len / terrain.horizontal_scale)
    platform_height_px = round(platform_height / terrain.vertical_scale)
    terrain.height_field_raw[0:platform_len_px, :] = platform_height_px

    stone_width_px = round(stone_width / terrain.horizontal_scale)
    last_stone_len_px = round(last_stone_len / terrain.horizontal_scale)
    incline_height_px = round(incline_height / terrain.vertical_scale)
    last_incline_height_px = round(last_incline_height / terrain.vertical_scale)

    dis_x = platform_len_px - np.random.randint(dis_x_min, dis_x_max) + stone_len_px // 2
    left_right_flag = np.random.randint(0, 2)
    dis_z = 0
    stone_info = []

    for i in range(num_stones):
        dis_x += np.random.randint(dis_x_min, dis_x_max)
        pos_neg = round(2 * (left_right_flag - 0.5))
        dis_y = mid_y + pos_neg * np.random.randint(dis_y_min, dis_y_max)
        if i == num_stones - 1:
            dis_x += last_stone_len_px // 4
            heights = np.tile(np.linspace(-last_incline_height_px, last_incline_height_px, stone_width_px),
                              (last_stone_len_px, 1)) * pos_neg
            x_s = dis_x - last_stone_len_px // 2
            x_e = dis_x + last_stone_len_px // 2
            y_s = dis_y - stone_width_px // 2
            y_e = dis_y + stone_width_px // 2
            terrain.height_field_raw[x_s:x_e, y_s:y_e] = heights.astype(int) + dis_z
        else:
            heights = np.tile(np.linspace(-incline_height_px, incline_height_px, stone_width_px),
                              (stone_len_px, 1)) * pos_neg
            x_s = dis_x - stone_len_px // 2
            x_e = dis_x + stone_len_px // 2
            y_s = dis_y - stone_width_px // 2
            y_e = dis_y + stone_width_px // 2
            terrain.height_field_raw[x_s:x_e, y_s:y_e] = heights.astype(int) + dis_z
        stone_info.append((dis_x, dis_y, pos_neg))
        left_right_flag = 1 - left_right_flag

    final_platform_start = dis_x + last_stone_len_px // 2 + round(0.05 // terrain.horizontal_scale)
    if final_platform_start < terrain.width:
        terrain.height_field_raw[final_platform_start:, :] = platform_height_px

    pad_width_px = int(pad_width // terrain.horizontal_scale)
    pad_height_px = int(pad_height // terrain.vertical_scale)
    terrain.height_field_raw[:, :pad_width_px] = pad_height_px
    terrain.height_field_raw[:, -pad_width_px:] = pad_height_px
    terrain.height_field_raw[:pad_width_px, :] = pad_height_px
    terrain.height_field_raw[-pad_width_px:, :] = pad_height_px

    return stone_info


# ==================== Height Map 采样 ====================

def sample_height_map(
    height_field_raw: np.ndarray,
    robot_x_m: float,
    robot_y_m: float,
    horizontal_scale: float = HORIZONTAL_SCALE,
    vertical_scale: float = VERTICAL_SCALE,
) -> torch.Tensor:
    """
    模拟训练中 _get_heights + _update_terrain_attention_data 的完整管线。
    
    与训练代码一致的采样方式:
    - 在 robot 周围的 17×11 网格采样地形高度
    - 使用 min-of-3 插值 (与 legged_robot._get_heights 一致)
    - 计算 height_map = robot_z - base_height - terrain_z
    - clip 到 [-1, 1]
    
    Returns:
        height_map: [1, GRID_H, GRID_W] tensor
    """
    hf = height_field_raw

    robot_px = robot_x_m / horizontal_scale
    robot_py = robot_y_m / horizontal_scale
    robot_terrain_z = _get_terrain_z(hf, robot_px, robot_py, vertical_scale)
    robot_z = robot_terrain_z + NOMINAL_STANDING_HEIGHT

    height_map = torch.zeros(1, GRID_H, GRID_W)
    for i, dx in enumerate(MEASURED_POINTS_X):
        for j, dy in enumerate(MEASURED_POINTS_Y):
            px = (robot_x_m + dx) / horizontal_scale
            py = (robot_y_m + dy) / horizontal_scale
            terrain_z = _get_terrain_z(hf, px, py, vertical_scale)
            height_map[0, i, j] = robot_z - NORMALIZATION_BASE_HEIGHT - terrain_z

    return torch.clip(height_map, -1.0, 1.0)


def _get_terrain_z(hf: np.ndarray, px: float, py: float, vertical_scale: float) -> float:
    """min-of-3 插值（与 legged_robot._get_heights 一致）"""
    ix = int(px)
    iy = int(py)
    ix = np.clip(ix, 0, hf.shape[0] - 2)
    iy = np.clip(iy, 0, hf.shape[1] - 2)
    h1 = hf[ix, iy]
    h2 = hf[ix + 1, iy]
    h3 = hf[ix, iy + 1]
    return min(h1, h2, h3) * vertical_scale


# ==================== 可视化 ====================

def visualize_scorer_output(
    height_map: torch.Tensor,
    debug_info: Dict,
    base_lin_vel: torch.Tensor,
    title: str = "",
    sample_idx: int = 0,
    save_path: Optional[str] = None,
):
    """可视化新版打分器的 8 个关键输出"""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        f"{title}\n"
        f"v=({base_lin_vel[sample_idx, 0]:.1f}, {base_lin_vel[sample_idx, 1]:.1f}) m/s",
        fontsize=13,
    )

    extent = [MEASURED_POINTS_Y[0], MEASURED_POINTS_Y[-1],
              MEASURED_POINTS_X[0], MEASURED_POINTS_X[-1]]
    kw = dict(extent=extent, aspect='auto', origin='lower')

    def mark(ax):
        ax.plot(0, 0, 'ko', markersize=6)
        ax.axhline(0, color='k', ls='--', alpha=0.2)
        ax.axvline(0, color='k', ls='--', alpha=0.2)

    def vel_arrow(ax, vel):
        vx, vy = vel[0].item(), vel[1].item()
        if (vx**2 + vy**2) > 0.01:
            ax.annotate('', xy=(vy * 0.3, vx * 0.3), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    data = {k: v[sample_idx].detach().cpu().numpy() if v.dim() > 1 else v.detach().cpu().numpy()
            for k, v in debug_info.items() if isinstance(v, torch.Tensor)}

    # Row 1
    ax = axes[0, 0]
    im = ax.imshow(data['residual_rms'], **kw, cmap='terrain_r')
    h = height_map[sample_idx].numpy()
    ax.set_title('Height Map (input)')
    im = ax.imshow(h, **kw, cmap='terrain_r')
    mark(ax); plt.colorbar(im, ax=ax, label='m')

    ax = axes[0, 1]
    im = ax.imshow(data['residual_rms'], **kw, cmap='hot')
    ax.set_title('Plane Fit Residual (RMS)')
    mark(ax); plt.colorbar(im, ax=ax, label='m')

    ax = axes[0, 2]
    im = ax.imshow(data['S_support'], **kw, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title('S_support [0,1]\n(1=flat, 0=discontinuity)')
    mark(ax); plt.colorbar(im, ax=ax)

    ax = axes[0, 3]
    im = ax.imshow(data['danger_mask'].astype(float), **kw, cmap='Reds')
    ax.set_title('Danger Mask\n(residual>thr OR pit)')
    mark(ax); plt.colorbar(im, ax=ax)

    # Row 2
    ax = axes[1, 0]
    im = ax.imshow(data['S_margin'], **kw, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title('S_margin [0,1]\n(edge distance)')
    mark(ax); plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(data['behind_mask'].astype(float), **kw, cmap='Blues')
    ax.set_title('Behind Mask')
    mark(ax); vel_arrow(ax, base_lin_vel[sample_idx])
    plt.colorbar(im, ax=ax)

    ax = axes[1, 2]
    im = ax.imshow(data['logits_2d'], **kw, cmap='RdYlGn')
    ax.set_title('Fused Logits')
    mark(ax); vel_arrow(ax, base_lin_vel[sample_idx])
    plt.colorbar(im, ax=ax)

    ax = axes[1, 3]
    prior = data['prior_dist'].reshape(GRID_H, GRID_W) if data['prior_dist'].ndim == 1 else data['prior_dist']
    im = ax.imshow(prior, **kw, cmap='hot')
    ax.set_title('Prior Distribution\n(softmax)')
    mark(ax); plt.colorbar(im, ax=ax, label='prob')

    for row in axes:
        for ax in row:
            ax.set_xlabel('Y (m)')
            ax.set_ylabel('X (m) ↑ fwd')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


def visualize_terrain_overview(
    height_field_raw: np.ndarray,
    robot_x_m: float,
    robot_y_m: float,
    title: str = "",
    save_path: Optional[str] = None,
):
    """鸟瞰全局地形 + 机器人位置 + 采样窗口"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    hf_m = height_field_raw.astype(float) * VERTICAL_SCALE
    extent = [0, hf_m.shape[1] * HORIZONTAL_SCALE,
              0, hf_m.shape[0] * HORIZONTAL_SCALE]
    im = ax.imshow(hf_m, extent=extent, aspect='auto', origin='lower', cmap='terrain')
    ax.plot(robot_y_m, robot_x_m, 'r*', markersize=15, label='Robot')

    x0 = robot_x_m + MEASURED_POINTS_X[0]
    x1 = robot_x_m + MEASURED_POINTS_X[-1]
    y0 = robot_y_m + MEASURED_POINTS_Y[0]
    y1 = robot_y_m + MEASURED_POINTS_Y[-1]
    rect = plt.Rectangle((y0, x0), y1 - y0, x1 - x0,
                          linewidth=2, edgecolor='red', facecolor='none', label='Sample window')
    ax.add_patch(rect)
    ax.legend(loc='upper right')
    ax.set_title(f'{title} — Terrain Overview')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ forward')
    plt.colorbar(im, ax=ax, label='Height (m)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ==================== 单元测试 ====================

def test_dimension_and_probability(scorer: TerrainSafetyScorer):
    """验证输出维度和概率归一性"""
    print("\n" + "=" * 60)
    print("Test: Dimension & Probability")
    print("=" * 60)

    for B in [1, 4, 32]:
        hm = torch.randn(B, GRID_H, GRID_W) * 0.1
        vel = torch.randn(B, 3) * 0.3
        prior, dbg = scorer(hm, vel, return_debug_info=True)

        assert prior.shape == (B, GRID_H * GRID_W), f"shape mismatch: {prior.shape}"
        assert torch.allclose(prior.sum(dim=-1), torch.ones(B), atol=1e-5), "sum != 1"
        assert (prior >= 0).all(), "negative probability"
        assert dbg['S_support'].shape == (B, GRID_H, GRID_W)
        assert dbg['S_margin'].shape == (B, GRID_H, GRID_W)
        print(f"  B={B}: shape OK, sum=1 OK, non-negative OK")

    print("  [PASS]")


def test_flat_terrain(scorer: TerrainSafetyScorer):
    """平地: 全部高分, 无危险点"""
    print("\n" + "=" * 60)
    print("Test: Flat Terrain")
    print("=" * 60)

    hm = torch.zeros(1, GRID_H, GRID_W)
    vel = torch.tensor([[0.5, 0.0, 0.0]])
    _, dbg = scorer(hm, vel, return_debug_info=True)

    assert not dbg['danger_mask'].any(), "flat terrain should have no danger"
    assert dbg['S_support'].min() > 0.9, f"S_support too low: {dbg['S_support'].min():.3f}"
    assert dbg['S_margin'].min() > 0.9, f"S_margin too low: {dbg['S_margin'].min():.3f}"

    print(f"  S_support: [{dbg['S_support'].min():.3f}, {dbg['S_support'].max():.3f}]")
    print(f"  S_margin:  [{dbg['S_margin'].min():.3f}, {dbg['S_margin'].max():.3f}]")
    print(f"  danger_mask: none")
    print("  [PASS]")


def test_uniform_slope(scorer: TerrainSafetyScorer):
    """均匀斜坡: 平面拟合残差应极小, 不应触发危险"""
    print("\n" + "=" * 60)
    print("Test: Uniform Slope (30°)")
    print("=" * 60)

    hm = torch.zeros(1, GRID_H, GRID_W)
    tan30 = 0.577
    for i in range(GRID_H):
        hm[0, i, :] = -i * 0.1 * tan30  # 上坡 → 负值

    vel = torch.tensor([[0.5, 0.0, 0.0]])
    _, dbg = scorer(hm, vel, return_debug_info=True)

    interior_residual = dbg['residual_rms'][0, 3:-3, 2:-2]
    assert interior_residual.max() < scorer.danger_threshold, \
        f"slope interior residual {interior_residual.max():.4f} exceeds danger_threshold"
    assert not dbg['danger_mask'][0, 3:-3, 2:-2].any(), "slope interior should have no danger"

    print(f"  Interior residual max: {interior_residual.max():.6f} (< {scorer.danger_threshold})")
    print(f"  S_support interior: [{dbg['S_support'][0,3:-3,2:-2].min():.3f}, {dbg['S_support'][0,3:-3,2:-2].max():.3f}]")
    print(f"  danger in interior: {dbg['danger_mask'][0,3:-3,2:-2].any().item()}")
    print("  [PASS]")


def test_step_edge(scorer: TerrainSafetyScorer):
    """台阶: 边缘处残差大、danger=True; 踏面上残差小"""
    print("\n" + "=" * 60)
    print("Test: Step Edge (15cm)")
    print("=" * 60)

    hm = torch.zeros(1, GRID_H, GRID_W)
    hm[0, 8:, :] = -0.15  # 上台阶

    vel = torch.tensor([[0.5, 0.0, 0.0]])
    _, dbg = scorer(hm, vel, return_debug_info=True)

    edge_support = dbg['S_support'][0, 7:9, 5].mean()
    tread_support = dbg['S_support'][0, [3, 13], 5].mean()

    assert edge_support < 0.3, f"edge S_support should be low, got {edge_support:.3f}"
    assert tread_support > 0.8, f"tread S_support should be high, got {tread_support:.3f}"

    edge_margin = dbg['S_margin'][0, 7:9, 5].mean()
    far_margin = dbg['S_margin'][0, [0, 15], 5].mean()

    assert edge_margin < 0.2, f"edge margin should be low, got {edge_margin:.3f}"
    assert far_margin > 0.8, f"far margin should be high, got {far_margin:.3f}"

    print(f"  Edge S_support: {edge_support:.3f} (< 0.3)")
    print(f"  Tread S_support: {tread_support:.3f} (> 0.8)")
    print(f"  Edge S_margin: {edge_margin:.3f}")
    print(f"  Far S_margin: {far_margin:.3f}")
    print("  [PASS]")


def test_behind_mask(scorer: TerrainSafetyScorer):
    """身后掩码: 前进时 x<0 被掩码; 斜向走时掩码方向正确"""
    print("\n" + "=" * 60)
    print("Test: Behind Mask")
    print("=" * 60)

    hm = torch.zeros(1, GRID_H, GRID_W)
    vel_fwd = torch.tensor([[0.5, 0.0, 0.0]])
    vel_diag = torch.tensor([[0.5, 0.5, 0.0]])

    _, dbg_fwd = scorer(hm, vel_fwd, return_debug_info=True)
    _, dbg_diag = scorer(hm, vel_diag, return_debug_info=True)

    assert dbg_fwd['behind_mask'][0, 0, 5].item(), "x=-0.8 should be behind when walking forward"
    assert not dbg_fwd['behind_mask'][0, -1, 5].item(), "x=+0.8 should NOT be behind"

    assert dbg_diag['behind_mask'][0, 0, 0].item(), "lower-left should be behind for diagonal"
    assert not dbg_diag['behind_mask'][0, -1, -1].item(), "upper-right should NOT be behind"

    print(f"  Forward: behind_count={dbg_fwd['behind_mask'].sum().item()}")
    print(f"  Diagonal: behind_count={dbg_diag['behind_mask'].sum().item()}")
    print("  [PASS]")


def test_pit_detection(scorer: TerrainSafetyScorer):
    """深坑: 正值 > pit_threshold 触发 danger"""
    print("\n" + "=" * 60)
    print("Test: Pit Detection")
    print("=" * 60)

    hm = torch.zeros(1, GRID_H, GRID_W)
    hm[0, 6:11, :] = 0.5  # 深坑

    vel = torch.tensor([[0.3, 0.0, 0.0]])
    prior, dbg = scorer(hm, vel, return_debug_info=True)

    pit_danger = dbg['danger_mask'][0, 6:11, :].all()
    fwd_safe = not dbg['danger_mask'][0, 12:16, :].any()

    assert pit_danger, "pit region should be all danger"
    assert fwd_safe, "forward safe region should have no danger"

    prior_2d = prior.view(1, GRID_H, GRID_W)
    pit_prob = prior_2d[0, 6:11, :].sum()
    fwd_safe_prob = prior_2d[0, 12:16, :].sum()
    assert fwd_safe_prob > pit_prob, \
        f"forward safe prob ({fwd_safe_prob:.3f}) should > pit prob ({pit_prob:.3f})"

    print(f"  Pit region danger: all True")
    print(f"  Forward safe prob: {fwd_safe_prob:.3f} > Pit prob: {pit_prob:.3f}")
    print("  [PASS]")


# ==================== 真实地形可视化测试 ====================

def run_visual_tests(scorer: TerrainSafetyScorer, save_dir: str):
    """使用训练真实地形参数生成地形, 采样 height_map 并可视化打分结果"""
    print("\n" + "=" * 60)
    print("Visual Tests (real terrain dimensions)")
    print("=" * 60)
    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(42)
    vel_fwd = torch.tensor([[0.5, 0.0, 0.0]])

    test_cases = []

    # --- 1. 平地 ---
    t = create_sub_terrain()
    t.height_field_raw[:] = 0
    rx, ry = 5.0, 2.0
    test_cases.append(("01_flat", t, rx, ry, "Flat Terrain"))

    # --- 2. 均匀斜坡 ---
    t = create_sub_terrain()
    for ix in range(t.width):
        t.height_field_raw[ix, :] = int(ix * 0.5)  # 缓坡
    rx, ry = 7.0, 2.0
    test_cases.append(("02_slope", t, rx, ry, "Uniform Slope"))

    # --- 3. 梅花桩 (difficulty=0.5) ---
    t = create_sub_terrain()
    difficulty = 0.5
    stone_centers = stepping_stones_terrain(
        t, stone_size=0.42 - 0.06 * difficulty,
        pitch_x=0.50, lane_offset=0.18, pad_height=0)
    if stone_centers:
        cx, cy = stone_centers[2]  # 第3个石块中心
        rx = cx * HORIZONTAL_SCALE
        ry = cy * HORIZONTAL_SCALE
    else:
        rx, ry = 4.0, 2.0
    test_cases.append(("03_stepping_stones", t, rx, ry, "Stepping Stones (d=0.5)"))

    # --- 4. 梅花桩 - 石块边缘 ---
    if stone_centers:
        cx, cy = stone_centers[2]
        stone_size_px = round((0.42 - 0.06 * difficulty) / HORIZONTAL_SCALE)
        rx_edge = (cx + stone_size_px // 2 - 1) * HORIZONTAL_SCALE
        ry_edge = cy * HORIZONTAL_SCALE
        test_cases.append(("04_stepping_stones_edge", t, rx_edge, ry_edge,
                           "Stepping Stones — Edge"))

    # --- 5. 台阶 (difficulty=0.5) ---
    t = create_sub_terrain()
    step_edges = parkour_stair_terrain(
        t, platform_len=1.5, platform_height=0., num_stones=10,
        x_range=0.31, y_range=[-0.01, 0.01], half_valid_width=1.5,
        step_height=0.10 + 0.15 * difficulty, pad_width=0.1, pad_height=0,
        num_groups=3, middle_platform_len=1.5)
    if step_edges and len(step_edges) > 2:
        rx = step_edges[2] * HORIZONTAL_SCALE
    else:
        rx = 5.0
    ry = 2.0
    test_cases.append(("05_stair", t, rx, ry, f"Stair (d={difficulty})"))

    # --- 6. 台阶 - 踏面中心 ---
    if step_edges and len(step_edges) > 3:
        mid_x = (step_edges[2] + step_edges[3]) / 2 * HORIZONTAL_SCALE
        test_cases.append(("06_stair_tread", t, mid_x, ry,
                           "Stair — Tread Center"))

    # --- 7. 间隙 (difficulty=0.5) ---
    t = create_sub_terrain()
    gap_pos = parkour_gap_terrain(
        t, platform_len=2.5, platform_height=0, num_gaps=7,
        gap_size=0.05 + 0.4 * difficulty,
        x_range=[0.75, 2], y_range=[-0.1, 0.1],
        half_valid_width=1.5, gap_depth=[0.2, 0.4],
        pad_width=0.1, pad_height=0, flat=False)
    if gap_pos:
        rx_gap = gap_pos[0] * HORIZONTAL_SCALE
    else:
        rx_gap = 5.0
    ry = 2.0
    test_cases.append(("07_gap", t, rx_gap, ry, f"Gap (d={difficulty})"))

    # --- 8. 交替斜坡 parkour (difficulty=0.5) ---
    t = create_sub_terrain()
    stone_info = parkour_terrain(
        t, num_stones=8,
        x_range=[0.1, 0.2 + 0.3 * difficulty],
        y_range=[0.2, 0.3 + 0.1 * difficulty],
        stone_len=[0.9 - 0.3 * difficulty, 1 - 0.2 * difficulty],
        incline_height=0.25 * difficulty,
        stone_width=1.0,
        last_incline_height=0.25 * difficulty + 0.1 - 0.1 * difficulty,
        pad_height=0,
        pit_depth=[0.2, 1])
    if stone_info:
        sx, sy, _ = stone_info[1]
        rx_pk = sx * HORIZONTAL_SCALE
        ry_pk = sy * HORIZONTAL_SCALE
    else:
        rx_pk, ry_pk = 5.0, 2.0
    test_cases.append(("08_parkour", t, rx_pk, ry_pk, f"Parkour (d={difficulty})"))

    # --- 运行每个测试 ---
    for name, terrain, rx, ry, title in test_cases:
        print(f"\n  [{name}] {title}")
        print(f"    Robot position: ({rx:.2f}, {ry:.2f}) m")

        hm = sample_height_map(terrain.height_field_raw, rx, ry)
        prior, dbg = scorer(hm, vel_fwd, return_debug_info=True)

        print(f"    height_map: [{hm.min():.3f}, {hm.max():.3f}]")
        print(f"    S_support:  [{dbg['S_support'].min():.3f}, {dbg['S_support'].max():.3f}]")
        print(f"    S_margin:   [{dbg['S_margin'].min():.3f}, {dbg['S_margin'].max():.3f}]")
        print(f"    danger_ratio: {dbg['danger_mask'].float().mean():.1%}")
        print(f"    prior entropy: {-(prior * torch.log(prior + 1e-8)).sum():.2f} "
              f"(uniform={np.log(GRID_H * GRID_W):.2f})")

        visualize_terrain_overview(
            terrain.height_field_raw, rx, ry, title=title,
            save_path=os.path.join(save_dir, f"{name}_overview.png"))

        visualize_scorer_output(
            hm, dbg, vel_fwd, title=title,
            save_path=os.path.join(save_dir, f"{name}_scorer.png"))

    print(f"\n  All {len(test_cases)} visual tests saved to: {save_dir}")


# ==================== Main ====================

def main():
    print("=" * 60)
    print("TerrainSafetyScorer v2 — Test Suite")
    print("=" * 60)
    print(f"\n符号语义: height_map = robot_z - base_height - terrain_z")
    print(f"  正值 = 坑 | 负值 = 凸起")
    print(f"  Grid: {GRID_H}x{GRID_W}, dx={HORIZONTAL_SCALE}m, dy={HORIZONTAL_SCALE}m")
    print(f"  Terrain: {TERRAIN_LENGTH}m x {TERRAIN_WIDTH}m "
          f"({LENGTH_PER_ENV_PX}x{WIDTH_PER_ENV_PX} px)")

    scorer = TerrainSafetyScorer(
        grid_h=GRID_H,
        grid_w=GRID_W,
        measured_points_x=MEASURED_POINTS_X,
        measured_points_y=MEASURED_POINTS_Y,
    )
    print(f"\nScorer: {scorer}\n")

    # === 单元测试 ===
    test_dimension_and_probability(scorer)
    test_flat_terrain(scorer)
    test_uniform_slope(scorer)
    test_step_edge(scorer)
    test_behind_mask(scorer)
    test_pit_detection(scorer)

    print("\n" + "=" * 60)
    print("All Unit Tests Passed!")
    print("=" * 60)

    # === 可视化测试 ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.abspath(
        os.path.join(project_root, f"terrain_scorer_v2_viz_{timestamp}"))
    run_visual_tests(scorer, save_dir=save_dir)


if __name__ == "__main__":
    main()
