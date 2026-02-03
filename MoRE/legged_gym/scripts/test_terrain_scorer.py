#!/usr/bin/env python3
"""
TerrainSafetyScorer 测试脚本

测试地形类型:
1. 平地 (flat)
2. 上升台阶 (step_up)
3. 下降台阶 (step_down)
4. 深坑 (pit)
5. 斜坡 (slope)
6. 粗糙地形 (rough)

测试场景:
- 静止状态 (v=0)
- 向前行走 (v_x > 0)
- 向侧面行走 (v_y > 0)
- 斜向行走 (v_x > 0, v_y > 0)
"""

import sys
import os

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '../..')  # MoRE/
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'rsl_rl'))

import torch
import numpy as np
import matplotlib
# 无屏幕/后台保存：使用非交互式后端，避免弹窗或依赖 X server
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from typing import Dict, Optional
from datetime import datetime

# 导入待测试模块
from rsl_rl.modules.terrain_safety_scorer import TerrainSafetyScorer


# ==================== 测试地形生成器 ====================

def create_flat_terrain(grid_h: int = 17, grid_w: int = 11, batch_size: int = 1) -> torch.Tensor:
    """创建平坦地形 (全零高度图)"""
    return torch.zeros(batch_size, grid_h, grid_w)


def create_step_up_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    step_height: float = 0.1,
    num_steps: int = 2
) -> torch.Tensor:
    """创建连续上升台阶地形 (前方多阶抬高)
    
    符号语义 (与 LeggedGym 一致):
    - 正值 = 地面更低（坑）
    - 负值 = 地面更高（凸起/台阶）
    
    Args:
        step_height: 每阶台阶的高度
        num_steps: 台阶数量
    """
    height_map = torch.zeros(batch_size, grid_h, grid_w)
    # 将网格分成 num_steps + 1 段 (身后平地 + num_steps 阶台阶)
    segment_len = grid_h // (num_steps + 1)
    for i in range(num_steps):
        start_idx = segment_len * (i + 1)
        # 台阶抬高 → 负值
        height_map[:, start_idx:, :] = -step_height * (i + 1)
    return height_map


def create_step_down_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    step_depth: float = 0.1,
    num_steps: int = 2
) -> torch.Tensor:
    """创建连续下降台阶地形 (前方多阶降低)
    
    符号语义 (与 LeggedGym 一致):
    - 正值 = 地面更低（坑/下台阶）
    - 负值 = 地面更高（凸起）
    
    Args:
        step_depth: 每阶台阶的深度
        num_steps: 台阶数量
    """
    height_map = torch.zeros(batch_size, grid_h, grid_w)
    # 将网格分成 num_steps + 1 段
    segment_len = grid_h // (num_steps + 1)
    for i in range(num_steps):
        start_idx = segment_len * (i + 1)
        # 台阶降低 → 正值
        height_map[:, start_idx:, :] = step_depth * (i + 1)
    return height_map


def create_pit_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    pit_depth: float = 0.5,
    pit_start: float = 0.4,
    pit_end: float = 0.7
) -> torch.Tensor:
    """创建深坑地形 (中间区域有深坑)
    
    符号语义 (与 LeggedGym 一致):
    - 正值 = 地面更低（坑）
    """
    height_map = torch.zeros(batch_size, grid_h, grid_w)
    start_idx = int(grid_h * pit_start)
    end_idx = int(grid_h * pit_end)
    # 坑 → 正值
    height_map[:, start_idx:end_idx, :] = pit_depth
    return height_map


def create_slope_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    max_height: float = 0.3
) -> torch.Tensor:
    """创建斜坡地形 (从后向前线性上升)
    
    符号语义 (与 LeggedGym 一致):
    - 负值 = 地面更高（上坡）
    """
    height_map = torch.zeros(batch_size, grid_h, grid_w)
    # 线性斜坡（上坡 → 负值）
    slope = torch.linspace(0, -max_height, grid_h).view(1, grid_h, 1)
    height_map = slope.expand(batch_size, grid_h, grid_w)
    return height_map


def create_stepping_stones_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    stone_height: float = 0.0,
    gap_depth: float = 0.5,
    stone_size: int = 2,
    gap_size: int = 1
) -> torch.Tensor:
    """创建踏脚石地形 (交替的石块和间隙)
    
    符号语义 (与 LeggedGym 一致):
    - 正值 = 地面更低（间隙/坑）
    - 0 = 石块（与机器人站立位置同高）
    
    Args:
        stone_height: 石块高度（相对基准）
        gap_depth: 间隙深度
        stone_size: 石块尺寸（网格单位）
        gap_size: 间隙尺寸（网格单位）
    """
    height_map = torch.zeros(batch_size, grid_h, grid_w)
    pattern_size = stone_size + gap_size
    
    for i in range(grid_h):
        for j in range(grid_w):
            # 计算在 pattern 中的位置
            i_in_pattern = i % pattern_size
            j_in_pattern = j % pattern_size
            
            # 如果在间隙区域 → 正值（坑）
            if i_in_pattern >= stone_size or j_in_pattern >= stone_size:
                height_map[:, i, j] = gap_depth
            else:
                height_map[:, i, j] = stone_height
    
    return height_map


def create_gap_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    gap_depth: float = 1.0,
    gap_width: int = 2,
    gap_position: float = 0.5
) -> torch.Tensor:
    """创建间隙地形 (窄而深的缝隙)
    
    符号语义 (与 LeggedGym 一致):
    - 正值 = 地面更低（间隙）
    """
    height_map = torch.zeros(batch_size, grid_h, grid_w)
    center_idx = int(grid_h * gap_position)
    start_idx = max(0, center_idx - gap_width // 2)
    end_idx = min(grid_h, center_idx + gap_width // 2 + 1)
    # 间隙 → 正值
    height_map[:, start_idx:end_idx, :] = gap_depth
    return height_map


def create_rough_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    roughness_scale: float = 0.1
) -> torch.Tensor:
    """创建粗糙地形 (随机高频起伏，用于测试 local_var)"""
    height_map = torch.randn(batch_size, grid_h, grid_w) * roughness_scale
    return height_map


# ==================== 可视化工具 ====================

def visualize_scorer_output(
    height_map: torch.Tensor,
    debug_info: Dict,
    base_lin_vel: torch.Tensor,
    title: str = "Terrain Safety Scorer Output",
    sample_idx: int = 0,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """
    可视化打分器输出
    
    Args:
        height_map: [B, H, W]
        debug_info: forward() 返回的调试信息
        base_lin_vel: [B, 3]
        title: 图标题
        sample_idx: 要可视化的 batch 索引
        save_path: 保存路径 (可选)
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(
        f"{title}\n"
        f"Velocity: v_x={base_lin_vel[sample_idx, 0]:.2f}, v_y={base_lin_vel[sample_idx, 1]:.2f} m/s | "
        f"α={debug_info['alpha']:.3f}", 
        fontsize=14
    )
    
    # 采样点坐标
    measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
                          0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    
    # extent: [left, right, bottom, top] with origin='lower'
    # 让 X 轴向上为正（上方 = 机器人前方）
    extent = [measured_points_y[0], measured_points_y[-1], 
              measured_points_x[0], measured_points_x[-1]]
    
    def add_robot_marker(ax):
        """在图上标记机器人位置"""
        ax.plot(0, 0, 'ko', markersize=10, label='Robot')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    def add_capture_point(ax, capture_point):
        """标记统一的理想捕获点"""
        # capture_point: [2] - 统一的捕获点位置
        cp_x = capture_point[0].item()
        cp_y = capture_point[1].item()
        ax.plot(cp_y, cp_x, 'r*', markersize=15, label='Capture Point')
    
    def add_velocity_arrow(ax, vel):
        """标记速度方向"""
        v_x, v_y = vel[0].item(), vel[1].item()
        v_norm = np.sqrt(v_x**2 + v_y**2)
        if v_norm > 0.05:
            # 箭头从原点指向速度方向
            ax.annotate('', xy=(v_y * 0.3, v_x * 0.3), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    # 使用 origin='lower' 让 X 轴向上为正（图示上方 = 机器人前方）
    imshow_kwargs = dict(extent=extent, aspect='auto', origin='lower')
    
    # ===== Row 1: 基础信息 =====
    
    # 1. Height Map (LeggedGym 语义: 正值=坑, 负值=凸起)
    ax = axes[0, 0]
    im = ax.imshow(height_map[sample_idx].detach().cpu().numpy(), **imshow_kwargs, cmap='terrain_r')
    ax.set_title('Height Map\n(+:pit, -:bump)')
    ax.set_xlabel('Y (m) ← Left | Right →')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='Height diff (m)')
    
    # 2. Slope (归一化坡度)
    ax = axes[0, 1]
    slope = debug_info['slope'][sample_idx].detach().cpu().numpy()
    im = ax.imshow(slope, **imshow_kwargs, cmap='hot')
    ax.set_title(f'Slope (tan θ)\n(for S_geo, no threshold)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='tan(θ)')
    
    # 3. Local Variance (局部方差)
    ax = axes[0, 2]
    local_var = debug_info['local_var'][sample_idx].detach().cpu().numpy()
    im = ax.imshow(local_var, **imshow_kwargs, cmap='hot')
    ax.set_title(f'Local Variance\nthreshold=0.03')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='Variance')
    
    # 4. z_ratio (VHIP 高度修正系数)
    ax = axes[0, 3]
    z_ratio = debug_info['z_ratio'][sample_idx].detach().cpu().numpy()
    im = ax.imshow(z_ratio, **imshow_kwargs, cmap='RdYlBu_r', vmin=0.5, vmax=1.5)
    ax.set_title('z_ratio (VHIP Height Correction)\n>1: pit (harder), <1: bump (easier)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='z_ratio')
    
    # ===== Row 2: 分数和掩码 =====
    
    # 5. S_geo (几何分数)
    ax = axes[1, 0]
    S_geo = debug_info['S_geo'][sample_idx].detach().cpu().numpy()
    im = ax.imshow(S_geo, **imshow_kwargs, cmap='RdYlGn')
    ax.set_title(f'S_geo (Geometric Score)\nσ_geo={debug_info["sigma_geo"]:.3f}')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='Score')
    
    # 6. S_dyn (动力学分数)
    ax = axes[1, 1]
    S_dyn = debug_info['S_dyn'][sample_idx].detach().cpu().numpy()
    im = ax.imshow(S_dyn, **imshow_kwargs, cmap='RdYlGn')
    ax.set_title(f'S_dyn (Dynamic Score)\nσ_dyn={debug_info["sigma_dyn"]:.3f}')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    add_capture_point(ax, debug_info['capture_point'][sample_idx])
    add_velocity_arrow(ax, base_lin_vel[sample_idx])
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='Score')
    
    # 7. Steep Mask (只用局部方差检测边缘)
    ax = axes[1, 2]
    steep_mask = debug_info['steep_mask'][sample_idx].detach().cpu().numpy().astype(float)
    im = ax.imshow(steep_mask, **imshow_kwargs, cmap='Reds')
    ax.set_title('Edge Mask\n(local_var > threshold)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='Masked')
    
    # 8. Behind Mask (身后区域 - 点积半平面)
    ax = axes[1, 3]
    behind_mask = debug_info['behind_mask'][sample_idx].detach().cpu().numpy().astype(float)
    im = ax.imshow(behind_mask, **imshow_kwargs, cmap='Blues')
    ax.set_title('Behind Mask\n(dot product half-plane)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    add_velocity_arrow(ax, base_lin_vel[sample_idx])
    plt.colorbar(im, ax=ax, label='Masked')
    
    # ===== Row 3: 最终结果 =====
    
    # 9. Pit Mask (深坑)
    ax = axes[2, 0]
    pit_mask = debug_info['pit_mask'][sample_idx].detach().cpu().numpy().astype(float)
    im = ax.imshow(pit_mask, **imshow_kwargs, cmap='Purples')
    ax.set_title('Pit Mask\n(height > drop_threshold)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='Masked')
    
    # 10. Combined Mask Types (可视化不同掩码的叠加)
    ax = axes[2, 1]
    steep = debug_info['steep_mask'][sample_idx].detach().cpu().numpy().astype(float) * 1
    pit = debug_info['pit_mask'][sample_idx].detach().cpu().numpy().astype(float) * 2
    behind = debug_info['behind_mask'][sample_idx].detach().cpu().numpy().astype(float) * 3
    combined = steep + pit + behind
    im = ax.imshow(combined, **imshow_kwargs, cmap='tab10', vmin=0, vmax=4)
    ax.set_title('Mask Types (overlapping)\n0:Safe, 1:Steep, 2:Pit, 3:Behind')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='Mask Type')
    
    # 11. Final Bias (2D)
    ax = axes[2, 2]
    bias_2d = debug_info['bias_2d'][sample_idx].detach().cpu().numpy()
    im = ax.imshow(bias_2d, **imshow_kwargs, cmap='RdYlGn')
    ax.set_title('Final Bias (with soft penalties)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    add_capture_point(ax, debug_info['capture_point'][sample_idx])
    add_velocity_arrow(ax, base_lin_vel[sample_idx])
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='Bias')
    
    # 12. Softmax Attention (bias-only, 仅展示偏置的相对分布)
    ax = axes[2, 3]
    bias_flat = bias_2d.flatten()
    attention = np.exp(bias_flat - bias_flat.max())  # 数值稳定的 softmax
    attention = attention / attention.sum()
    attention_2d = attention.reshape(bias_2d.shape)
    im = ax.imshow(attention_2d, **imshow_kwargs, cmap='hot')
    ax.set_title('Softmax Attention (bias-only)\nfinal = softmax(QK^T/√d + bias)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    add_capture_point(ax, debug_info['capture_point'][sample_idx])
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def test_dimension_alignment(scorer: TerrainSafetyScorer):
    """测试维度对齐"""
    print("\n" + "="*60)
    print("Test: Dimension Alignment")
    print("="*60)
    
    batch_sizes = [1, 4, 16, 128]
    
    for B in batch_sizes:
        height_map = torch.randn(B, 17, 11)
        base_lin_vel = torch.randn(B, 3)
        
        bias, debug_info = scorer(height_map, base_lin_vel, return_debug_info=True)
        
        # 验证维度
        assert bias.shape == (B, 187), f"bias shape mismatch: {bias.shape}"
        assert debug_info['S_geo'].shape == (B, 17, 11), f"S_geo shape mismatch"
        assert debug_info['S_dyn'].shape == (B, 17, 11), f"S_dyn shape mismatch"
        assert debug_info['slope'].shape == (B, 17, 11), f"slope shape mismatch"
        assert debug_info['local_var'].shape == (B, 17, 11), f"local_var shape mismatch"
        assert debug_info['omega'].shape == (B, 17, 11), f"omega shape mismatch"
        assert debug_info['capture_point'].shape == (B, 2), f"capture_point shape mismatch: {debug_info['capture_point'].shape}"
        assert debug_info['z_ratio'].shape == (B, 17, 11), f"z_ratio shape mismatch"
        
        print(f"  Batch size {B}: ✓ All dimensions correct")
    
    print("  [PASS] Dimension alignment test passed!")


def test_device_handling(scorer: TerrainSafetyScorer):
    """测试 Device 管理"""
    print("\n" + "="*60)
    print("Test: Device Handling")
    print("="*60)
    
    # CPU 测试
    height_map_cpu = torch.randn(2, 17, 11)
    vel_cpu = torch.randn(2, 3)
    
    bias_cpu = scorer(height_map_cpu, vel_cpu)
    assert bias_cpu.device.type == 'cpu', "CPU output should be on CPU"
    print("  CPU test: ✓")
    
    # GPU 测试 (如果可用)
    if torch.cuda.is_available():
        scorer_gpu = scorer.cuda()
        height_map_gpu = height_map_cpu.cuda()
        vel_gpu = vel_cpu.cuda()
        
        bias_gpu = scorer_gpu(height_map_gpu, vel_gpu)
        assert bias_gpu.device.type == 'cuda', "GPU output should be on GPU"
        print("  GPU test: ✓")
        
        # 移回 CPU
        scorer.cpu()
    else:
        print("  GPU test: skipped (CUDA not available)")
    
    print("  [PASS] Device handling test passed!")


def test_pit_detection(scorer: TerrainSafetyScorer):
    """测试深坑检测（软掩码）"""
    print("\n" + "="*60)
    print("Test: Pit Detection (Soft Penalty)")
    print("="*60)
    
    # 创建深坑地形（LeggedGym 语义：正值 = 坑）
    height_map = create_pit_terrain(pit_depth=0.5)
    base_lin_vel = torch.zeros(1, 3)
    
    bias, debug_info = scorer(height_map, base_lin_vel, return_debug_info=True)
    
    # 检查深坑区域是否被标记（正值 > threshold = 坑）
    pit_mask = debug_info['pit_mask']
    pit_region = height_map > scorer.height_drop_threshold
    
    assert torch.all(pit_mask[pit_region]), "Pit area should be masked"
    
    # 检查深坑区域的 bias 是否应用了 pit_penalty
    bias_2d = bias.view(1, 17, 11)
    flat_region = torch.abs(height_map) < 0.1
    
    # 深坑区域的 bias 应该显著低于平坦区域
    pit_bias_mean = bias_2d[pit_region].mean().item()
    flat_bias_mean = bias_2d[flat_region].mean().item()
    
    assert pit_bias_mean < flat_bias_mean + scorer.pit_penalty, \
        f"Pit bias ({pit_bias_mean:.2f}) should be lower than flat bias ({flat_bias_mean:.2f}) by pit_penalty ({scorer.pit_penalty})"
    
    print(f"  Pit region correctly penalized: ✓ (bias diff: {flat_bias_mean - pit_bias_mean:.2f})")
    print(f"  Pit penalty: {scorer.pit_penalty}")
    print("  [PASS] Pit detection test passed!")


def test_behind_mask_dot_product(scorer: TerrainSafetyScorer):
    """测试身后掩码（点积半平面）"""
    print("\n" + "="*60)
    print("Test: Behind Mask (Dot Product Half-Plane)")
    print("="*60)
    
    height_map = create_flat_terrain()
    
    # 向前行走
    vel_forward = torch.tensor([[0.5, 0.0, 0.0]])
    _, debug_forward = scorer(height_map, vel_forward, return_debug_info=True)
    
    # 向右行走
    vel_right = torch.tensor([[0.0, 0.5, 0.0]])
    _, debug_right = scorer(height_map, vel_right, return_debug_info=True)
    
    # 斜向行走 (45度)
    vel_diagonal = torch.tensor([[0.5, 0.5, 0.0]])
    _, debug_diagonal = scorer(height_map, vel_diagonal, return_debug_info=True)
    
    behind_forward = debug_forward['behind_mask'][0]
    behind_right = debug_right['behind_mask'][0]
    behind_diagonal = debug_diagonal['behind_mask'][0]
    
    # 向前走时，身后区域应该在 x < 0 的区域
    # 使用网格坐标来验证
    grid_h, grid_w = 17, 11
    measured_points_x = torch.tensor([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
                                       0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    
    # 向前走时，x < -behind_threshold 的点应该被掩码
    x_behind_threshold = measured_points_x < -scorer.behind_threshold
    forward_expected_count = x_behind_threshold.sum().item() * grid_w
    forward_actual_count = behind_forward.sum().item()
    
    print(f"  Forward walking:")
    print(f"    Expected ~{forward_expected_count} masked points, got {forward_actual_count}")
    
    # 斜向走时，掩码应该是相对于速度方向的半平面
    diagonal_count = behind_diagonal.sum().item()
    print(f"  Diagonal walking (45°):")
    print(f"    Behind mask count: {diagonal_count}")
    
    # 验证斜向走时掩码形状是斜的（而不是轴对齐的）
    # 左下角 (x<0, y<0) 应该被掩码，右上角 (x>0, y>0) 不应该被掩码
    assert behind_diagonal[0, 0].item() == True, "Lower-left corner should be masked for diagonal movement"
    assert behind_diagonal[-1, -1].item() == False, "Upper-right corner should NOT be masked for diagonal movement"
    
    print("  Diagonal mask is correctly oriented: ✓")
    print("  [PASS] Behind mask (dot product) test passed!")


def test_capture_point_unified(scorer: TerrainSafetyScorer):
    """测试统一捕获点 (方案A)"""
    print("\n" + "="*60)
    print("Test: Unified Capture Point (Plan A)")
    print("="*60)
    
    height_map = create_flat_terrain()
    
    # 静止状态
    vel_zero = torch.zeros(1, 3)
    _, debug_zero = scorer(height_map, vel_zero, return_debug_info=True)
    
    # 向前行走
    vel_forward = torch.tensor([[0.5, 0.0, 0.0]])
    _, debug_forward = scorer(height_map, vel_forward, return_debug_info=True)
    
    # 向侧面行走
    vel_side = torch.tensor([[0.0, 0.3, 0.0]])
    _, debug_side = scorer(height_map, vel_side, return_debug_info=True)
    
    # 验证捕获点位置 (现在是统一的 [2] 向量)
    cap_zero = debug_zero['capture_point'][0]  # [2]
    cap_forward = debug_forward['capture_point'][0]  # [2]
    cap_side = debug_side['capture_point'][0]  # [2]
    
    # 静止时捕获点在原点附近
    assert torch.abs(cap_zero).max() < 0.01, "Zero velocity should have capture point at origin"
    
    # 向前行走时捕获点向前偏移 (x 方向)
    assert cap_forward[0] > 0, "Forward velocity should have positive x capture point"
    assert torch.abs(cap_forward[1]) < 0.01, "Forward velocity should have ~zero y capture point"
    
    # 向侧面行走时捕获点向侧面偏移 (y 方向)
    assert cap_side[1] > 0, "Side velocity should have positive y capture point"
    assert torch.abs(cap_side[0]) < 0.01, "Side velocity should have ~zero x capture point"
    
    # 验证 z_ratio
    z_ratio_flat = debug_forward['z_ratio'][0]
    assert torch.allclose(z_ratio_flat, torch.ones_like(z_ratio_flat), atol=0.01), \
        "Flat terrain should have z_ratio ≈ 1.0"
    
    print(f"  Alpha value: {debug_forward['alpha']:.3f}")
    print(f"  Capture point (forward): ({cap_forward[0]:.3f}, {cap_forward[1]:.3f})")
    print(f"  Capture point (side): ({cap_side[0]:.3f}, {cap_side[1]:.3f})")
    print("  Zero velocity → capture at origin: ✓")
    print("  Forward velocity → capture ahead: ✓")
    print("  Side velocity → capture to side: ✓")
    print("  Flat terrain z_ratio ≈ 1.0: ✓")
    print("  [PASS] Unified capture point test passed!")


def test_z_ratio_correction(scorer: TerrainSafetyScorer):
    """测试 VHIP 高度修正系数 z_ratio"""
    print("\n" + "="*60)
    print("Test: VHIP Height Correction (z_ratio)")
    print("="*60)
    
    # 创建深坑地形
    pit_map = create_pit_terrain(pit_depth=0.5)
    # 创建上台阶地形
    step_up_map = create_step_up_terrain(step_height=0.2)
    
    vel_forward = torch.tensor([[0.3, 0.0, 0.0]])
    
    _, debug_pit = scorer(pit_map, vel_forward, return_debug_info=True)
    _, debug_step = scorer(step_up_map, vel_forward, return_debug_info=True)
    
    z_ratio_pit = debug_pit['z_ratio'][0]
    z_ratio_step = debug_step['z_ratio'][0]
    
    # 坑区域 (height_map > 0) 应该有 z_ratio > 1
    pit_region = pit_map[0] > 0.3
    flat_region_pit = pit_map[0].abs() < 0.1
    
    z_ratio_in_pit = z_ratio_pit[pit_region].mean().item()
    z_ratio_flat_pit = z_ratio_pit[flat_region_pit].mean().item()
    
    print(f"  Pit terrain:")
    print(f"    z_ratio in pit region: {z_ratio_in_pit:.3f} (expected > 1)")
    print(f"    z_ratio in flat region: {z_ratio_flat_pit:.3f} (expected ≈ 1)")
    
    assert z_ratio_in_pit > 1.0, f"z_ratio in pit should be > 1, got {z_ratio_in_pit}"
    assert abs(z_ratio_flat_pit - 1.0) < 0.1, f"z_ratio in flat region should be ≈ 1, got {z_ratio_flat_pit}"
    
    # 上台阶区域 (height_map < 0) 应该有 z_ratio < 1
    step_region = step_up_map[0] < -0.1
    z_ratio_on_step = z_ratio_step[step_region].mean().item()
    
    print(f"  Step-up terrain:")
    print(f"    z_ratio on step region: {z_ratio_on_step:.3f} (expected < 1)")
    
    assert z_ratio_on_step < 1.0, f"z_ratio on step should be < 1, got {z_ratio_on_step}"
    
    # 验证 z_ratio 对 S_dyn 的影响
    # 坑区域的 S_dyn 应该更负（因为距离被放大）
    S_dyn_pit = debug_pit['S_dyn'][0]
    S_dyn_step = debug_step['S_dyn'][0]
    
    print(f"  S_dyn effect:")
    print(f"    S_dyn in pit region: {S_dyn_pit[pit_region].mean().item():.3f}")
    print(f"    S_dyn in flat region: {S_dyn_pit[flat_region_pit].mean().item():.3f}")
    
    print("  Pit has z_ratio > 1 (harder to reach): ✓")
    print("  Step-up has z_ratio < 1 (easier to reach): ✓")
    print("  [PASS] VHIP height correction test passed!")


def test_slope_normalization(scorer: TerrainSafetyScorer):
    """测试坡度归一化（中心差分）"""
    print("\n" + "="*60)
    print("Test: Slope Normalization (Central Difference)")
    print("="*60)
    
    # 创建已知坡度的斜坡
    # 从 x=0 到 x=1.6m，高度变化 0.32m，坡度 tan(θ) = 0.32/1.6 = 0.2
    height_map = create_slope_terrain(max_height=0.32)
    base_lin_vel = torch.zeros(1, 3)
    
    _, debug_info = scorer(height_map, base_lin_vel, return_debug_info=True)
    
    slope = debug_info['slope'][0]
    
    # 中间区域的坡度应该接近理论值 0.2
    # 边界由于 padding 可能略有偏差
    center_slope = slope[5:12, 3:8].mean().item()
    expected_slope = 0.32 / 1.6  # = 0.2
    
    print(f"  Expected slope: {expected_slope:.4f}")
    print(f"  Measured center slope: {center_slope:.4f}")
    print(f"  dx={scorer.dx:.3f}, dy={scorer.dy:.3f}")
    
    assert abs(center_slope - expected_slope) < 0.05, \
        f"Center slope {center_slope:.4f} should be close to {expected_slope:.4f}"
    
    print("  Slope normalization correct: ✓")
    print("  [PASS] Slope normalization test passed!")


def test_local_variance(scorer: TerrainSafetyScorer):
    """测试局部方差检测"""
    print("\n" + "="*60)
    print("Test: Local Variance Detection")
    print("="*60)
    
    # 创建粗糙地形
    rough_map = create_rough_terrain(roughness_scale=0.2)
    # 创建平坦地形
    flat_map = create_flat_terrain()
    
    base_lin_vel = torch.zeros(1, 3)
    
    _, debug_rough = scorer(rough_map, base_lin_vel, return_debug_info=True)
    _, debug_flat = scorer(flat_map, base_lin_vel, return_debug_info=True)
    
    rough_var = debug_rough['local_var'][0].mean().item()
    flat_var = debug_flat['local_var'][0].mean().item()
    
    print(f"  Rough terrain local variance: {rough_var:.6f}")
    print(f"  Flat terrain local variance: {flat_var:.6f}")
    
    assert rough_var > flat_var, "Rough terrain should have higher local variance"
    assert flat_var < 1e-6, "Flat terrain should have near-zero local variance"
    
    # 检查粗糙地形是否触发 steep_mask
    steep_mask_rough = debug_rough['steep_mask'][0]
    steep_mask_flat = debug_flat['steep_mask'][0]
    
    print(f"  Rough terrain steep_mask ratio: {steep_mask_rough.float().mean():.2%}")
    print(f"  Flat terrain steep_mask ratio: {steep_mask_flat.float().mean():.2%}")
    
    assert steep_mask_rough.float().mean() > steep_mask_flat.float().mean(), \
        "Rough terrain should have more steep_mask coverage"
    
    print("  Local variance detection correct: ✓")
    print("  [PASS] Local variance test passed!")


def test_soft_penalty_values(scorer: TerrainSafetyScorer):
    """测试分层软惩罚值"""
    print("\n" + "="*60)
    print("Test: Soft Penalty Values")
    print("="*60)
    
    print(f"  behind_penalty: {scorer.behind_penalty}")
    print(f"  steep_penalty: {scorer.steep_penalty}")
    print(f"  pit_penalty: {scorer.pit_penalty}")
    
    # 验证惩罚值的相对大小
    assert scorer.behind_penalty > scorer.steep_penalty, \
        "behind_penalty should be less severe than steep_penalty"
    assert scorer.steep_penalty > scorer.pit_penalty, \
        "steep_penalty should be less severe than pit_penalty"
    
    # 验证惩罚值都是负数
    assert scorer.behind_penalty < 0, "behind_penalty should be negative"
    assert scorer.steep_penalty < 0, "steep_penalty should be negative"
    assert scorer.pit_penalty < 0, "pit_penalty should be negative"
    
    # 验证惩罚值不会导致数值问题
    assert scorer.pit_penalty > -100, "pit_penalty should not be too extreme"
    
    print("  Penalty hierarchy correct: behind > steep > pit (less severe to more severe)")
    print("  All penalties are negative and reasonable: ✓")
    print("  [PASS] Soft penalty values test passed!")


def run_visual_tests(scorer: TerrainSafetyScorer, save_dir: Optional[str] = None):
    """运行可视化测试"""
    print("\n" + "="*60)
    print("Visual Tests")
    print("="*60)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    test_cases = [
        ("Flat Terrain - Stationary", create_flat_terrain(), torch.zeros(1, 3)),
        ("Flat Terrain - Forward", create_flat_terrain(), torch.tensor([[0.5, 0.0, 0.0]])),
        ("Flat Terrain - Sideways", create_flat_terrain(), torch.tensor([[0.0, 0.3, 0.0]])),
        ("Flat Terrain - Diagonal", create_flat_terrain(), torch.tensor([[0.4, 0.3, 0.0]])),
        ("Two-Step Up", create_step_up_terrain(step_height=0.1, num_steps=2), torch.tensor([[0.3, 0.0, 0.0]])),
        ("Two-Step Down", create_step_down_terrain(step_depth=0.1, num_steps=2), torch.tensor([[0.3, 0.0, 0.0]])),
        ("Deep Pit", create_pit_terrain(pit_depth=0.5), torch.tensor([[0.3, 0.0, 0.0]])),
        ("Slope", create_slope_terrain(max_height=0.3), torch.tensor([[0.3, 0.0, 0.0]])),
        ("Stepping Stones", create_stepping_stones_terrain(stone_size=3, gap_size=1, gap_depth=0.4), torch.tensor([[0.3, 0.0, 0.0]])),
        ("Gap", create_gap_terrain(gap_depth=0.8), torch.tensor([[0.3, 0.0, 0.0]])),
        ("Rough Terrain", create_rough_terrain(roughness_scale=0.15), torch.tensor([[0.3, 0.0, 0.0]])),
    ]
    
    for idx, (name, height_map, vel) in enumerate(test_cases):
        print(f"  Visualizing: {name}")
        _, debug_info = scorer(height_map, vel, return_debug_info=True)
        
        save_path = None
        if save_dir:
            safe_name = name.replace(" ", "_").replace("-", "").lower()
            save_path = os.path.join(save_dir, f"{idx:02d}_{safe_name}.png")
        
        visualize_scorer_output(
            height_map, debug_info, vel,
            title=name, save_path=save_path, show=False
        )


def main():
    """主测试函数"""
    print("="*60)
    print("TerrainSafetyScorer Test Suite")
    print("="*60)
    print("\n符号语义 (与 LeggedGym 管线一致):")
    print("  height_map = base_z - base_height - terrain_z")
    print("  正值 = 地面更低（坑）")
    print("  负值 = 地面更高（凸起/台阶）")
    
    # 创建打分器（使用更新后的参数）
    scorer = TerrainSafetyScorer(
        grid_h=17,
        grid_w=11,
        z_nominal=0.75,
        local_var_threshold=0.03,      # 局部方差阈值（用于边缘检测）
        height_drop_threshold=0.3,
        behind_threshold=0.3,
        behind_vel_threshold=0.1,
        behind_penalty=-1.5,           # 身后区域轻惩罚
        steep_penalty=-4.0,            # 边缘中等惩罚
        pit_penalty=-6.0,             # 深坑重惩罚
        learnable_sigma=True,
        init_sigma_dyn=0.3,
        init_sigma_geo=0.5,
        learnable_alpha=True,
        init_alpha=1.0,
    )
    
    print(f"\nScorer Configuration:")
    print(f"  {scorer}")
    
    # 运行单元测试
    test_dimension_alignment(scorer)
    test_device_handling(scorer)
    test_pit_detection(scorer)
    test_behind_mask_dot_product(scorer)
    test_capture_point_unified(scorer)
    test_z_ratio_correction(scorer)
    test_slope_normalization(scorer)
    test_local_variance(scorer)
    test_soft_penalty_values(scorer)
    
    print("\n" + "="*60)
    print("All Unit Tests Passed! ✓")
    print("="*60)

    # ===== 无交互：自动运行可视化并保存到 /home/nubot/ssd/my_MoRE/MoRE 下的时间戳子目录 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.abspath(os.path.join(project_root, f"terrain_safety_scorer_viz_{timestamp}"))
    print(f"\nRunning visual tests (headless) and saving figures to:\n  {save_dir}\n")
    run_visual_tests(scorer, save_dir=save_dir)


if __name__ == "__main__":
    main()
