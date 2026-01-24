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
    step_height: float = 0.15,
    step_position: float = 0.5  # 台阶位置 (0-1, 相对于网格)
) -> torch.Tensor:
    """创建上升台阶地形 (前方抬高)"""
    height_map = torch.zeros(batch_size, grid_h, grid_w)
    step_idx = int(grid_h * step_position)
    height_map[:, step_idx:, :] = step_height
    return height_map


def create_step_down_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    step_depth: float = 0.2,
    step_position: float = 0.5
) -> torch.Tensor:
    """创建下降台阶地形 (前方降低)"""
    height_map = torch.zeros(batch_size, grid_h, grid_w)
    step_idx = int(grid_h * step_position)
    height_map[:, step_idx:, :] = -step_depth
    return height_map


def create_pit_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    pit_depth: float = 0.5,
    pit_start: float = 0.4,
    pit_end: float = 0.7
) -> torch.Tensor:
    """创建深坑地形 (中间区域有深坑)"""
    height_map = torch.zeros(batch_size, grid_h, grid_w)
    start_idx = int(grid_h * pit_start)
    end_idx = int(grid_h * pit_end)
    height_map[:, start_idx:end_idx, :] = -pit_depth
    return height_map


def create_slope_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    max_height: float = 0.3
) -> torch.Tensor:
    """创建斜坡地形 (从后向前线性上升)"""
    height_map = torch.zeros(batch_size, grid_h, grid_w)
    # 线性斜坡
    slope = torch.linspace(0, max_height, grid_h).view(1, grid_h, 1)
    height_map = slope.expand(batch_size, grid_h, grid_w)
    return height_map


def create_rough_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    roughness: float = 0.1
) -> torch.Tensor:
    """创建粗糙地形 (随机噪声)"""
    height_map = torch.randn(batch_size, grid_h, grid_w) * roughness
    return height_map


def create_gap_terrain(
    grid_h: int = 17, 
    grid_w: int = 11, 
    batch_size: int = 1,
    gap_depth: float = 1.0,
    gap_width: int = 2,
    gap_position: float = 0.5
) -> torch.Tensor:
    """创建间隙地形 (窄而深的缝隙)"""
    height_map = torch.zeros(batch_size, grid_h, grid_w)
    center_idx = int(grid_h * gap_position)
    start_idx = max(0, center_idx - gap_width // 2)
    end_idx = min(grid_h, center_idx + gap_width // 2 + 1)
    height_map[:, start_idx:end_idx, :] = -gap_depth
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
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{title}\nVelocity: v_x={base_lin_vel[sample_idx, 0]:.2f}, "
                 f"v_y={base_lin_vel[sample_idx, 1]:.2f} m/s", fontsize=14)
    
    # 采样点坐标
    measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
                          0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    
    extent = [measured_points_y[0], measured_points_y[-1], 
              measured_points_x[-1], measured_points_x[0]]
    
    def add_robot_marker(ax):
        """在图上标记机器人位置"""
        ax.plot(0, 0, 'ko', markersize=10, label='Robot')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    def add_capture_point(ax, capture_offset):
        """标记理想捕获点"""
        # 捕获点偏移是相对于每个网格点的，我们取中心点的偏移作为示意
        center_h, center_w = capture_offset.shape[1] // 2, capture_offset.shape[2] // 2
        cp_x = capture_offset[0, center_h, center_w].item()
        cp_y = capture_offset[1, center_h, center_w].item()
        ax.plot(cp_y, cp_x, 'r*', markersize=15, label='Capture Point (center)')
    
    # 1. Height Map
    ax = axes[0, 0]
    im = ax.imshow(height_map[sample_idx].detach().cpu().numpy(), extent=extent, aspect='auto', cmap='terrain')
    ax.set_title('Height Map')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='Height (m)')
    
    # 2. S_geo (几何分数)
    ax = axes[0, 1]
    S_geo = debug_info['S_geo'][sample_idx].detach().cpu().numpy()
    im = ax.imshow(S_geo, extent=extent, aspect='auto', cmap='RdYlGn')
    ax.set_title(f'S_geo (Geometric Score)\nσ_geo={debug_info["sigma_geo"]:.3f}')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='Score')
    
    # 3. S_dyn (动力学分数)
    ax = axes[0, 2]
    S_dyn = debug_info['S_dyn'][sample_idx].detach().cpu().numpy()
    im = ax.imshow(S_dyn, extent=extent, aspect='auto', cmap='RdYlGn')
    ax.set_title(f'S_dyn (Dynamic Score)\nσ_dyn={debug_info["sigma_dyn"]:.3f}')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    add_robot_marker(ax)
    add_capture_point(ax, debug_info['capture_offset'][sample_idx])
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='Score')
    
    # 4. Roughness
    ax = axes[0, 3]
    roughness = debug_info['roughness'][sample_idx].detach().cpu().numpy()
    im = ax.imshow(roughness, extent=extent, aspect='auto', cmap='hot')
    ax.set_title('Roughness (Gradient Magnitude)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='Roughness')
    
    # 5. Omega (自然频率)
    ax = axes[1, 0]
    omega = debug_info['omega'][sample_idx].detach().cpu().numpy()
    im = ax.imshow(omega, extent=extent, aspect='auto', cmap='viridis')
    ax.set_title('ω (Natural Frequency)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='ω (rad/s)')
    
    # 6. Unsafe Mask
    ax = axes[1, 1]
    unsafe_mask = debug_info['unsafe_mask'][sample_idx].detach().cpu().numpy().astype(float)
    im = ax.imshow(unsafe_mask, extent=extent, aspect='auto', cmap='Reds')
    ax.set_title('Unsafe Mask (Combined)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='Unsafe')
    
    # 7. Individual Masks
    ax = axes[1, 2]
    steep = debug_info['steep_mask'][sample_idx].detach().cpu().numpy().astype(float) * 1
    too_low = debug_info['too_low_mask'][sample_idx].detach().cpu().numpy().astype(float) * 2
    too_high = debug_info['too_high_mask'][sample_idx].detach().cpu().numpy().astype(float) * 3
    combined = steep + too_low + too_high
    im = ax.imshow(combined, extent=extent, aspect='auto', cmap='tab10', vmin=0, vmax=4)
    ax.set_title('Mask Types\n0:Safe, 1:Steep, 2:TooLow, 3:TooHigh')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    add_robot_marker(ax)
    plt.colorbar(im, ax=ax, label='Mask Type')
    
    # 8. Final Bias (融合后)
    ax = axes[1, 3]
    bias_2d = debug_info['bias_2d'][sample_idx].detach().cpu().numpy()
    # 将 -1e9 替换为 NaN 以便可视化
    bias_vis = np.where(bias_2d < -1e8, np.nan, bias_2d)
    im = ax.imshow(bias_vis, extent=extent, aspect='auto', cmap='RdYlGn')
    ax.set_title('Final Bias (masked areas = NaN)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m)')
    add_robot_marker(ax)
    add_capture_point(ax, debug_info['capture_offset'][sample_idx])
    ax.legend(loc='upper right')
    plt.colorbar(im, ax=ax, label='Bias')
    
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
        assert debug_info['omega'].shape == (B, 17, 11), f"omega shape mismatch"
        assert debug_info['capture_offset'].shape == (B, 2, 17, 11), f"capture_offset shape mismatch"
        
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
    """测试深坑检测"""
    print("\n" + "="*60)
    print("Test: Pit Detection")
    print("="*60)
    
    # 创建深坑地形
    height_map = create_pit_terrain(pit_depth=0.5)
    base_lin_vel = torch.zeros(1, 3)
    
    bias, debug_info = scorer(height_map, base_lin_vel, return_debug_info=True)
    
    # 检查深坑区域是否被掩码
    too_low_mask = debug_info['too_low_mask']
    pit_region = height_map < -scorer.height_drop_threshold
    
    assert torch.all(too_low_mask[pit_region]), "Pit area should be masked"
    assert torch.all(bias.view(1, 17, 11)[pit_region] < -1e8), "Pit bias should be -inf"
    
    # 检查平坦区域是否安全
    flat_region = torch.abs(height_map) < 0.1
    assert not torch.all(too_low_mask[flat_region]), "Flat areas should not all be masked"
    
    print("  Pit region correctly masked: ✓")
    print("  Flat region correctly safe: ✓")
    print("  [PASS] Pit detection test passed!")


def test_capture_point_offset(scorer: TerrainSafetyScorer):
    """测试捕获点偏移"""
    print("\n" + "="*60)
    print("Test: Capture Point Offset")
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
    
    # 验证捕获点偏移方向
    cap_zero = debug_zero['capture_offset'][0]  # [2, H, W]
    cap_forward = debug_forward['capture_offset'][0]
    cap_side = debug_side['capture_offset'][0]
    
    # 静止时捕获点在原点附近
    assert torch.abs(cap_zero).mean() < 0.01, "Zero velocity should have zero capture offset"
    
    # 向前行走时捕获点向前偏移 (x 方向)
    assert cap_forward[0].mean() > 0, "Forward velocity should have positive x capture offset"
    
    # 向侧面行走时捕获点向侧面偏移 (y 方向)
    assert cap_side[1].mean() > 0, "Side velocity should have positive y capture offset"
    
    print("  Zero velocity → capture at origin: ✓")
    print("  Forward velocity → capture ahead: ✓")
    print("  Side velocity → capture to side: ✓")
    print("  [PASS] Capture point offset test passed!")


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
        ("Step Up", create_step_up_terrain(step_height=0.2), torch.tensor([[0.3, 0.0, 0.0]])),
        ("Step Down", create_step_down_terrain(step_depth=0.25), torch.tensor([[0.3, 0.0, 0.0]])),
        ("Deep Pit", create_pit_terrain(pit_depth=0.5), torch.tensor([[0.3, 0.0, 0.0]])),
        ("Slope", create_slope_terrain(max_height=0.3), torch.tensor([[0.3, 0.0, 0.0]])),
        ("Rough Terrain", create_rough_terrain(roughness=0.15), torch.tensor([[0.3, 0.0, 0.0]])),
        ("Gap", create_gap_terrain(gap_depth=0.8), torch.tensor([[0.3, 0.0, 0.0]])),
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
    
    # 创建打分器
    scorer = TerrainSafetyScorer(
        grid_h=17,
        grid_w=11,
        z_nominal=0.75,
        roughness_threshold=0.3,
        height_drop_threshold=0.3,
        height_climb_threshold=0.4,
        learnable_sigma=True,
        init_sigma_dyn=0.3,
        init_sigma_geo=0.1,
    )
    
    print(f"\nScorer Configuration:")
    print(f"  {scorer}")
    
    # 运行单元测试
    test_dimension_alignment(scorer)
    test_device_handling(scorer)
    test_pit_detection(scorer)
    test_capture_point_offset(scorer)
    
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

