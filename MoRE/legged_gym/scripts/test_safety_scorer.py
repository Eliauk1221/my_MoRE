# -*- coding: utf-8 -*-
"""
测试 TerrainSafetyScorer 在不同地形下的输出是否合理

测试场景:
1. 平地 (Flat): 所有点高度相同 → 应该均匀或偏向前方
2. 向前上坡 (Uphill): 前方点更高 → 分数应该偏向前方
3. 向前下坡 (Downhill): 前方点更低 → 分数应该偏向前方
4. 台阶边缘 (Step Edge): 某处有突然的高度变化 → 边缘点分数应该低
5. 侧向斜坡 (Side Slope): 左右高度不同 → 应该偏向较平的一侧
"""

import sys
import os

# 直接导入 terrain_safety_scorer，避免 IsaacGym 导入顺序问题
script_dir = os.path.dirname(os.path.abspath(__file__))
terrain_scorer_path = os.path.join(script_dir, '..', 'envs', 'base')
sys.path.insert(0, terrain_scorer_path)

import torch
import numpy as np
import matplotlib.pyplot as plt

# 直接从文件导入，绕过 legged_gym.envs 的 __init__.py
from terrain_safety_scorer import TerrainSafetyScorer, TerrainSafetyScorerCfg


def create_height_points(num_x=17, num_y=11, x_range=(-0.8, 0.8), y_range=(-0.5, 0.5)):
    """创建标准的采样点网格 (base frame)"""
    x = torch.linspace(x_range[0], x_range[1], num_x)
    y = torch.linspace(y_range[0], y_range[1], num_y)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    # [num_x, num_y, 3]
    height_points = torch.zeros(num_x, num_y, 3)
    height_points[:, :, 0] = grid_x
    height_points[:, :, 1] = grid_y
    # z 初始化为 0
    
    return height_points


def test_flat_terrain():
    """测试平地"""
    print("=" * 60)
    print("测试 1: 平地 (Flat Terrain)")
    print("=" * 60)
    
    height_points = create_height_points()
    height_points = height_points.unsqueeze(0)  # [1, 17, 11, 3]
    
    # 所有点高度相同 (z = 0)
    height_points[:, :, :, 2] = 0.0
    
    # 测试不同速度
    velocities = [
        (0.0, 0.0, "静止"),
        (0.5, 0.0, "向前走"),
        (0.0, 0.3, "向左走"),
        (-0.3, 0.0, "向后走"),
    ]
    
    cfg = TerrainSafetyScorerCfg()
    scorer = TerrainSafetyScorer(cfg, num_points_x=17, num_points_y=11)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (vx, vy, label) in enumerate(velocities):
        base_lin_vel = torch.tensor([[vx, vy, 0.0]])
        safety_scores, target_attention = scorer(height_points, base_lin_vel)
        
        # 重塑为网格形式可视化
        attn_grid = target_attention.view(17, 11).numpy()
        
        ax = axes[idx]
        im = ax.imshow(attn_grid.T, origin='lower', cmap='hot', aspect='auto')
        ax.set_title(f'{label} (vx={vx}, vy={vy})')
        ax.set_xlabel('X (前后)')
        ax.set_ylabel('Y (左右)')
        plt.colorbar(im, ax=ax)
        
        print(f"  {label}: max={target_attention.max():.4f}, min={target_attention.min():.6f}, "
              f"entropy={compute_entropy(target_attention):.2f}")
    
    plt.suptitle('平地测试 - 目标注意力分布', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_flat_terrain.png', dpi=150)
    print("  图片保存到: test_flat_terrain.png")
    plt.close()


def test_uphill():
    """测试向前上坡"""
    print("\n" + "=" * 60)
    print("测试 2: 向前上坡 (Uphill)")
    print("=" * 60)
    
    height_points = create_height_points()
    
    # 前方 (x > 0) 逐渐升高
    x_coords = height_points[:, :, 0]
    height_points[:, :, 2] = 0.3 * x_coords  # 斜率 0.3
    
    height_points = height_points.unsqueeze(0)  # [1, 17, 11, 3]
    
    cfg = TerrainSafetyScorerCfg()
    scorer = TerrainSafetyScorer(cfg, num_points_x=17, num_points_y=11)
    
    base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])  # 向前走
    safety_scores, target_attention = scorer(height_points, base_lin_vel)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 地形高度
    ax = axes[0]
    z_grid = height_points[0, :, :, 2].numpy()
    im = ax.imshow(z_grid.T, origin='lower', cmap='terrain', aspect='auto')
    ax.set_title('地形高度 (z)')
    ax.set_xlabel('X (前后)')
    ax.set_ylabel('Y (左右)')
    plt.colorbar(im, ax=ax)
    
    # Safety Scores
    ax = axes[1]
    score_grid = safety_scores.view(17, 11).numpy()
    im = ax.imshow(score_grid.T, origin='lower', cmap='RdYlGn', aspect='auto')
    ax.set_title('Safety Scores')
    plt.colorbar(im, ax=ax)
    
    # Target Attention
    ax = axes[2]
    attn_grid = target_attention.view(17, 11).numpy()
    im = ax.imshow(attn_grid.T, origin='lower', cmap='hot', aspect='auto')
    ax.set_title('Target Attention')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('向前上坡测试 (vx=0.5)', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_uphill.png', dpi=150)
    print(f"  Safety scores: max={safety_scores.max():.4f}, min={safety_scores.min():.6f}")
    print(f"  Target attention: max={target_attention.max():.4f}, entropy={compute_entropy(target_attention):.2f}")
    print("  图片保存到: test_uphill.png")
    plt.close()


def test_step_edge():
    """测试台阶边缘"""
    print("\n" + "=" * 60)
    print("测试 3: 台阶边缘 (Step Edge)")
    print("=" * 60)
    
    height_points = create_height_points()
    
    # 在 x=0.2 处有一个台阶 (高度差 0.15m)
    x_coords = height_points[:, :, 0]
    height_points[:, :, 2] = torch.where(x_coords > 0.2, 
                                          torch.tensor(0.15), 
                                          torch.tensor(0.0))
    
    height_points = height_points.unsqueeze(0)  # [1, 17, 11, 3]
    
    cfg = TerrainSafetyScorerCfg()
    scorer = TerrainSafetyScorer(cfg, num_points_x=17, num_points_y=11)
    
    base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])  # 向前走
    safety_scores, target_attention = scorer(height_points, base_lin_vel)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 地形高度
    ax = axes[0]
    z_grid = height_points[0, :, :, 2].numpy()
    im = ax.imshow(z_grid.T, origin='lower', cmap='terrain', aspect='auto')
    ax.set_title('地形高度 (z) - 台阶在 x≈0.2')
    ax.set_xlabel('X (前后)')
    ax.set_ylabel('Y (左右)')
    plt.colorbar(im, ax=ax)
    
    # Safety Scores
    ax = axes[1]
    score_grid = safety_scores.view(17, 11).numpy()
    im = ax.imshow(score_grid.T, origin='lower', cmap='RdYlGn', aspect='auto')
    ax.set_title('Safety Scores (边缘处应该低)')
    plt.colorbar(im, ax=ax)
    
    # Target Attention
    ax = axes[2]
    attn_grid = target_attention.view(17, 11).numpy()
    im = ax.imshow(attn_grid.T, origin='lower', cmap='hot', aspect='auto')
    ax.set_title('Target Attention')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('台阶边缘测试 (vx=0.5)', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_step_edge.png', dpi=150)
    print(f"  Safety scores: max={safety_scores.max():.4f}, min={safety_scores.min():.6f}")
    print(f"  Target attention: max={target_attention.max():.4f}, entropy={compute_entropy(target_attention):.2f}")
    print("  图片保存到: test_step_edge.png")
    plt.close()


def test_gap():
    """测试缺口/悬崖"""
    print("\n" + "=" * 60)
    print("测试 4: 缺口/悬崖 (Gap)")
    print("=" * 60)
    
    height_points = create_height_points()
    
    # 在 x=0.2~0.4 处有一个深坑
    x_coords = height_points[:, :, 0]
    mask = (x_coords > 0.2) & (x_coords < 0.4)
    height_points[:, :, 2] = torch.where(mask, 
                                          torch.tensor(-0.5),  # 深坑
                                          torch.tensor(0.0))
    
    height_points = height_points.unsqueeze(0)  # [1, 17, 11, 3]
    
    cfg = TerrainSafetyScorerCfg()
    scorer = TerrainSafetyScorer(cfg, num_points_x=17, num_points_y=11)
    
    base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])  # 向前走
    safety_scores, target_attention = scorer(height_points, base_lin_vel)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 地形高度
    ax = axes[0]
    z_grid = height_points[0, :, :, 2].numpy()
    im = ax.imshow(z_grid.T, origin='lower', cmap='terrain', aspect='auto')
    ax.set_title('地形高度 (z) - 缺口在 0.2<x<0.4')
    plt.colorbar(im, ax=ax)
    
    # Safety Scores
    ax = axes[1]
    score_grid = safety_scores.view(17, 11).numpy()
    im = ax.imshow(score_grid.T, origin='lower', cmap='RdYlGn', aspect='auto')
    ax.set_title('Safety Scores (缺口处应该低)')
    plt.colorbar(im, ax=ax)
    
    # Target Attention
    ax = axes[2]
    attn_grid = target_attention.view(17, 11).numpy()
    im = ax.imshow(attn_grid.T, origin='lower', cmap='hot', aspect='auto')
    ax.set_title('Target Attention (应避开缺口)')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('缺口测试 (vx=0.5)', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_gap.png', dpi=150)
    print(f"  Safety scores: max={safety_scores.max():.4f}, min={safety_scores.min():.6f}")
    print(f"  Target attention: max={target_attention.max():.4f}, entropy={compute_entropy(target_attention):.2f}")
    print("  图片保存到: test_gap.png")
    plt.close()


def test_side_slope():
    """测试侧向斜坡"""
    print("\n" + "=" * 60)
    print("测试 5: 侧向斜坡 (Side Slope)")
    print("=" * 60)
    
    height_points = create_height_points()
    
    # 左侧 (y > 0) 更高
    y_coords = height_points[:, :, 1]
    height_points[:, :, 2] = 0.3 * y_coords  # 侧向斜坡
    
    height_points = height_points.unsqueeze(0)  # [1, 17, 11, 3]
    
    cfg = TerrainSafetyScorerCfg()
    scorer = TerrainSafetyScorer(cfg, num_points_x=17, num_points_y=11)
    
    # 向前走
    base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])
    safety_scores, target_attention = scorer(height_points, base_lin_vel)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 地形高度
    ax = axes[0]
    z_grid = height_points[0, :, :, 2].numpy()
    im = ax.imshow(z_grid.T, origin='lower', cmap='terrain', aspect='auto')
    ax.set_title('地形高度 (z) - 左高右低')
    plt.colorbar(im, ax=ax)
    
    # Safety Scores
    ax = axes[1]
    score_grid = safety_scores.view(17, 11).numpy()
    im = ax.imshow(score_grid.T, origin='lower', cmap='RdYlGn', aspect='auto')
    ax.set_title('Safety Scores')
    plt.colorbar(im, ax=ax)
    
    # Target Attention
    ax = axes[2]
    attn_grid = target_attention.view(17, 11).numpy()
    im = ax.imshow(attn_grid.T, origin='lower', cmap='hot', aspect='auto')
    ax.set_title('Target Attention')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('侧向斜坡测试 (vx=0.5, 向前走)', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_side_slope.png', dpi=150)
    print(f"  Safety scores: max={safety_scores.max():.4f}, min={safety_scores.min():.6f}")
    print(f"  Target attention: max={target_attention.max():.4f}, entropy={compute_entropy(target_attention):.2f}")
    print("  图片保存到: test_side_slope.png")
    plt.close()


def test_component_scores():
    """测试各个分量的分数（LIP, Flatness, Heading）"""
    print("\n" + "=" * 60)
    print("测试 6: 分量分数分析 (台阶边缘)")
    print("=" * 60)
    
    height_points = create_height_points()
    
    # 台阶
    x_coords = height_points[:, :, 0]
    height_points[:, :, 2] = torch.where(x_coords > 0.2, 
                                          torch.tensor(0.15), 
                                          torch.tensor(0.0))
    
    height_points = height_points.unsqueeze(0)  # [1, 17, 11, 3]
    
    cfg = TerrainSafetyScorerCfg()
    scorer = TerrainSafetyScorer(cfg, num_points_x=17, num_points_y=11)
    
    base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])
    
    # 手动计算各个分量
    height_points_flat = height_points.view(1, -1, 3)
    
    vhip_score = scorer.compute_vhip_score(height_points_flat, base_lin_vel)
    flatness_score = scorer.compute_flatness_score(height_points)
    heading_factor = scorer.compute_heading_factor(height_points_flat, base_lin_vel)

    # ========== 额外：检查 logit 的原始量级（你关心的 scale 问题） ==========
    vhip_logit = scorer.compute_vhip_logit(height_points_flat, base_lin_vel)          # [1, N]
    flatness_logit = scorer.compute_flatness_logit(height_points)                    # [1, N]
    safety_logits_raw = vhip_logit + flatness_logit                                  # [1, N]（不含 heading mask）
    heading_mask = scorer.compute_heading_mask(height_points_flat, base_lin_vel, neg_inf=-1e9) \
        if cfg.use_heading_awareness else torch.zeros_like(safety_logits_raw)
    # 含 heading mask 的最终 logits（masked 点为 -1e9）
    safety_logits = safety_logits_raw + heading_mask

    # 打印统计（忽略 masked 点，避免 -1e9 拉爆范围）
    valid_mask = (heading_mask > -1e8)
    print("\n[Logit Scale Check]")
    print(f"  sigma_capture={float(cfg.sigma_capture):.3f}, sigma_flatness={float(cfg.sigma_flatness):.3f}, "
          f"temperature={float(cfg.temperature):.3f}, use_heading_awareness={cfg.use_heading_awareness}")
    _describe_tensor("vhip_logit", vhip_logit, mask=valid_mask)
    _describe_tensor("flatness_logit", flatness_logit, mask=valid_mask)
    _describe_tensor("safety_logits_raw", safety_logits_raw, mask=valid_mask)
    _describe_tensor("safety_logits(final, masked removed)", safety_logits, mask=valid_mask)

    # 对同一份 logits，比较不同 temperature 下的 teacher 分布尖锐度
    for temp in (0.1, 0.5):
        teacher = torch.softmax(safety_logits / temp, dim=-1)
        print(f"  teacher(T={temp:.1f}): max={teacher.max().item():.4f}, entropy={compute_entropy(teacher):.2f}")
    # ==============================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 地形高度
    ax = axes[0, 0]
    z_grid = height_points[0, :, :, 2].numpy()
    im = ax.imshow(z_grid.T, origin='lower', cmap='terrain', aspect='auto')
    ax.set_title('地形高度 (z)')
    plt.colorbar(im, ax=ax)
    
    # LIP Score
    ax = axes[0, 1]
    lip_grid = vhip_score.view(17, 11).numpy()
    im = ax.imshow(lip_grid.T, origin='lower', cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'VHIP Score (max={vhip_score.max():.3f}, min={vhip_score.min():.3f})')
    plt.colorbar(im, ax=ax)
    
    # Flatness Score
    ax = axes[0, 2]
    flat_grid = flatness_score.view(17, 11).numpy()
    im = ax.imshow(flat_grid.T, origin='lower', cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'Flatness Score (边缘应该低)\nmax={flatness_score.max():.3f}, min={flatness_score.min():.3f}')
    plt.colorbar(im, ax=ax)
    
    # Heading Factor
    ax = axes[1, 0]
    head_grid = heading_factor.view(17, 11).numpy()
    im = ax.imshow(head_grid.T, origin='lower', cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'Heading Factor (前方应该高)\nmax={heading_factor.max():.3f}')
    plt.colorbar(im, ax=ax)
    
    # Combined = LIP * Flatness
    ax = axes[1, 1]
    combined = vhip_score * flatness_score
    comb_grid = combined.view(17, 11).numpy()
    im = ax.imshow(comb_grid.T, origin='lower', cmap='RdYlGn', aspect='auto')
    ax.set_title('VHIP × Flatness')
    plt.colorbar(im, ax=ax)
    
    # Final = Combined * Heading
    ax = axes[1, 2]
    final = combined * heading_factor
    final_grid = final.view(17, 11).numpy()
    im = ax.imshow(final_grid.T, origin='lower', cmap='hot', aspect='auto')
    ax.set_title('Final Safety Score')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('分量分数分析 (台阶边缘, vx=0.5)', fontsize=14)
    plt.tight_layout()
    plt.savefig('test_component_scores.png', dpi=150)
    print(f"  VHIP Score: max={vhip_score.max():.4f}, min={vhip_score.min():.6f}")
    print(f"  Flatness Score: max={flatness_score.max():.4f}, min={flatness_score.min():.6f}")
    print(f"  Heading Factor: max={heading_factor.max():.4f}, min={heading_factor.min():.6f}")
    print("  图片保存到: test_component_scores.png")
    plt.close()


def compute_entropy(probs):
    """计算注意力分布的熵"""
    probs = probs + 1e-8
    entropy = -torch.sum(probs * torch.log(probs))
    return entropy.item()


def _describe_tensor(name: str, t: torch.Tensor, mask: torch.Tensor = None):
    """打印张量的数值范围/统计，用于检查 logits scale 是否合理。"""
    with torch.no_grad():
        if t is None:
            print(f"  {name}: None")
            return
        x = t.detach()
        if mask is not None:
            x = x[mask]
        x = x.float().cpu()
        if x.numel() == 0:
            print(f"  {name}: empty")
            return
        # percentiles for additional signal
        q = torch.quantile(x, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0]))
        std = x.std(unbiased=False).item()
        print(
            f"  {name}: "
            f"min={q[0].item():.3f}, p5={q[1].item():.3f}, p50={q[2].item():.3f}, "
            f"p95={q[3].item():.3f}, max={q[4].item():.3f}, mean={x.mean().item():.3f}, std={std:.3f}"
        )


def main():
    print("=" * 60)
    print("TerrainSafetyScorer 测试")
    print("=" * 60)
    print(f"采样网格: 17 x 11 = 187 点")
    print(f"X 范围: [-0.8, 0.8] m (前后)")
    print(f"Y 范围: [-0.5, 0.5] m (左右)")
    print()
    
    # 设置中文字体（如果可用）
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    test_flat_terrain()
    test_uphill()
    test_step_edge()
    test_gap()
    test_side_slope()
    test_component_scores()
    
    print("\n" + "=" * 60)
    print("所有测试完成！请查看生成的图片文件。")
    print("=" * 60)
    
    # 显示预期结果
    print("\n预期结果:")
    print("  1. 平地: 注意力应该根据速度方向分布（静止时均匀，前进时偏前）")
    print("  2. 上坡: 前方高度虽高，但如果平坦，分数应该不低")
    print("  3. 台阶边缘: 边缘处 Flatness 应该很低，导致该区域分数低")
    print("  4. 缺口: 缺口处和边缘处都应该分数很低")
    print("  5. 侧向斜坡: 应该偏向较平坦的区域")
    print("  6. 分量分析: 帮助诊断哪个分量计算有问题")


if __name__ == '__main__':
    main()

