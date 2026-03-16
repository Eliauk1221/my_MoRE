"""
训练时地形可视化工具

提供简化的可视化函数，用于在训练过程中定期保存注意力分布和 KL prior 的可视化图像。
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional


def visualize_terrain_attention(
    height_map: torch.Tensor,
    attn_weights: torch.Tensor,
    prior_dist: Optional[torch.Tensor] = None,
    base_lin_vel: Optional[torch.Tensor] = None,
    sample_idx: int = 0,
    iteration: int = 0,
    save_path: Optional[str] = None,
    terrain_type: str = "unknown",
):
    """
    可视化地形注意力和 KL prior 分布

    Args:
        height_map: [B, 17, 11]
        attn_weights: [B, 187]
        prior_dist: [B, 187] TerrainSafetyScorer 的先验分布 (可选)
        base_lin_vel: [B, 3] (可选, 用于绘制速度箭头)
        sample_idx: batch 索引
        iteration: 当前迭代
        save_path: 保存路径
        terrain_type: 地形类型名称
    """
    num_cols = 3 if prior_dist is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 5))
    if num_cols == 2:
        axes = list(axes)

    measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                          0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
    grid_h, grid_w = 17, 11

    extent = [measured_points_y[0], measured_points_y[-1],
              measured_points_x[0], measured_points_x[-1]]
    imshow_kwargs = dict(extent=extent, aspect='auto', origin='lower')

    def add_robot_marker(ax):
        ax.plot(0, 0, 'ko', markersize=8)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    def add_velocity_arrow(ax, vel):
        if vel is not None:
            v_x, v_y = vel[0].item(), vel[1].item()
            v_norm = np.sqrt(v_x**2 + v_y**2)
            if v_norm > 0.05:
                ax.annotate('', xy=(v_y * 0.3, v_x * 0.3), xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    vel = base_lin_vel[sample_idx] if base_lin_vel is not None else None
    vel_str = f"v=({vel[0].item():.2f}, {vel[1].item():.2f})" if vel is not None else ""

    fig.suptitle(f"Iter {iteration} | {terrain_type} | {vel_str}", fontsize=14)

    # 1. Height Map
    ax = axes[0]
    h = height_map[sample_idx].detach().cpu().numpy()
    im = ax.imshow(h, **imshow_kwargs, cmap='terrain_r')
    ax.set_title('Height Map\n(+:pit, -:bump)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    add_velocity_arrow(ax, vel)
    plt.colorbar(im, ax=ax, label='m')

    # 2. Attention Weights
    ax = axes[1]
    attn_2d = attn_weights[sample_idx].detach().cpu().numpy().reshape(grid_h, grid_w)
    im = ax.imshow(attn_2d, **imshow_kwargs, cmap='hot')
    ax.set_title('Attention Weights')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    add_velocity_arrow(ax, vel)
    plt.colorbar(im, ax=ax, label='Weight')

    # 3. Prior Distribution (KL target)
    if prior_dist is not None:
        ax = axes[2]
        prior_2d = prior_dist[sample_idx].detach().cpu().numpy().reshape(grid_h, grid_w)
        im = ax.imshow(prior_2d, **imshow_kwargs, cmap='RdYlGn')
        ax.set_title('Prior Distribution (KL target)')
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('X (m) ↑ Forward')
        add_robot_marker(ax)
        add_velocity_arrow(ax, vel)
        plt.colorbar(im, ax=ax, label='Prob')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')

    plt.close(fig)


def visualize_training_sample(
    env,
    actor_critic,
    iteration: int,
    save_dir: str,
    sample_indices: list = None,
    terrain_names: list = None,
):
    """
    在训练过程中可视化采样

    Args:
        env: 环境实例
        actor_critic: ActorCriticDepth 实例
        iteration: 当前迭代
        save_dir: 保存目录
        sample_indices: 要可视化的 env 索引列表
        terrain_names: 地形类型名称列表
    """
    if terrain_names is None:
        terrain_names = ['stepping_stones', 'parkour', 'pit', 'gap', 'stair']

    if sample_indices is None:
        sample_indices = []
        if hasattr(env, 'env_class'):
            for t_type in range(len(terrain_names)):
                mask = (env.env_class == t_type).nonzero(as_tuple=False)
                if len(mask) > 0:
                    sample_indices.append(mask[0, 0].item())
                    if len(sample_indices) >= 2:
                        break
        if len(sample_indices) == 0:
            sample_indices = [0]

    if not hasattr(env, 'height_map') or not hasattr(actor_critic, 'terrain_attention'):
        return

    height_map = env.height_map
    terrain_attn = actor_critic.terrain_attention
    terrain_scorer = getattr(actor_critic, 'terrain_safety_scorer', None)

    if terrain_attn is None or terrain_attn.last_attention_weights is None:
        return

    attn_weights = terrain_attn.last_attention_weights
    base_lin_vel = getattr(env, 'base_lin_vel', None)

    prior_dist = None
    if terrain_scorer is not None and base_lin_vel is not None:
        try:
            with torch.no_grad():
                prior_dist = terrain_scorer(height_map, base_lin_vel)
        except Exception:
            pass

    iter_dir = os.path.join(save_dir, f'iter_{iteration:06d}')
    os.makedirs(iter_dir, exist_ok=True)

    for idx in sample_indices:
        if idx >= height_map.shape[0]:
            continue

        terrain_type = "unknown"
        if hasattr(env, 'env_class'):
            t_idx = int(env.env_class[idx].item())
            if 0 <= t_idx < len(terrain_names):
                terrain_type = terrain_names[t_idx]

        save_path = os.path.join(iter_dir, f'env_{idx}_{terrain_type}.png')

        visualize_terrain_attention(
            height_map=height_map,
            attn_weights=attn_weights,
            prior_dist=prior_dist,
            base_lin_vel=base_lin_vel,
            sample_idx=idx,
            iteration=iteration,
            save_path=save_path,
            terrain_type=terrain_type,
        )
