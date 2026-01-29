"""
训练时地形可视化工具

提供简化的可视化函数，用于在训练过程中定期保存打分机制和注意力分布的可视化图像。
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免依赖 X server
import matplotlib.pyplot as plt
from typing import Optional, Dict


def visualize_terrain_attention(
    height_map: torch.Tensor,
    attn_weights: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    base_lin_vel: Optional[torch.Tensor] = None,
    debug_info: Optional[Dict] = None,
    sample_idx: int = 0,
    iteration: int = 0,
    beta: float = 1.0,
    save_path: Optional[str] = None,
    terrain_type: str = "unknown",
):
    """
    可视化地形注意力和打分偏置
    
    Args:
        height_map: [B, 17, 11] 高度图
        attn_weights: [B, 187] 注意力权重
        attn_bias: [B, 187] 偏置分数 (可选)
        base_lin_vel: [B, 3] 机体线速度 (可选)
        debug_info: TerrainSafetyScorer 的调试信息 (可选)
        sample_idx: 要可视化的 batch 索引
        iteration: 当前迭代次数
        beta: 当前 β 值
        save_path: 保存路径
        terrain_type: 地形类型名称
    """
    # 确定子图数量
    num_cols = 3 if attn_bias is not None else 2
    fig, axes = plt.subplots(2, num_cols, figsize=(6 * num_cols, 10))
    
    # 采样点坐标
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
    
    fig.suptitle(f"Iteration {iteration} | Terrain: {terrain_type} | β={beta:.3f} | {vel_str}", fontsize=14)
    
    # ===== Row 1: 高度图、注意力、偏置 =====
    
    # 1. Height Map
    ax = axes[0, 0]
    h = height_map[sample_idx].detach().cpu().numpy()
    im = ax.imshow(h, **imshow_kwargs, cmap='terrain_r')
    ax.set_title('Height Map\n(+:pit, -:bump)')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    add_velocity_arrow(ax, vel)
    plt.colorbar(im, ax=ax, label='m')
    
    # 2. Attention Weights
    ax = axes[0, 1]
    attn_2d = attn_weights[sample_idx].detach().cpu().numpy().reshape(grid_h, grid_w)
    im = ax.imshow(attn_2d, **imshow_kwargs, cmap='hot')
    ax.set_title('Attention Weights')
    ax.set_xlabel('Y (m)')
    ax.set_ylabel('X (m) ↑ Forward')
    add_robot_marker(ax)
    add_velocity_arrow(ax, vel)
    plt.colorbar(im, ax=ax, label='Weight')
    
    # 3. Bias (if available)
    if attn_bias is not None:
        ax = axes[0, 2]
        bias_2d = attn_bias[sample_idx].detach().cpu().numpy().reshape(grid_h, grid_w)
        im = ax.imshow(bias_2d, **imshow_kwargs, cmap='RdYlGn')
        ax.set_title('Safety Bias (β × scorer)')
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('X (m) ↑ Forward')
        add_robot_marker(ax)
        add_velocity_arrow(ax, vel)
        plt.colorbar(im, ax=ax, label='Bias')
    
    # ===== Row 2: 调试信息 (如果有) =====
    
    if debug_info is not None:
        # S_geo
        ax = axes[1, 0]
        S_geo = debug_info['S_geo'][sample_idx].detach().cpu().numpy()
        im = ax.imshow(S_geo, **imshow_kwargs, cmap='RdYlGn')
        ax.set_title(f'S_geo (σ={debug_info["sigma_geo"]:.3f})')
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('X (m)')
        add_robot_marker(ax)
        plt.colorbar(im, ax=ax)
        
        # S_dyn
        ax = axes[1, 1]
        S_dyn = debug_info['S_dyn'][sample_idx].detach().cpu().numpy()
        im = ax.imshow(S_dyn, **imshow_kwargs, cmap='RdYlGn')
        ax.set_title(f'S_dyn (σ={debug_info["sigma_dyn"]:.3f}, α={debug_info["alpha"]:.3f})')
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('X (m)')
        add_robot_marker(ax)
        add_velocity_arrow(ax, vel)
        plt.colorbar(im, ax=ax)
        
        if num_cols > 2:
            # Masks
            ax = axes[1, 2]
            steep = debug_info['steep_mask'][sample_idx].detach().cpu().numpy().astype(float) * 1
            pit = debug_info['pit_mask'][sample_idx].detach().cpu().numpy().astype(float) * 2
            behind = debug_info['behind_mask'][sample_idx].detach().cpu().numpy().astype(float) * 3
            combined = steep + pit + behind
            im = ax.imshow(combined, **imshow_kwargs, cmap='tab10', vmin=0, vmax=4)
            ax.set_title('Masks (0:OK 1:Edge 2:Pit 3:Behind)')
            ax.set_xlabel('Y (m)')
            ax.set_ylabel('X (m)')
            add_robot_marker(ax)
            plt.colorbar(im, ax=ax)
    else:
        # 如果没有 debug_info，用空白或重复显示
        for col in range(num_cols):
            ax = axes[1, col]
            ax.text(0.5, 0.5, 'No debug info', ha='center', va='center', transform=ax.transAxes)
            ax.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.close(fig)


def visualize_training_sample(
    env,
    actor_critic,
    iteration: int,
    beta: float,
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
        beta: 当前 β 值
        save_dir: 保存目录
        sample_indices: 要可视化的 env 索引列表
        terrain_names: 地形类型名称列表
    """
    if terrain_names is None:
        terrain_names = ['stepping_stones', 'parkour', 'pit', 'gap', 'stair']
    
    # 如果没有指定索引，尝试从每种地形类型中选一个
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
    
    # 获取数据
    if not hasattr(env, 'height_map') or not hasattr(actor_critic, 'terrain_attention'):
        return
    
    height_map = env.height_map
    terrain_attn = actor_critic.terrain_attention
    terrain_scorer = getattr(actor_critic, 'terrain_safety_scorer', None)
    
    if terrain_attn is None or terrain_attn.last_attention_weights is None:
        return
    
    attn_weights = terrain_attn.last_attention_weights
    attn_bias = getattr(terrain_attn, 'last_attn_bias', None)
    
    # 获取速度
    base_lin_vel = getattr(env, 'base_lin_vel', None)
    
    # 获取调试信息 (如果有 scorer)
    debug_info = None
    if terrain_scorer is not None and base_lin_vel is not None:
        try:
            with torch.no_grad():
                _, debug_info = terrain_scorer(height_map, base_lin_vel, return_debug_info=True)
        except:
            pass
    
    # 保存目录
    iter_dir = os.path.join(save_dir, f'iter_{iteration:06d}')
    os.makedirs(iter_dir, exist_ok=True)
    
    # 为每个采样可视化
    for idx in sample_indices:
        if idx >= height_map.shape[0]:
            continue
        
        # 获取地形类型
        terrain_type = "unknown"
        if hasattr(env, 'env_class'):
            t_idx = int(env.env_class[idx].item())
            if 0 <= t_idx < len(terrain_names):
                terrain_type = terrain_names[t_idx]
        
        save_path = os.path.join(iter_dir, f'env_{idx}_{terrain_type}.png')
        
        visualize_terrain_attention(
            height_map=height_map,
            attn_weights=attn_weights,
            attn_bias=attn_bias,
            base_lin_vel=base_lin_vel,
            debug_info=debug_info,
            sample_idx=idx,
            iteration=iteration,
            beta=beta,
            save_path=save_path,
            terrain_type=terrain_type,
        )

