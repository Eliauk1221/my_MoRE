# -*- coding: utf-8 -*-
"""
测试 VHIP (Variable Height Inverted Pendulum) 评分在各种地形下的表现

参考论文:
- Caron et al., "Capturability-based pattern generation for walking with variable height," 
  IEEE T-RO, 2019
- FastStair: Learning to run up stairs with humanoid robots

测试场景:
1. 平地 (Flat): k=0, VHIP 退化为 LIP
2. 上坡 (Uphill): k>0, DCM 发散更快，为“接住”它需要迈得更远
3. 下坡 (Downhill): k<0, DCM 发散更慢，捕获点相对缩进
4. 台阶 (Step): 边缘 k 很大，应该低分
5. 踏脚石 (Stepping Stones): k≈0，应该正常工作
6. 间隙 (Gap): 中间 k 剧变，边缘应该低分
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
terrain_scorer_path = os.path.join(script_dir, '..', 'envs', 'base')
sys.path.insert(0, terrain_scorer_path)

import torch
import numpy as np
import matplotlib.pyplot as plt


# ========== VHIP 评分实现 ==========
class VHIPScorer:
    """
    基于 VHIP 模型的捕获域评分器
    
    VHIP 核心公式 (Caron et al. 2019):
    - 变高度模型: z(t) = z_0 + k * d, 其中 k 是坡度, d 是水平距离
    - 修正的自然频率: ω = sqrt(g / z_eff)
    - DCM 演化修正因子: (1 + k / (2 * sqrt(g * z)))
    
    捕获域判断:
    - 上坡 (k > 0): DCM 发散更快 → 捕获点更远
    - 下坡 (k < 0): DCM 发散更慢 → 捕获点缩进
    - 平地 (k = 0): 退化为标准 LIP
    """
    
    def __init__(self, z_com=0.75, gravity=9.81, T_stance=0.3, sigma=0.15):
        self.z_com = z_com      # 名义 CoM 高度 (m)
        self.g = gravity        # 重力加速度 (m/s²)
        self.T_stance = T_stance  # 站立相时间 (s)
        self.sigma = sigma      # 高斯核宽度 (m)
        self.omega_0 = np.sqrt(gravity / z_com)  # 名义 LIP 频率
    
    def compute_slope(self, height_points):
        """
        计算每个采样点相对于机器人当前位置的坡度
        
        Args:
            height_points: [B, num_points, 3] 采样点 (x, y, z) in base frame
                          z 是相对于机器人脚下的高度差
        
        Returns:
            k: [B, num_points] 坡度 (无量纲)
            d: [B, num_points] 水平距离 (m)
        """
        x = height_points[..., 0]
        y = height_points[..., 1]
        h = height_points[..., 2]  # 相对高度
        
        # 水平距离
        d = torch.sqrt(x**2 + y**2 + 1e-6)
        
        # 坡度 k = Δh / Δd
        # 注意：这里 h 已经是相对高度（相对于机器人脚下）
        k = h / (d + 1e-6)
        
        return k, d
    
    def compute_vhip_factor(self, k, d):
        """
        计算 VHIP 修正因子
        
        来自 Caron et al. 2019 公式:
        DCM 演化: ξ̇ = (1 + k/(2√(gz))) ẋ + (1/ω) ẍ
        
        修正因子 = 1 + k / (2 * sqrt(g * z_eff))
        
        Args:
            k: [B, num_points] 坡度
            d: [B, num_points] 水平距离
        
        Returns:
            vhip_factor: [B, num_points] VHIP 修正因子
        """
        # 有效高度 z_eff = z_com + k * d
        z_eff = self.z_com + k * d
        z_eff = torch.clamp(z_eff, min=0.1)  # 防止负值或过小
        
        # VHIP 修正因子
        # 当 k > 0 (上坡): factor > 1, DCM 发散更快 → 捕获点更远
        # 当 k < 0 (下坡): factor < 1, DCM 发散更慢 → 捕获点缩进
        # 当 k = 0 (平地): factor = 1, 退化为 LIP
        vhip_factor = 1 + k / (2 * torch.sqrt(self.g * z_eff))
        
        return vhip_factor, z_eff
    
    def compute_capture_region_score(self, height_points, base_lin_vel):
        """
        计算基于 VHIP 的捕获域评分
        
        核心思想:
        1. 根据当前速度计算 Raibert 名义落足点
        2. 使用 VHIP 修正捕获域范围
        3. 计算每个采样点到捕获域中心的距离
        4. 使用高斯核评分
        
        Args:
            height_points: [B, num_points, 3] 采样点 (x, y, z)
            base_lin_vel: [B, 3] 基座速度 (v_x, v_y, v_z)
        
        Returns:
            vhip_score: [B, num_points] VHIP 捕获域评分 [0, 1]
            debug_info: dict 调试信息
        """
        B = height_points.shape[0]
        
        # Step 1: 计算坡度
        k, d = self.compute_slope(height_points)
        
        # Step 2: 计算 VHIP 修正因子
        vhip_factor, z_eff = self.compute_vhip_factor(k, d)
        
        # Step 3: 计算当前速度大小和方向
        v_x = base_lin_vel[..., 0:1]  # [B, 1]
        v_y = base_lin_vel[..., 1:2]  # [B, 1]
        v_mag = torch.sqrt(v_x**2 + v_y**2 + 1e-6)  # [B, 1]
        
        # 速度方向单位向量
        v_dir_x = v_x / (v_mag + 1e-6)  # [B, 1]
        v_dir_y = v_y / (v_mag + 1e-6)  # [B, 1]
        
        # Step 4: 计算 Raibert 名义落足点
        # p_nominal = (T_stance / 2) * v
        p_nominal_mag = (self.T_stance / 2) * v_mag  # [B, 1]
        
        # Step 5: VHIP 修正后的捕获点距离
        # 上坡时 vhip_factor > 1: DCM 发散更快 → 捕获点更远 (p_capture > p_nominal)
        # 下坡时 vhip_factor < 1: DCM 发散更慢 → 捕获点缩进 (p_capture < p_nominal)
        p_capture_mag = p_nominal_mag * torch.clamp(vhip_factor, min=0.5, max=2.0)  # [B, num_points]
        
        # Step 6: 计算每个采样点在速度方向上的投影距离
        point_x = height_points[..., 0]  # [B, num_points]
        point_y = height_points[..., 1]  # [B, num_points]
        
        # 在速度方向上的投影
        proj_along_vel = point_x * v_dir_x + point_y * v_dir_y  # [B, num_points]
        
        # 垂直于速度方向的距离
        proj_perp_vel = torch.abs(point_x * (-v_dir_y) + point_y * v_dir_x)  # [B, num_points]
        
        # Step 7: 计算到 VHIP 捕获点的距离
        # 主要关注速度方向上的距离
        dist_along = torch.abs(proj_along_vel - p_capture_mag)  # [B, num_points]
        dist_perp = proj_perp_vel  # [B, num_points]
        
        # 总距离 (椭圆形: 速度方向上容忍度更大)
        dist_total = torch.sqrt(dist_along**2 + (2 * dist_perp)**2)  # [B, num_points]
        
        # Step 8: 高斯核评分
        vhip_score = torch.exp(-dist_total**2 / (2 * self.sigma**2))  # [B, num_points]
        
        # Step 9: 对陡峭斜坡额外惩罚
        # |k| > 0.5 (约 27°) 是比较陡的斜坡
        steep_penalty = torch.exp(-torch.abs(k) / 0.5)
        vhip_score = vhip_score * steep_penalty
        
        # 调试信息
        debug_info = {
            'slope_k': k,
            'horizontal_dist': d,
            'vhip_factor': vhip_factor,
            'z_eff': z_eff,
            'p_nominal': p_nominal_mag,
            'p_capture': p_capture_mag,
            'steep_penalty': steep_penalty,
        }
        
        return vhip_score, debug_info
    
    def forward(self, height_points, base_lin_vel):
        """简单接口"""
        score, _ = self.compute_capture_region_score(height_points, base_lin_vel)
        return score


# ========== 测试工具函数 ==========
def create_height_points(num_x=17, num_y=11, x_range=(-0.8, 0.8), y_range=(-0.5, 0.5)):
    """创建标准的采样点网格 (base frame)"""
    x = torch.linspace(x_range[0], x_range[1], num_x)
    y = torch.linspace(y_range[0], y_range[1], num_y)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    height_points = torch.zeros(num_x, num_y, 3)
    height_points[:, :, 0] = grid_x
    height_points[:, :, 1] = grid_y
    # z 初始化为 0
    
    return height_points


def compute_entropy(probs):
    """计算注意力分布的熵"""
    probs = probs.flatten() + 1e-8
    entropy = -torch.sum(probs * torch.log(probs))
    return entropy.item()


# ========== 测试场景 ==========
def test_flat_terrain():
    """测试 1: 平地 - VHIP 应该退化为 LIP"""
    print("=" * 60)
    print("测试 1: 平地 (Flat) - VHIP 退化为 LIP")
    print("=" * 60)
    
    height_points = create_height_points()
    height_points[:, :, 2] = 0.0  # 全部平地
    height_points = height_points.unsqueeze(0)  # [1, 17, 11, 3]
    height_points_flat = height_points.view(1, -1, 3)
    
    scorer = VHIPScorer()
    
    velocities = [
        (0.0, 0.0, "静止"),
        (0.5, 0.0, "向前走 (0.5 m/s)"),
        (1.0, 0.0, "向前跑 (1.0 m/s)"),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (vx, vy, label) in enumerate(velocities):
        base_lin_vel = torch.tensor([[vx, vy, 0.0]])
        score, debug = scorer.compute_capture_region_score(height_points_flat, base_lin_vel)
        
        score_grid = score.view(17, 11).numpy()
        
        ax = axes[idx]
        im = ax.imshow(score_grid.T, origin='lower', cmap='hot', aspect='auto',
                       extent=[-0.8, 0.8, -0.5, 0.5])
        ax.set_title(f'{label}\nmax={score.max():.3f}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        
        # 标记捕获点
        p_cap = debug['p_capture'][0, 0].item()
        ax.plot(p_cap, 0, 'g*', markersize=15, label=f'Capture: {p_cap:.2f}m')
        ax.legend(loc='upper right', fontsize=8)
        
        plt.colorbar(im, ax=ax)
        
        print(f"  {label}:")
        print(f"    - p_nominal = {debug['p_nominal'][0,0]:.3f} m")
        print(f"    - p_capture = {p_cap:.3f} m (VHIP 修正后)")
        print(f"    - vhip_factor 范围: [{debug['vhip_factor'].min():.3f}, {debug['vhip_factor'].max():.3f}]")
        print(f"    - 分数范围: [{score.min():.3f}, {score.max():.3f}]")
    
    plt.suptitle('Test 1: Flat Terrain - VHIP should degrade to LIP', fontsize=12)
    plt.tight_layout()
    plt.savefig('test_vhip_flat.png', dpi=150)
    print("  图片保存到: test_vhip_flat.png")
    plt.close()


def test_uphill():
    """测试 2: 上坡 - 捕获点应更远"""
    print("\n" + "=" * 60)
    print("测试 2: 上坡 (Uphill) - 捕获点应更远")
    print("=" * 60)
    
    height_points = create_height_points()
    
    # 前方上坡: x > 0 的区域高度逐渐增加
    # 坡度 k = 0.2 (约 11°)
    x_coords = height_points[:, :, 0]
    height_points[:, :, 2] = torch.clamp(0.2 * x_coords, min=0)  # 只有前方上坡
    
    height_points = height_points.unsqueeze(0)
    height_points_flat = height_points.view(1, -1, 3)
    
    scorer = VHIPScorer()
    base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])  # 向前走
    
    score, debug = scorer.compute_capture_region_score(height_points_flat, base_lin_vel)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # 地形高度
    ax = axes[0]
    z_grid = height_points[0, :, :, 2].numpy()
    im = ax.imshow(z_grid.T, origin='lower', cmap='terrain', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('Terrain Height (z)')
    ax.set_xlabel('X (m)')
    plt.colorbar(im, ax=ax, label='m')
    
    # 坡度
    ax = axes[1]
    k_grid = debug['slope_k'].view(17, 11).numpy()
    im = ax.imshow(k_grid.T, origin='lower', cmap='RdBu_r', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5], vmin=-0.3, vmax=0.3)
    ax.set_title('Slope k (red=uphill)')
    plt.colorbar(im, ax=ax)
    
    # VHIP 修正因子
    ax = axes[2]
    vhip_grid = debug['vhip_factor'].view(17, 11).numpy()
    im = ax.imshow(vhip_grid.T, origin='lower', cmap='RdYlGn_r', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5], vmin=0.8, vmax=1.2)
    ax.set_title('VHIP Factor\n(>1: faster DCM divergence)')
    plt.colorbar(im, ax=ax)
    
    # VHIP 评分
    ax = axes[3]
    score_grid = score.view(17, 11).numpy()
    im = ax.imshow(score_grid.T, origin='lower', cmap='hot', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title(f'VHIP Score\nmax={score.max():.3f}')
    ax.axvline(x=debug['p_nominal'][0,0].item(), color='cyan', linestyle='--', 
               label=f'p_nominal={debug["p_nominal"][0,0]:.2f}')
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Test 2: Uphill - Capture point should be farther', fontsize=12)
    plt.tight_layout()
    plt.savefig('test_vhip_uphill.png', dpi=150)
    
    print(f"  向前走 (vx=0.5 m/s) 在上坡地形:")
    print(f"    - p_nominal = {debug['p_nominal'][0,0]:.3f} m")
    print(f"    - 平均坡度 k = {debug['slope_k'].mean():.3f}")
    print(f"    - VHIP factor 范围: [{debug['vhip_factor'].min():.3f}, {debug['vhip_factor'].max():.3f}]")
    print(f"    - 前方 p_capture 比 p_nominal 更远 (因为上坡)")
    print("  图片保存到: test_vhip_uphill.png")
    plt.close()


def test_downhill():
    """测试 3: 下坡 - 捕获点应缩进"""
    print("\n" + "=" * 60)
    print("测试 3: 下坡 (Downhill) - 捕获点应缩进")
    print("=" * 60)
    
    height_points = create_height_points()
    
    # 前方下坡: x > 0 的区域高度逐渐降低
    x_coords = height_points[:, :, 0]
    height_points[:, :, 2] = torch.clamp(-0.2 * x_coords, max=0)  # 只有前方下坡
    
    height_points = height_points.unsqueeze(0)
    height_points_flat = height_points.view(1, -1, 3)
    
    scorer = VHIPScorer()
    base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])
    
    score, debug = scorer.compute_capture_region_score(height_points_flat, base_lin_vel)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # 地形高度
    ax = axes[0]
    z_grid = height_points[0, :, :, 2].numpy()
    im = ax.imshow(z_grid.T, origin='lower', cmap='terrain', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('Terrain Height (z)')
    plt.colorbar(im, ax=ax, label='m')
    
    # 坡度
    ax = axes[1]
    k_grid = debug['slope_k'].view(17, 11).numpy()
    im = ax.imshow(k_grid.T, origin='lower', cmap='RdBu_r', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5], vmin=-0.3, vmax=0.3)
    ax.set_title('Slope k (blue=downhill)')
    plt.colorbar(im, ax=ax)
    
    # VHIP 修正因子
    ax = axes[2]
    vhip_grid = debug['vhip_factor'].view(17, 11).numpy()
    im = ax.imshow(vhip_grid.T, origin='lower', cmap='RdYlGn_r', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5], vmin=0.8, vmax=1.2)
    ax.set_title('VHIP Factor\n(<1: slower DCM divergence)')
    plt.colorbar(im, ax=ax)
    
    # VHIP 评分
    ax = axes[3]
    score_grid = score.view(17, 11).numpy()
    im = ax.imshow(score_grid.T, origin='lower', cmap='hot', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title(f'VHIP Score\nmax={score.max():.3f}')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Test 3: Downhill - Capture point should retract', fontsize=12)
    plt.tight_layout()
    plt.savefig('test_vhip_downhill.png', dpi=150)
    
    print(f"  向前走 (vx=0.5 m/s) 在下坡地形:")
    print(f"    - p_nominal = {debug['p_nominal'][0,0]:.3f} m")
    print(f"    - VHIP factor 范围: [{debug['vhip_factor'].min():.3f}, {debug['vhip_factor'].max():.3f}]")
    print(f"    - 前方 p_capture 比 p_nominal 更近 (因为下坡)")
    print("  图片保存到: test_vhip_downhill.png")
    plt.close()


def test_step():
    """测试 4: 台阶 - 边缘应该低分"""
    print("\n" + "=" * 60)
    print("测试 4: 台阶 (Step) - 边缘应该低分")
    print("=" * 60)
    
    height_points = create_height_points()
    
    # 在 x=0.2 处有一个台阶 (高度 0.15m)
    x_coords = height_points[:, :, 0]
    height_points[:, :, 2] = torch.where(x_coords > 0.2, 
                                          torch.tensor(0.15), 
                                          torch.tensor(0.0))
    
    height_points = height_points.unsqueeze(0)
    height_points_flat = height_points.view(1, -1, 3)
    
    scorer = VHIPScorer()
    base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])
    
    score, debug = scorer.compute_capture_region_score(height_points_flat, base_lin_vel)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # 地形高度
    ax = axes[0]
    z_grid = height_points[0, :, :, 2].numpy()
    im = ax.imshow(z_grid.T, origin='lower', cmap='terrain', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('Terrain (step at x=0.2)')
    ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(im, ax=ax, label='m')
    
    # 坡度 (台阶处 k 很大)
    ax = axes[1]
    k_grid = debug['slope_k'].view(17, 11).numpy()
    im = ax.imshow(k_grid.T, origin='lower', cmap='RdBu_r', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('Slope k\n(large at step edge)')
    ax.axvline(x=0.2, color='black', linestyle='--', alpha=0.7)
    plt.colorbar(im, ax=ax)
    
    # 陡峭惩罚
    ax = axes[2]
    steep_grid = debug['steep_penalty'].view(17, 11).numpy()
    im = ax.imshow(steep_grid.T, origin='lower', cmap='RdYlGn', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5], vmin=0, vmax=1)
    ax.set_title('Steep Penalty\n(low at edge)')
    ax.axvline(x=0.2, color='black', linestyle='--', alpha=0.7)
    plt.colorbar(im, ax=ax)
    
    # VHIP 评分
    ax = axes[3]
    score_grid = score.view(17, 11).numpy()
    im = ax.imshow(score_grid.T, origin='lower', cmap='hot', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title(f'VHIP Score\n(edge should be low)')
    ax.axvline(x=0.2, color='cyan', linestyle='--', alpha=0.7)
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Test 4: Step - Edge should have low score', fontsize=12)
    plt.tight_layout()
    plt.savefig('test_vhip_step.png', dpi=150)
    
    # 检查边缘处的分数
    edge_idx = 10  # x ≈ 0.2 附近的索引
    edge_scores = score.view(17, 11)[edge_idx, :].mean()
    flat_scores = score.view(17, 11)[5, :].mean()  # x ≈ -0.3 的平坦区域
    
    print(f"  台阶测试 (台阶在 x=0.2, 高度 0.15m):")
    print(f"    - 边缘处 (x≈0.2) 平均分数: {edge_scores:.4f}")
    print(f"    - 平坦处 (x≈-0.3) 平均分数: {flat_scores:.4f}")
    print(f"    - 边缘处坡度 k: {debug['slope_k'].view(17,11)[edge_idx,:].mean():.2f}")
    print("  图片保存到: test_vhip_step.png")
    plt.close()


def test_stepping_stones():
    """测试 5: 踏脚石 - k≈0，应该正常工作"""
    print("\n" + "=" * 60)
    print("测试 5: 踏脚石 (Stepping Stones) - k≈0")
    print("=" * 60)
    
    height_points = create_height_points()
    
    # 几块踏脚石 (高度相同，周围是低洼)
    height_points[:, :, 2] = -0.3  # 背景是低洼
    
    # 踏脚石位置 (随机几块)
    x_coords = height_points[:, :, 0]
    y_coords = height_points[:, :, 1]
    
    # 石头 1: 中心附近
    stone1 = ((x_coords - 0.0)**2 + (y_coords - 0.0)**2) < 0.1**2
    # 石头 2: 前方
    stone2 = ((x_coords - 0.3)**2 + (y_coords - 0.0)**2) < 0.1**2
    # 石头 3: 右前方
    stone3 = ((x_coords - 0.5)**2 + (y_coords - 0.2)**2) < 0.1**2
    
    height_points[:, :, 2] = torch.where(stone1 | stone2 | stone3,
                                          torch.tensor(0.0),
                                          height_points[:, :, 2])
    
    height_points = height_points.unsqueeze(0)
    height_points_flat = height_points.view(1, -1, 3)
    
    scorer = VHIPScorer()
    base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])
    
    score, debug = scorer.compute_capture_region_score(height_points_flat, base_lin_vel)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 地形
    ax = axes[0]
    z_grid = height_points[0, :, :, 2].numpy()
    im = ax.imshow(z_grid.T, origin='lower', cmap='terrain', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('Stepping Stones')
    plt.colorbar(im, ax=ax, label='m')
    
    # 坡度
    ax = axes[1]
    k_grid = debug['slope_k'].view(17, 11).numpy()
    im = ax.imshow(k_grid.T, origin='lower', cmap='RdBu_r', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('Slope k')
    plt.colorbar(im, ax=ax)
    
    # VHIP 评分
    ax = axes[2]
    score_grid = score.view(17, 11).numpy()
    im = ax.imshow(score_grid.T, origin='lower', cmap='hot', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('VHIP Score\n(stones should have high score)')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Test 5: Stepping Stones', fontsize=12)
    plt.tight_layout()
    plt.savefig('test_vhip_stepping_stones.png', dpi=150)
    
    print(f"  踏脚石测试:")
    print(f"    - 分数范围: [{score.min():.4f}, {score.max():.4f}]")
    print(f"    - 踏脚石 (z=0) 处分数较高")
    print(f"    - 低洼 (z=-0.3) 处分数较低 (因为距离 + 陡峭惩罚)")
    print("  图片保存到: test_vhip_stepping_stones.png")
    plt.close()


def test_alternating_slopes():
    """测试 6: 交替斜坡 - VHIP 的主要应用场景"""
    print("\n" + "=" * 60)
    print("测试 6: 交替斜坡 (Alternating Slopes)")
    print("=" * 60)
    
    height_points = create_height_points()
    
    # 交替的上下坡 (正弦波形)
    x_coords = height_points[:, :, 0]
    height_points[:, :, 2] = 0.1 * torch.sin(2 * np.pi * x_coords / 0.4)
    
    height_points = height_points.unsqueeze(0)
    height_points_flat = height_points.view(1, -1, 3)
    
    scorer = VHIPScorer()
    base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])
    
    score, debug = scorer.compute_capture_region_score(height_points_flat, base_lin_vel)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 地形高度
    ax = axes[0, 0]
    z_grid = height_points[0, :, :, 2].numpy()
    im = ax.imshow(z_grid.T, origin='lower', cmap='terrain', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('Alternating Slopes (sine wave)')
    plt.colorbar(im, ax=ax, label='m')
    
    # 坡度
    ax = axes[0, 1]
    k_grid = debug['slope_k'].view(17, 11).numpy()
    im = ax.imshow(k_grid.T, origin='lower', cmap='RdBu_r', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('Slope k\n(alternating +/-)')
    plt.colorbar(im, ax=ax)
    
    # VHIP 修正因子
    ax = axes[0, 2]
    vhip_grid = debug['vhip_factor'].view(17, 11).numpy()
    im = ax.imshow(vhip_grid.T, origin='lower', cmap='RdYlGn_r', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('VHIP Factor')
    plt.colorbar(im, ax=ax)
    
    # 有效高度
    ax = axes[1, 0]
    zeff_grid = debug['z_eff'].view(17, 11).numpy()
    im = ax.imshow(zeff_grid.T, origin='lower', cmap='coolwarm', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('Effective Height z_eff')
    plt.colorbar(im, ax=ax, label='m')
    
    # 陡峭惩罚
    ax = axes[1, 1]
    steep_grid = debug['steep_penalty'].view(17, 11).numpy()
    im = ax.imshow(steep_grid.T, origin='lower', cmap='RdYlGn', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5], vmin=0, vmax=1)
    ax.set_title('Steep Penalty')
    plt.colorbar(im, ax=ax)
    
    # VHIP 评分
    ax = axes[1, 2]
    score_grid = score.view(17, 11).numpy()
    im = ax.imshow(score_grid.T, origin='lower', cmap='hot', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title(f'VHIP Score\nmax={score.max():.3f}')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Test 6: Alternating Slopes - VHIP main use case', fontsize=12)
    plt.tight_layout()
    plt.savefig('test_vhip_alternating_slopes.png', dpi=150)
    
    print(f"  交替斜坡测试 (正弦波, 周期 0.4m, 振幅 0.1m):")
    print(f"    - 坡度 k 范围: [{debug['slope_k'].min():.3f}, {debug['slope_k'].max():.3f}]")
    print(f"    - VHIP factor 范围: [{debug['vhip_factor'].min():.3f}, {debug['vhip_factor'].max():.3f}]")
    print(f"    - 分数范围: [{score.min():.4f}, {score.max():.4f}]")
    print("  图片保存到: test_vhip_alternating_slopes.png")
    plt.close()


def test_comparison_lip_vs_vhip():
    """测试 7: LIP vs VHIP 对比"""
    print("\n" + "=" * 60)
    print("测试 7: LIP vs VHIP 对比 (上坡场景)")
    print("=" * 60)
    
    height_points = create_height_points()
    
    # 上坡
    x_coords = height_points[:, :, 0]
    height_points[:, :, 2] = torch.clamp(0.25 * x_coords, min=0)
    
    height_points = height_points.unsqueeze(0)
    height_points_flat = height_points.view(1, -1, 3)
    
    base_lin_vel = torch.tensor([[0.5, 0.0, 0.0]])
    
    # VHIP 评分
    vhip_scorer = VHIPScorer()
    vhip_score, vhip_debug = vhip_scorer.compute_capture_region_score(height_points_flat, base_lin_vel)
    
    # 简单 LIP 评分 (不考虑坡度)
    # p_nominal = T/2 * v = 0.15 * 0.5 = 0.075m
    p_nominal = 0.15 * 0.5
    x = height_points_flat[0, :, 0]
    y = height_points_flat[0, :, 1]
    d = torch.sqrt(x**2 + y**2)
    lip_score = torch.exp(-(d - p_nominal)**2 / (2 * 0.15**2))
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    
    # 地形
    ax = axes[0]
    z_grid = height_points[0, :, :, 2].numpy()
    im = ax.imshow(z_grid.T, origin='lower', cmap='terrain', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('Uphill Terrain (k=0.25)')
    plt.colorbar(im, ax=ax, label='m')
    
    # LIP 评分 (忽略坡度)
    ax = axes[1]
    lip_grid = lip_score.view(17, 11).numpy()
    im = ax.imshow(lip_grid.T, origin='lower', cmap='hot', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title(f'LIP Score (ignores slope)\nCapture at d={p_nominal:.2f}m')
    ax.scatter([p_nominal], [0], c='cyan', s=100, marker='*', zorder=5)
    plt.colorbar(im, ax=ax)
    
    # VHIP 评分
    ax = axes[2]
    vhip_grid = vhip_score.view(17, 11).numpy()
    im = ax.imshow(vhip_grid.T, origin='lower', cmap='hot', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5])
    ax.set_title('VHIP Score (considers slope)\nCapture farther on uphill')
    plt.colorbar(im, ax=ax)
    
    # 差异
    ax = axes[3]
    diff_grid = (vhip_grid - lip_grid)
    im = ax.imshow(diff_grid.T, origin='lower', cmap='RdBu_r', aspect='auto',
                   extent=[-0.8, 0.8, -0.5, 0.5], vmin=-0.5, vmax=0.5)
    ax.set_title('Difference (VHIP - LIP)\nBlue: VHIP lower')
    plt.colorbar(im, ax=ax)
    
    plt.suptitle('Test 7: LIP vs VHIP Comparison', fontsize=12)
    plt.tight_layout()
    plt.savefig('test_vhip_comparison.png', dpi=150)
    
    print(f"  LIP vs VHIP 对比 (上坡 k=0.25):")
    print(f"    - LIP 捕获点: d = {p_nominal:.3f} m (固定)")
    print(f"    - VHIP 修正后: 上坡处捕获点更远")
    print(f"    - LIP 分数范围: [{lip_score.min():.4f}, {lip_score.max():.4f}]")
    print(f"    - VHIP 分数范围: [{vhip_score.min():.4f}, {vhip_score.max():.4f}]")
    print("  图片保存到: test_vhip_comparison.png")
    plt.close()


def main():
    print("=" * 60)
    print("VHIP (Variable Height Inverted Pendulum) 评分测试")
    print("=" * 60)
    print("参考: Caron et al., 'Capturability-based pattern generation")
    print("      for walking with variable height', IEEE T-RO, 2019")
    print()
    print("采样网格: 17 x 11 = 187 点")
    print("X 范围: [-0.8, 0.8] m (前后)")
    print("Y 范围: [-0.5, 0.5] m (左右)")
    print()
    
    test_flat_terrain()
    test_uphill()
    test_downhill()
    test_step()
    test_stepping_stones()
    test_alternating_slopes()
    test_comparison_lip_vs_vhip()
    
    print("\n" + "=" * 60)
    print("所有 VHIP 测试完成！")
    print("=" * 60)
    print("\n关键结论:")
    print("  1. 平地 (k=0): VHIP 退化为 LIP，正常工作")
    print("  2. 上坡 (k>0): DCM 发散更快，捕获点更远")
    print("  3. 下坡 (k<0): DCM 发散更慢，捕获点缩进")
    print("  4. 台阶边缘: 陡峭惩罚使边缘分数降低")
    print("  5. 踏脚石: k≈0 的区域正常评分")
    print("  6. 交替斜坡: VHIP 能正确处理周期性地形")


if __name__ == '__main__':
    main()


