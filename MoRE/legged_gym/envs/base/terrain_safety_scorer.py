# -*- coding: utf-8 -*-
"""
基于 LIP + 平坦度的地形安全评分器
参考 PLANC 论文设计，使用物理模型为每个地形采样点计算"安全分数"

核心功能：
1. LIP Score: 基于线性倒立摆的 Orbital Energy 评估动力学稳定性
2. Flatness Score: 基于局部高度梯度检测边缘/空隙
3. Heading Factor: 速度方向感知，只关注前进方向的点
4. 综合评分: combined_score = lip_score × flatness_score × heading_factor

用于：
- 注意力偏置：直接引导注意力关注安全区域
- 注意力监督：KL 散度 loss 强化学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TerrainSafetyScorerCfg:
    """TerrainSafetyScorer 配置参数"""
    
    # LIP 参数
    z0: float = 0.75                    # 名义 CoM 高度 (m)
    gravity: float = 9.81               # 重力加速度 (m/s²)
    
    # 平坦度参数
    sigma_flatness: float = 0.05        # 平坦度评分的梯度阈值
    
    # 方向感知参数
    use_heading_awareness: bool = True  # 是否启用速度方向感知
    min_vel_for_heading: float = 0.1    # 低于此速度时不使用方向过滤 (m/s)
    
    # 目标分布参数
    temperature: float = 0.1            # softmax 温度 (越小分布越尖锐)


class AttentionGuidanceCfg:
    """注意力引导机制配置参数"""
    
    use_safety_bias: bool = True        # 是否使用安全偏置
    use_attention_loss: bool = True     # 是否使用注意力监督 loss
    attention_loss_coef: float = 0.5    # 注意力 loss 权重 λ_attn
    
    # 课程式 β 衰减
    use_curriculum_decay: bool = True   # 是否启用课程式衰减
    beta_max: float = 2.0               # 初期偏置强度
    beta_min: float = 0.0               # 后期偏置强度
    decay_curriculum_levels: int = 10   # 衰减所需的课程等级数


class TerrainSafetyScorer(nn.Module):
    """
    地形安全评分器
    
    为每个地形采样点计算安全分数，用于引导注意力机制。
    
    输入:
        - height_points: [B, num_x, num_y, 3] 地形采样点 (x, y, z) in base frame
        - base_lin_vel: [B, 3] 基座线速度 (body frame)
        - z_com: CoM 高度（可选，默认使用配置中的 z0）
    
    输出:
        - safety_scores: [B, num_points] 每个点的最终安全分数（含方向感知）
        - target_attention: [B, num_points] 目标注意力分布
    """
    
    def __init__(self, cfg: TerrainSafetyScorerCfg = None, 
                 num_points_x: int = 17, num_points_y: int = 11):
        super().__init__()
        
        self.cfg = cfg if cfg is not None else TerrainSafetyScorerCfg()
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.num_points = num_points_x * num_points_y
        
        # 注册常量
        self.register_buffer('gravity', torch.tensor(self.cfg.gravity))
        self.register_buffer('z0', torch.tensor(self.cfg.z0))
        self.register_buffer('sigma_flatness', torch.tensor(self.cfg.sigma_flatness))
        self.register_buffer('temperature', torch.tensor(self.cfg.temperature))
        self.register_buffer('min_vel_for_heading', torch.tensor(self.cfg.min_vel_for_heading))
    
    def compute_lip_score(self, height_points: torch.Tensor, 
                          base_lin_vel: torch.Tensor,
                          z_com: torch.Tensor = None) -> torch.Tensor:
        """
        计算 LIP Score (基于 Orbital Energy)
        
        LIP 模型核心公式:
            λ = sqrt(g / z_eff)           # LIP 自然频率
            E = (v_x/λ)² + (v_y/λ)²       # Orbital Energy
        
        E > 0 表示有足够动量跨过支撑点，动力学稳定。
        
        Args:
            height_points: [B, num_points, 3] 采样点 (x, y, z)
            base_lin_vel: [B, 3] 基座速度 (v_x, v_y, v_z) in body frame
            z_com: [B] 或标量，CoM 高度
        
        Returns:
            lip_score: [B, num_points] 归一化到 [0, 1] 的 LIP 分数
        """
        B = height_points.shape[0]
        device = height_points.device
        
        if z_com is None:
            z_com = self.z0.expand(B)
        elif z_com.dim() == 0:
            z_com = z_com.expand(B)
        
        # 获取每个采样点的高度
        z_points = height_points[:, :, 2]  # [B, num_points]
        
        # 有效 CoM 高度 = CoM 高度 - 地形高度
        # z_points 是相对高度（已归一化），需要反归一化
        # 这里假设 z_points 已经是相对于机器人脚下的高度差
        z_eff = z_com.unsqueeze(-1) - z_points  # [B, num_points]
        z_eff = torch.clamp(z_eff, min=0.1)  # 防止除零
        
        # LIP 自然频率
        lambda_lip = torch.sqrt(self.gravity / z_eff)  # [B, num_points]
        
        # 基座速度
        v_x = base_lin_vel[:, 0:1]  # [B, 1]
        v_y = base_lin_vel[:, 1:2]  # [B, 1]
        
        # Orbital Energy
        E = (v_x / lambda_lip) ** 2 + (v_y / lambda_lip) ** 2  # [B, num_points]
        
        # 归一化到 [0, 1]
        # E 越大越稳定，使用 sigmoid 或 tanh 归一化
        lip_score = torch.tanh(E)  # [B, num_points]
        
        return lip_score
    
    def compute_flatness_score(self, height_points: torch.Tensor) -> torch.Tensor:
        """
        计算 Flatness Score (基于局部高度梯度)
        
        梯度大 → 边缘或空隙 → 低分
        梯度小 → 平坦区域 → 高分
        
        公式:
            gradient_mag = sqrt(grad_x² + grad_y²)
            flatness = exp(-gradient_mag / σ)
        
        Args:
            height_points: [B, num_x, num_y, 3] 采样点网格
        
        Returns:
            flatness_score: [B, num_points] 归一化到 [0, 1] 的平坦度分数
        """
        B = height_points.shape[0]
        num_x = self.num_points_x
        num_y = self.num_points_y
        
        # 提取高度 [B, num_x, num_y]
        z = height_points[:, :, :, 2]
        
        # 计算 x 方向梯度 (前向差分)
        grad_x = torch.zeros_like(z)
        grad_x[:, :-1, :] = z[:, 1:, :] - z[:, :-1, :]
        grad_x[:, -1, :] = grad_x[:, -2, :]  # 边界填充
        
        # 计算 y 方向梯度
        grad_y = torch.zeros_like(z)
        grad_y[:, :, :-1] = z[:, :, 1:] - z[:, :, :-1]
        grad_y[:, :, -1] = grad_y[:, :, -2]  # 边界填充
        
        # 梯度幅值
        gradient_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # [B, num_x, num_y]
        
        # 平坦度分数：梯度越小越好
        flatness = torch.exp(-gradient_mag / self.sigma_flatness)  # [B, num_x, num_y]
        
        # 展平
        flatness_score = flatness.view(B, -1)  # [B, num_points]
        
        return flatness_score
    
    def compute_heading_factor(self, height_points: torch.Tensor,
                               base_lin_vel: torch.Tensor) -> torch.Tensor:
        """
        计算速度方向因子 (Heading Factor)
        
        只关注速度方向前方（±90°扇区内）的点：
            θ = arctan2(point_y, point_x) - arctan2(v_y, v_x)
            heading_factor = ReLU(cos(θ))
        
        物理含义：
        - cos(θ) > 0: 点在速度方向前方 → 保留分数
        - cos(θ) ≤ 0: 点在速度方向后方 → 分数归零
        - 低速时禁用方向过滤，避免原地站立时注意力消失
        
        Args:
            height_points: [B, num_points, 3] 采样点 (x, y, z) in body frame
            base_lin_vel: [B, 3] 基座速度 in body frame
        
        Returns:
            heading_factor: [B, num_points] 方向因子 [0, 1]
        """
        B = height_points.shape[0]
        device = height_points.device
        
        # 速度大小
        v_x = base_lin_vel[:, 0]  # [B]
        v_y = base_lin_vel[:, 1]  # [B]
        vel_magnitude = torch.sqrt(v_x ** 2 + v_y ** 2 + 1e-8)  # [B]
        
        # 低速时禁用方向过滤
        low_speed_mask = vel_magnitude < self.min_vel_for_heading  # [B]
        
        # 速度方向角
        vel_angle = torch.atan2(v_y, v_x)  # [B]
        
        # 采样点方向角 (相对于机器人)
        point_x = height_points[:, :, 0]  # [B, num_points]
        point_y = height_points[:, :, 1]  # [B, num_points]
        point_angle = torch.atan2(point_y, point_x)  # [B, num_points]
        
        # 相对角度
        theta = point_angle - vel_angle.unsqueeze(-1)  # [B, num_points]
        
        # 方向因子：ReLU(cos(θ))
        heading_factor = F.relu(torch.cos(theta))  # [B, num_points]
        
        # 低速时设为 1（保留所有点）
        heading_factor = torch.where(
            low_speed_mask.unsqueeze(-1).expand_as(heading_factor),
            torch.ones_like(heading_factor),
            heading_factor
        )
        
        return heading_factor
    
    def forward(self, height_points: torch.Tensor,
                base_lin_vel: torch.Tensor,
                z_com: torch.Tensor = None) -> tuple:
        """
        计算综合安全分数和目标注意力分布
        
        Args:
            height_points: [B, num_x, num_y, 3] 地形采样点
            base_lin_vel: [B, 3] 基座速度
            z_com: [B] 或标量，CoM 高度（可选）
        
        Returns:
            safety_scores: [B, num_points] 最终安全分数
            target_attention: [B, num_points] 目标注意力分布
        """
        B = height_points.shape[0]
        device = height_points.device
        
        # 展平 height_points 用于 LIP 计算
        height_points_flat = height_points.view(B, -1, 3)  # [B, num_points, 3]
        
        # ========== 1. 计算 LIP Score ==========
        lip_score = self.compute_lip_score(height_points_flat, base_lin_vel, z_com)
        
        # ========== 2. 计算 Flatness Score ==========
        flatness_score = self.compute_flatness_score(height_points)
        
        # ========== 3. 计算综合分数 ==========
        combined_score = lip_score * flatness_score  # [B, num_points]
        
        # ========== 4. 应用速度方向因子 ==========
        if self.cfg.use_heading_awareness:
            heading_factor = self.compute_heading_factor(height_points_flat, base_lin_vel)
            safety_scores = combined_score * heading_factor
        else:
            safety_scores = combined_score
        
        # ========== 5. 生成目标注意力分布 ==========
        # 使用 softmax 将分数转换为概率分布
        target_attention = F.softmax(safety_scores / self.temperature, dim=-1)
        
        return safety_scores, target_attention
    
    def get_stats(self, safety_scores: torch.Tensor) -> dict:
        """获取安全分数统计量，用于 tensorboard 记录"""
        with torch.no_grad():
            return {
                'safety_score_mean': safety_scores.mean().item(),
                'safety_score_max': safety_scores.max().item(),
                'safety_score_min': safety_scores.min().item(),
                'safety_score_std': safety_scores.std().item(),
            }


def get_beta(curriculum_level: int, cfg: AttentionGuidanceCfg) -> float:
    """
    根据课程等级计算当前的偏置强度 β
    
    公式: β = β_max - (β_max - β_min) × (level / max_level)
    
    Args:
        curriculum_level: 当前课程等级
        cfg: 注意力引导配置
    
    Returns:
        beta: 当前偏置强度
    
    示例:
        Level 0  → β = 2.0 (强引导)
        Level 5  → β = 1.0 (中等引导)
        Level 10 → β = 0.0 (无引导，完全自主)
    """
    if not cfg.use_curriculum_decay:
        return cfg.beta_max
    
    # 线性衰减
    progress = min(curriculum_level / cfg.decay_curriculum_levels, 1.0)
    beta = cfg.beta_max - (cfg.beta_max - cfg.beta_min) * progress
    return beta




