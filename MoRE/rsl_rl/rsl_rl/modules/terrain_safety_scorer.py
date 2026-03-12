"""
TerrainSafetyScorer: 纯物理先验的地形安全打分模块（无可学习参数）

输出 softmax 概率分布，用作 attention 的 KL 监督目标。

核心组件:
- S_support: 支撑面积分数（3×3 局部最小二乘平面拟合，计算 RMS 残差）
             均匀斜坡 → 残差≈0 → 高分；台阶边缘 → 残差大 → 低分
- S_margin:  边缘裕度分数（基于 S_support 的 danger_map 做形态学腐蚀距离变换）
             离边缘远 → 高分；贴边 → 低分
- behind_mask: 身后区域惩罚（速度方向投影）

height_map 符号语义 (与 LeggedGym 管线一致):
- height_map = base_z - base_height - terrain_z
- 正值 = 地面更低（坑）
- 负值 = 地面更高（凸起/台阶）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, List


class TerrainSafetyScorer(nn.Module):
    """
    纯物理先验的地形安全打分模块
    
    无任何可学习参数，所有计算均为固定物理规则。
    输出 softmax 概率分布，作为 attention 的 KL 监督目标（detach 后使用）。
    
    输入:
        height_map: [B, grid_h, grid_w] 采样点相对基准高度的差值
        base_lin_vel: [B, 3] 机体坐标系下的质心线速度
        
    输出:
        prior_dist: [B, grid_h * grid_w] softmax 概率分布
        debug_info (可选): 包含中间结果的字典
    """
    
    def __init__(
        self,
        grid_h: int = 17,
        grid_w: int = 11,
        measured_points_x: Optional[List[float]] = None,
        measured_points_y: Optional[List[float]] = None,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        # ===== 支撑面积分数参数 =====
        support_scale: float = 0.02,
        # ===== 边缘裕度参数 =====
        danger_threshold: float = 0.02,
        pit_threshold: float = 0.3,
        max_margin_steps: int = 3,
        # ===== 融合参数 =====
        w_support: float = 1.0,
        w_margin: float = 1.0,
        temperature: float = 1.0,
        # ===== 惩罚参数 =====
        danger_penalty: float = -3.0,
        behind_threshold: float = 0.3,
        behind_vel_threshold: float = 0.1,
        behind_penalty: float = -2.0,
        # ===== Label Smoothing =====
        label_smooth: float = 0.05,
    ):
        """
        Args:
            grid_h, grid_w: 采样网格尺寸
            measured_points_x/y: 采样点坐标列表
            dx, dy: 采样间距（若 None 则从坐标自动计算）
            support_scale: S_support 归一化尺度 (meter)。
                           exp(-residual/scale) 将 RMS 残差映射到 [0,1]
            danger_threshold: 平面拟合 RMS 残差超过此值 → 标记为危险点
            pit_threshold: 深坑检测阈值 (meter，正值表示坑深)
            max_margin_steps: 形态学腐蚀最大步数，决定裕度的最大感知范围
            w_support, w_margin: S_support 与 S_margin 的融合权重
            temperature: softmax 温度，越小分布越尖锐
            danger_penalty: 危险区域的 logit 惩罚值（确保坑底等区域获得极低概率）
            behind_threshold: 身后区域掩码的投影距离阈值
            behind_vel_threshold: 触发身后掩码的最小速度
            behind_penalty: 身后区域的 logit 惩罚值
            label_smooth: label smoothing 系数 ε ∈ [0, 1)。
                          p_smooth = (1-ε)·p_sharp + ε·uniform。
                          确保所有点概率非零，改善 KL 梯度覆盖。
                          ε=0 退化为无平滑（原始尖锐分布）。
        """
        super().__init__()
        
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_points = grid_h * grid_w
        self.support_scale = support_scale
        self.danger_threshold = danger_threshold
        self.pit_threshold = pit_threshold
        self.max_margin_steps = max_margin_steps
        self.w_support = w_support
        self.w_margin = w_margin
        self.temperature = temperature
        self.danger_penalty = danger_penalty
        self.behind_threshold = behind_threshold
        self.behind_vel_threshold = behind_vel_threshold
        self.behind_penalty = behind_penalty
        self.label_smooth = label_smooth
        
        # ===== 默认采样点坐标 (G1 配置) =====
        if measured_points_x is None:
            measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
                                  0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if measured_points_y is None:
            measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        # ===== 计算采样间距 =====
        if dx is None:
            dx = (measured_points_x[-1] - measured_points_x[0]) / (len(measured_points_x) - 1)
        if dy is None:
            dy = (measured_points_y[-1] - measured_points_y[0]) / (len(measured_points_y) - 1)
        
        self.dx = dx
        self.dy = dy
        
        # ===== 生成 xy 坐标网格 (用于 behind_mask) =====
        x = torch.tensor(measured_points_x, dtype=torch.float32)
        y = torch.tensor(measured_points_y, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        grid_xy = torch.stack([xx, yy], dim=0)  # [2, grid_h, grid_w]
        self.register_buffer('grid_xy', grid_xy)
        
        # ===== 预计算平面拟合卷积核 =====
        self._register_plane_fit_kernels(dx, dy)
    
    def _register_plane_fit_kernels(self, dx: float, dy: float):
        """
        预计算 3×3 局部最小二乘平面拟合所需的固定卷积核。
        
        对于中心对称的 3×3 正则网格，设计矩阵 X = [x, y, 1]，
        其法方程 X^T X 为对角阵，残差方差有封闭解析公式:
            var = (1/9)[Σz² − (Σxz)²/(6dx²) − (Σyz)²/(6dy²) − (Σz)²/9]
        
        三个卷积核分别计算 Σ(x_i·z_i)、Σ(y_i·z_i)、Σz_i。
        """
        # 计算 Σ(x_i · z_i) 的核：每行乘以对应的 x 偏移量
        x_kernel = torch.tensor([
            [-dx, -dx, -dx],
            [  0,   0,   0],
            [ dx,  dx,  dx]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 计算 Σ(y_i · z_i) 的核：每列乘以对应的 y 偏移量
        y_kernel = torch.tensor([
            [-dy,  0,  dy],
            [-dy,  0,  dy],
            [-dy,  0,  dy]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        # 计算 Σz_i 的核
        ones_kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32)
        
        self.register_buffer('x_kernel', x_kernel)
        self.register_buffer('y_kernel', y_kernel)
        self.register_buffer('ones_kernel', ones_kernel)
        
        self.inv_6dx2 = 1.0 / (6.0 * dx * dx)
        self.inv_6dy2 = 1.0 / (6.0 * dy * dy)
    
    @torch.no_grad()
    def _compute_plane_residual(self, height_map: torch.Tensor) -> torch.Tensor:
        """
        计算 3×3 窗口内局部平面拟合的 RMS 残差。
        
        物理含义：残差反映窗口内地面偏离平面的程度。
        - 平坦地面 / 均匀斜坡 → 完美拟合 → 残差 ≈ 0
        - 台阶边缘 / 缝隙边缘 → 拟合困难 → 残差大
        
        推导（利用正则网格对称性 Σx=Σy=Σxy=0）:
            residual_var = (1/9)[Σz² − (Σxz)²/(6dx²) − (Σyz)²/(6dy²) − (Σz)²/9]
        
        Args:
            height_map: [B, grid_h, grid_w]
        Returns:
            residual_rms: [B, grid_h, grid_w] RMS 残差 (meter)
        """
        h = height_map.unsqueeze(1)  # [B, 1, H, W]
        h_pad = F.pad(h, (1, 1, 1, 1), mode='replicate')
        
        sum_z  = F.conv2d(h_pad, self.ones_kernel)       # Σz_i      [B, 1, H, W]
        sum_z2 = F.conv2d(h_pad ** 2, self.ones_kernel)   # Σz_i²
        sum_xz = F.conv2d(h_pad, self.x_kernel)           # Σ(x_i·z_i)
        sum_yz = F.conv2d(h_pad, self.y_kernel)           # Σ(y_i·z_i)
        
        residual_var = (1.0 / 9.0) * (
            sum_z2
            - sum_xz ** 2 * self.inv_6dx2
            - sum_yz ** 2 * self.inv_6dy2
            - sum_z ** 2 / 9.0
        )
        residual_var = residual_var.clamp(min=0)  # 数值保护
        
        return torch.sqrt(residual_var + 1e-8).squeeze(1)  # [B, H, W]
    
    @torch.no_grad()
    def _compute_margin(self, danger_mask: torch.Tensor) -> torch.Tensor:
        """
        基于形态学腐蚀计算到最近危险区域的 Chebyshev 距离。
        
        每轮腐蚀将安全区域向内收缩一个像素（3×3 min pooling）。
        某点存活的轮数 = 到最近危险点的距离。
        
        使用 replicate padding 避免网格边界被误判为危险区域。
        
        Args:
            danger_mask: [B, grid_h, grid_w] bool, True = 危险
        Returns:
            S_margin: [B, grid_h, grid_w] 归一化裕度分数 [0, 1]
                      1 = 远离边缘（存活所有腐蚀轮次）
                      0 = 在危险点上或紧邻危险点
        """
        safe = (~danger_mask).float().unsqueeze(1)  # [B, 1, H, W]
        margin = torch.zeros_like(safe)
        
        for _ in range(self.max_margin_steps):
            margin += safe
            # min pooling (= erosion): 3×3 邻域中有任一 0 → 输出 0
            safe_pad = F.pad(safe, (1, 1, 1, 1), mode='replicate')
            safe = -F.max_pool2d(-safe_pad, 3, 1, 0)
        
        return (margin / self.max_margin_steps).squeeze(1)  # [B, H, W]
    
    @torch.no_grad()
    def compute_behind_mask(
        self,
        base_lin_vel: torch.Tensor,
        B: int
    ) -> torch.Tensor:
        """
        计算身后区域掩码（速度方向的点积半平面判断）。
        
        Args:
            base_lin_vel: [B, 3] 机体坐标系下的线速度
            B: batch size
        Returns:
            behind_mask: [B, grid_h, grid_w] bool
        """
        grid_xy = self.grid_xy.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        grid_x = grid_xy[:, 0, :, :]
        grid_y = grid_xy[:, 1, :, :]
        
        v_x = base_lin_vel[:, 0].view(B, 1, 1)
        v_y = base_lin_vel[:, 1].view(B, 1, 1)
        
        v_norm = torch.sqrt(v_x ** 2 + v_y ** 2 + 1e-8)
        moving_mask = v_norm > self.behind_vel_threshold
        
        v_norm_safe = v_norm.clamp(min=1e-6)
        v_hat_x = v_x / v_norm_safe
        v_hat_y = v_y / v_norm_safe
        
        proj = grid_x * v_hat_x + grid_y * v_hat_y
        
        return moving_mask & (proj < -self.behind_threshold)
    
    @torch.no_grad()
    def forward(
        self,
        height_map: torch.Tensor,
        base_lin_vel: torch.Tensor,
        return_debug_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        计算地形安全先验分布。
        
        流程:
            1. 平面拟合残差 → S_support (exp 归一化到 [0,1])
            2. 残差 + 深坑 → danger_map → 形态学腐蚀 → S_margin [0,1]
            3. 身后掩码 → behind_penalty
            4. 加权融合 → softmax → label smoothing → prior_dist
        
        Args:
            height_map: [B, grid_h, grid_w]
            base_lin_vel: [B, 3]
            return_debug_info: 是否返回中间结果
            
        Returns:
            prior_dist: [B, num_points] softmax 概率分布
            debug_info (可选): 中间结果字典
        """
        B = height_map.shape[0]
        
        # ===== 1. 平面拟合残差 =====
        residual_rms = self._compute_plane_residual(height_map)  # [B, H, W]
        
        # ===== 2. 支撑面积分数 =====
        S_support = torch.exp(-residual_rms / self.support_scale)  # [B, H, W], ∈ [0, 1]
        
        # ===== 3. 危险区域检测 =====
        danger_mask = (
            (residual_rms > self.danger_threshold) |  # 平面拟合残差过大
            (height_map > self.pit_threshold)          # 深坑
        )
        
        # ===== 4. 边缘裕度分数 =====
        S_margin = self._compute_margin(danger_mask)  # [B, H, W], ∈ [0, 1]
        
        # ===== 5. 身后掩码 =====
        behind_mask = self.compute_behind_mask(base_lin_vel, B)
        
        # ===== 6. 加权融合 =====
        # danger 区域抹零 S_support（消除"平坑高分"假象：坑底虽平但不可踩）
        safe_float = (~danger_mask).float()
        logits = self.w_support * S_support * safe_float + self.w_margin * S_margin
        # 为 danger 区域施加显式惩罚（拉开安全/危险区的 logit 差距）
        logits = logits + danger_mask.float() * self.danger_penalty
        logits = logits + behind_mask.float() * self.behind_penalty
        
        # ===== 7. softmax → 概率分布 =====
        prior_dist = F.softmax(logits.view(B, -1) / self.temperature, dim=-1)
        
        # ===== 8. Label Smoothing =====
        if self.label_smooth > 0:
            uniform = 1.0 / self.num_points
            prior_dist = (1.0 - self.label_smooth) * prior_dist + self.label_smooth * uniform
        
        if return_debug_info:
            debug_info = {
                'residual_rms': residual_rms.detach(),   # [B, H, W] 平面拟合残差
                'S_support': S_support.detach(),         # [B, H, W] 支撑面积分数
                'danger_mask': danger_mask.detach(),     # [B, H, W] 危险区域掩码
                'S_margin': S_margin.detach(),           # [B, H, W] 边缘裕度分数
                'behind_mask': behind_mask.detach(),     # [B, H, W] 身后掩码
                'logits_2d': logits.detach(),            # [B, H, W] 融合后 logits
                'prior_dist': prior_dist.detach(),       # [B, num_points] 概率分布
            }
            return prior_dist, debug_info
        
        return prior_dist
    
    def extra_repr(self) -> str:
        return (
            f"grid={self.grid_h}x{self.grid_w}, "
            f"dx={self.dx:.3f}, dy={self.dy:.3f}, "
            f"support_scale={self.support_scale}, "
            f"danger_threshold={self.danger_threshold}, "
            f"pit_threshold={self.pit_threshold}, "
            f"max_margin_steps={self.max_margin_steps}, "
            f"w_support={self.w_support}, w_margin={self.w_margin}, "
            f"temperature={self.temperature}, "
            f"danger_penalty={self.danger_penalty}, "
            f"behind_penalty={self.behind_penalty}, "
            f"label_smooth={self.label_smooth}"
        )
