"""
TerrainSafetyScorer V2: "信息先验" 替代 "安全先验"

核心改动:
1. 将 danger_penalty 拆分为 edge_bonus + pit_penalty
   - 边缘/过渡区 (高残差非深坑) → 正向权重 (信息丰富)
   - 深坑 → 保留惩罚
2. 新增 forward_bias: 基于 x 坐标的高斯前向偏置
3. 调整各分量权重: 降低 support/margin, 新增 edge/forward
4. 温和的 temperature 和 label_smooth

改动位置全部在 forward() 和 __init__() 中，
其余方法 (_compute_plane_residual, _compute_margin, compute_behind_mask) 完全不变。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, List


class TerrainSafetyScorer(nn.Module):
    
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
        # ===== 融合参数 (V2: 调整默认值) =====
        w_support: float = 0.3,       # V1: 1.0 → V2: 0.3 降低"平坦=好"偏见
        w_margin: float = 0.3,        # V1: 1.0 → V2: 0.3 降低"远离边缘=好"偏见
        w_edge: float = 0.5,          # V2 新增: 边缘/过渡区信息加分
        w_forward: float = 0.8,       # V2 新增: 前向位置偏置
        forward_peak: float = 0.3,    # V2 新增: 前向高斯峰值位置 (m)
        forward_sigma: float = 0.4,   # V2 新增: 前向高斯标准差 (m)
        temperature: float = 1.0,     # V1: 1.0, 绿色实验: 0.5 → V2: 回到 1.0
        # ===== 惩罚参数 (V2: 只惩罚深坑) =====
        pit_penalty: float = -3.0,    # V1 的 danger_penalty 重命名为 pit_penalty
        behind_threshold: float = 0.3,
        behind_vel_threshold: float = 0.1,
        behind_penalty: float = -2.0,
        # ===== Label Smoothing =====
        label_smooth: float = 0.05,   # V1: 0.05, 绿色实验: 0.01 → V2: 0.05
        # ===== 边缘显著性上限 =====
        edge_salience_max: float = 5.0,  # clamp 上限，防止极端值
    ):
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
        self.w_edge = w_edge
        self.w_forward = w_forward
        self.temperature = temperature
        self.pit_penalty = pit_penalty
        self.behind_threshold = behind_threshold
        self.behind_vel_threshold = behind_vel_threshold
        self.behind_penalty = behind_penalty
        self.label_smooth = label_smooth
        self.edge_salience_max = edge_salience_max
        
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
        
        # ===== V2 新增: 前向位置偏置 =====
        forward_bias = torch.exp(-(x - forward_peak)**2 / (2 * forward_sigma**2))
        forward_bias_2d = forward_bias.unsqueeze(1).expand(-1, len(measured_points_y))
        self.register_buffer('forward_bias', forward_bias_2d)  # [grid_h, grid_w]
        
        # ===== 预计算平面拟合卷积核 =====
        self._register_plane_fit_kernels(dx, dy)
    
    def _register_plane_fit_kernels(self, dx: float, dy: float):
        """预计算 3×3 局部最小二乘平面拟合卷积核（与 V1 完全相同）"""
        x_kernel = torch.tensor([
            [-dx, -dx, -dx],
            [  0,   0,   0],
            [ dx,  dx,  dx]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        y_kernel = torch.tensor([
            [-dy,  0,  dy],
            [-dy,  0,  dy],
            [-dy,  0,  dy]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        ones_kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32)
        
        self.register_buffer('x_kernel', x_kernel)
        self.register_buffer('y_kernel', y_kernel)
        self.register_buffer('ones_kernel', ones_kernel)
        
        self.inv_6dx2 = 1.0 / (6.0 * dx * dx)
        self.inv_6dy2 = 1.0 / (6.0 * dy * dy)
    
    @torch.no_grad()
    def _compute_plane_residual(self, height_map: torch.Tensor) -> torch.Tensor:
        """计算 3×3 窗口内局部平面拟合的 RMS 残差（与 V1 完全相同）"""
        h = height_map.unsqueeze(1)
        h_pad = F.pad(h, (1, 1, 1, 1), mode='replicate')
        
        sum_z  = F.conv2d(h_pad, self.ones_kernel)
        sum_z2 = F.conv2d(h_pad ** 2, self.ones_kernel)
        sum_xz = F.conv2d(h_pad, self.x_kernel)
        sum_yz = F.conv2d(h_pad, self.y_kernel)
        
        residual_var = (1.0 / 9.0) * (
            sum_z2
            - sum_xz ** 2 * self.inv_6dx2
            - sum_yz ** 2 * self.inv_6dy2
            - sum_z ** 2 / 9.0
        )
        residual_var = residual_var.clamp(min=0)
        
        return torch.sqrt(residual_var + 1e-8).squeeze(1)
    
    @torch.no_grad()
    def _compute_margin(self, danger_mask: torch.Tensor) -> torch.Tensor:
        """形态学腐蚀距离变换（与 V1 完全相同）"""
        safe = (~danger_mask).float().unsqueeze(1)
        margin = torch.zeros_like(safe)
        
        for _ in range(self.max_margin_steps):
            margin += safe
            safe_pad = F.pad(safe, (1, 1, 1, 1), mode='replicate')
            safe = -F.max_pool2d(-safe_pad, 3, 1, 0)
        
        return (margin / self.max_margin_steps).squeeze(1)
    
    @torch.no_grad()
    def compute_behind_mask(self, base_lin_vel: torch.Tensor, B: int) -> torch.Tensor:
        """身后区域掩码（与 V1 完全相同）"""
        grid_xy = self.grid_xy.unsqueeze(0).expand(B, -1, -1, -1)
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
        V2 计算地形信息先验分布。
        
        与 V1 的关键区别:
        1. 边缘/过渡区获得正向权重 (edge_bonus) 而非惩罚
        2. 只有深坑保留惩罚 (pit_penalty)
        3. 新增前向位置偏置 (forward_bias)
        4. 降低 support/margin 权重，突出 edge/forward
        """
        B = height_map.shape[0]
        
        # ===== 1. 平面拟合残差 =====
        residual_rms = self._compute_plane_residual(height_map)
        
        # ===== 2. 支撑面积分数 =====
        S_support = torch.exp(-residual_rms / self.support_scale)
        
        # ===== 3. V2: 区分 "边缘" 和 "深坑" =====
        is_pit = (height_map > self.pit_threshold)
        is_edge = (residual_rms > self.danger_threshold) & ~is_pit
        
        # 边缘显著性: 残差越大 → 地形变化越剧烈 → 越需要关注
        edge_salience = (residual_rms / self.support_scale).clamp(max=self.edge_salience_max)
        
        # ===== 4. 边缘裕度 (只基于 pit 做 danger_mask) =====
        # V2: margin 只计算到深坑的距离，不把边缘也当 danger
        S_margin = self._compute_margin(is_pit)
        
        # ===== 5. 身后掩码 =====
        behind_mask = self.compute_behind_mask(base_lin_vel, B)
        
        # ===== 6. V2: 融合 logits =====
        logits = (
            # 基础: 平坦区域的支撑质量 (权重降低)
            self.w_support * S_support * (~is_pit).float()
            # 裕度: 离深坑的距离 (权重降低)
            + self.w_margin * S_margin
            # V2 新增: 边缘/过渡区加分 (核心改动)
            + self.w_edge * is_edge.float() * edge_salience
            # V2 新增: 前向位置偏置
            + self.w_forward * self.forward_bias.unsqueeze(0)
            # 深坑惩罚 (只惩罚真正不可踩的区域)
            + is_pit.float() * self.pit_penalty
            # 身后惩罚
            + behind_mask.float() * self.behind_penalty
        )
        
        # ===== 7. softmax → 概率分布 =====
        prior_dist = F.softmax(logits.view(B, -1) / self.temperature, dim=-1)
        
        # ===== 8. Label Smoothing =====
        if self.label_smooth > 0:
            uniform = 1.0 / self.num_points
            prior_dist = (1.0 - self.label_smooth) * prior_dist + self.label_smooth * uniform
        
        if return_debug_info:
            debug_info = {
                'residual_rms': residual_rms.detach(),
                'S_support': S_support.detach(),
                'is_pit': is_pit.detach(),            # V2: 替代 danger_mask
                'is_edge': is_edge.detach(),           # V2 新增
                'edge_salience': edge_salience.detach(),  # V2 新增
                'S_margin': S_margin.detach(),
                'behind_mask': behind_mask.detach(),
                'logits_2d': logits.detach(),
                'prior_dist': prior_dist.detach(),
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
            f"w_edge={self.w_edge}, w_forward={self.w_forward}, "
            f"temperature={self.temperature}, "
            f"pit_penalty={self.pit_penalty}, "
            f"behind_penalty={self.behind_penalty}, "
            f"label_smooth={self.label_smooth}"
        )