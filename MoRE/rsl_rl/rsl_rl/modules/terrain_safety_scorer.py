"""
TerrainSafetyScorer: 物理引导的地形安全打分模块

基于 FastStair VHIP 模型和几何平坦度分析，为地形采样点生成注意力偏置。

核心功能:
- S_geo: 几何平坦度分数（归一化中心差分坡度 + 局部方差）
- S_dyn: 动力学可行性分数（逐点 VHIP 捕获点，可学习 alpha）
- 分层软掩码: 陡峭/深坑/身后区域的差异化惩罚

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
    物理引导的地形安全打分模块
    
    输入:
        height_map: [B, grid_h, grid_w] 采样点相对基准高度的差值
        base_lin_vel: [B, 3] 机体坐标系下的质心线速度
        
    输出:
        bias_map: [B, grid_h * grid_w] 未归一化的 logits
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
        g: float = 9.81,
        z_nominal: float = 0.75,
        C_scaling: float = 1.0,
        local_var_threshold: float = 0.03,
        height_drop_threshold: float = 0.3,
        behind_threshold: float = 0.3,
        behind_vel_threshold: float = 0.1,
        behind_penalty: float = -1.5,   # 降低：原 -3.0，防止注意力崩塌
        steep_penalty: float = -4.0,    # 降低：原 -8.0
        pit_penalty: float = -6.0,      # 降低：原 -12.0
        learnable_sigma: bool = True,
        init_sigma_dyn: float = 0.3,
        init_sigma_geo: float = 0.5,
        learnable_alpha: bool = True,
        init_alpha: float = 1.0,
    ):
        """
        Args:
            grid_h: 采样网格高度 (对应 x 方向)
            grid_w: 采样网格宽度 (对应 y 方向)
            measured_points_x: x 方向采样点坐标列表
            measured_points_y: y 方向采样点坐标列表
            dx: x 方向采样间距 (米)，若为 None 则从 measured_points_x 自动计算
            dy: y 方向采样间距 (米)，若为 None 则从 measured_points_y 自动计算
            g: 重力加速度
            z_nominal: 名义站立高度 (G1 约 0.75m)
            C_scaling: 捕获点基础缩放系数
            local_var_threshold: 局部方差阈值 (用于检测碎石/台阶边缘的高频起伏)
            height_drop_threshold: 深坑检测阈值 (米，正值表示坑深)
            behind_threshold: 身后区域掩码阈值 (米，相对速度方向的投影距离)
            behind_vel_threshold: 触发身后掩码的最小速度 (m/s)
            behind_penalty: 身后区域惩罚值 (轻，不优先)
            steep_penalty: 陡坡/边缘惩罚值 (中等风险)
            pit_penalty: 深坑惩罚值 (灾难性风险)
            learnable_sigma: 是否学习温度参数
            init_sigma_dyn: 动力学温度初始值
            init_sigma_geo: 几何温度初始值
            learnable_alpha: 是否学习捕获点缩放因子
            init_alpha: 捕获点缩放因子初始值
        """
        super().__init__()
        
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_points = grid_h * grid_w
        self.g = g
        self.z_nominal = z_nominal
        self.C_scaling = C_scaling
        self.local_var_threshold = local_var_threshold
        self.height_drop_threshold = height_drop_threshold
        self.behind_threshold = behind_threshold
        self.behind_vel_threshold = behind_vel_threshold
        self.behind_penalty = behind_penalty
        self.steep_penalty = steep_penalty
        self.pit_penalty = pit_penalty
        
        # ===== 默认采样点坐标 (G1 配置) =====
        if measured_points_x is None:
            measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
                                  0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if measured_points_y is None:
            measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        # ===== 计算采样间距 (用于归一化梯度) =====
        if dx is None:
            # 从采样点坐标自动计算间距
            dx = (measured_points_x[-1] - measured_points_x[0]) / (len(measured_points_x) - 1)
        if dy is None:
            dy = (measured_points_y[-1] - measured_points_y[0]) / (len(measured_points_y) - 1)
        
        self.dx = dx
        self.dy = dy
        
        # ===== 生成 xy 坐标网格 (register_buffer 自动管理 device) =====
        x = torch.tensor(measured_points_x, dtype=torch.float32)
        y = torch.tensor(measured_points_y, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')  # [grid_h, grid_w]
        grid_xy = torch.stack([xx, yy], dim=0)  # [2, grid_h, grid_w]
        self.register_buffer('grid_xy', grid_xy)
        
        # ===== 可学习温度参数 (使用 softplus 保证正值) =====
        # 初始化: softplus(x) ≈ x when x > 0, 所以用 inverse softplus
        init_raw_dyn = self._inverse_softplus(init_sigma_dyn)
        init_raw_geo = self._inverse_softplus(init_sigma_geo)
        
        if learnable_sigma:
            self.raw_sigma_dyn = nn.Parameter(torch.tensor(init_raw_dyn))
            self.raw_sigma_geo = nn.Parameter(torch.tensor(init_raw_geo))
        else:
            self.register_buffer('raw_sigma_dyn', torch.tensor(init_raw_dyn))
            self.register_buffer('raw_sigma_geo', torch.tensor(init_raw_geo))
        
        # ===== 可学习捕获点缩放因子 =====
        init_raw_alpha = self._inverse_softplus(init_alpha)
        
        if learnable_alpha:
            self.raw_alpha = nn.Parameter(torch.tensor(init_raw_alpha))
        else:
            self.register_buffer('raw_alpha', torch.tensor(init_raw_alpha))
    
    @staticmethod
    def _inverse_softplus(y: float, beta: float = 1.0) -> float:
        """计算 softplus 的逆函数: x = log(exp(y) - 1)"""
        if y > 20:  # 避免数值溢出
            return y
        return torch.log(torch.exp(torch.tensor(y)) - 1).item()
    
    def compute_geometric_score(
        self, 
        height_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算几何平坦度分数 S_geo
        
        使用归一化中心差分计算坡度，以及 3x3 局部方差检测高频起伏。
        
        Args:
            height_map: [B, grid_h, grid_w]
            
        Returns:
            S_geo: [B, grid_h, grid_w] 几何分数 (负的坡度)
            slope: [B, grid_h, grid_w] 归一化坡度 (tan(θ))
            local_var: [B, grid_h, grid_w] 局部方差
        """
        # 添加通道维度: [B, 1, H, W]
        h = height_map.unsqueeze(1)
        
        # ===== 归一化中心差分 (replicate padding) =====
        # padding 顺序: (left, right, top, bottom)
        h_pad = F.pad(h, (1, 1, 1, 1), mode='replicate')  # [B, 1, H+2, W+2]
        
        # 中心差分计算梯度，除以采样间距得到真实坡度
        dh_dx = (h_pad[:, :, 2:, 1:-1] - h_pad[:, :, :-2, 1:-1]) / (2 * self.dx)  # [B, 1, H, W]
        dh_dy = (h_pad[:, :, 1:-1, 2:] - h_pad[:, :, 1:-1, :-2]) / (2 * self.dy)  # [B, 1, H, W]
        
        # 坡度幅值 (tan(θ))
        slope = torch.sqrt(dh_dx ** 2 + dh_dy ** 2 + 1e-8).squeeze(1)  # [B, H, W]
        
        # ===== 3x3 局部方差 (replicate padding，避免边界伪高方差) =====
        # 使用 replicate padding 计算局部均值
        local_mean = F.avg_pool2d(h_pad, kernel_size=3, stride=1, padding=0)  # [B, 1, H, W]
        
        # 计算 (h - local_mean)^2，需要重新 pad
        h_centered = h - local_mean
        h_centered_pad = F.pad(h_centered, (1, 1, 1, 1), mode='replicate')
        local_var = F.avg_pool2d(h_centered_pad ** 2, kernel_size=3, stride=1, padding=0).squeeze(1)  # [B, H, W]
        
        # 几何分数 = 负的坡度
        S_geo = -slope
        
        return S_geo, slope, local_var
    
    def compute_dynamic_score(
        self,
        height_map: torch.Tensor,
        base_lin_vel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        计算动力学可行性分数 S_dyn (方案A: 统一捕获点 + VHIP高度修正)
        
        物理意义:
        1. 使用名义高度计算 唯一的 理想捕获点位置
        2. 计算每个采样点到这个统一捕获点的距离
        3. 根据每个点的实际高度微调: 坑 = 更难到达, 凸起 = 更容易到达
        
        Args:
            height_map: [B, grid_h, grid_w]
            base_lin_vel: [B, 3] 机体坐标系下的线速度
            
        Returns:
            S_dyn: [B, grid_h, grid_w] 动力学分数 (负的修正距离平方)
            omega: [B, grid_h, grid_w] 逐点自然频率 (用于调试)
            capture_point: [B, 2] 统一的理想捕获点位置
            z_ratio: [B, grid_h, grid_w] VHIP 高度修正系数
            alpha: float 当前 alpha 值
        """
        B = height_map.shape[0]
        
        # ===== 可学习的 alpha 缩放因子 =====
        alpha = F.softplus(self.raw_alpha) + 1e-6
        
        # ===== 1. 使用名义高度计算统一的捕获点 =====
        # omega_nominal = sqrt(g / z_nominal) 是固定值
        omega_nominal = (self.g / self.z_nominal) ** 0.5
        
        # 提取 xy 方向速度
        v_xy = base_lin_vel[:, :2]  # [B, 2]
        
        # 统一捕获点: P_capture = alpha * C_scaling * v / omega_nominal
        capture_point = alpha * self.C_scaling * v_xy / omega_nominal  # [B, 2]
        
        # ===== 2. 计算每个采样点到统一捕获点的基础距离 =====
        # 扩展 grid_xy 到 batch
        grid_xy = self.grid_xy.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        # 扩展 capture_point 到网格形状
        capture_point_expanded = capture_point.view(B, 2, 1, 1)  # [B, 2, 1, 1]
        
        # 基础距离平方 (所有点到同一个捕获点的距离)
        base_dist_sq = ((grid_xy - capture_point_expanded) ** 2).sum(dim=1)  # [B, H, W]
        
        # ===== 3. VHIP 高度修正 =====
        # 物理意义: 坑内的点需要更大的能量/步幅才能跨越，等效于"更难到达"
        # 有效高度 = z_nominal + height_map (正值=坑, 有效高度增加)
        effective_z = self.z_nominal + height_map  # [B, H, W]
        effective_z = effective_z.clamp(min=0.1)  # 防止除零
        
        # 高度比率: z_ratio > 1 表示坑 (更难到达), z_ratio < 1 表示凸起 (更容易到达)
        z_ratio = (effective_z / self.z_nominal).clamp(0.5, 2.0)  # [B, H, W]
        
        # 修正后的距离: 坑区域距离放大, 凸起区域距离缩小
        adjusted_dist_sq = base_dist_sq * z_ratio  # [B, H, W]
        
        # ===== 4. 计算逐点 omega (用于调试/可视化) =====
        omega = torch.sqrt(self.g / effective_z)  # [B, H, W]
        
        # ===== 5. 动力学分数 = 负的修正距离平方 =====
        S_dyn = -adjusted_dist_sq
        
        return S_dyn, omega, capture_point, z_ratio, alpha.item()
    
    def compute_behind_mask(
        self,
        base_lin_vel: torch.Tensor,
        B: int
    ) -> torch.Tensor:
        """
        计算身后区域掩码 (点积半平面)
        
        使用速度方向的投影判断，而非轴对齐判断。
        
        Args:
            base_lin_vel: [B, 3] 机体坐标系下的线速度
            B: batch size
            
        Returns:
            behind_mask: [B, grid_h, grid_w] 身后区域布尔掩码
        """
        # 动态扩展网格
        grid_xy = self.grid_xy.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        grid_x = grid_xy[:, 0, :, :]  # [B, H, W]
        grid_y = grid_xy[:, 1, :, :]  # [B, H, W]
        
        v_x = base_lin_vel[:, 0].view(B, 1, 1)  # [B, 1, 1]
        v_y = base_lin_vel[:, 1].view(B, 1, 1)  # [B, 1, 1]
        
        # 计算速度方向的单位向量
        v_norm = torch.sqrt(v_x ** 2 + v_y ** 2 + 1e-8)  # [B, 1, 1]
        moving_mask = v_norm > self.behind_vel_threshold  # [B, 1, 1]
        
        # 防止除零
        v_norm_safe = v_norm.clamp(min=1e-6)
        v_hat_x = v_x / v_norm_safe  # [B, 1, 1]
        v_hat_y = v_y / v_norm_safe  # [B, 1, 1]
        
        # 每个点在速度方向上的投影
        proj = grid_x * v_hat_x + grid_y * v_hat_y  # [B, H, W]
        
        # 身后区域 = 正在移动 & 投影 < -threshold
        behind_mask = moving_mask & (proj < -self.behind_threshold)
        
        return behind_mask
    
    def forward(
        self,
        height_map: torch.Tensor,
        base_lin_vel: torch.Tensor,
        return_debug_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        计算地形安全偏置
        
        Args:
            height_map: [B, grid_h, grid_w] 采样点高度图
            base_lin_vel: [B, 3] 机体坐标系下的线速度
            return_debug_info: 是否返回调试信息
            
        Returns:
            bias_map: [B, num_points] 未归一化的注意力偏置
            debug_info (可选): 包含中间结果的字典
        """
        B = height_map.shape[0]
        
        # ===== 1. 几何平坦度分数 =====
        S_geo, slope, local_var = self.compute_geometric_score(height_map)
        
        # ===== 2. 动力学可行性分数 (方案A: 统一捕获点 + VHIP修正) =====
        S_dyn, omega, capture_point, z_ratio, alpha = self.compute_dynamic_score(
            height_map, base_lin_vel
        )
        
        # ===== 3. 温度参数 (softplus 保证正值) =====
        sigma_dyn = F.softplus(self.raw_sigma_dyn) + 1e-6
        sigma_geo = F.softplus(self.raw_sigma_geo) + 1e-6
        
        # ===== 4. 融合分数 =====
        bias = S_dyn / (2 * sigma_dyn ** 2) + S_geo / (sigma_geo ** 2)
        
        # ===== 5. 分层软掩码 (差异化惩罚) =====
        # 5.1 边缘检测 (只用局部方差检测高频起伏/台阶边缘)
        # 移除 slope > threshold 条件，因为平滑斜坡坡度大但可行走
        steep_mask = local_var > self.local_var_threshold
        
        # 5.2 深坑检测 (LeggedGym 语义: 正值表示坑)
        pit_mask = height_map > self.height_drop_threshold
        
        # 5.3 身后区域掩码 (点积半平面)
        behind_mask = self.compute_behind_mask(base_lin_vel, B)
        
        # ===== 6. 应用分层惩罚 (从 bias 中减去) =====
        # 注意：掩码可能重叠，惩罚会累加
        bias = bias + behind_mask.float() * self.behind_penalty
        bias = bias + steep_mask.float() * self.steep_penalty
        bias = bias + pit_mask.float() * self.pit_penalty
        
        # ===== 7. 展平输出 =====
        bias_flat = bias.view(B, -1)  # [B, num_points]
        
        if return_debug_info:
            debug_info = {
                'S_dyn': S_dyn.detach(),                    # [B, H, W]
                'S_geo': S_geo.detach(),                    # [B, H, W]
                'slope': slope.detach(),                    # [B, H, W] 归一化坡度
                'local_var': local_var.detach(),            # [B, H, W] 局部方差
                'omega': omega.detach(),                    # [B, H, W] 逐点自然频率
                'capture_point': capture_point.detach(),    # [B, 2] 统一捕获点位置
                'z_ratio': z_ratio.detach(),                # [B, H, W] VHIP 高度修正系数
                'steep_mask': steep_mask.detach(),          # [B, H, W]
                'pit_mask': pit_mask.detach(),              # [B, H, W]
                'behind_mask': behind_mask.detach(),        # [B, H, W]
                'sigma_dyn': sigma_dyn.detach().item(),
                'sigma_geo': sigma_geo.detach().item(),
                'alpha': alpha,                             # 当前 alpha 值
                'bias_2d': bias.detach(),                   # [B, H, W] 融合后的偏置
            }
            return bias_flat, debug_info
        
        return bias_flat
    
    def get_sigma_values(self) -> Tuple[float, float]:
        """获取当前温度参数值"""
        sigma_dyn = (F.softplus(self.raw_sigma_dyn) + 1e-6).item()
        sigma_geo = (F.softplus(self.raw_sigma_geo) + 1e-6).item()
        return sigma_dyn, sigma_geo
    
    def get_alpha_value(self) -> float:
        """获取当前 alpha 缩放因子值"""
        return (F.softplus(self.raw_alpha) + 1e-6).item()
    
    def extra_repr(self) -> str:
        sigma_dyn, sigma_geo = self.get_sigma_values()
        alpha = self.get_alpha_value()
        return (
            f"grid={self.grid_h}x{self.grid_w}, "
            f"dx={self.dx:.3f}, dy={self.dy:.3f}, "
            f"z_nominal={self.z_nominal}, "
            f"sigma_dyn={sigma_dyn:.3f}, sigma_geo={sigma_geo:.3f}, "
            f"alpha={alpha:.3f}, "
            f"thresholds=(var={self.local_var_threshold}, "
            f"drop={self.height_drop_threshold}, behind={self.behind_threshold}), "
            f"penalties=(behind={self.behind_penalty}, steep={self.steep_penalty}, "
            f"pit={self.pit_penalty})"
        )
