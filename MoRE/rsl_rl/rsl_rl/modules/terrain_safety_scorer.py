"""
TerrainSafetyScorer: 物理引导的地形安全打分模块

基于 FastStair VHIP 模型和几何平坦度分析，为地形采样点生成注意力偏置。

核心功能:
- S_geo: 几何平坦度分数（Sobel 梯度）
- S_dyn: 动力学可行性分数（逐点 VHIP 捕获点）
- 三重安全掩码: 陡峭/深坑/高台检测
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
        g: float = 9.81,
        z_nominal: float = 0.75,
        C_scaling: float = 1.0,
        roughness_threshold: float = 0.5,
        height_drop_threshold: float = 0.3,
        height_climb_threshold: float = 0.4,
        learnable_sigma: bool = True,
        init_sigma_dyn: float = 0.3,
        init_sigma_geo: float = 0.1,
    ):
        """
        Args:
            grid_h: 采样网格高度 (对应 x 方向)
            grid_w: 采样网格宽度 (对应 y 方向)
            measured_points_x: x 方向采样点坐标列表
            measured_points_y: y 方向采样点坐标列表
            g: 重力加速度
            z_nominal: 名义站立高度 (G1 约 0.75m)
            C_scaling: 捕获点缩放系数
            roughness_threshold: 粗糙度安全阈值
            height_drop_threshold: 深坑检测阈值 (米)
            height_climb_threshold: 可攀爬高度阈值 (米)
            learnable_sigma: 是否学习温度参数
            init_sigma_dyn: 动力学温度初始值
            init_sigma_geo: 几何温度初始值
        """
        super().__init__()
        
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_points = grid_h * grid_w
        self.g = g
        self.z_nominal = z_nominal
        self.C_scaling = C_scaling
        self.roughness_threshold = roughness_threshold
        self.height_drop_threshold = height_drop_threshold
        self.height_climb_threshold = height_climb_threshold
        
        # ===== 默认采样点坐标 (G1 配置) =====
        if measured_points_x is None:
            measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 
                                  0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        if measured_points_y is None:
            measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        # ===== 生成 xy 坐标网格 (register_buffer 自动管理 device) =====
        x = torch.tensor(measured_points_x, dtype=torch.float32)
        y = torch.tensor(measured_points_y, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')  # [grid_h, grid_w]
        grid_xy = torch.stack([xx, yy], dim=0)  # [2, grid_h, grid_w]
        self.register_buffer('grid_xy', grid_xy)
        
        # ===== Sobel 卷积核 (3x3, padding=1 保持维度) =====
        sobel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1., -2., -1.],
            [ 0.,  0.,  0.],
            [ 1.,  2.,  1.]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
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
    
    @staticmethod
    def _inverse_softplus(y: float, beta: float = 1.0) -> float:
        """计算 softplus 的逆函数: x = log(exp(y) - 1)"""
        if y > 20:  # 避免数值溢出
            return y
        return torch.log(torch.exp(torch.tensor(y)) - 1).item()
    
    def compute_geometric_score(
        self, 
        height_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算几何平坦度分数 S_geo
        
        Args:
            height_map: [B, grid_h, grid_w]
            
        Returns:
            S_geo: [B, grid_h, grid_w] 几何分数 (负的粗糙度)
            roughness: [B, grid_h, grid_w] 粗糙度 (梯度幅值)
        """
        # 添加通道维度: [B, 1, H, W]
        h = height_map.unsqueeze(1)
        
        # Sobel 卷积 (padding=1 保持维度)
        G_x = F.conv2d(h, self.sobel_x, padding=1)  # [B, 1, H, W]
        G_y = F.conv2d(h, self.sobel_y, padding=1)  # [B, 1, H, W]
        
        # 梯度幅值 (粗糙度)
        roughness = torch.sqrt(G_x ** 2 + G_y ** 2 + 1e-8).squeeze(1)  # [B, H, W]
        
        # 几何分数 = 负的粗糙度
        S_geo = -roughness
        
        return S_geo, roughness
    
    def compute_dynamic_score(
        self,
        height_map: torch.Tensor,
        base_lin_vel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算动力学可行性分数 S_dyn (基于逐点 VHIP)
        
        Args:
            height_map: [B, grid_h, grid_w]
            base_lin_vel: [B, 3] 机体坐标系下的线速度
            
        Returns:
            S_dyn: [B, grid_h, grid_w] 动力学分数 (负的距离平方)
            omega: [B, grid_h, grid_w] 逐点自然频率
            capture_offset: [B, 2, grid_h, grid_w] 理想捕获点偏移
        """
        B = height_map.shape[0]
        
        # 逐点计算有效高度 (质心到地面的高度)
        # height_map 存储的是 "采样点高度 - 基准高度"
        # 有效高度 = z_nominal - height_map (当 height_map 为正表示地面抬高)
        effective_z = self.z_nominal - height_map  # [B, H, W]
        effective_z = effective_z.clamp(min=0.1)  # 防止除零
        
        # 逐点自然频率 omega = sqrt(g / z)
        omega = torch.sqrt(self.g / effective_z)  # [B, H, W]
        
        # 提取 xy 方向速度
        v_xy = base_lin_vel[:, :2]  # [B, 2]
        v_xy = v_xy.unsqueeze(-1).unsqueeze(-1)  # [B, 2, 1, 1]
        
        # 逐点计算理想捕获点偏移
        # P_ideal = v / omega * C_scaling
        omega_expanded = omega.unsqueeze(1)  # [B, 1, H, W]
        capture_offset = v_xy / omega_expanded * self.C_scaling  # [B, 2, H, W]
        
        # 动态扩展网格到当前 batch size
        grid_xy = self.grid_xy.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 2, H, W]
        
        # 计算每个网格点到其对应理想捕获点的距离平方
        dist_sq = ((grid_xy - capture_offset) ** 2).sum(dim=1)  # [B, H, W]
        
        # 动力学分数 = 负的距离平方
        S_dyn = -dist_sq
        
        return S_dyn, omega, capture_offset
    
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
        S_geo, roughness = self.compute_geometric_score(height_map)
        
        # ===== 2. 动力学可行性分数 =====
        S_dyn, omega, capture_offset = self.compute_dynamic_score(
            height_map, base_lin_vel
        )
        
        # ===== 3. 温度参数 (softplus 保证正值) =====
        sigma_dyn = F.softplus(self.raw_sigma_dyn) + 1e-6
        sigma_geo = F.softplus(self.raw_sigma_geo) + 1e-6
        
        # ===== 4. 融合分数 =====
        bias = S_dyn / (2 * sigma_dyn ** 2) + S_geo / (sigma_geo ** 2)
        
        # ===== 5. 安全掩码 (三重检测) =====
        # 5.1 粗糙度过大 (悬崖边缘、陡坡)
        steep_mask = roughness > self.roughness_threshold
        
        # 5.2 高度差过大 (深坑底部)
        too_low_mask = height_map < -self.height_drop_threshold
        
        # 5.3 高度差过高 (无法跨越的台阶)
        too_high_mask = height_map > self.height_climb_threshold
        
        # 合并掩码
        unsafe_mask = steep_mask | too_low_mask | too_high_mask
        
        # 危险区域设为 -inf (使用 -1e9 避免数值问题)
        bias = bias.clone()  # 避免 in-place 操作影响梯度
        bias[unsafe_mask] = -1e9
        
        # ===== 6. 展平输出 =====
        bias_flat = bias.view(B, -1)  # [B, num_points]
        
        if return_debug_info:
            debug_info = {
                'S_dyn': S_dyn.detach(),                    # [B, H, W]
                'S_geo': S_geo.detach(),                    # [B, H, W]
                'roughness': roughness.detach(),            # [B, H, W]
                'omega': omega.detach(),                    # [B, H, W]
                'capture_offset': capture_offset.detach(),  # [B, 2, H, W]
                'steep_mask': steep_mask.detach(),          # [B, H, W]
                'too_low_mask': too_low_mask.detach(),      # [B, H, W]
                'too_high_mask': too_high_mask.detach(),    # [B, H, W]
                'unsafe_mask': unsafe_mask.detach(),        # [B, H, W]
                'sigma_dyn': sigma_dyn.detach().item(),
                'sigma_geo': sigma_geo.detach().item(),
                'bias_2d': bias.detach(),                   # [B, H, W] 融合后的偏置
            }
            return bias_flat, debug_info
        
        return bias_flat
    
    def get_sigma_values(self) -> Tuple[float, float]:
        """获取当前温度参数值"""
        sigma_dyn = (F.softplus(self.raw_sigma_dyn) + 1e-6).item()
        sigma_geo = (F.softplus(self.raw_sigma_geo) + 1e-6).item()
        return sigma_dyn, sigma_geo
    
    def extra_repr(self) -> str:
        sigma_dyn, sigma_geo = self.get_sigma_values()
        return (
            f"grid={self.grid_h}x{self.grid_w}, "
            f"z_nominal={self.z_nominal}, "
            f"sigma_dyn={sigma_dyn:.3f}, sigma_geo={sigma_geo:.3f}, "
            f"thresholds=(rough={self.roughness_threshold}, "
            f"drop={self.height_drop_threshold}, climb={self.height_climb_threshold})"
        )

