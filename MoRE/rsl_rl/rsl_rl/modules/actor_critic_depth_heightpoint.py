# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
#
# 本文件实现了基于地形高度点的空间感知注意力机制
# 备份文件 actor_critic_depth_backup.py 包含原始的基于深度图像的实现

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, output_dim, output_activation=None, in_channels=1):
        super().__init__()

        self.in_channels = in_channels
        self.output_dim = output_dim
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 64, 64]
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
            # [32, 15, 15]
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            activation,
            # [32, 7, 7]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
            # [64, 5, 5]
            nn.Flatten(),
            # [64 * 5 * 5]
            nn.Linear(64 * 5 * 5, 128),
            activation,
            nn.Linear(128, output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: Tensor of shape [N, H, W] (no explicit channel dim).
                Note: despite the historical class name containing "58x87", the current
                `image_compression` MLP is built assuming a 64x64 input (so that the
                conv stack ends in 64x5x5 -> 1600 features).
                To make the model robust to different upstream depth resolutions,
                we resize to (64, 64) when needed.
        """
        if images.ndim != 3:
            raise ValueError(f"DepthOnlyFCBackbone expects images [N,H,W] but got shape {tuple(images.shape)}")

        # `image_compression` is designed for 64x64.
        target_hw = (64, 64)
        if tuple(images.shape[-2:]) != target_hw:
            # Resize only when mismatch happens; keep this path cheap for the common (64,64) case.
            images = F.interpolate(
                images.unsqueeze(1),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        images_compressed = self.image_compression(images.unsqueeze(1))  # [N, 1, 64, 64] -> [N, output_dim]
        latent = self.output_activation(images_compressed)

        return latent

class StackDepthEncoder(nn.Module):  # 堆叠深度编码器：处理多帧深度图像，提取时间序列特征
    def __init__(self, base_backbone: DepthOnlyFCBackbone58x87, buffer_len) -> None:
        super().__init__()
        activation = nn.ELU()
        self.base_backbone = base_backbone

        self.conv1d = nn.Sequential(nn.Conv1d(in_channels=buffer_len, out_channels=16, kernel_size=4, stride=2),
                                    activation,
                                    nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2),
                                    activation)
        self.mlp = nn.Sequential(nn.Linear(16*62, 128), activation)
        
    def forward(self, depth_image):
        # depth_image shape: [batch_size, num, H, W] (typically 64x64 in current env configs)
        depth_latent = self.base_backbone(depth_image.flatten(0, 1))  # [batch_size * num, 128]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 128]
        depth_latent = self.conv1d(depth_latent) # [batch_size, 16, 62]
        depth_latent = self.mlp(depth_latent.flatten(1, 2))
        return depth_latent


class HeightPointAttentionEncoder(nn.Module):
    """
    基于地形高度点的空间感知注意力编码器 (Height Point Spatial Attention Encoder)
    
    核心特性：
    - 使用显式的地形高度采样点代替深度图像特征图
    - **两路特征（几何 + 位置）**：
      - 几何视觉路：仅取 z，高度图 [B, 1, num_x, num_y] 经过 1~2 层 Conv2d，得到每个点的几何特征（捕捉邻域高差）
      - 物理位置路：保留显式 (x, y, z) 绝对坐标
      - 拼接后作为 K/V 输入：既表达“这里有坑/边缘”，也保留“它在什么位置”
    - 注意力权重具有明确的空间可解释性
    - Sim2Real 友好：几何信息在仿真和真实世界中一致
    
    基于 VHIP + 平坦度的注意力引导机制：
    - Query 仅使用本体感知 (proprioception)
    - 注意力偏置：将 VHIP 安全分数作为偏置加入，直接引导注意力
    - 注意力监督：使用 KL 散度作为 loss，进一步强化学习
    
    输入:
        - height_points: [B, num_x, num_y, 3] 地形高度采样点 (x, y, z) in base frame
        - proprioception: [B, Proprio_Dim] 本体感知 (关节角、IMU等)
        - safety_scores: [B, num_points] 物理安全分数 (可选，用于注意力偏置)
        - beta: float 偏置强度 (可选，用于课程式衰减)
        
    输出:
        - context_vector: [B, output_dim] 注意力加权的地形特征
        - attn_weights: [B, num_x, num_y] 注意力权重 (用于可视化和监督)
    """
    
    def __init__(self,
                 num_points_x: int = 17,              # x方向采样点数
                 num_points_y: int = 11,              # y方向采样点数
                 proprio_dim: int = 57,               # 本体感知维度
                 point_feature_dim: int = 64,         # 几何特征维度（Conv2d输出通道数）
                 hidden_dim: int = 128,               # 隐藏层维度
                 num_heads: int = 4,                  # 注意力头数
                 output_dim: int = 64,                # 输出特征维度
                 use_safety_bias: bool = True,        # 是否使用安全偏置
                 geom_cnn_hidden_channels: int = 16,  # 几何卷积隐藏通道数
                 geom_cnn_layers: int = 2,            # 1 或 2 层 Conv2d
                 **kwargs):                           # 兼容旧配置
        super().__init__()
        
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        self.num_points = num_points_x * num_points_y
        self.proprio_dim = proprio_dim
        self.point_feature_dim = point_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_safety_bias = use_safety_bias
        
        # ========== 新设计: z→Conv2d(几何) + (x,y,z)(位置) 拼接 ==========
        # 上路：仅使用 z 构造高度图像，并用卷积捕捉局部高差
        # 下路：显式保留 (x,y,z) 坐标作为绝对位置
        self.geom_cnn_layers = int(geom_cnn_layers)
        if self.geom_cnn_layers not in (1, 2):
            raise ValueError(f"geom_cnn_layers must be 1 or 2, got {geom_cnn_layers}")
        
        if self.geom_cnn_layers == 1:
            self.geom_cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=point_feature_dim, kernel_size=3, padding=1),
                nn.ELU(),
            )
        else:
            self.geom_cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=geom_cnn_hidden_channels, kernel_size=3, padding=1),
                nn.ELU(),
                nn.Conv2d(in_channels=geom_cnn_hidden_channels, out_channels=point_feature_dim, kernel_size=3, padding=1),
                nn.ELU(),
            )
        
        # K/V 的输入维度 = 几何特征 + 显式位置坐标 (x,y,z)
        self.kv_input_dim = point_feature_dim + 3
        
        # ========== Query 生成器 ==========
        # Query = MLP(proprioception)
        # 只使用本体感知
        query_input_dim = proprio_dim
        self.query_mlp = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ========== Key 和 Value 投影 ==========
        # Key: 用于计算注意力分数
        # Value: 用于加权聚合
        self.key_proj = nn.Linear(self.kv_input_dim, hidden_dim)
        self.value_proj = nn.Linear(self.kv_input_dim, hidden_dim)
        
        # ========== 注意力分数缩放 ==========
        self.num_heads = num_heads
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        self.head_dim = hidden_dim // num_heads
        self.scale = (self.head_dim) ** -0.5
        
        # ========== 输出融合层 ==========
        # 融合 attended feature + global pooled feature
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 保存注意力权重用于可视化和监督
        self.last_attn_weights = None
        self.last_raw_attn_scores = None  # 保存原始注意力分数（用于 loss 计算）
    
    def forward(self, 
                height_points: torch.Tensor,
                proprioception: torch.Tensor,
                safety_scores: torch.Tensor = None,
                beta=0.0,
                **kwargs) -> tuple:
        """
        前向传播
        
        Args:
            height_points: [B, num_x, num_y, 3] 地形高度点 (x, y, z) in base frame
                          或 [B, num_points, 3] 展平形式
            proprioception: [B, Proprio_Dim] 本体感知
            safety_scores: [B, num_points] 物理安全分数 (可选)
            beta: float 偏置强度 (可选，用于课程式衰减)
            
        Returns:
            context_vector: [B, output_dim] 注意力加权的地形特征
            attn_weights: [B, num_x, num_y] 注意力权重热力图
        """
        B = height_points.shape[0]
        device = height_points.device
        
        # 处理输入形状
        if height_points.dim() == 4:
            # [B, num_x, num_y, 3] -> [B, num_points, 3]
            num_x, num_y = height_points.shape[1], height_points.shape[2]
            height_points_flat = height_points.view(B, -1, 3)
        else:
            # 已经是 [B, num_points, 3]
            height_points_flat = height_points
            num_x, num_y = self.num_points_x, self.num_points_y
        
        num_points = height_points_flat.shape[1]
        
        # ========== Step A: 两路构建每点特征 (几何Conv + 显式坐标) ==========
        # A.0: 将高度点整理成网格形式，便于构造高度图
        if height_points.dim() == 4:
            height_points_grid = height_points  # [B, num_x, num_y, 3]
        else:
            if num_points != num_x * num_y:
                raise ValueError(f"height_points has {num_points} points but expected {num_x}x{num_y}={num_x*num_y}")
            height_points_grid = height_points_flat.view(B, num_x, num_y, 3)
        
        # 上路（几何视觉路）：z -> [B,1,num_x,num_y] -> Conv2d -> 每点几何特征
        z_grid = height_points_grid[:, :, :, 2]  # [B, num_x, num_y]
        z_img = z_grid.unsqueeze(1)  # [B, 1, num_x, num_y]
        geom_feat_map = self.geom_cnn(z_img)  # [B, point_feature_dim, num_x, num_y]
        geom_feat = geom_feat_map.permute(0, 2, 3, 1).contiguous().view(B, num_points, self.point_feature_dim)  # [B, N, C]
        
        # 下路（物理位置路）：显式 (x,y,z)
        coords = height_points_flat.to(dtype=geom_feat.dtype)  # [B, N, 3]
        
        # 汇合：拼接几何特征 + 坐标
        point_features = torch.cat([geom_feat, coords], dim=-1)  # [B, N, C+3]
        
        # ========== Step B: 构建 Query (仅本体感知) ==========
        # 多头：hidden_dim = num_heads * head_dim
        query = self.query_mlp(proprioception)  # [B, hidden_dim]
        query = query.view(B, self.num_heads, self.head_dim).unsqueeze(2)  # [B, Heads, 1, head_dim]
        
        # ========== Step C: 生成 Key 和 Value ==========
        keys = self.key_proj(point_features)      # [B, N, hidden_dim]
        values = self.value_proj(point_features)  # [B, N, hidden_dim]
        keys = keys.view(B, num_points, self.num_heads, self.head_dim).transpose(1, 2)    # [B, Heads, N, head_dim]
        values = values.view(B, num_points, self.num_heads, self.head_dim).transpose(1, 2)  # [B, Heads, N, head_dim]
        
        # ========== Step D: 计算注意力分数（真正多头并行） ==========
        # raw_attention = (Q × K^T) / √d_head
        raw_attn_scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale  # [B, Heads, 1, N]
        raw_attn_scores = raw_attn_scores.squeeze(2)  # [B, Heads, N]
        
        # 保存原始注意力分数（用于调试/分析；保持兼容：取 heads 平均后存为 [B,N]）
        self.last_raw_attn_scores = raw_attn_scores.mean(dim=1).detach()
        
        # ========== Step E: 添加安全偏置 (VHIP + 平坦度引导) ==========
        # safety_scores: [B, N] -> [B, 1, N]，广播到每个 head: [B, Heads, N]
        if self.use_safety_bias and safety_scores is not None:
            safety = safety_scores.unsqueeze(1)  # [B, 1, N]
            if isinstance(beta, torch.Tensor):
                beta_term = beta.view(-1, 1, 1).to(dtype=safety.dtype, device=safety.device)  # [B,1,1]
            else:
                beta_term = float(beta)
            biased_attn_scores = raw_attn_scores + beta_term * safety  # [B, Heads, N]
        else:
            biased_attn_scores = raw_attn_scores
        
        # ========== Step F: Softmax 获得注意力权重 ==========
        attn_probs = torch.softmax(biased_attn_scores, dim=-1)  # [B, Heads, N]
        
        # ========== Step G: 加权聚合 Value（每个 head 独立） ==========
        attended = torch.matmul(attn_probs.unsqueeze(2), values)  # [B, Heads, 1, head_dim]
        attended = attended.squeeze(2)  # [B, Heads, head_dim]
        attended = attended.reshape(B, self.hidden_dim)  # [B, hidden_dim]  拼回多头输出
        
        # ========== Step H: 输出融合 ==========
        # H.1: 全局地形特征作为补充 (平均池化)
        global_feature = point_features.mean(dim=1)  # [B, point_feature_dim]
        global_feature = self.key_proj(global_feature)  # [B, hidden_dim]
        
        # H.2: 融合注意力输出和全局特征
        fused_output = torch.cat([attended, global_feature], dim=-1)  # [B, hidden_dim * 2]
        context_vector = self.output_mlp(fused_output)  # [B, output_dim]
        
        # H.3: 重塑注意力权重用于可视化/监督
        # 多头下返回一个单通道 attention：对 heads 取平均，保持兼容 [B, N]
        attn_weights = attn_probs.mean(dim=1)  # [B, N]
        self.last_attn_weights = attn_weights.view(B, num_x, num_y)
        
        return context_vector, self.last_attn_weights
    
    def get_attention_weights(self) -> torch.Tensor:
        """获取最近一次的注意力权重，用于可视化"""
        return self.last_attn_weights
    
    def get_attention_weights_flat(self) -> torch.Tensor:
        """获取展平的注意力权重 [B, num_points]，用于 loss 计算"""
        if self.last_attn_weights is None:
            return None
        B = self.last_attn_weights.shape[0]
        return self.last_attn_weights.view(B, -1)


class ActorCriticDepth(nn.Module):
    """
    Actor-Critic 网络，使用深度图像 + 基于高度点的空间注意力
    
    地形感知方式：
    - 深度图像: 通过 CNN 编码，提取全局地形语义特征
    - 高度点空间注意力: 在显式的地形高度采样点上进行注意力，
                       获得具有明确空间可解释性的地形特征
    """
    is_recurrent = False
    
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        his_encoder_dims=[1024, 512, 128],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        his_latent_dim = 64 + 3,
                        history_dim = 570,
                        activation='elu',
                        init_noise_std=1.0,
                        max_grad_norm=10.0,
                        # ========== 空间感知注意力参数 (基于高度点) ==========
                        use_spatial_attention=False,        # 是否启用空间感知注意力
                        num_points_x=17,                    # x方向高度点数 (对应 measured_points_x)
                        num_points_y=11,                    # y方向高度点数 (对应 measured_points_y)
                        point_feature_dim=64,               # 每个点的特征维度
                        attention_hidden_dim=128,           # 注意力隐藏维度
                        attention_heads=4,                  # 注意力头数
                        attention_output_dim=64,            # 注意力输出维度
                        # ==========================================
                        # ========== LIP + 平坦度注意力引导参数 ==========
                        use_safety_bias=True,               # 是否使用物理安全偏置
                        use_attention_loss=True,            # 是否使用注意力监督 loss
                        use_curriculum_decay=True,          # 是否启用课程式衰减
                        beta_max=2.0,                       # 初期偏置强度
                        beta_min=0.0,                       # 后期偏置强度
                        decay_curriculum_levels=10,         # 衰减所需的课程等级数
                        # ==========================================
                        **kwargs):
        if kwargs:
            print("ActorCriticDepth.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDepth, self).__init__()
        activation = get_activation(activation)

        self.his_latent_dim = his_latent_dim
        self.max_grad_norm = max_grad_norm
        self.use_spatial_attention = use_spatial_attention
        self.num_actor_obs = num_actor_obs
        self.num_points_x = num_points_x
        self.num_points_y = num_points_y
        
        # ========== LIP + 平坦度注意力引导参数 ==========
        self.use_safety_bias = use_safety_bias
        self.use_attention_loss = use_attention_loss
        self.use_curriculum_decay = use_curriculum_decay
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.decay_curriculum_levels = decay_curriculum_levels
        self.current_beta = beta_max  # 当前偏置强度（会随课程衰减）
        # ================================================

        # depth encoder (保留用于提取全局深度特征)
        depth_backbone = DepthOnlyFCBackbone58x87(output_dim=128, output_activation=activation)
        self.depth_encoder = StackDepthEncoder(depth_backbone, buffer_len=2)
        self.depth_output_dim = depth_backbone.output_dim  # 128

        # ========== 基于高度点的空间感知注意力模块 ==========
        if use_spatial_attention:
            print("=" * 60)
            print("  基于地形高度点的空间感知注意力 (Height Point Attention) ENABLED")
            print("  使用 LIP + 平坦度注意力引导机制 (Query 仅使用本体感知)")
            print("=" * 60)
            print(f"  高度点网格: {num_points_x} x {num_points_y} = {num_points_x * num_points_y} 点")
            print(f"  每点特征维度: {point_feature_dim}")
            print(f"  注意力头数: {attention_heads}")
            print(f"  输出维度: {attention_output_dim}")
            print(f"  安全偏置: {'ENABLED' if use_safety_bias else 'DISABLED'}")
            print(f"  注意力监督 Loss: {'ENABLED' if use_attention_loss else 'DISABLED'}")
            if use_curriculum_decay:
                print(f"  课程式 β 衰减: β_max={beta_max} → β_min={beta_min} (levels={decay_curriculum_levels})")
            
            # 基于高度点的空间感知注意力编码器
            self.spatial_attention = HeightPointAttentionEncoder(
                num_points_x=num_points_x,
                num_points_y=num_points_y,
                proprio_dim=num_actor_obs,
                point_feature_dim=point_feature_dim,
                hidden_dim=attention_hidden_dim,
                num_heads=attention_heads,
                output_dim=attention_output_dim,
                use_safety_bias=use_safety_bias
            )
            
            # Actor输入维度: obs + history + depth + spatial_attention
            mlp_input_dim_a = num_actor_obs + his_latent_dim + self.depth_output_dim + attention_output_dim
            print(f"  Actor输入维度: {mlp_input_dim_a}")
            print("=" * 60)
        else:
            print("=" * 60)
            print("  空间感知注意力 DISABLED")
            print("=" * 60)
            self.spatial_attention = None
            mlp_input_dim_a = num_actor_obs + his_latent_dim + self.depth_output_dim
            print(f"  Actor输入维度 (无空间注意力): {mlp_input_dim_a}")
        
        mlp_input_dim_c = num_critic_obs + his_latent_dim
        
        # 保存最新的注意力权重，供监督 loss 使用
        self.last_attn_weights_flat = None
        
        # History Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(history_dim, his_encoder_dims[0]))
        encoder_layers.append(activation)
        for l in range(len(his_encoder_dims)):
            if l == len(his_encoder_dims) - 1:
                encoder_layers.append(nn.Linear(his_encoder_dims[l], his_latent_dim))
            else:
                encoder_layers.append(nn.Linear(his_encoder_dims[l], his_encoder_dims[l + 1]))
                encoder_layers.append(activation)
        self.history_encoder = nn.Sequential(*encoder_layers)
        
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, history, depth, height_points=None, safety_scores=None, 
            curriculum_level=0, **kwargs):
        """
        根据观测生成动作
        
        Args:
            observations: [B, obs_dim] 本体感知
            history: [B, history_len, obs_dim] 历史观测
            depth: [B, buffer_len, H, W] 深度图像
            height_points: [B, num_x, num_y, 3] 地形高度采样点 (启用空间注意力时必须)
            safety_scores: [B, num_points] 物理安全分数 (用于注意力偏置)
            curriculum_level: float 当前课程等级 (用于动态 β 衰减)
            
        Returns:
            actions: [B, num_actions] 采样的动作
        """
        # 编码历史信息
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        
        # 编码深度图像
        depth_feature = self.depth_encoder(depth)  # [B, 128]
        
        # 基于高度点的空间感知注意力
        if self.use_spatial_attention and height_points is not None:
            # 计算当前 β (课程式衰减)
            beta = self._get_beta(curriculum_level)
            # 记录一个标量用于日志/对比（避免 tensor 导致 float() 失败）
            if isinstance(beta, torch.Tensor):
                self.current_beta = float(beta.mean().item())
            else:
                self.current_beta = float(beta)
            
            # 使用高度点空间注意力 (带安全偏置)
            attn_feature, attn_weights = self.spatial_attention(
                height_points=height_points,
                proprioception=observations,
                safety_scores=safety_scores,
                beta=beta
            )
            
            # 保存注意力权重用于监督 loss
            B = attn_weights.shape[0]
            self.last_attn_weights_flat = attn_weights.view(B, -1)
            
            # 拼接所有特征
            actor_input = torch.cat((observations, his_feature, depth_feature, attn_feature), dim=-1)
        else:
            # 原始逻辑（无空间注意力）
            actor_input = torch.cat((observations, his_feature, depth_feature), dim=-1)
            self.last_attn_weights_flat = None
        
        self.update_distribution(actor_input)
        return self.distribution.sample()
    
    def _get_beta(self, curriculum_level):
        """
        根据课程等级计算当前的偏置强度 β
        
        公式: β = β_max - (β_max - β_min) × (level / max_level)
        """
        if not self.use_curriculum_decay:
            return self.beta_max
        
        # 线性衰减（支持标量或逐样本 Tensor）
        max_level = float(self.decay_curriculum_levels)
        if isinstance(curriculum_level, torch.Tensor):
            level = curriculum_level.to(dtype=torch.float)
            progress = torch.clamp(level / max_level, 0.0, 1.0)
            beta = self.beta_max - (self.beta_max - self.beta_min) * progress
            return beta
        else:
            progress = min(float(curriculum_level) / max_level, 1.0)
            beta = self.beta_max - (self.beta_max - self.beta_min) * progress
            return beta
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, depth, height_points=None, safety_scores=None, 
                      beta_inference=0.5, **kwargs):
        """
        推理时生成动作（无采样噪声）
        
        Args:
            observations: [B, obs_dim] 本体感知
            history: [B, history_len, obs_dim] 历史观测
            depth: [B, buffer_len, H, W] 深度图像
            height_points: [B, num_x, num_y, 3] 地形高度采样点
            safety_scores: [B, num_points] 物理安全分数 (用于注意力偏置)
            beta_inference: float 推理时的偏置强度 (默认 0.5 作为安全保障)
            
        Returns:
            actions_mean: [B, num_actions] 动作均值
        """
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        depth_feature = self.depth_encoder(depth)  # [B, 128]
        
        # 基于高度点的空间感知注意力
        if self.use_spatial_attention and height_points is not None:
            # 推理时使用固定的 β 值
            attn_feature, attn_weights = self.spatial_attention(
                height_points=height_points,
                proprioception=observations,
                safety_scores=safety_scores,
                beta=beta_inference
            )
            
            actor_input = torch.cat((observations, his_feature, depth_feature, attn_feature), dim=-1)
        else:
            actor_input = torch.cat((observations, his_feature, depth_feature), dim=-1)
        
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def get_attention_weights(self):
        """获取最近一次的注意力权重，用于可视化"""
        if self.use_spatial_attention and self.spatial_attention is not None:
            return self.spatial_attention.get_attention_weights()
        return None
    
    def get_attention_weights_flat(self):
        """获取展平的注意力权重 [B, num_points]，用于 loss 计算"""
        return self.last_attn_weights_flat
    
    def get_attention_stats(self):
        """
        获取注意力统计量，用于tensorboard记录
        """
        if not self.use_spatial_attention or self.spatial_attention is None:
            return None
        
        attn_weights = self.spatial_attention.last_attn_weights  # [B, num_x, num_y]
        if attn_weights is None:
            return None
        
        with torch.no_grad():
            # 展平为 [B, num_points]
            attn_flat = attn_weights.view(attn_weights.shape[0], -1)
            num_elements = attn_flat.shape[-1]
            
            # 1. 注意力熵: 衡量分布集中程度
            attn_probs = attn_flat + 1e-8  # 防止log(0)
            entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1).mean()
            
            # 2. 注意力峰值: 最大注意力值
            peak_value = attn_flat.max(dim=-1)[0].mean()
            
            # 3. 注意力稀疏度: top-k的注意力占总注意力的比例
            k = min(10, num_elements)  # top-10
            topk_values, _ = torch.topk(attn_flat, k=k, dim=-1)
            sparsity = (topk_values.sum(dim=-1) / (attn_flat.sum(dim=-1) + 1e-8)).mean()
            
            # 4. 当前 β 值
            current_beta = self.current_beta
            
            return {
                'entropy': entropy.item(),
                'peak_value': peak_value.item(),
                'sparsity': sparsity.item(),
                'current_beta': current_beta,
            }
    
    def evaluate(self, critic_observations, history, **kwargs):
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        actor_input = torch.cat((critic_observations, his_feature), dim=-1)
        value = self.critic(actor_input)
        return value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
