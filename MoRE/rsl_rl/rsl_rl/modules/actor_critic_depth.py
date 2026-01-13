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

import numpy as np

import torch
import torch.nn as nn
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
        images_compressed = self.image_compression(images.unsqueeze(1)) # [bs * 2, 1 64 64]
        latent = self.output_activation(images_compressed)

        return latent

class StackDepthEncoder(nn.Module):
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
        # depth_image shape: [batch_size, num, 58, 87]
        depth_latent = self.base_backbone(depth_image.flatten(0, 1))  # [batch_size * num, 128]
        depth_latent = depth_latent.reshape(depth_image.shape[0], depth_image.shape[1], -1)  # [batch_size, num, 128]
        depth_latent = self.conv1d(depth_latent) # [batch_size, 16, 62]
        depth_latent = self.mlp(depth_latent.flatten(1, 2))
        return depth_latent


class FootholdPredictor(nn.Module):
    """
    落足点预测辅助头 (Foothold Prediction Auxiliary Head)
    
    预测相对于 Raibert 启发式点的修正量 (Residual Prediction)。
    
    输入: context_vector [B, input_dim] - 来自 SpatialAttentionEncoder 的输出
    输出: delta_footholds [B, 2, 3] - 左右腿的 (Δx, Δy, Δz) 修正量
    
    坐标系: Base Frame (基座坐标系)
        - X: 机器人前方
        - Y: 机器人左侧  
        - Z: 垂直向上
    
    梯度流说明:
        当 loss_foot.backward() 执行时:
        loss_foot → FootholdPredictor → context_vector → SpatialAttentionEncoder 
        → feature_map → DepthEncoder
        
        这会强迫整个视觉编码管道学习地形几何特征。
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 2 legs * 3 coordinates = 6
        )
        
        # 初始化最后一层为小值，使初始输出接近 0（即接近 Raibert 点）
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
    
    def forward(self, context_vector: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            context_vector: [B, input_dim] 来自 SpatialAttentionEncoder 的输出
            
        Returns:
            delta_footholds: [B, 2, 3] 落足点修正量
                - dim 1: 0=Left leg, 1=Right leg
                - dim 2: (Δx, Δy, Δz) in base frame
        """
        delta = self.mlp(context_vector)  # [B, 6]
        return delta.view(-1, 2, 3)  # [B, 2, 3]


class SpatialAttentionEncoder(nn.Module):
    """
    空间感知注意力编码器 (Spatially-Aware Query Attention Encoder)
    
    核心思想：基于 Raibert 启发式公式计算"名义落足点"作为空间先验，
    结合本体感知信息生成 Query，主动查询地形特征图中相关区域。
    
    输入:
        - feature_map: [B, C, H, W] 上游 CNN/Backbone 输出的地形特征图
        - proprioception: [B, Proprio_Dim] 本体感知 (关节角、IMU等)
        - cmd_vel: [B, 3] 用户指令速度
        
    输出:
        - context_vector: [B, output_dim] 注意力加权的地形特征
        - attn_weights: [B, H, W] 注意力权重 (用于可视化)
    
    Raibert 公式 (简化版，部署友好):
        p_nominal = (T_stance / 2) * v_cmd
    """
    
    def __init__(self,
                 feature_channels: int = 128,       # 特征图通道数 C
                 feature_height: int = 5,           # 特征图高度 H (来自depth encoder)
                 feature_width: int = 5,            # 特征图宽度 W
                 proprio_dim: int = 57,             # 本体感知维度
                 hidden_dim: int = 128,             # 隐藏层维度
                 num_heads: int = 4,                # 注意力头数
                 output_dim: int = 64,              # 输出特征维度
                 T_stance: float = 0.25,            # 站立相时间 (秒)
                 learnable_T_stance: bool = True):  # 是否让 T_stance 可学习
        super().__init__()
        
        self.feature_channels = feature_channels
        self.feature_height = feature_height
        self.feature_width = feature_width
        self.proprio_dim = proprio_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # ========== Raibert 参数 ==========
        if learnable_T_stance:
            self.T_stance = nn.Parameter(torch.tensor(T_stance))
        else:
            self.register_buffer('T_stance', torch.tensor(T_stance))
        
        # ========== Step B: Query 生成器 ==========
        # Query = MLP(proprioception + p_nominal)
        # p_nominal: [B, 2] (只用 x, y 两个维度)
        query_input_dim = proprio_dim + 2  # proprio + nominal foothold (x, y)
        self.query_mlp = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ========== Step C: 位置编码 + 特征融合 ==========
        # 1x1 卷积将 (C + 2) 通道融合为 hidden_dim
        self.coord_fusion = nn.Conv2d(
            in_channels=feature_channels + 2,  # C + 2 (x, y 坐标)
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # ========== Step D: Key 和 Value 投影 ==========
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # ========== Step D: 多头交叉注意力 ==========
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # ========== Step E: 输出融合层 ==========
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # attended + global pooling
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 缓存：预生成坐标网格 (需要在 forward 时根据设备动态创建)
        self._coord_grid = None
        self._grid_shape = None
        
        # 保存注意力权重用于可视化
        self.last_attn_weights = None
        self.last_nominal_foothold = None
    
    def _get_coord_grid(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        生成归一化的 2D 坐标网格 [-1, 1]
        
        Args:
            H: 特征图高度
            W: 特征图宽度
            device: 目标设备
            
        Returns:
            coord_grid: [1, 2, H, W] 坐标网格
        """
        # 检查缓存是否有效
        if self._coord_grid is not None and self._grid_shape == (H, W) and self._coord_grid.device == device:
            return self._coord_grid
        
        # 生成归一化坐标 [-1, 1]
        # y 坐标: 从 -1 (顶部) 到 1 (底部)
        # x 坐标: 从 -1 (左侧) 到 1 (右侧)
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        
        # 创建网格 meshgrid
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')  # [H, W], [H, W]
        
        # 堆叠为 [2, H, W] 并添加 batch 维度
        coord_grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)  # [1, 2, H, W]
        
        # 缓存
        self._coord_grid = coord_grid
        self._grid_shape = (H, W)
        
        return coord_grid
    
    def compute_nominal_foothold(self, cmd_vel: torch.Tensor) -> torch.Tensor:
        """
        使用 Raibert 启发式公式计算名义落足点
        
        公式: p_nominal = (T_stance / 2) * v_cmd
        
        Args:
            cmd_vel: [B, 3] 指令速度 (x, y, yaw)
            
        Returns:
            p_nominal: [B, 2] 名义落足点位置 (x, y)
        """
        # 只使用 x, y 分量
        v_cmd = cmd_vel[:, :2]  # [B, 2]
        
        # 根据指令速度计算期望落足点
        p_nominal = (self.T_stance / 2.0) * v_cmd
        
        return p_nominal  # [B, 2]
    
    def forward(self, 
                feature_map: torch.Tensor,
                proprioception: torch.Tensor,
                cmd_vel: torch.Tensor) -> tuple:
        """
        前向传播
        
        Args:
            feature_map: [B, C, H, W] 上游特征图
            proprioception: [B, Proprio_Dim] 本体感知
            cmd_vel: [B, 3] 指令速度
            
        Returns:
            context_vector: [B, output_dim] 注意力加权的地形特征
            attn_weights: [B, H, W] 注意力权重热力图
        """
        B, C, H, W = feature_map.shape
        device = feature_map.device
        
        # ========== Step A: 计算名义落足点 (Raibert Heuristic) ==========
        p_nominal = self.compute_nominal_foothold(cmd_vel)  # [B, 2]
        self.last_nominal_foothold = p_nominal  # 保存用于可视化/奖励
        
        # ========== Step B: 构建空间感知 Query ==========
        # 拼接 proprioception 和 p_nominal
        query_input = torch.cat([proprioception, p_nominal], dim=-1)  # [B, proprio_dim + 2]
        query = self.query_mlp(query_input).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # ========== Step C: 构建包含位置信息的 Key 和 Value ==========
        # C.1: 生成坐标网格
        coord_grid = self._get_coord_grid(H, W, device)  # [1, 2, H, W]
        coord_grid = coord_grid.expand(B, -1, -1, -1)    # [B, 2, H, W]
        
        # C.2: 拼接特征图和坐标网格
        feature_with_coords = torch.cat([feature_map, coord_grid], dim=1)  # [B, C+2, H, W]
        
        # C.3: 通过 1x1 卷积融合坐标信息
        fused_features = self.coord_fusion(feature_with_coords)  # [B, hidden_dim, H, W]
        
        # C.4: 展平并重排维度 [B, hidden_dim, H, W] -> [B, H*W, hidden_dim]
        fused_features = fused_features.flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]
        
        # C.5: 生成 Key 和 Value
        keys = self.key_proj(fused_features)    # [B, H*W, hidden_dim]
        values = self.value_proj(fused_features)  # [B, H*W, hidden_dim]
        
        # ========== Step D: 多头注意力机制 ==========
        # Query: [B, 1, hidden_dim]
        # Keys:  [B, H*W, hidden_dim]
        # Values: [B, H*W, hidden_dim]
        attended, attn_weights = self.cross_attention(
            query=query,
            key=keys,
            value=values
        )  # attended: [B, 1, hidden_dim], attn_weights: [B, 1, H*W]
        
        # ========== Step E: 输出融合 ==========
        # E.1: 全局地形特征作为补充 (平均池化)
        global_feature = fused_features.mean(dim=1)  # [B, hidden_dim]
        
        # E.2: 融合注意力输出和全局特征
        attended = attended.squeeze(1)  # [B, hidden_dim]
        fused_output = torch.cat([attended, global_feature], dim=-1)  # [B, hidden_dim * 2]
        context_vector = self.output_mlp(fused_output)  # [B, output_dim]
        
        # E.3: 重塑注意力权重用于可视化 [B, 1, H*W] -> [B, H, W]
        self.last_attn_weights = attn_weights.squeeze(1).view(B, H, W)
        
        return context_vector, self.last_attn_weights
    
    def get_nominal_foothold(self) -> torch.Tensor:
        """获取最近一次计算的名义落足点"""
        return self.last_nominal_foothold
    
    def get_attention_weights(self) -> torch.Tensor:
        """获取最近一次的注意力权重，用于可视化"""
        return self.last_attn_weights


class ActorCriticDepth(nn.Module):
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
                        # ========== 空间感知注意力参数 ==========
                        use_spatial_attention=False,        # 是否启用空间感知注意力
                        spatial_feature_channels=128,       # 深度编码器输出通道数
                        spatial_feature_height=5,           # 特征图高度 (根据depth encoder)
                        spatial_feature_width=5,            # 特征图宽度
                        attention_hidden_dim=128,           # 注意力隐藏维度
                        attention_heads=4,                  # 注意力头数
                        attention_output_dim=64,            # 注意力输出维度
                        T_stance=0.25,                      # Raibert站立相时间 (秒)
                        learnable_T_stance=True,            # 是否让T_stance可学习
                        # ==========================================
                        # ========== 落足点预测辅助任务参数 ==========
                        use_foothold_predictor=False,       # 是否启用落足点预测辅助任务
                        foothold_predictor_hidden_dim=256,  # FootholdPredictor 隐藏层维度
                        # ==========================================
                        **kwargs):
        if kwargs:
            print("ActorCriticEst.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDepth, self).__init__()
        activation = get_activation(activation)

        self.his_latent_dim = his_latent_dim
        self.max_grad_norm = max_grad_norm
        self.use_spatial_attention = use_spatial_attention
        self.use_foothold_predictor = use_foothold_predictor
        self.num_actor_obs = num_actor_obs

        # depth encoder
        depth_backbone = DepthOnlyFCBackbone58x87(output_dim=128, output_activation=activation)
        self.depth_encoder = StackDepthEncoder(depth_backbone, buffer_len=2)
        self.depth_output_dim = depth_backbone.output_dim  # 128

        # ========== 空间感知注意力模块 ==========
        if use_spatial_attention:
            print("========== 空间感知注意力 (Spatial Attention) ENABLED ==========")
            
            # 空间感知注意力编码器
            self.spatial_attention = SpatialAttentionEncoder(
                feature_channels=spatial_feature_channels,
                feature_height=spatial_feature_height,
                feature_width=spatial_feature_width,
                proprio_dim=num_actor_obs,
                hidden_dim=attention_hidden_dim,
                num_heads=attention_heads,
                output_dim=attention_output_dim,
                T_stance=T_stance,
                learnable_T_stance=learnable_T_stance
            )
            
            # 特征图投影层：将 depth_encoder 的 1D 输出重塑为 2D 特征图
            # depth_encoder 输出 128 维，需要投影到 [C, H, W] = [128, 5, 5]
            self.feature_reshape_dim = spatial_feature_channels * spatial_feature_height * spatial_feature_width
            self.feature_proj = nn.Linear(self.depth_output_dim, self.feature_reshape_dim)
            self.spatial_feature_shape = (spatial_feature_channels, spatial_feature_height, spatial_feature_width)
            
            # Actor输入维度: obs + history + depth + spatial_attention
            mlp_input_dim_a = num_actor_obs + his_latent_dim + self.depth_output_dim + attention_output_dim
            print(f"Actor输入维度 (带空间注意力): {mlp_input_dim_a}")
            
            # ========== 落足点预测辅助头 ==========
            if use_foothold_predictor:
                print("========== 落足点预测辅助任务 (Foothold Predictor) ENABLED ==========")
                self.foothold_predictor = FootholdPredictor(
                    input_dim=attention_output_dim,
                    hidden_dim=foothold_predictor_hidden_dim
                )
                print(f"FootholdPredictor: input_dim={attention_output_dim}, hidden_dim={foothold_predictor_hidden_dim}, output=[B,2,3]")
            else:
                print("========== 落足点预测辅助任务 DISABLED ==========")
                self.foothold_predictor = None
            # =====================================
        else:
            print("========== 空间感知注意力 DISABLED ==========")
            self.spatial_attention = None
            self.feature_proj = None
            self.foothold_predictor = None  # 没有空间注意力就没有落足点预测
            mlp_input_dim_a = num_actor_obs + his_latent_dim + self.depth_output_dim
            print(f"Actor输入维度 (无空间注意力): {mlp_input_dim_a}")
        # ============================================
        
        mlp_input_dim_c = num_critic_obs + his_latent_dim
        
        # 保存名义落足点，供环境计算奖励使用
        self.last_nominal_foothold = None
        # 保存预测的落足点修正量，供训练时计算 Aux Loss 使用
        self.last_pred_footholds = None
        
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
    # not used at the moment
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

    def act(self, observations, history, depth, cmd_vel=None, **kwargs):
        """
        根据观测生成动作
        
        Args:
            observations: [B, obs_dim] 本体感知
            history: [B, history_len, obs_dim] 历史观测
            depth: [B, buffer_len, H, W] 深度图像
            cmd_vel: [B, 3] 指令速度 (启用空间注意力时必须)
            
        Returns:
            actions: [B, num_actions] 采样的动作
        """
        # 编码历史信息
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        
        # 编码深度图像
        depth_feature = self.depth_encoder(depth)  # [B, 128]
        
        # 空间感知注意力
        if self.use_spatial_attention and cmd_vel is not None:
            # 将深度特征投影并重塑为 2D 特征图
            feature_flat = self.feature_proj(depth_feature)  # [B, C*H*W]
            B = feature_flat.shape[0]
            feature_map = feature_flat.view(B, *self.spatial_feature_shape)  # [B, C, H, W]
            
            # 使用空间感知注意力
            attn_feature, _ = self.spatial_attention(
                feature_map=feature_map,
                proprioception=observations,
                cmd_vel=cmd_vel
            )
            self.last_nominal_foothold = self.spatial_attention.get_nominal_foothold()
            
            # ========== 落足点预测辅助任务 ==========
            # 从 attn_feature (context_vector) 预测落足点修正量
            # 梯度流: loss_foot → FootholdPredictor → attn_feature → SpatialAttentionEncoder → feature_map → DepthEncoder
            if self.foothold_predictor is not None:
                self.last_pred_footholds = self.foothold_predictor(attn_feature)  # [B, 2, 3]
            else:
                self.last_pred_footholds = None
            # =========================================
            
            # 拼接所有特征
            actor_input = torch.cat((observations, his_feature, depth_feature, attn_feature), dim=-1)
        else:
            # 原始逻辑
            actor_input = torch.cat((observations, his_feature, depth_feature), dim=-1)
            self.last_nominal_foothold = None
            self.last_pred_footholds = None
        
        self.update_distribution(actor_input)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, depth, cmd_vel=None, **kwargs):
        """
        推理时生成动作（无采样噪声）
        
        Args:
            observations: [B, obs_dim] 本体感知
            history: [B, history_len, obs_dim] 历史观测
            depth: [B, buffer_len, H, W] 深度图像
            cmd_vel: [B, 3] 指令速度
            
        Returns:
            actions_mean: [B, num_actions] 动作均值
        """
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        depth_feature = self.depth_encoder(depth)  # [B, 128]
        
        # 空间感知注意力
        if self.use_spatial_attention and cmd_vel is not None:
            # 将深度特征投影并重塑为 2D 特征图
            feature_flat = self.feature_proj(depth_feature)  # [B, C*H*W]
            B = feature_flat.shape[0]
            feature_map = feature_flat.view(B, *self.spatial_feature_shape)  # [B, C, H, W]
            
            # 使用空间感知注意力
            attn_feature, _ = self.spatial_attention(
                feature_map=feature_map,
                proprioception=observations,
                cmd_vel=cmd_vel
            )
            self.last_nominal_foothold = self.spatial_attention.get_nominal_foothold()
            
            # ========== 落足点预测（推理时也计算，用于可视化） ==========
            if self.foothold_predictor is not None:
                self.last_pred_footholds = self.foothold_predictor(attn_feature)  # [B, 2, 3]
            else:
                self.last_pred_footholds = None
            # =========================================================
            
            actor_input = torch.cat((observations, his_feature, depth_feature, attn_feature), dim=-1)
        else:
            actor_input = torch.cat((observations, his_feature, depth_feature), dim=-1)
            self.last_nominal_foothold = None
            self.last_pred_footholds = None
        
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def get_nominal_foothold(self):
        """获取最近一次计算的名义落足点 (Raibert Heuristic)"""
        return self.last_nominal_foothold
    
    def get_pred_footholds(self):
        """
        获取最近一次预测的落足点修正量 [B, 2, 3]
        
        用于训练时计算 Aux Loss:
            target_footholds = P_oracle - P_raibert  (来自环境的 Oracle 搜索)
            loss_foot = masked_mse(pred_footholds, target_footholds)
        """
        return self.last_pred_footholds
    
    def get_attention_weights(self):
        """获取最近一次的注意力权重，用于可视化"""
        if self.use_spatial_attention and self.spatial_attention is not None:
            return self.spatial_attention.get_attention_weights()
        return None
    
    def get_attention_stats(self):
        """
        获取注意力统计量，用于tensorboard记录
        
        返回:
            dict: 包含以下指标:
                - entropy: 注意力分布熵 (越低说明注意力越集中)
                - peak_value: 注意力最大值 (越高说明有明确的关注焦点)
                - sparsity: 注意力稀疏度 (top-k占总注意力的比例)
                - T_stance: 当前的站立相时间参数
        """
        if not self.use_spatial_attention or self.spatial_attention is None:
            return None
        
        attn_weights = self.spatial_attention.last_attn_weights  # [B, H, W]
        if attn_weights is None:
            return None
        
        with torch.no_grad():
            # 展平为 [B, H*W]
            attn_flat = attn_weights.view(attn_weights.shape[0], -1)
            num_elements = attn_flat.shape[-1]
            
            # 1. 注意力熵: 衡量分布集中程度
            attn_probs = attn_flat + 1e-8  # 防止log(0)
            entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1).mean()
            
            # 2. 注意力峰值: 最大注意力值
            peak_value = attn_flat.max(dim=-1)[0].mean()
            
            # 3. 注意力稀疏度: top-k的注意力占总注意力的比例
            k = min(5, num_elements)  # 对于小特征图，使用 top-5
            topk_values, _ = torch.topk(attn_flat, k=k, dim=-1)
            sparsity = (topk_values.sum(dim=-1) / (attn_flat.sum(dim=-1) + 1e-8)).mean()
            
            # 4. T_stance 参数
            T_stance = self.spatial_attention.T_stance.item()
            
            return {
                'entropy': entropy.item(),
                'peak_value': peak_value.item(),
                'sparsity': sparsity.item(),
                'T_stance': T_stance,
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
