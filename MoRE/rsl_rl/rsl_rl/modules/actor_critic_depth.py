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

from rsl_rl.modules.terrain_safety_scorer import TerrainSafetyScorer


class TerrainAttentionEncoder(nn.Module):
    """
    双路地形特征提取 + 多头交叉注意力
    - 上路: 5x5 CNN 提取几何特征（高度差、梯度等）
    - 下路: 原始 (x,y,z) 坐标直接使用
    - MHA 内部处理 K/V 投影，q_proj 仅做维度对齐 (query_dim → hidden_dim)
    - use_pre_ln: 可选，在 Attention 之前对 Q/KV 做 LayerNorm
    """
    def __init__(self, 
                 grid_h=17, 
                 grid_w=11,
                 obs_dim=57,
                 query_dim=None,
                 hidden_dim=128,
                 num_heads=8,
                 output_dim=64,
                 use_pre_ln=True):  # 新增参数：是否使用 Pre-LN
        super().__init__()
        
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_points = grid_h * grid_w  # 187
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads  # 保存用于 attn_mask 形状扩展
        self.use_pre_ln = use_pre_ln  # 保存配置
        
        # ===== 上路: 5x5 CNN 提取几何特征 =====
        # CNN输出维度 = hidden_dim - 3，留3维给xyz坐标
        cnn_out_dim = hidden_dim - 3
        self.height_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # [B,1,17,11] -> [B,32,17,11]
            nn.ReLU(),
            nn.Conv2d(32, cnn_out_dim, kernel_size=5, padding=2),  # -> [B,hidden-3,17,11]
            nn.ReLU(),
        )
        
        # ===== Query 投影 (query_dim → hidden_dim) =====
        # 为兼容旧参数名，若未显式传入 query_dim，则回退到 obs_dim
        self.query_dim = query_dim if query_dim is not None else obs_dim
        self.q_proj = nn.Linear(self.query_dim, hidden_dim)
        
        # ===== Pre-LN: 在 Attention 之前对 Q 和 K/V 分别归一化 =====
        # 修正 CNN(ReLU, ≥0) 与 xyz([-1,1]) 拼接后的尺度不对称
        if use_pre_ln:
            self.q_norm = nn.LayerNorm(hidden_dim)
            self.kv_norm = nn.LayerNorm(hidden_dim)
        
        # ===== 多头注意力 (MHA内部处理K/V投影) =====
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            kdim=hidden_dim,  # K的输入维度
            vdim=hidden_dim,  # V的输入维度
            batch_first=True
        )
        
        # ===== 输出投影 =====
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # 存储注意力权重用于可视化和监控
        self.last_attention_weights = None       # [B, 187] 平均后权重（detached，用于日志/可视化）
        self.last_attention_weights_per_head = None  # [B, num_heads, 187] per-head 权重（detached）
        self.last_attention_weights_raw = None   # [B, 187] 平均后权重（保留计算图，用于 KL loss）
        
    def forward(self, height_map, terrain_xyz, query_context):
        """
        Args:
            height_map: [B, 17, 11] 高度图
            terrain_xyz: [B, 187, 3] 每个采样点的 (x,y,z) 坐标（机体坐标系）
            query_context: [B, query_dim] Query 输入（可由 obs / history 特征拼接得到）
            
        Returns:
            terrain_feature: [B, output_dim] 注意力加权后的地形特征
            
        数据流 (use_pre_ln=True 时 Q/KV 经过 LayerNorm):
            query_context ──► q_proj ──► [LN] ──► Q ──────────────┐
                                                                  │
            terrain ──► CNN+xyz ──► kv_input ──► [LN] ──► K,V ────┤
                                                                  │
                                                    MultiheadAttention(Q, K, V)
                                                                  │
                                                                  ▼
                                            attn_output ──► output_proj ──► terrain_feature
        """
        B = height_map.shape[0]
        
        # ===== 上路: CNN 提取几何特征 =====
        h = height_map.unsqueeze(1)  # [B, 1, 17, 11]
        cnn_feat = self.height_cnn(h)  # [B, hidden-3, 17, 11]
        cnn_feat = cnn_feat.permute(0, 2, 3, 1)  # [B, 17, 11, hidden-3]
        cnn_feat = cnn_feat.reshape(B, self.num_points, -1)  # [B, 187, hidden-3]
        
        # ===== 下路: 直接使用原始坐标 xyz =====
        xyz = terrain_xyz  # [B, 187, 3]
        
        # ===== 拼接: CNN几何特征 + xyz位置坐标 =====
        kv_input = torch.cat([cnn_feat, xyz], dim=-1)  # [B, 187, hidden]
        
        # ===== Query 投影 =====
        Q = self.q_proj(query_context).unsqueeze(1)  # [B, 1, hidden]
        
        # ===== Pre-LN: 在 Attention 之前归一化 (可选，用于兼容旧模型) =====
        if self.use_pre_ln:
            Q_input = self.q_norm(Q)           # [B, 1, hidden] - Query 归一化
            kv_input_normed = self.kv_norm(kv_input)  # [B, 187, hidden] - K/V 归一化
        else:
            Q_input = Q
            kv_input_normed = kv_input
        
        # ===== 多头交叉注意力 =====
        attn_output, attn_weights_per_head = self.multihead_attn(
            query=Q_input,
            key=kv_input_normed,
            value=kv_input_normed,
            average_attn_weights=False
        )  # attn_output: [B, 1, hidden], attn_weights_per_head: [B, num_heads, 1, 187]
        
        # 保存 per-head 权重 (detached): [B, num_heads, 187]
        self.last_attention_weights_per_head = attn_weights_per_head.squeeze(2).detach()
        # 保存平均权重 (detached): [B, 187]
        self.last_attention_weights = self.last_attention_weights_per_head.mean(dim=1)
        # 保留计算图的平均权重，供 KL loss 回传梯度
        self.last_attention_weights_raw = attn_weights_per_head.squeeze(2).mean(dim=1)  # [B, 187]
        
        # ===== 输出投影 =====
        terrain_feature = self.output_proj(attn_output.squeeze(1))  # [B, output_dim]
        
        return terrain_feature


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
                        # ===== 地形注意力参数 =====
                        use_terrain_attention=False,
                        terrain_attn_grid_h=17,
                        terrain_attn_grid_w=11,
                        terrain_attn_hidden_dim=128,
                        terrain_attn_num_heads=8,
                        terrain_attn_output_dim=64,
                        terrain_attn_use_pre_ln=True,   # Pre-LN: 对 Q/KV 做 LayerNorm
                        # ===== 消融实验开关 =====
                        include_depth_in_actor=True,  # 是否在 actor 输入中包含 depth_feature
                        terrain_attn_query_with_history=True,  # 注意力 Query 是否包含历史特征
                        # ===== KL 先验引导 =====
                        use_attn_kl_loss=False,       # 是否使用 KL 散度辅助损失引导注意力
                        **kwargs):
        if kwargs:
            print("ActorCriticEst.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDepth, self).__init__()
        activation = get_activation(activation)

        self.his_latent_dim = his_latent_dim
        self.max_grad_norm = max_grad_norm
        self.use_terrain_attention = use_terrain_attention
        self.include_depth_in_actor = include_depth_in_actor
        self.use_attn_kl_loss = use_attn_kl_loss
        self.terrain_attn_query_with_history = terrain_attn_query_with_history

        # depth encoder
        depth_backbone = DepthOnlyFCBackbone58x87(output_dim=128, output_activation=activation)
        self.depth_encoder = StackDepthEncoder(depth_backbone, buffer_len=2)

        # ===== 地形注意力编码器 (Actor 用) =====
        terrain_attn_dim = 0
        if use_terrain_attention:
            if terrain_attn_query_with_history:
                terrain_attn_query_dim = num_actor_obs + his_latent_dim
            else:
                terrain_attn_query_dim = num_actor_obs
            self.terrain_attention = TerrainAttentionEncoder(
                grid_h=terrain_attn_grid_h,
                grid_w=terrain_attn_grid_w,
                query_dim=terrain_attn_query_dim,
                hidden_dim=terrain_attn_hidden_dim,
                num_heads=terrain_attn_num_heads,
                output_dim=terrain_attn_output_dim,
                use_pre_ln=terrain_attn_use_pre_ln
            )
            terrain_attn_dim = terrain_attn_output_dim
            print(f"Actor TerrainAttentionEncoder enabled: grid={terrain_attn_grid_h}x{terrain_attn_grid_w}, "
                  f"heads={terrain_attn_num_heads}, output_dim={terrain_attn_output_dim}, "
                  f"query_dim={terrain_attn_query_dim}, query_with_history={terrain_attn_query_with_history}, "
                  f"use_pre_ln={terrain_attn_use_pre_ln}")
        else:
            self.terrain_attention = None
        
        # ===== KL 先验引导打分器（无可学习参数） =====
        if use_attn_kl_loss and use_terrain_attention:
            self.terrain_safety_scorer = TerrainSafetyScorer(
                grid_h=terrain_attn_grid_h,
                grid_w=terrain_attn_grid_w,
                temperature=0.5,
                label_smooth=0.01,
            )
            print(f"TerrainSafetyScorer enabled for KL loss guidance")
        else:
            self.terrain_safety_scorer = None
            if use_attn_kl_loss and not use_terrain_attention:
                print("[Warning] use_attn_kl_loss=True but use_terrain_attention=False, KL loss disabled")

        # Actor 输入维度: obs + his_feature + (depth_feature if include) + (terrain_feature if attention)
        depth_dim = depth_backbone.output_dim if include_depth_in_actor else 0
        mlp_input_dim_a = num_actor_obs + his_latent_dim + depth_dim + terrain_attn_dim
        
        if not include_depth_in_actor:
            print(f"[Ablation] depth_feature EXCLUDED from actor input (include_depth_in_actor=False)")
        
        # Critic 输入: critic_obs (privileged_obs，已包含完整 heights 187维) + his_feature
        mlp_input_dim_c = num_critic_obs + his_latent_dim
        
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

    def act(self, observations, history, depth, height_map=None, terrain_xyz=None, 
            base_lin_vel=None, **kwargs):
        """
        Args:
            observations: [B, obs_dim] 本体感知观测
            history: [B, history_len, history_dim] 历史观测
            depth: [B, buffer_len, H, W] 深度图
            height_map: [B, grid_h, grid_w] 高度图（可选）
            terrain_xyz: [B, num_points, 3] 地形采样点坐标（可选）
            base_lin_vel: [B, 3] 机体线速度（用于 KL loss 中的 TerrainSafetyScorer，可选）
        """
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        
        depth_feature = self.depth_encoder(depth)
        
        actor_input_parts = [observations, his_feature]
        
        if self.include_depth_in_actor:
            actor_input_parts.append(depth_feature)
        
        if self.use_terrain_attention and height_map is not None and terrain_xyz is not None:
            if self.terrain_attn_query_with_history:
                query_context = torch.cat([observations, his_feature], dim=-1)
            else:
                query_context = observations
            terrain_feature = self.terrain_attention(height_map, terrain_xyz, query_context)
            actor_input_parts.append(terrain_feature)
        
        actor_input = torch.cat(actor_input_parts, dim=-1)

        self.update_distribution(actor_input)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, depth, height_map=None, terrain_xyz=None,
                       **kwargs):
        """推理模式（无采样，直接返回均值动作）"""
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        depth_feature = self.depth_encoder(depth)
        
        actor_input_parts = [observations, his_feature]
        
        if self.include_depth_in_actor:
            actor_input_parts.append(depth_feature)
        
        if self.use_terrain_attention and height_map is not None and terrain_xyz is not None:
            if self.terrain_attn_query_with_history:
                query_context = torch.cat([observations, his_feature], dim=-1)
            else:
                query_context = observations
            terrain_feature = self.terrain_attention(height_map, terrain_xyz, query_context)
            actor_input_parts.append(terrain_feature)
        
        actor_input = torch.cat(actor_input_parts, dim=-1)
            
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def compute_terrain_kl_loss(self, height_map, base_lin_vel):
        """
        计算 KL(prior || attn) 辅助损失。
        
        prior 来自 TerrainSafetyScorer（detach，仅作目标分布），
        attn 来自注意力权重（保留计算图，梯度回传更新 CNN / q_proj 等）。
        
        Args:
            height_map: [B, grid_h, grid_w]
            base_lin_vel: [B, 3]
        Returns:
            kl_loss: scalar tensor (batch 平均)
        """
        if self.terrain_safety_scorer is None or self.terrain_attention is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        attn_weights = self.terrain_attention.last_attention_weights_raw  # [B, 187]，带梯度
        if attn_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        with torch.no_grad():  # 下面缩进块里的运算不记录梯度（只是目标分布，不是要训练的东西）
            prior_dist = self.terrain_safety_scorer(height_map, base_lin_vel)  # [B, 187]
        
        # KL(prior || attn) = Σ prior * log(prior / attn)
        kl = torch.sum(
            prior_dist * (torch.log(prior_dist + 1e-8) - torch.log(attn_weights + 1e-8)),
            dim=-1
        )
        return kl.mean()

    def evaluate(self, critic_observations, history, **kwargs):
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        critic_input = torch.cat((critic_observations, his_feature), dim=-1)
        value = self.critic(critic_input)
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
