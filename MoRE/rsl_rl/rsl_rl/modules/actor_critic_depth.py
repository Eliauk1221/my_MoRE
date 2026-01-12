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
    落足点预测器
    
    功能：根据本体感知和当前脚位置，预测下一步左右脚的落足点位置
    
    输入:
        - obs: 本体感知 [B, obs_dim] (包含速度指令、关节状态等)
        - foot_pos: 当前左右脚在身体坐标系下的位置 [B, 6] (左脚xyz + 右脚xyz)
        
    输出:
        - pred_foothold: 预测的落足点偏移 [B, 4] (左脚dx,dy + 右脚dx,dy)
          偏移是相对于机器人身体中心的坐标
    """
    def __init__(self, 
                 obs_dim=57,                    # 本体感知维度
                 foot_pos_dim=6,                # 脚位置维度 (左脚xyz + 右脚xyz)
                 hidden_dims=[256, 128],        # 隐藏层维度
                 output_dim=4,                  # 输出维度 (左脚dx,dy + 右脚dx,dy)
                 foothold_scale=0.5):           # 落足点预测范围 (米)
        super().__init__()
        
        self.foothold_scale = foothold_scale
        input_dim = obs_dim + foot_pos_dim
        
        # 构建MLP网络
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ELU())
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ELU())
        
        # 输出层使用tanh限制范围
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Tanh())
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, obs, foot_pos):
        """
        前向传播
        
        Args:
            obs: [B, obs_dim] 本体感知
            foot_pos: [B, 6] 当前脚位置 (身体坐标系)
            
        Returns:
            pred_foothold: [B, 4] 预测落足点 (左dx,dy, 右dx,dy)
        """
        # 拼接输入
        x = torch.cat([obs, foot_pos], dim=-1)
        
        # 通过MLP预测
        pred = self.mlp(x)
        
        # 缩放到实际物理范围
        pred_foothold = pred * self.foothold_scale
        
        return pred_foothold


class FootholdGuidedAttention(nn.Module):
    """
    落足点引导注意力机制
    
    创新点：使用 本体感知 + 预测落足点 作为Query，引导对地形的注意力
    
    输入:
        - obs: 本体感知 [B, obs_dim]
        - pred_foothold: 预测的落足点 [B, 4]
        - terrain_heights: 地形高度采样 [B, 187]
        
    输出:
        - attended_feature: 注意力加权的地形特征 [B, output_dim]
        - attn_weights: 注意力权重 [B, 17, 11] (用于可视化)
    """
    def __init__(self,
                 obs_dim=57,                # 本体感知维度
                 foothold_dim=4,            # 预测落足点维度
                 terrain_dim=187,           # 地形采样点数 (17x11)
                 hidden_dim=128,            # 隐藏层维度
                 num_heads=4,               # 注意力头数
                 output_dim=64):            # 输出特征维度
        super().__init__()
        
        self.terrain_dim = terrain_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # ========== 1. Query生成器 ==========
        # Query = 本体感知 + 预测落足点
        query_input_dim = obs_dim + foothold_dim
        self.query_net = nn.Sequential(
            nn.Linear(query_input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ========== 2. 地形编码器 ==========
        # 将每个高度点编码为高维特征
        self.terrain_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # ========== 3. 可学习位置编码 ==========
        # 让网络知道每个采样点的空间位置
        self.position_encoding = nn.Parameter(
            torch.randn(1, terrain_dim, hidden_dim) * 0.02
        )
        
        # ========== 4. Key和Value投影 ==========
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # ========== 5. 多头交叉注意力 ==========
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # ========== 6. 输出融合层 ==========
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 保存注意力权重用于可视化
        self.last_attn_weights = None
        
    def forward(self, obs, pred_foothold, terrain_heights):
        """
        前向传播
        
        Args:
            obs: [B, obs_dim] 本体感知
            pred_foothold: [B, 4] 预测落足点
            terrain_heights: [B, 187] 地形高度采样
            
        Returns:
            output: [B, output_dim] 注意力加权的地形特征
            attn_weights: [B, 17, 11] 注意力权重热力图
        """
        B = obs.shape[0]
        
        # Step 1: 生成Query (本体感知 + 预测落足点)
        query_input = torch.cat([obs, pred_foothold], dim=-1)
        query = self.query_net(query_input).unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Step 2: 编码地形高度点
        heights_expanded = terrain_heights.unsqueeze(-1)  # [B, 187, 1]
        terrain_features = self.terrain_encoder(heights_expanded)  # [B, 187, hidden_dim]
        
        # Step 3: 添加位置编码
        terrain_features = terrain_features + self.position_encoding
        
        # Step 4: 生成Key和Value
        keys = self.key_proj(terrain_features)    # [B, 187, hidden_dim]
        values = self.value_proj(terrain_features)  # [B, 187, hidden_dim]
        
        # Step 5: 交叉注意力
        attended, attn_weights = self.cross_attention(
            query=query,
            key=keys,
            value=values
        )  # attended: [B, 1, hidden_dim], attn_weights: [B, 1, 187]
        
        # Step 6: 全局地形特征作为补充
        global_terrain = terrain_features.mean(dim=1)  # [B, hidden_dim]
        
        # Step 7: 融合注意力输出和全局特征
        attended = attended.squeeze(1)  # [B, hidden_dim]
        fused = torch.cat([attended, global_terrain], dim=-1)  # [B, hidden_dim * 2]
        output = self.output_mlp(fused)  # [B, output_dim]
        
        # 保存并reshape注意力权重用于可视化
        self.last_attn_weights = attn_weights.squeeze(1).view(B, 17, 11)
        
        return output, self.last_attn_weights


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
                        # ========== 落足点引导注意力参数 ==========
                        use_foothold_attention=False,       # 是否启用落足点引导注意力
                        foothold_predictor_hidden=[256, 128],  # 落足点预测器隐藏层
                        attention_hidden_dim=128,           # 注意力隐藏维度
                        attention_heads=4,                  # 注意力头数
                        attention_output_dim=64,            # 注意力输出维度
                        foothold_scale=0.5,                 # 落足点预测范围 (米)
                        terrain_dim=187,                    # 地形采样点数
                        foot_pos_dim=6,                     # 脚位置维度
                        # ==========================================
                        **kwargs):
        if kwargs:
            print("ActorCriticEst.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticDepth, self).__init__()
        activation = get_activation(activation)

        self.his_latent_dim = his_latent_dim
        self.max_grad_norm = max_grad_norm
        self.use_foothold_attention = use_foothold_attention
        self.num_actor_obs = num_actor_obs

        # depth encoder
        depth_backbone = DepthOnlyFCBackbone58x87(output_dim=128, output_activation=activation)
        self.depth_encoder = StackDepthEncoder(depth_backbone, buffer_len=2)

        # ========== 落足点引导注意力模块 ==========
        if use_foothold_attention:
            print("========== 落足点引导注意力 ENABLED ==========")
            
            # 落足点预测器
            self.foothold_predictor = FootholdPredictor(
                obs_dim=num_actor_obs,
                foot_pos_dim=foot_pos_dim,
                hidden_dims=foothold_predictor_hidden,
                output_dim=4,  # 左脚dx,dy + 右脚dx,dy
                foothold_scale=foothold_scale
            )
            
            # 落足点引导注意力
            self.foothold_attention = FootholdGuidedAttention(
                obs_dim=num_actor_obs,
                foothold_dim=4,
                terrain_dim=terrain_dim,
                hidden_dim=attention_hidden_dim,
                num_heads=attention_heads,
                output_dim=attention_output_dim
            )
            
            # Actor输入维度: obs + history + depth + attention
            mlp_input_dim_a = num_actor_obs + his_latent_dim + depth_backbone.output_dim + attention_output_dim
            print(f"Actor输入维度 (带注意力): {mlp_input_dim_a}")
        else:
            print("========== 落足点引导注意力 DISABLED ==========")
            self.foothold_predictor = None
            self.foothold_attention = None
            mlp_input_dim_a = num_actor_obs + his_latent_dim + depth_backbone.output_dim
            print(f"Actor输入维度 (无注意力): {mlp_input_dim_a}")
        # ============================================
        
        mlp_input_dim_c = num_critic_obs + his_latent_dim
        
        # 保存预测的落足点，供环境计算奖励使用
        self.last_pred_foothold = None
        
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

    def act(self, observations, history, depth, foot_pos=None, terrain_heights=None, **kwargs):
        """
        根据观测生成动作
        
        Args:
            observations: [B, obs_dim] 本体感知
            history: [B, history_len, obs_dim] 历史观测
            depth: [B, buffer_len, H, W] 深度图像
            foot_pos: [B, 6] 当前脚位置 (启用注意力时需要)
            terrain_heights: [B, 187] 地形高度采样 (启用注意力时需要)
            
        Returns:
            actions: [B, num_actions] 采样的动作
        """
        # 编码历史信息
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        
        # 编码深度图像
        depth_feature = self.depth_encoder(depth)
        
        # 落足点引导注意力
        if self.use_foothold_attention and foot_pos is not None and terrain_heights is not None:
            # 预测落足点
            pred_foothold = self.foothold_predictor(observations, foot_pos)
            self.last_pred_foothold = pred_foothold  # 保存供环境计算奖励
            
            # 使用落足点引导注意力
            attn_feature, _ = self.foothold_attention(observations, pred_foothold, terrain_heights)
            
            # 拼接所有特征
            actor_input = torch.cat((observations, his_feature, depth_feature, attn_feature), dim=-1)
        else:
            # 原始逻辑
            actor_input = torch.cat((observations, his_feature, depth_feature), dim=-1)
            self.last_pred_foothold = None
        
        self.update_distribution(actor_input)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, history, depth, foot_pos=None, terrain_heights=None, **kwargs):
        """
        推理时生成动作（无采样噪声）
        
        Args:
            observations: [B, obs_dim] 本体感知
            history: [B, history_len, obs_dim] 历史观测
            depth: [B, buffer_len, H, W] 深度图像
            foot_pos: [B, 6] 当前脚位置
            terrain_heights: [B, 187] 地形高度采样
            
        Returns:
            actions_mean: [B, num_actions] 动作均值
        """
        history = history.flatten(1)
        his_feature = self.history_encoder(history)
        depth_feature = self.depth_encoder(depth)
        
        # 落足点引导注意力
        if self.use_foothold_attention and foot_pos is not None and terrain_heights is not None:
            pred_foothold = self.foothold_predictor(observations, foot_pos)
            self.last_pred_foothold = pred_foothold
            attn_feature, _ = self.foothold_attention(observations, pred_foothold, terrain_heights)
            actor_input = torch.cat((observations, his_feature, depth_feature, attn_feature), dim=-1)
        else:
            actor_input = torch.cat((observations, his_feature, depth_feature), dim=-1)
            self.last_pred_foothold = None
        
        actions_mean = self.actor(actor_input)
        return actions_mean
    
    def get_pred_foothold(self):
        """获取最近一次预测的落足点"""
        return self.last_pred_foothold
    
    def get_attention_weights(self):
        """获取最近一次的注意力权重，用于可视化"""
        if self.use_foothold_attention and self.foothold_attention is not None:
            return self.foothold_attention.last_attn_weights
        return None
    
    def get_attention_stats(self):
        """
        获取注意力统计量，用于tensorboard记录
        
        返回:
            dict: 包含以下指标:
                - entropy: 注意力分布熵 (越低说明注意力越集中)
                - peak_value: 注意力最大值 (越高说明有明确的关注焦点)
                - sparsity: 注意力稀疏度 (top-10%占总注意力的比例)
        """
        if not self.use_foothold_attention or self.foothold_attention is None:
            return None
        
        attn_weights = self.foothold_attention.last_attn_weights  # [B, 17, 11]
        if attn_weights is None:
            return None
        
        with torch.no_grad():
            # 展平为 [B, 187]
            attn_flat = attn_weights.view(attn_weights.shape[0], -1)
            
            # 1. 注意力熵: 衡量分布集中程度
            # H = -sum(p * log(p)), 最大值为 log(187) ≈ 5.23 (均匀分布)
            attn_probs = attn_flat + 1e-8  # 防止log(0)
            entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1).mean()
            
            # 2. 注意力峰值: 最大注意力值
            peak_value = attn_flat.max(dim=-1)[0].mean()
            
            # 3. 注意力稀疏度: top-10的注意力占总注意力的比例
            # 如果注意力集中在少数点上，这个值接近1
            topk_values, _ = torch.topk(attn_flat, k=min(10, attn_flat.shape[-1]), dim=-1)
            sparsity = (topk_values.sum(dim=-1) / attn_flat.sum(dim=-1)).mean()
            
            return {
                'entropy': entropy.item(),
                'peak_value': peak_value.item(),
                'sparsity': sparsity.item(),
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
