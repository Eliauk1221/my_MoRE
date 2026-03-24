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

import time
import os
from collections import deque
import statistics
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import wandb

from rsl_rl.algorithms.amp_ppo_multi import AMPPPOMulti
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.actor_critic_depth import ActorCriticDepth
from rsl_rl.env import VecEnv
from rsl_rl.algorithms.amp_discriminator_multi import AMPDiscriminatorMulti
from legged_gym.datasets.motion_loader_g1 import G1_AMPLoader
from legged_gym.utils.utils import Normalizer
from rsl_rl.utils.terrain_visualizer import visualize_training_sample

class AMPOnPolicyRunnerMulti:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.wandb_run_name = (
            datetime.now().strftime("%b%d_%H-%M-%S")
            + "_"
            + train_cfg["runner"]["experiment_name"]
            + "_"
            + train_cfg["runner"]["run_name"]
        )
        self.policy_cfg = train_cfg["policy"]
        self.all_cfg = train_cfg
        self.device = device
        self.env = env
        self.obs_history_len = self.env.obs_history_len
        self.use_depth = False
        if self.env.cfg.depth.use_camera or self.env.cfg.depth.warp_camera:  # 如果环境配置里说要用相机深度图，或者 warp camera 开启
            self.use_depth = True
            self.depth_shape = self.env.cfg.depth.resized

        self.use_amp = self.alg_cfg["use_amp"]
            
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        num_actor_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.env.num_actions,
                                                        history_dim=self.obs_history_len * (num_actor_obs),
                                                        # history_dim=self.obs_history_len * num_actor_obs,
                                                        **self.policy_cfg).to(self.device)
        # prepare for AMP
        if self.use_amp:
            self.amp_loader_type = self.alg_cfg['amp_loader_type']
            self.amp_loader_class_name = self.alg_cfg['amp_loader_class_name']; del self.alg_cfg['amp_loader_class_name']
            if 'lafan_16dof' in self.alg_cfg['amp_loader_type']:
                self.amp_indices = [i for i in range(16)]
                self.num_amp_obs = self.env.num_amp_obs

            amp_loader_class = eval(self.amp_loader_class_name)
            amp_data = amp_loader_class(
                device, time_between_frames=self.env.dt, preload_transitions=True,
                num_preload_transitions=train_cfg['runner']['amp_num_preload_transitions'],
                motion_dir=self.env.amp_motion_files)
            amp_normalizer = Normalizer(self.num_amp_obs)

            discriminator = AMPDiscriminatorMulti(
                self.num_amp_obs,
                train_cfg['runner']['amp_reward_coef'],
                train_cfg['runner']['amp_discr_hidden_dims'], device,
                train_cfg['runner']['num_amp_frames'],
                train_cfg['runner']['amp_task_reward_lerp'],
                train_cfg['runner']['use_lerp'],
                ).to(self.device)
        else:
            amp_data = None
            amp_normalizer = None
            discriminator = None

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: AMPPPOMulti = alg_class(actor_critic, discriminator, amp_data, amp_normalizer,
                                             use_depth=self.use_depth, device=self.device, num_amp_frames=train_cfg['runner']['num_amp_frames'], **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.num_amp_frames = train_cfg['runner']['num_amp_frames']

        # init storage and model
        # 获取地形注意力配置
        terrain_attn_grid_h = None
        terrain_attn_grid_w = None
        if hasattr(self.env.cfg, 'terrain_attention') and self.env.cfg.terrain_attention.use_attention:
            terrain_attn_grid_h = self.env.cfg.terrain_attention.grid_h
            terrain_attn_grid_w = self.env.cfg.terrain_attention.grid_w
        
        # ===== KL 先验引导配置 =====
        self.use_attn_kl_loss = False
        self.attn_kl_coef = 0.1
        self.attn_kl_anneal_start = 0
        self.attn_kl_anneal_end = 0
        
        if hasattr(self.env.cfg, 'terrain_attention'):
            ta_cfg = self.env.cfg.terrain_attention
            self.use_attn_kl_loss = getattr(ta_cfg, 'use_attn_kl_loss', False)
            self.attn_kl_coef = getattr(ta_cfg, 'attn_kl_coef', 0.1)
            self.attn_kl_anneal_start = getattr(ta_cfg, 'attn_kl_anneal_start', 0)
            self.attn_kl_anneal_end = getattr(ta_cfg, 'attn_kl_anneal_end', 0)
        
        if self.use_attn_kl_loss:
            print(f"[AttnKL] Enabled: coef={self.attn_kl_coef}, "
                  f"anneal_start={self.attn_kl_anneal_start}, anneal_end={self.attn_kl_anneal_end}")
        
        # ===== 训练可视化配置 =====
        self.viz_enabled = False
        self.viz_interval = 500
        self.viz_num_samples = 2
        
        if hasattr(self.env.cfg, 'terrain_attention'):
            ta_cfg = self.env.cfg.terrain_attention
            self.viz_enabled = getattr(ta_cfg, 'viz_enabled', False)
            self.viz_interval = getattr(ta_cfg, 'viz_interval', 500)
            self.viz_num_samples = getattr(ta_cfg, 'viz_num_samples', 2)
        
        if self.viz_enabled:
            print(f"[TrainViz] Enabled: interval={self.viz_interval}, num_samples={self.viz_num_samples}")
        
        self.alg.init_storage(self.env.num_envs, 
                              self.num_steps_per_env, 
                              [num_actor_obs], 
                              [num_critic_obs], 
                              [self.env.num_actions], 
                              self.obs_history_len, 
                              self.env.num_obs,
                              depth_shape=self.depth_shape if self.use_depth else None,
                              depth_buffer_len=self.env.cfg.depth.buffer_len if self.use_depth else None,
                              terrain_attn_grid_h=terrain_attn_grid_h,
                              terrain_attn_grid_w=terrain_attn_grid_w)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        # ===== 地形指标相关 =====
        # 地形类型名称 (与 terrain_dict 顺序一致)
        self.terrain_names = ['stepping_stones', 'parkour', 'pit', 'gap', 'stair']
        self.num_terrain_types = len(self.terrain_names)
        # 获取地形长度用于计算 traverse rate
        self.terrain_length = getattr(self.env.cfg.terrain, 'terrain_length', 14.0)
        # success 判定阈值（traverse_rate >= 0.9 视为"真正通关"）
        self.success_threshold = getattr(self.env.cfg.terrain, 'success_threshold', 0.9)
        # 阶段性里程碑阈值，用于观察训练中期进展
        self.success_milestones = [0.5, 0.8]

        _, _ = self.env.reset()
    
    def compute_effective_kl_coef(self, current_iter: int) -> float:
        """
        计算当前迭代有效的 KL loss 系数（支持可选的线性 warmup）。
        
        - anneal_start=0, anneal_end=0: 直接返回 attn_kl_coef（不退火）
        - anneal_start > 0: 在该 iteration 之前返回 0
        - anneal_end > anneal_start: 从 start 到 end 线性增长到 attn_kl_coef
        """
        if not self.use_attn_kl_loss:
            return 0.0  # 当前 KL loss 的有效权重为 0，完全不参与总 loss
        
        if self.attn_kl_anneal_end <= self.attn_kl_anneal_start:
            if current_iter < self.attn_kl_anneal_start:
                return 0.0
            return self.attn_kl_coef  # 如果已经不小于 anneal_start，直接返回 attn_kl_coef
        
        if current_iter < self.attn_kl_anneal_start:
            return 0.0
        
        progress = min(1.0, (current_iter - self.attn_kl_anneal_start) /
                       (self.attn_kl_anneal_end - self.attn_kl_anneal_start))  # 计算当前 warmup 进度，保持在 0-1 之间
        return self.attn_kl_coef * progress
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        if self.use_amp:
            amp_obs = self.env.get_amp_observations(); amp_obs = amp_obs[:, self.amp_indices]
            amp_obs = amp_obs.to(self.device)
            self.alg.discriminator.train()
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        # process trajectory history
        self.trajectory_history = torch.zeros(size=(self.env.num_envs, self.obs_history_len, self.env.num_obs), device=self.device)  # 用来存每个环境最近若干步地观测历史
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs.unsqueeze(1)), dim=1)  # 将当前 obs 插入历史队列末尾，同时丢掉最旧的一帧
        if self.use_amp:
            self.amp_obs_frames = torch.zeros(size=(self.env.num_envs, self.num_amp_frames, self.env.num_amp_obs), device=self.device)
            self.amp_obs_frames = torch.concat((self.amp_obs_frames[:, 1:], amp_obs.unsqueeze(1)), dim=1)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        discrewbuffer = deque(maxlen=100)
        step_discrewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        
        # ===== 按地形类型细分的指标 buffer =====
        # traverse_rate: 前进方向位移 / 地形长度（连续值）
        # survival: episode 存活到 max_episode_length（未跌倒）
        # success: traverse_rate >= success_threshold（真正通关）
        # milestone: traverse_rate >= 阶段阈值（观察中期进展）
        traverse_buffers = {name: deque(maxlen=50) for name in self.terrain_names}
        survival_buffers = {name: deque(maxlen=50) for name in self.terrain_names}
        success_buffers = {name: deque(maxlen=50) for name in self.terrain_names}
        milestone_buffers = {
            ms: {name: deque(maxlen=50) for name in self.terrain_names}
            for ms in self.success_milestones
        }

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_disc_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_single_step_disc_rew = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        tot_iter = self.current_learning_iteration + num_learning_iterations
        infos = {}
        infos["depth"] = self.env.warp_depth_buffer.clone().to(self.device) if self.use_depth else None
        
        # ===== 地形注意力支持 =====
        self.use_terrain_attention = getattr(self.env, 'use_terrain_attention', False)

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            
            # ===== 计算当前有效 KL 系数 =====
            effective_kl_coef = self.compute_effective_kl_coef(it)
            
            # Rollout
            with torch.inference_mode():  # 只做前向推理，不计算梯度
                for i in range(self.num_steps_per_env):
                    history = self.trajectory_history
                    if infos["depth"] is not None:  # 检查深度图数据是否存在
                        depth_image = infos['depth']
                    if self.use_depth:  # 检查模型是否要用深度图
                        obs = (obs, depth_image)  # 将原来的 obs 改为一个元组
                    
                    # ===== 获取地形注意力数据 =====
                    terrain_data = None
                    if self.use_terrain_attention:
                        terrain_data = {
                            'height_map': self.env.height_map.clone().to(self.device),
                            'terrain_xyz': self.env.terrain_xyz.clone().to(self.device),
                        }
                        if self.use_attn_kl_loss:
                            terrain_data['base_lin_vel'] = self.env.base_lin_vel.clone().to(self.device)

                    actions = self.alg.act(obs, critic_obs, history, terrain_data=terrain_data)

                    obs, privileged_obs, rewards, dones, infos, _, terminal_amp_states, terminal_obs, terminal_critic_obs = self.env.step(actions)
                    
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)

                    # Account for terminal states.
                    env_ids = dones.nonzero(as_tuple=False).flatten()                    

                    next_obs = torch.clone(obs)
                    next_obs[env_ids] = terminal_obs

                    next_critic_obs = torch.clone(privileged_obs)
                    next_critic_obs[env_ids] = terminal_critic_obs

                    if self.use_amp:
                        next_amp_obs = self.env.get_amp_observations(); next_amp_obs = next_amp_obs[:, self.amp_indices]
                        next_amp_obs = next_amp_obs.to(self.device)
                        terminal_amp_states = terminal_amp_states[:, self.amp_indices]
                        next_amp_obs_with_term = torch.clone(next_amp_obs)
                        next_amp_obs_with_term[env_ids] = terminal_amp_states
                        self.amp_obs_frames = torch.concat((self.amp_obs_frames[:, 1:], next_amp_obs_with_term.unsqueeze(1)), dim=1)
                        rewards, logit, disc_reward = self.alg.discriminator.predict_amp_reward(
                            self.amp_obs_frames, rewards, normalizer=self.alg.amp_normalizer)

                        amp_obs = torch.clone(next_amp_obs)
                        self.alg.process_env_step(rewards, dones, infos, next_obs, next_critic_obs, self.amp_obs_frames)
                    else:
                        self.alg.process_env_step(rewards, dones, infos, next_obs, next_critic_obs)

                    # process trajectory history
                    self.trajectory_history[env_ids] = 0
                    self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs.unsqueeze(1)), dim=1)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        if self.use_amp:
                            cur_disc_reward_sum += disc_reward
                            cur_single_step_disc_rew += disc_reward
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        discrewbuffer.extend(cur_disc_reward_sum[new_ids][:, 0].cpu().numpy().tolist())

                        to_extend_disc = (cur_single_step_disc_rew[new_ids] / self.env.max_episode_length_s)[:, 0].cpu().numpy()
                        step_discrewbuffer.extend(to_extend_disc.tolist())
                        
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        # ===== 按地形类型记录指标 =====
                        if len(new_ids) > 0 and "terminal_root_pos_xy" in infos:
                            done_env_ids = new_ids[:, 0]
                            
                            terminal_pos = infos["terminal_root_pos_xy"].cpu()
                            terminal_origins = infos["terminal_env_origins_xy"].cpu()
                            terrain_types = infos["terminal_env_class"].long().cpu().numpy()
                            
                            forward_distance = (terminal_pos[:, 0] - terminal_origins[:, 0]).clamp(min=0).numpy()
                            traverse_rates = forward_distance / self.terrain_length
                            
                            is_survived = (cur_episode_length[done_env_ids].cpu().numpy() >= self.env.max_episode_length - 1)
                            is_success = (traverse_rates >= self.success_threshold)
                            
                            for idx, (t_type, t_rate, survived, success) in enumerate(
                                    zip(terrain_types, traverse_rates, is_survived, is_success)):
                                if 0 <= t_type < self.num_terrain_types:
                                    terrain_name = self.terrain_names[int(t_type)]
                                    traverse_buffers[terrain_name].append(t_rate)
                                    survival_buffers[terrain_name].append(float(survived))
                                    success_buffers[terrain_name].append(float(success))
                                    for ms in self.success_milestones:
                                        milestone_buffers[ms][terrain_name].append(float(t_rate >= ms))
                        
                        cur_reward_sum[new_ids] = 0
                        cur_disc_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                        cur_single_step_disc_rew[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                history = self.trajectory_history
                self.alg.compute_returns(critic_obs, history)
            
            mean_value_loss, mean_surrogate_loss, mean_amp_loss, mean_grad_pen_loss, \
            mean_policy_pred, mean_expert_pred, mean_agent_acc, mean_demo_acc, \
            mean_terrain_attn_grad_norm, mean_terrain_kl_loss, \
            mean_prior_attn_cosine, mean_prior_entropy = self.alg.update(
                attn_kl_coef=effective_kl_coef)
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            
            # ===== 训练可视化 =====
            if self.viz_enabled and it % self.viz_interval == 0 and self.use_terrain_attention:
                try:
                    viz_save_dir = os.path.join(self.log_dir, 'visualizations')
                    visualize_training_sample(
                        env=self.env,
                        actor_critic=self.alg.actor_critic,
                        iteration=it,
                        save_dir=viz_save_dir,
                        sample_indices=None,
                        terrain_names=self.terrain_names,
                    )
                except Exception as e:
                    print(f"[TrainViz] Warning: visualization failed at iter {it}: {e}")
            
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP', locs['mean_amp_loss'], locs['it'])
        self.writer.add_scalar('Loss/AMP_grad', locs['mean_grad_pen_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.policy_learning_rate, locs['it'])
        self.writer.add_scalar('Disc/agent_acc', locs['mean_agent_acc'], locs['it'])
        self.writer.add_scalar('Disc/demo_acc', locs['mean_demo_acc'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if 'mean_terrain_attn_grad_norm' in locs:
            self.writer.add_scalar('Attention/grad_norm', locs['mean_terrain_attn_grad_norm'], locs['it'])
        if self.use_terrain_attention and hasattr(self.env, 'height_clip_ratio') and self.env.height_clip_ratio is not None:
            clip_ratio = self.env.height_clip_ratio
            if isinstance(clip_ratio, torch.Tensor):
                clip_ratio = clip_ratio.detach().item()
            self.writer.add_scalar('Attention/height_clip_ratio', clip_ratio, locs['it'])
        
        # ===== 地形注意力指标 =====
        if self.use_terrain_attention and hasattr(self.alg.actor_critic, 'terrain_attention'):
            terrain_attn = self.alg.actor_critic.terrain_attention
            if terrain_attn is not None and terrain_attn.last_attention_weights is not None:
                attn_weights = terrain_attn.last_attention_weights  # [B, 187]
                
                # 注意力熵 (越低越专注) — 平均后权重
                entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1).mean()
                self.writer.add_scalar('Attention/entropy_avg', entropy.item(), locs['it'])
                
                # ===== Per-head 注意力熵 =====
                per_head_w = getattr(terrain_attn, 'last_attention_weights_per_head', None)
                if per_head_w is not None:
                    # per_head_w: [B, num_heads, 187]
                    per_head_entropy = -torch.sum(
                        per_head_w * torch.log(per_head_w + 1e-8), dim=-1
                    )  # [B, num_heads]
                    head_entropy_mean = per_head_entropy.mean().item()
                    head_entropy_std = per_head_entropy.std().item()
                    head_entropy_min = per_head_entropy.min(dim=-1).values.mean().item()
                    head_entropy_max = per_head_entropy.max(dim=-1).values.mean().item()
                    self.writer.add_scalar('Attention/per_head_entropy_mean', head_entropy_mean, locs['it'])
                    self.writer.add_scalar('Attention/per_head_entropy_std', head_entropy_std, locs['it'])
                    self.writer.add_scalar('Attention/per_head_entropy_min', head_entropy_min, locs['it'])
                    self.writer.add_scalar('Attention/per_head_entropy_max', head_entropy_max, locs['it'])
                
                # Top-10 权重占比 (越高越集中)
                top10_values, _ = torch.topk(attn_weights, 10, dim=-1)
                top10_ratio = top10_values.sum(dim=-1).mean()
                self.writer.add_scalar('Attention/top10_ratio', top10_ratio.item(), locs['it'])
                
                # 最大权重 (单点最大关注度)
                max_weight = attn_weights.max(dim=-1).values.mean()
                self.writer.add_scalar('Attention/max_weight', max_weight.item(), locs['it'])
                
                # 前方区域权重 (假设前半部分是前方)
                num_points = attn_weights.shape[-1]
                front_weight = attn_weights[:, :num_points//2].sum(dim=-1).mean()
                self.writer.add_scalar('Attention/front_region_weight', front_weight.item(), locs['it'])
        
        # ===== KL 先验引导指标 =====
        if self.use_attn_kl_loss:
            self.writer.add_scalar('Loss/terrain_kl', locs['mean_terrain_kl_loss'], locs['it'])
            self.writer.add_scalar('Loss/terrain_kl_weighted',
                                   locs['effective_kl_coef'] * locs['mean_terrain_kl_loss'], locs['it'])
            self.writer.add_scalar('Attention/kl_coef_effective', locs['effective_kl_coef'], locs['it'])
            
            # Prior-Attention 对齐指标（来自 update() 中配对数据的累积均值）
            self.writer.add_scalar('Attention/prior_attn_cosine', locs['mean_prior_attn_cosine'], locs['it'])
            self.writer.add_scalar('Attention/prior_entropy', locs['mean_prior_entropy'], locs['it'])
        
        # ===== 按地形类型的 Traverse Rate 和 Success Rate =====
        if 'traverse_buffers' in locs and 'success_buffers' in locs:
            traverse_buffers = locs['traverse_buffers']
            success_buffers = locs['success_buffers']
            
            total_traverse = []
            total_success = []
            
            for terrain_name in self.terrain_names:
                if len(traverse_buffers[terrain_name]) > 0:
                    mean_traverse = statistics.mean(traverse_buffers[terrain_name])
                    self.writer.add_scalar(f'Terrain/{terrain_name}/traverse_rate', mean_traverse, locs['it'])
                    total_traverse.extend(traverse_buffers[terrain_name])
                    
                if len(success_buffers[terrain_name]) > 0:
                    mean_success = statistics.mean(success_buffers[terrain_name])
                    self.writer.add_scalar(f'Terrain/{terrain_name}/success_rate', mean_success, locs['it'])
                    total_success.extend(success_buffers[terrain_name])
            
            # 总体指标
            if len(total_traverse) > 0:
                self.writer.add_scalar('Terrain/overall/traverse_rate', statistics.mean(total_traverse), locs['it'])
            if len(total_success) > 0:
                self.writer.add_scalar('Terrain/overall/success_rate', statistics.mean(total_success), locs['it'])
            
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_disc_reward', statistics.mean(locs['discrewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_step_disc_reward', statistics.mean(locs['step_discrewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'AMP loss:':>{pad}} {locs['mean_amp_loss']:.4f}\n"""
                          f"""{'AMP grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                          f"""{'AMP mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                          f"""{'AMP mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                          f"""{'AMP mean policy acc:':>{pad}} {locs['mean_agent_acc']:.4f}\n"""
                          f"""{'AMP mean demo acc:':>{pad}} {locs['mean_demo_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean disc reward:':>{pad}} {statistics.mean(locs['discrewbuffer']):.2f}\n"""
                          f"""{'Step disc reward:':>{pad}} {statistics.mean(locs['step_discrewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict() if self.alg.discriminator is not None else None,
            'amp_normalizer': self.alg.amp_normalizer if self.alg.amp_normalizer is not None else None,
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=lambda storage, loc:storage.cuda(0))
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)
        if loaded_dict['discriminator_state_dict'] is not None:
            self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        if self.alg.discriminator is not None:
            self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
