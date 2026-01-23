
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.terrain_safety_scorer import (
    TerrainSafetyScorer, 
    TerrainSafetyScorerCfg,
    AttentionGuidanceCfg,
    get_beta
)

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.datasets.motion_loader_g1 import G1_AMPLoader
import torch
import cv2
import numpy as np
import torch.nn.functional as F

class G1_16Dof_Loco_Robot(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.amp_motion_files = self.cfg.env.amp_motion_files
        self.num_amp_obs = self.cfg.env.num_amp_obs
        if self.cfg.env.reference_state_initialization: # NOTE only for visualize reference motion
            self.amp_loader = G1_AMPLoader(motion_dir=self.amp_motion_files, device=self.device, time_between_frames=self.dt)
            self.motion_reference = self.amp_loader.get_joint_pose_batch_16dof(torch.cat(self.amp_loader.trajectories_full, dim=0))
        
    def get_amp_observations(self):
        return self.dof_pos

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_feet_contact_force = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_feet_contact_force = torch.zeros(self.num_envs, 2, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_indicator_offset = torch.tensor(self.cfg.asset.feet_indicator_offset, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_indicator_pos = torch.zeros(self.num_envs, len(self.feet_indices), *self.feet_indicator_offset.shape,dtype=torch.float, device=self.device, requires_grad=False)
        
        # 存储注意力权重，尺寸根据方案不同:
        #   - 高度点方案: [num_envs, 17, 11]
        # 初始化为 None，在第一次 set_attention_weights 时自动分配
        self.attention_weights = None
        self.attention_shape = None  # 记录注意力权重的形状
        
        # ========== LIP + 平坦度注意力引导相关 ==========
        # 初始化 TerrainSafetyScorer
        self._init_terrain_safety_scorer()
        # 存储安全分数 [num_envs, num_points]
        num_points = len(self.cfg.terrain.measured_points_x) * len(self.cfg.terrain.measured_points_y)
        self.safety_scores = torch.zeros(self.num_envs, num_points, dtype=torch.float, device=self.device, requires_grad=False)
        # 存储目标注意力分布 [num_envs, num_points]
        self.target_attention = torch.zeros(self.num_envs, num_points, dtype=torch.float, device=self.device, requires_grad=False)
        # =============================================
        
        # 复用 buffer：避免每个 step 在 `get_height_points_with_heights()` 里 clone 大张量
        # 该 buffer 的 (x,y) 与 `self.height_points` 一致，只在每步更新 z。
        self._height_points_with_z_buf = None

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_feet_contact_force[env_ids] = 0.
        self.last_last_feet_contact_force[env_ids] = 0.
        if self.attention_weights is not None:
            self.attention_weights[env_ids] = 0.  # 重置注意力权重
        # 重置安全分数和目标注意力
        self.safety_scores[env_ids] = 0.
        self.target_attention[env_ids] = 0.
    
    def _init_terrain_safety_scorer(self):
        """
        初始化 TerrainSafetyScorer
        基于 LIP + 平坦度的注意力引导机制
        """
        # 创建配置
        scorer_cfg = TerrainSafetyScorerCfg()
        scorer_cfg.z0 = 0.75  # 名义 CoM 高度
        scorer_cfg.gravity = 9.81
        scorer_cfg.sigma_flatness = 0.05
        scorer_cfg.use_heading_awareness = True
        scorer_cfg.min_vel_for_heading = 0.1
        # 默认温度（训练时建议由 runner 从 train_cfg.policy.safety_temperature 覆盖）
        scorer_cfg.temperature = 0.5
        
        # 获取采样点数量
        num_points_x = len(self.cfg.terrain.measured_points_x)
        num_points_y = len(self.cfg.terrain.measured_points_y)
        
        # 创建评分器
        self.terrain_safety_scorer = TerrainSafetyScorer(
            cfg=scorer_cfg,
            num_points_x=num_points_x,
            num_points_y=num_points_y
        ).to(self.device)
        
        print(f"[Env] TerrainSafetyScorer 初始化完成: {num_points_x}x{num_points_y} = {num_points_x * num_points_y} 点")

    def configure_terrain_safety_scorer(self,
                                        safety_temperature: float = None,
                                        z0: float = None,
                                        sigma_flatness: float = None,
                                        use_heading_awareness: bool = None,
                                        min_vel_for_heading: float = None):
        """在 env 创建后动态配置 TerrainSafetyScorer（通常由 runner 在训练开始前调用）。

        说明：
            - `TerrainSafetyScorer` 内部大部分参数以 buffer 形式注册，需要同时更新 buffer 才会生效。
            - 该方法仅在初始化/训练启动时调用一次，不属于热路径。
        """
        if not hasattr(self, "terrain_safety_scorer") or self.terrain_safety_scorer is None:
            return

        scorer = self.terrain_safety_scorer

        if safety_temperature is not None:
            t = float(safety_temperature)
            scorer.cfg.temperature = t
            scorer.temperature.fill_(t)

        if z0 is not None:
            z0_f = float(z0)
            scorer.cfg.z0 = z0_f
            scorer.z0.fill_(z0_f)

        if sigma_flatness is not None:
            sf = float(sigma_flatness)
            scorer.cfg.sigma_flatness = sf
            scorer.sigma_flatness.fill_(sf)

        if min_vel_for_heading is not None:
            mv = float(min_vel_for_heading)
            scorer.cfg.min_vel_for_heading = mv
            scorer.min_vel_for_heading.fill_(mv)

        if use_heading_awareness is not None:
            scorer.cfg.use_heading_awareness = bool(use_heading_awareness)
    
    def compute_safety_scores(self, height_points: torch.Tensor = None):
        """
        计算地形安全分数和目标注意力分布
        
        基于 LIP + 平坦度 + 速度方向感知
        
        Returns:
            safety_scores: [num_envs, num_points] 安全分数
            target_attention: [num_envs, num_points] 目标注意力分布
        """
        if not self.cfg.terrain.measure_heights:
            return self.safety_scores, self.target_attention
        
        if not isinstance(self.measured_heights, torch.Tensor):
            return self.safety_scores, self.target_attention
        
        # 获取带高度的地形采样点（若调用方已提供则复用，避免重复构建）
        if height_points is None:
            height_points = self.get_height_points_with_heights()
        if height_points is None:
            return self.safety_scores, self.target_attention
        
        # 获取基座速度（水平航向坐标系 / Horizontal Heading Frame）
        # - X 指向机头方向（yaw）
        # - Z 严格与重力反方向平行（不随 pitch/roll 旋转）
        #
        # 说明：
        #   `self.base_lin_vel` 是用完整姿态 quat 做 inverse rotate 得到的 body frame 速度，
        #   当机器人有 pitch/roll 时其 (x,y) 不是“水平”速度分量，会导致 TerrainSafetyScorer
        #   的 heading_factor / VHIP 在几何点云(heading frame)下失配。
        #
        # 因此这里用 **严格 yaw-only quaternion** 把 world velocity 转到 heading frame。
        # 注意：不能用“把 quat 的 (x,y) 置零再 normalize”的近似法；那会让 yaw 仍随 roll/pitch 耦合变化。
        world_lin_vel = self.root_states[:, 7:10]  # [num_envs, 3] in world frame
        q = self.base_quat
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        half = 0.5 * yaw
        base_quat_yaw = torch.zeros_like(q)
        base_quat_yaw[:, 2] = torch.sin(half)
        base_quat_yaw[:, 3] = torch.cos(half)
        base_lin_vel_hh = quat_rotate_inverse(base_quat_yaw, world_lin_vel)  # [num_envs, 3]
        
        # 计算安全分数
        with torch.no_grad():
            safety_scores, target_attention = self.terrain_safety_scorer(
                height_points, base_lin_vel_hh
            )
        
        # 更新 buffer
        self.safety_scores[:] = safety_scores
        self.target_attention[:] = target_attention
        
        return safety_scores, target_attention
    
    def get_safety_scores(self):
        """获取当前的安全分数 [num_envs, num_points]"""
        return self.safety_scores
    
    def get_target_attention(self):
        """获取目标注意力分布 [num_envs, num_points]"""
        return self.target_attention
    
    def get_curriculum_level(self):
        """获取当前课程等级，用于计算动态 β。

        说明：
            - 为了避免在 rollout 阶段频繁触发 GPU→CPU 同步，这里**不返回 python 标量**，
              而是返回位于 device 上的 Tensor。
            - 推荐返回逐环境的课程等级向量 [num_envs]，这样策略端可实现逐 env 的 β。
        """
        if hasattr(self, 'terrain_levels') and isinstance(self.terrain_levels, torch.Tensor):
            # [num_envs] on device
            return self.terrain_levels.to(dtype=torch.float)
        # fallback: keep shape-compatible tensor on device
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

    def set_attention_weights(self, attn_weights):
        """
        从策略网络接收注意力权重，用于可视化
        自动适配不同尺寸:
          - 高度点方案: [num_envs, 17, 11]
          - 深度图像方案: [num_envs, 5, 5]
        
        Args:
            attn_weights: [num_envs, H, W] 注意力权重热力图
        """
        if attn_weights is not None:
            # 首次调用时自动初始化 buffer
            if self.attention_weights is None or self.attention_shape != attn_weights.shape[1:]:
                self.attention_shape = attn_weights.shape[1:]
                self.attention_weights = torch.zeros(
                    self.num_envs, *self.attention_shape, 
                    dtype=torch.float, device=self.device, requires_grad=False
                )
                print(f"[Env] 注意力权重 buffer 初始化为: {self.attention_weights.shape}")
            self.attention_weights[:] = attn_weights.detach()
    
    def get_height_points_with_heights(self):
        """
        获取带有测量高度的地形采样点，用于空间注意力模块
        
        Returns:
            height_points_with_z: [num_envs, num_x, num_y, 3] 地形高度采样点
                - 前两维 (x, y) 是局部坐标系下的采样点位置
                - 第三维 z 是归一化后的相对高度
        
        注意: 
            - height_points 是在 base frame 下定义的采样点网格
            - measured_heights 是每个采样点相对于机器人脚下的高度差
            - 归一化: z = clip(base_height - measured_height, -1, 1)
        """
        if not self.cfg.terrain.measure_heights or not hasattr(self, 'height_points'):
            return None
        
        if not isinstance(self.measured_heights, torch.Tensor):
            return None
        
        # height_points: [num_envs, 187, 3] - 局部坐标系下的采样点 (x, y, 0)
        # measured_heights: [num_envs, 187] - 每个点的高度测量值
        
        # 计算相对高度并归一化
        # 使用与 privileged_obs_buf 相同的归一化方式
        relative_heights = self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_heights
        normalized_heights = torch.clip(relative_heights, -1.0, 1.0)  # [num_envs, 187]
        
        # 构建完整的 (x, y, z) 数据
        # height_points 的前两列是 x, y 坐标
        if self._height_points_with_z_buf is None or self._height_points_with_z_buf.shape != self.height_points.shape:
            # 只在首次/尺寸变化时分配；后续 step 复用该 buffer
            self._height_points_with_z_buf = self.height_points.clone()  # [num_envs, 187, 3]
        # 仅更新 z 坐标（in-place），避免 clone + 分配
        self._height_points_with_z_buf[:, :, 2].copy_(normalized_heights)
        
        # 重塑为 [num_envs, num_x, num_y, 3]
        # 配置中: measured_points_x 有 17 个点, measured_points_y 有 11 个点
        num_x = len(self.cfg.terrain.measured_points_x)  # 17
        num_y = len(self.cfg.terrain.measured_points_y)  # 11
        height_points_reshaped = self._height_points_with_z_buf.view(self.num_envs, num_x, num_y, 3)
        
        return height_points_reshaped

    def _draw_foot_indicator(self):
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.01, 10, 10, None, color=(1, 0, 0))
        indicator_pos = self.feet_indicator_pos.reshape(-1, 3)
        for i, point in enumerate(indicator_pos):
            pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)
            gymutil.draw_lines(
                sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose
            )

    def _draw_spatial_attention_viz(self):
        """
        可视化空间注意力机制（自适应不同尺寸）
        
        当前仅保留高度点注意力方案:
          - 高度点方案: [17, 11] 注意力权重 -> 可直接映射到高度点
        
        包含两部分：
        1. 2D热力图窗口：显示注意力权重分布
        2. 3D场景点云：在真实位置显示注意力权重（与高度点一一对应）
        """
        env_id = self.lookat_id
        
        # 检查注意力权重是否已初始化
        if self.attention_weights is None:
            return
        
        # ========== Part 1: 2D 注意力热力图 ==========
        attn = self.attention_weights[env_id].cpu().numpy()  # [H, W]
        H, W = attn.shape
        attn_num_elements = H * W
        
        # 高度点方案: 17x11 = 187
        is_heightpoint_attention = (attn_num_elements == 187)
        
        if attn.max() > 0:
            # 归一化到 [0, 255]
            attn_normalized = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
            attn_uint8 = (attn_normalized * 255).astype(np.uint8)
            
            # 上采样到更大尺寸便于观察 (10x 放大)
            target_h, target_w = H * 10, W * 10
            attn_upsampled = cv2.resize(attn_uint8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            
            # 应用颜色映射 (蓝→红: COLORMAP_JET)
            attn_colored = cv2.applyColorMap(attn_upsampled, cv2.COLORMAP_JET)
            
            # 添加标题和数值信息
            mode_str = "HeightPoint" if is_heightpoint_attention else "Image"
            info_text = f"{mode_str} max:{attn.max():.3f}"
            cv2.putText(attn_colored, info_text, (5, target_h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 显示 (窗口标题动态显示尺寸)
            window_name = f"Spatial Attention ({H}x{W})"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, attn_colored)
            cv2.waitKey(1)
        
        # ========== Part 2: 3D 场景中显示注意力权重点云 ==========
        # 仅在高度点方案下可视化（注意力权重与高度点一一对应）
        self.gym.clear_lines(self.viewer)
        
        if is_heightpoint_attention and hasattr(self, 'height_points') and self.height_points is not None:
            from legged_gym.utils.math import quat_apply_yaw
            
            base_quat = self.base_quat[env_id]
            robot_pos = self.root_states[env_id, :3]
            
            # 获取局部坐标系下的高度点 [187, 3]
            local_points = self.height_points[env_id]
            
            # 转换到世界坐标系
            quat_expanded = base_quat.unsqueeze(0).repeat(local_points.shape[0], 1)
            world_points = quat_apply_yaw(quat_expanded, local_points) + robot_pos
            
            # 获取地形高度
            if isinstance(self.measured_heights, torch.Tensor):
                world_points[:, 2] = self.measured_heights[env_id]
            
            # 只显示前方的点 (x >= 0)，并根据注意力权重着色
            attn_flat = self.attention_weights[env_id].flatten()  # [187]
            
            # 计算阈值（top 20%）
            threshold = torch.quantile(attn_flat, 0.8) if attn_flat.max() > 0 else 0.0
            
            # 绘制所有点
            small_radius = 0.008
            for i in range(local_points.shape[0]):
                # 只显示 x >= 0 的点
                if local_points[i, 0] < 0:
                    continue
                
                point = world_points[i]
                attn_val = attn_flat[i].item()
                
                if attn_val >= threshold and attn_flat.max() > 0:
                    # 高注意力：蓝→红渐变
                    t = (attn_val - threshold.item()) / (attn_flat.max().item() - threshold.item() + 1e-6)
                    t = min(max(t, 0), 1)
                    # 蓝(0,0,1) → 白(1,1,1) → 红(1,0,0)
                    if t < 0.5:
                        color = (t*2, t*2, 1.0)
                    else:
                        color = (1.0, 1.0-(t-0.5)*2, 1.0-(t-0.5)*2)
                    radius = small_radius * 1.5
                else:
                    # 低注意力：蓝色
                    color = (0.2, 0.4, 1.0)
                    radius = small_radius
                
                sphere_geom = gymutil.WireframeSphereGeometry(radius, 6, 6, None, color=color)
                pose = gymapi.Transform(gymapi.Vec3(point[0].item(), point[1].item(), point[2].item()), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], pose)

        # ========== 调试信息（每100帧打印一次） ==========
        if self.common_step_counter % 100 == 0:
            mode_str = "HeightPoint" if is_heightpoint_attention else "Unknown"
            print(f"[Attention-{mode_str}] shape={H}x{W}, max={attn.max():.4f}, min={attn.min():.4f}")

    def _reset_dofs(self, env_ids):
        if self.cfg.init_state.random_default_pos:
            rand_default_pos = self.motion_reference[np.random.randint(0, self.motion_reference.shape[0], size=(env_ids.shape[0], )), :]
            self.dof_pos[env_ids] = rand_default_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.contact_over = torch.logical_and(~contact, self.last_contacts)
        self.last_contacts = contact

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        terminal_obs, terminal_critic_obs = self.compute_observations()
        self.reset_idx(env_ids)

        self.update_depth_buffer()
        self.warp_update_depth_buffer()

        self._post_physics_step_callback()
        
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self.last_last_actions[:] = torch.clone(self.last_actions[:])
        self.last_actions[:] = self.actions[:]
        self.last_last_feet_contact_force[:] = torch.clone(self.last_feet_contact_force[:])
        self.last_feet_contact_force[:] = self.contact_forces[:, self.feet_indices]

        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            if self.cfg.depth.use_camera and self.cfg.depth.warp_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
                window_name = "Warp Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Warp Depth Image", self.warp_depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
            elif self.cfg.depth.warp_camera:
                window_name = "Warp Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Warp Depth Image", self.warp_depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)
            elif self.cfg.depth.use_camera:
                window_name = "Depth Image"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow("Depth Image", self.depth_buffer[self.lookat_id, -1].cpu().numpy() + 0.5)
                cv2.waitKey(1)

            # self._draw_foot_indicator()
            
            # 空间注意力可视化 (2D热力图 + 3D落足点)
            self._draw_spatial_attention_viz()
    
        return env_ids, terminal_amp_states, terminal_obs[env_ids], terminal_critic_obs[env_ids]

    def _post_physics_step_callback(self):
        self.compute_both_feet_info()
        self.compute_feet_indicator_pos()
        
        return super()._post_physics_step_callback()
    
    def compute_both_feet_info(self):
        # compute both feet swing length
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        for i in range(len(self.feet_indices)):
            self.footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            self.footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
    
    def compute_feet_indicator_pos(self):
        num_dot = self.feet_indicator_offset.shape[0]
        ankle_quat = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        feet_offset = self.feet_indicator_offset.view(1, 1, num_dot, 3).expand(self.num_envs, 2, num_dot, 3)
        quat_expanded = ankle_quat.unsqueeze(2).expand(-1, -1, num_dot, -1)  # (num_envs, 2, num_dot, 4)
        rotated_points = quat_apply(quat_expanded.reshape(-1, 4), feet_offset.reshape(-1, 3))
        rotated_points = rotated_points.view(self.num_envs, 2, num_dot, 3)
        self.feet_indicator_pos = rotated_points + self.feet_pos.unsqueeze(2)  # (num_envs, 2, num_dot, 3)

    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1000., dim=1)
        
        # 姿态终止条件（参考 extreme-parkour）
        self.reset_buf |= torch.abs(self.rpy[:, 0]) > 1.5  # roll
        self.reset_buf |= torch.abs(self.rpy[:, 1]) > 1.5  # pitch
        
        # 相对高度终止条件：机器人身体相对于脚下地形太低
        self.reset_buf |= (self._get_base_heights() < 0.35)
        
        # 绝对高度终止条件（防止深坑中继续行走）
        # stepping stones 间隙深度约1米，掉落后z应该很低
        absolute_height_cutoff = self.root_states[:, 2] < -0.2
        self.reset_buf |= absolute_height_cutoff

        if self.cfg.terrain.mesh_type == "trimesh":
            offset_y = torch.abs(self.root_states[:, 1] - self.origin_y)
            only_forward_env = torch.logical_and(self.env_class != 0, self.env_class != 1)
            self.reset_buf |= torch.logical_and(only_forward_env, offset_y>1.0)
        
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ),dim=-1)
        
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    ),dim=-1)

        if self.cfg.env.feet_info:  # 6 * 2 = 12
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.footpos_in_body_frame.reshape(self.num_envs, -1), 
                                                 self.footvel_in_body_frame.reshape(self.num_envs, -1)), dim=-1)
        
        if self.cfg.env.foot_force_info:  # 6
            contact_force = self.sensor_forces.flatten(1) * self.obs_scales.contact_force
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, contact_force), dim=-1)
        
        if self.cfg.env.priv_info:  # 32 + 1 + 1 + 1 + 3 = 38
            self.privileged_obs_buf= torch.cat((self.privileged_obs_buf, self.root_states[:, 2].unsqueeze(-1)), dim=-1)

            if self.cfg.domain_rand.randomize_friction:  # 1
                self.privileged_obs_buf= torch.cat((self.privileged_obs_buf, self.randomized_frictions), dim=-1)

            if (self.cfg.domain_rand.randomize_base_mass):  # 1
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.randomized_added_masses), dim=-1)

            if (self.cfg.domain_rand.randomize_com_pos):  # 3
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, self.randomized_com_pos * self.obs_scales.com_pos), dim=-1)

            if (self.cfg.domain_rand.randomize_gains):  # 16 * 2
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, (self.randomized_p_gains / self.p_gains - 1) * self.obs_scales.pd_gains), dim=-1)
                self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, (self.randomized_d_gains / self.d_gains - 1) * self.obs_scales.pd_gains), dim=-1)
        
        if self.cfg.terrain.measure_heights:  # 187
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.cfg.normalization.base_height - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
            
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        return self.obs_buf, self.privileged_obs_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        super()._resample_commands(env_ids)

        only_forward_env = torch.logical_and(self.env_class != 0, self.env_class != 1)
        self.commands[only_forward_env, 3] = 0
        self.commands[only_forward_env, 2] = 0
        self.commands[only_forward_env, 1] = 0
        self.commands[only_forward_env, 0] = torch.abs(self.commands[only_forward_env, 0])
    
    #------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_action_smoothness(self):
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        """
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        return term_1 + term_2 + term_3

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_joint_power(self):
        # Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_feet_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        rew = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             3 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        return rew.float()
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_arm_joint_deviation(self):
        return torch.square(torch.norm(torch.abs(self.dof_pos[:, 12:] - self.default_dof_pos[:, 12:]), dim=1))

    def _reward_hip_joint_deviation(self):
        return torch.square(torch.norm(torch.abs(self.dof_pos[:, [1, 2, 7, 8]]), dim=1))
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    
    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_no_fly(self):
        is_jump =  torch.all(self.contact_forces[:, self.feet_indices, 2] < 1, dim=1)
        return is_jump.float()
    
    def _reward_feet_lateral_distance(self):
        # Penalize feet lateral distance (too narrow)
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        rew = (footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1]) - self.cfg.rewards.feet_min_lateral_distance_target
        return rew
    
    def _reward_feet_lateral_distance_max(self):
        """
        惩罚两脚横向距离过大（防止劈叉站姿）
        
        使用软边界惩罚，只惩罚超出最大距离的部分
        """
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
        
        # 计算两脚横向距离（左脚y - 右脚y）
        lateral_distance = footpos_in_body_frame[:, 0, 1] - footpos_in_body_frame[:, 1, 1]
        
        # 软边界惩罚：只惩罚超出最大距离的部分
        max_distance = self.cfg.rewards.feet_max_lateral_distance_target
        penalty = F.relu(lateral_distance - max_distance)
        
        return penalty
    
    def _reward_feet_slippage(self):
        return torch.sum(torch.norm(self.feet_vel, dim=-1) * (torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.), dim=1)
    
    def _reward_feet_contact_force(self):
        # penalize high contact forces
        return torch.sum(F.relu(self.contact_forces[:, self.feet_indices, 2] - self.cfg.rewards.feet_contact_force_range[0]), dim=-1)
    
    def _reward_feet_force_rate(self):
        return torch.sum(F.relu(self.contact_forces[:, self.feet_indices, 2] - self.last_feet_contact_force[..., 2]), dim=-1)
    
    def _reward_feet_contact_momentum(self):
        """
        Penalizes the momentum of the feet contact forces, encouraging a more stable and controlled motion.
        foot vel * contact force
        """
        feet_contact_force = self.contact_forces[:, self.feet_indices, 2]
        feet_vertical_vel = self.feet_vel[:, :, 2]
        rew = torch.sum(torch.abs(feet_contact_force * feet_vertical_vel), dim=-1)
        return rew
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~self.contact_filt
        return rew_airTime
    
    def _reward_stuck(self):
        # Penalize stuck
        return (torch.abs(self.base_lin_vel[:, 0]) < 0.1) * (torch.abs(self.commands[:, 0]) > 0.1)
    
    def _reward_cheat(self):
        # penalty cheating to bypass the obstacle
        no_cheat_env = torch.logical_and(self.env_class != 0, self.env_class != 1)
        forward = quat_apply(self.base_quat[no_cheat_env], self.forward_vec[no_cheat_env])
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        cheat = (heading > 1.0) | (heading < -1.0)
        cheat_penalty = torch.zeros(self.num_envs, device=self.device)
        cheat_penalty[no_cheat_env] = cheat.float()
        return cheat_penalty
    
    def _reward_feet_edge(self):
        foot_indicators_pos_xy = ((self.feet_indicator_pos[..., :2]+self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()
        foot_indicators_pos_xy[..., 0] = torch.clip(foot_indicators_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
        foot_indicators_pos_xy[..., 1] = torch.clip(foot_indicators_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)

        feet_at_edge = self.x_edge_mask[foot_indicators_pos_xy[..., 0], foot_indicators_pos_xy[..., 1]]
        feet_at_edge = torch.sum(feet_at_edge, dim=-1) >= 3
        feet_at_edge = self.contact_filt & feet_at_edge
        rew = (self.terrain_levels > 3) * torch.sum(feet_at_edge, dim=1)
        return rew

    # def _reward_feet_edge(self):
    #     feet_pos_xy = ((self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, :2] + self.terrain.cfg.border_size) / self.cfg.terrain.horizontal_scale).round().long()  # (num_envs, 4, 2)
    #     feet_pos_xy[..., 0] = torch.clip(feet_pos_xy[..., 0], 0, self.x_edge_mask.shape[0]-1)
    #     feet_pos_xy[..., 1] = torch.clip(feet_pos_xy[..., 1], 0, self.x_edge_mask.shape[1]-1)
    #     feet_at_edge = self.x_edge_mask[feet_pos_xy[..., 0], feet_pos_xy[..., 1]]
    
    #     self.feet_at_edge = self.contact_filt & feet_at_edge
    #     rew = (self.terrain_levels > 3) * torch.sum(self.feet_at_edge, dim=-1)
    #     return rew
    
    def _reward_y_offset_pen(self):
        pen = torch.abs(self.root_states[:, 1] - self.origin_y) * torch.logical_and(self.env_class != 0, self.env_class != 1)
        return pen
