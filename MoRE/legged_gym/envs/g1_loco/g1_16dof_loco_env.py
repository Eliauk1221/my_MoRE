
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.datasets.motion_loader_g1 import G1_AMPLoader
import torch
import cv2
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
        
        # ========== 落足点引导注意力相关 ==========
        # 存储策略网络预测的落足点 [num_envs, 4] (左脚dx,dy + 右脚dx,dy)
        self.pred_foothold = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        # 存储注意力权重 [num_envs, 17, 11] 用于可视化
        self.attention_weights = torch.zeros(self.num_envs, 17, 11, dtype=torch.float, device=self.device, requires_grad=False)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_feet_contact_force[env_ids] = 0.
        self.last_last_feet_contact_force[env_ids] = 0.
        self.pred_foothold[env_ids] = 0.  # 重置预测落足点
        self.attention_weights[env_ids] = 0.  # 重置注意力权重

    def set_attention_weights(self, attn_weights):
        """
        从策略网络接收注意力权重，用于可视化
        
        Args:
            attn_weights: [num_envs, 17, 11] 注意力权重热力图
        """
        if attn_weights is not None:
            self.attention_weights[:] = attn_weights.detach()

    def _draw_foot_indicator(self):
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.01, 10, 10, None, color=(1, 0, 0))
        indicator_pos = self.feet_indicator_pos.reshape(-1, 3)
        for i, point in enumerate(indicator_pos):
            pose = gymapi.Transform(gymapi.Vec3(point[0], point[1], point[2]), r=None)
            gymutil.draw_lines(
                sphere_geom, self.gym, self.viewer, self.envs[self.lookat_id], pose
            )

    def _draw_foothold_attention(self):
        """
        可视化落足点注意力机制
        
        三层可视化：
        1. 第一层（基础）：x > 0 的采样点用小蓝点显示
        2. 第二层（叠加）：高注意力的点变色（蓝→红渐变）
        3. 第三层（叠加）：预测落脚点用大红点标记
        """
        self.gym.clear_lines(self.viewer)
        
        # 只可视化 lookat_id 对应的环境
        env_id = self.lookat_id
        
        # ========== 获取采样点信息 ==========
        # height_points: [num_envs, 187, 3] 身体坐标系下的采样点
        # measured_heights: [num_envs, 187] 每个点的地形高度
        
        if not hasattr(self, 'height_points') or self.height_points is None:
            return
        
        # 获取身体坐标系下的采样点 [187, 3]
        local_points = self.height_points[env_id]  # [187, 3]
        
        # 筛选 x > 0 的点 (机器人前方)
        # measured_points_x = [-0.8, ..., 0.8], 共17个点
        # x > 0 的点对应索引 9-16 (共8个)，加上 x=0 的点共9个
        # 网格是 17x11，每行11个点
        # x 索引从 0 到 16，x > 0 对应索引 9-16
        x_coords = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device)
        x_positive_mask = x_coords >= 0  # [17] 布尔掩码
        
        # 计算需要显示的点的索引
        # height_points 排列方式: [x0y0, x0y1, ..., x0y10, x1y0, x1y1, ..., x16y10]
        # 即外层是 x，内层是 y
        num_y = len(self.cfg.terrain.measured_points_y)  # 11
        
        # 生成 x > 0 的点的索引
        x_positive_indices = torch.where(x_positive_mask)[0]  # x >= 0 的索引
        point_indices = []
        for xi in x_positive_indices:
            for yi in range(num_y):
                point_indices.append(xi * num_y + yi)
        point_indices = torch.tensor(point_indices, device=self.device, dtype=torch.long)
        
        # 获取筛选后的点 [num_filtered, 3]
        filtered_local_points = local_points[point_indices]  # 身体坐标系
        
        # ========== 转换到世界坐标系 ==========
        # 使用 yaw 旋转 + 机器人位置
        base_quat = self.base_quat[env_id]  # [4]
        robot_pos = self.root_states[env_id, :3]  # [3]
        
        # 旋转采样点到世界坐标系
        from isaacgym.torch_utils import quat_apply_yaw
        quat_expanded = base_quat.unsqueeze(0).repeat(len(point_indices), 1)  # [num_filtered, 4]
        world_points_xy = quat_apply_yaw(quat_expanded, filtered_local_points) + robot_pos
        
        # 获取每个点的地形高度作为 z 坐标
        if hasattr(self, 'measured_heights') and isinstance(self.measured_heights, torch.Tensor):
            filtered_heights = self.measured_heights[env_id, point_indices]  # [num_filtered]
            # measured_heights 是相对高度，需要加上地形基准
            world_points_z = filtered_heights + (robot_pos[2] - self.cfg.normalization.base_height)
        else:
            world_points_z = torch.zeros(len(point_indices), device=self.device)
        
        world_points = world_points_xy.clone()
        world_points[:, 2] = world_points_z
        
        # ========== 获取注意力权重 ==========
        # attention_weights: [num_envs, 17, 11]
        attn = self.attention_weights[env_id]  # [17, 11]
        
        # 筛选 x > 0 的注意力权重
        filtered_attn = attn[x_positive_mask, :].flatten()  # [num_filtered]
        
        # 计算 top 20% 阈值
        if filtered_attn.max() > 0:
            threshold = torch.quantile(filtered_attn, 0.8)
        else:
            threshold = 0.0
        
        # ========== 第一层：所有点用小蓝点 ==========
        small_radius = 0.01
        for i, point in enumerate(world_points):
            # 默认蓝色
            color = (0.2, 0.4, 1.0)  # 蓝色
            sphere_geom = gymutil.WireframeSphereGeometry(small_radius, 6, 6, None, color=color)
            pose = gymapi.Transform(gymapi.Vec3(point[0].item(), point[1].item(), point[2].item()), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], pose)
        
        # ========== 第二层：高注意力点变色 ==========
        for i, point in enumerate(world_points):
            if filtered_attn[i] >= threshold and filtered_attn[i] > 0:
                # 根据注意力值计算颜色 (蓝→红渐变)
                attn_normalized = (filtered_attn[i] - threshold) / (filtered_attn.max() - threshold + 1e-6)
                attn_normalized = attn_normalized.clamp(0, 1).item()
                
                # 渐变色: 蓝(0,0,1) -> 白(1,1,1) -> 红(1,0,0)
                if attn_normalized < 0.5:
                    # 蓝 -> 白
                    t = attn_normalized * 2
                    color = (t, t, 1.0)
                else:
                    # 白 -> 红
                    t = (attn_normalized - 0.5) * 2
                    color = (1.0, 1.0 - t, 1.0 - t)
                
                # 用稍大的球体覆盖
                sphere_geom = gymutil.WireframeSphereGeometry(small_radius * 1.5, 8, 8, None, color=color)
                pose = gymapi.Transform(gymapi.Vec3(point[0].item(), point[1].item(), point[2].item()), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], pose)
        
        # ========== 第三层：预测落脚点用大红点 ==========
        pred_foothold = self.pred_foothold[env_id]  # [4]: 左脚dx,dy + 右脚dx,dy
        
        if torch.any(pred_foothold != 0):
            # 左脚预测位置
            pred_left_local = torch.tensor([pred_foothold[0], pred_foothold[1], 0], device=self.device)
            pred_left_world = quat_apply_yaw(base_quat.unsqueeze(0), pred_left_local.unsqueeze(0)).squeeze() + robot_pos
            # 采样地形高度
            pred_left_z = self._sample_terrain_height(pred_left_world[:2].unsqueeze(0)).squeeze()
            pred_left_world[2] = pred_left_z
            
            # 右脚预测位置
            pred_right_local = torch.tensor([pred_foothold[2], pred_foothold[3], 0], device=self.device)
            pred_right_world = quat_apply_yaw(base_quat.unsqueeze(0), pred_right_local.unsqueeze(0)).squeeze() + robot_pos
            pred_right_z = self._sample_terrain_height(pred_right_world[:2].unsqueeze(0)).squeeze()
            pred_right_world[2] = pred_right_z
            
            # 绘制大红点
            big_radius = 0.02
            red_color = (1.0, 0.0, 0.0)
            
            # 左脚
            sphere_geom = gymutil.WireframeSphereGeometry(big_radius, 10, 10, None, color=red_color)
            pose = gymapi.Transform(gymapi.Vec3(
                pred_left_world[0].item(), pred_left_world[1].item(), pred_left_world[2].item()
            ), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], pose)
            
            # 右脚
            pose = gymapi.Transform(gymapi.Vec3(
                pred_right_world[0].item(), pred_right_world[1].item(), pred_right_world[2].item()
            ), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], pose)

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
            
            # 落足点注意力可视化
            if self.cfg.terrain.measure_heights:
                self._draw_foothold_attention()
    
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
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.reset_buf |= (self._get_base_heights() < 0.4)

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
    
    # ========== 落足点引导注意力奖励 ==========
    def set_pred_foothold(self, pred_foothold):
        """
        从策略网络接收预测的落足点
        
        Args:
            pred_foothold: [num_envs, 4] 预测的落足点偏移 (左脚dx,dy + 右脚dx,dy)
        """
        if pred_foothold is not None:
            self.pred_foothold[:] = pred_foothold.detach()
    
    def _sample_terrain_height(self, world_xy):
        """
        从地形高度图采样指定世界xy位置的高度
        
        对于台阶、坑洞等有高度变化的地形，这个函数可以获取预测落足点位置的真实地形高度，
        避免使用当前脚高度导致的坐标转换误差。
        
        Args:
            world_xy: [num_envs, 2] 世界坐标系的xy位置
        
        Returns:
            heights: [num_envs] 该位置的地形高度 (米)
        """
        h_scale = self.cfg.terrain.horizontal_scale
        v_scale = self.cfg.terrain.vertical_scale
        border_size = self.terrain.cfg.border_size if hasattr(self.terrain, 'cfg') else 0
        
        # 转换到网格坐标
        grid_xy = ((world_xy + border_size) / h_scale).long()
        
        # 限制在有效范围内
        px = torch.clip(grid_xy[:, 0], 0, self.height_samples.shape[0] - 1)
        py = torch.clip(grid_xy[:, 1], 0, self.height_samples.shape[1] - 1)
        
        # 采样高度并转换为米
        heights = self.height_samples[px, py] * v_scale
        
        return heights
    
    def _reward_foothold_sampling(self):
        """
        BeamDojo风格的采样落足点奖励
        
        评估策略网络预测的落足点位置是否安全（地形平坦、非边缘）
        只在脚即将落地时给予奖励
        
        思路：
        1. 将预测落足点从身体坐标系转换到世界坐标系
        2. 采样预测位置的地形高度
        3. 评估该位置的地形质量（平坦度和边缘惩罚）
        4. 只在脚处于摆动相且即将着地时计算奖励
        
        Returns:
            reward: [num_envs] 落足点质量奖励
        """
        # 提前返回：如果没有地形高度测量
        if not self.cfg.terrain.measure_heights:
            return torch.zeros(self.num_envs, device=self.device)
        
        # 提前返回：如果 measured_heights 尚未初始化（第一次 reset 时）
        if not isinstance(self.measured_heights, torch.Tensor):
            return torch.zeros(self.num_envs, device=self.device)
        
        # ========== 1. 计算预测落足点在世界坐标系的位置 ==========
        # pred_foothold: [num_envs, 4] -> 左脚(dx,dy), 右脚(dx,dy)
        # 需要将其转换到世界坐标系
        
        # 预测是相对于机器人身体中心的偏移
        pred_left_xy = self.pred_foothold[:, :2]   # [num_envs, 2]
        pred_right_xy = self.pred_foothold[:, 2:4]  # [num_envs, 2]
        
        # === 改进的z坐标处理：使用地形高度图而非当前脚高度 ===
        # 对于台阶、坑洞等有高度变化的地形，直接使用当前脚高度会导致坐标转换误差
        # 步骤1：先用xy（z=0）进行旋转，得到粗略的世界xy位置
        pred_left_xy_3d = torch.cat([pred_left_xy, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        pred_right_xy_3d = torch.cat([pred_right_xy, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)
        
        pred_left_world_rough = quat_apply(self.base_quat, pred_left_xy_3d) + self.root_states[:, 0:3]
        pred_right_world_rough = quat_apply(self.base_quat, pred_right_xy_3d) + self.root_states[:, 0:3]
        
        # 步骤2：从地形高度图采样该位置的真实高度
        pred_left_z = self._sample_terrain_height(pred_left_world_rough[:, :2])
        pred_right_z = self._sample_terrain_height(pred_right_world_rough[:, :2])
        
        # 步骤3：使用真实地形高度重新进行坐标转换
        # 计算预测点相对于机器人的z偏移（地形高度 - 机器人高度）
        robot_height = self.root_states[:, 2]
        pred_left_dz = (pred_left_z - robot_height).unsqueeze(-1)   # [num_envs, 1]
        pred_right_dz = (pred_right_z - robot_height).unsqueeze(-1)  # [num_envs, 1]
        
        pred_left_3d = torch.cat([pred_left_xy, pred_left_dz], dim=-1)   # [num_envs, 3]
        pred_right_3d = torch.cat([pred_right_xy, pred_right_dz], dim=-1) # [num_envs, 3]
        
        # 从身体坐标系转换到世界坐标系（使用正确的z坐标）
        pred_left_world = quat_apply(self.base_quat, pred_left_3d) + self.root_states[:, 0:3]
        pred_right_world = quat_apply(self.base_quat, pred_right_3d) + self.root_states[:, 0:3]
        
        # ========== 2. 计算预测位置在地形网格中的索引 ==========
        # 地形分辨率
        h_scale = self.cfg.terrain.horizontal_scale  # 默认0.1
        border_size = self.terrain.cfg.border_size if hasattr(self.terrain, 'cfg') else 0
        
        # 转换到网格坐标
        pred_left_grid = ((pred_left_world[:, :2] + border_size) / h_scale).round().long()
        pred_right_grid = ((pred_right_world[:, :2] + border_size) / h_scale).round().long()
        
        # 限制在有效范围内
        max_x = self.x_edge_mask.shape[0] - 1
        max_y = self.x_edge_mask.shape[1] - 1
        
        pred_left_grid[:, 0] = torch.clip(pred_left_grid[:, 0], 0, max_x)
        pred_left_grid[:, 1] = torch.clip(pred_left_grid[:, 1], 0, max_y)
        pred_right_grid[:, 0] = torch.clip(pred_right_grid[:, 0], 0, max_x)
        pred_right_grid[:, 1] = torch.clip(pred_right_grid[:, 1], 0, max_y)
        
        # ========== 3. 检查预测位置是否在边缘（危险区域） ==========
        left_at_edge = self.x_edge_mask[pred_left_grid[:, 0], pred_left_grid[:, 1]]
        right_at_edge = self.x_edge_mask[pred_right_grid[:, 0], pred_right_grid[:, 1]]
        
        # 边缘惩罚
        edge_penalty = left_at_edge.float() + right_at_edge.float()
        
        # ========== 4. 采样预测位置的地形高度 ==========
        # 使用 measured_heights 进行评估
        # measured_heights 是一个 17x11 的网格，共187个点
        # 我们需要将预测的落足点位置映射到这个网格
        
        # 获取机器人局部坐标系下的预测位置
        local_left = pred_left_3d[:, :2]   # [num_envs, 2] - 相对于机器人的xy偏移
        local_right = pred_right_3d[:, :2]  # [num_envs, 2]
        
        # measured_heights 的采样范围 (需要从配置获取)
        # 假设采样范围 x: [-0.8, 0.8], y: [-0.5, 0.5]
        mh_x_range = 1.6  # 17 points over ~1.6m
        mh_y_range = 1.0  # 11 points over ~1.0m
        
        # 归一化到 [0, 1] 范围
        norm_left_x = (local_left[:, 0] + mh_x_range/2) / mh_x_range
        norm_left_y = (local_left[:, 1] + mh_y_range/2) / mh_y_range
        norm_right_x = (local_right[:, 0] + mh_x_range/2) / mh_x_range
        norm_right_y = (local_right[:, 1] + mh_y_range/2) / mh_y_range
        
        # 转换到网格索引 (17x11)
        mh_grid_left_x = (norm_left_x * 16).round().long().clamp(0, 16)
        mh_grid_left_y = (norm_left_y * 10).round().long().clamp(0, 10)
        mh_grid_right_x = (norm_right_x * 16).round().long().clamp(0, 16)
        mh_grid_right_y = (norm_right_y * 10).round().long().clamp(0, 10)
        
        # 计算线性索引
        mh_idx_left = mh_grid_left_x * 11 + mh_grid_left_y
        mh_idx_right = mh_grid_right_x * 11 + mh_grid_right_y
        
        # 安全裁剪索引
        mh_idx_left = mh_idx_left.clamp(0, 186)
        mh_idx_right = mh_idx_right.clamp(0, 186)
        
        # 采样高度（需要在update之后使用measured_heights）
        # measured_heights: [num_envs, 187]
        batch_indices = torch.arange(self.num_envs, device=self.device)
        
        height_left = self.measured_heights[batch_indices, mh_idx_left]
        height_right = self.measured_heights[batch_indices, mh_idx_right]
        
        # ========== 5. 计算高度变化作为平坦度指标 ==========
        # 对相邻点进行采样评估地形平坦度
        # 简化：使用当前点与机器人高度的差异
        base_height = self.root_states[:, 2]
        
        # 高度差异（越小越好）
        height_diff_left = torch.abs(height_left - (base_height - self.cfg.normalization.base_height))
        height_diff_right = torch.abs(height_right - (base_height - self.cfg.normalization.base_height))
        
        flatness_penalty = height_diff_left + height_diff_right
        
        # ========== 6. 只在脚摆动时给奖励 ==========
        # 使用接触检测判断哪只脚正在摆动
        is_left_swing = ~self.contact_filt[:, 0]   # 左脚摆动
        is_right_swing = ~self.contact_filt[:, 1]  # 右脚摆动
        
        # 综合奖励
        # 负数：惩罚边缘位置和不平坦地形
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # 左脚摆动时评估左脚落足点
        reward += is_left_swing.float() * (-edge_penalty * 0.5 - flatness_penalty * 0.5)
        # 右脚摆动时评估右脚落足点
        reward += is_right_swing.float() * (-edge_penalty * 0.5 - flatness_penalty * 0.5)
        
        # 归一化和裁剪
        reward = torch.clamp(reward, -2.0, 0.5)
        
        return reward
