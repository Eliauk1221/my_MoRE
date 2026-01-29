from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from isaacgym import gymapi, gymutil
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit_resi, export_policy_as_jit_depth, task_registry
import torch
import numpy as np


def draw_attention_points(env, attention_weights):
    """
    根据注意力权重绘制彩色采样点（在真实地形表面上）
    - 蓝色 (0, 0, 1): 低权重
    - 红色 (1, 0, 0): 高权重
    
    Args:
        env: 环境对象
        attention_weights: [num_envs, 187] 注意力权重
    """
    if env.viewer is None:
        return
        
    env.gym.clear_lines(env.viewer)
    
    lookat_id = env.lookat_id if hasattr(env, 'lookat_id') else 0
    
    # 获取当前环境的权重
    weights = attention_weights[lookat_id].cpu().numpy()
    # 归一化到 [0, 1]
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min > 1e-8:
        weights_norm = (weights - w_min) / (w_max - w_min)
    else:
        weights_norm = np.zeros_like(weights)
    
    # 获取机体位置和姿态
    base_pos = env.root_states[lookat_id, :3].cpu().numpy()
    base_quat = env.root_states[lookat_id, 3:7].cpu().numpy()
    
    # 获取机体坐标系下的采样点 xy 坐标（未归一化的原始坐标）
    # height_points 存储的是原始的机体坐标系下的 xy 位置
    points_body_xy = env.height_points[lookat_id, :, :2].cpu().numpy()  # [187, 2]
    
    # 获取真实地形高度（世界坐标系下的绝对高度）
    # measured_heights 是 _get_heights() 返回的地形绝对高度
    terrain_z = env.measured_heights[lookat_id].cpu().numpy()  # [187]
    
    # 计算 yaw 角
    # quat = [x, y, z, w]
    qx, qy, qz, qw = base_quat
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    
    # 将机体坐标系的 xy 旋转到世界坐标系
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rot_x = points_body_xy[:, 0] * cos_yaw - points_body_xy[:, 1] * sin_yaw
    rot_y = points_body_xy[:, 0] * sin_yaw + points_body_xy[:, 1] * cos_yaw
    
    # 世界坐标
    world_x = base_pos[0] + rot_x
    world_y = base_pos[1] + rot_y
    world_z = terrain_z  # 直接使用真实地形高度
    
    # 绘制每个点
    for i in range(len(weights_norm)):
        w = weights_norm[i]
        # 蓝色 → 红色渐变
        color = (w, 0.0, 1.0 - w)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 6, 6, None, color=color)
        pose = gymapi.Transform(gymapi.Vec3(world_x[i], world_y[i], world_z[i]), r=None)
        gymutil.draw_lines(sphere_geom, env.gym, env.viewer, env.envs[lookat_id], pose)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 100
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    env_cfg.terrain.difficulty_level = 0.3
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_com_pos = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 8
    env_cfg.domain_rand.push_interval_min_s = 8
    env_cfg.domain_rand.max_push_vel_xy = 1
    env_cfg.domain_rand.min_push_vel_xy = 1

    env_cfg.asset.self_collisions = 0

    env_cfg.env.test = True

    env_cfg.depth.y_angle = [45, 45]
    env_cfg.depth.x_angle = [0, 0]
    env_cfg.depth.z_angle = [0, 0]
    env_cfg.depth.x_pos_range = [0, 0]
    env_cfg.depth.y_pos_range = [0, 0]
    env_cfg.depth.z_pos_range = [0, 0]
    env_cfg.depth.use_camera = False
    env_cfg.depth.warp_camera = True
    env_cfg.depth.add_body_mask = False
    env_cfg.depth.dis_noise = 0
    env_cfg.depth.gaussian_noise = False
    env_cfg.depth.gaussian_noise_std = 0.05
    env_cfg.depth.gaussian_filter = False
    env_cfg.depth.gaussian_filter_kernel = [5]
    env_cfg.depth.gaussian_filter_sigma = 1.5
    
    env_cfg.commands.ranges.lin_vel_y = [0, 0]
    env_cfg.commands.ranges.lin_vel_x = [0, 0]
    env_cfg.commands.ranges.heading = [0, 0]
    env_cfg.commands.ranges.ang_vel_yaw = [0, 0]
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = 100

    env_cfg.terrain.terrain_dict = {"stepping_stones": 1, 
                                    "parkour": 1,
                                    "pit": 1,
                                    "gap": 1,
                                    "stair": 1,}
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    train_cfg.runner.zero_init = False
    train_cfg.runner.load_delta_policy = False
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # process trajectory history
    num_gait = env.cfg.env.num_gait if hasattr(env.cfg.env, 'num_gait') else 0
    trajectory_history = torch.zeros(size=(env.num_envs, env.obs_history_len, env.num_obs-num_gait), device=env.device)
    trajectory_history = torch.concat((trajectory_history[:, 1:], obs[:, num_gait:].unsqueeze(1)), dim=1)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.load_run, 'exported')
        if 'Resi' in train_cfg.runner_class_name:
            export_policy_as_jit_resi(ppo_runner.alg.actor_critic, path)
        else:
            export_policy_as_jit_depth(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    infos = {}
    if env.cfg.depth.warp_camera:
        infos["depth"] = env.warp_depth_buffer.clone().to(env.device)  
    elif env.cfg.depth.use_camera:
        infos["depth"] = env.depth_buffer.clone().to(env.device)  
    else:
        infos["depth"] = None
    
    # ===== 检查是否使用地形注意力和物理引导偏置 =====
    use_terrain_attention = getattr(env, 'use_terrain_attention', False)
    use_safety_bias = False
    if hasattr(env.cfg, 'terrain_attention'):
        use_safety_bias = getattr(env.cfg.terrain_attention, 'use_safety_bias', False)
    
    actor_critic = ppo_runner.alg.actor_critic
    
    # 推理时 β=0.0（训练完成后不再需要物理先验，让模型自主决策）
    attn_bias_beta = 0.0
    
    print(f"[Play] use_terrain_attention={use_terrain_attention}, use_safety_bias={use_safety_bias}, attn_bias_beta={attn_bias_beta}")

    for i in range(int(env.max_episode_length)):
        # get depth image
        if infos["depth"] is not None:
            depth_image = infos['depth']
        if env.cfg.depth.warp_camera or env.cfg.depth.use_camera:
            obs = (obs, depth_image)
        
        # ===== 准备地形注意力数据 =====
        height_map = None
        terrain_xyz = None
        base_lin_vel = None
        
        if use_terrain_attention and env.height_map is not None:
            height_map = env.height_map.clone().to(env.device)
            terrain_xyz = env.terrain_xyz.clone().to(env.device)
            
            # 只有当 use_safety_bias=True 时才传递 base_lin_vel
            if use_safety_bias:
                base_lin_vel = env.base_lin_vel.clone().to(env.device)

        if isinstance(obs, tuple):
            actions = policy(obs[0].detach(), trajectory_history.detach(), obs[1][:, :2, ...].detach(),
                           height_map=height_map, terrain_xyz=terrain_xyz,
                           base_lin_vel=base_lin_vel, attn_bias_beta=attn_bias_beta)
        else:
            actions = policy(obs.detach(), trajectory_history,
                           height_map=height_map, terrain_xyz=terrain_xyz,
                           base_lin_vel=base_lin_vel, attn_bias_beta=attn_bias_beta)
        obs, _, _, dones, infos, *_= env.step(actions.detach())
        
        # ===== 绘制注意力可视化 =====
        if use_terrain_attention and hasattr(actor_critic, 'terrain_attention'):
            terrain_attn = actor_critic.terrain_attention
            if terrain_attn is not None and terrain_attn.last_attention_weights is not None:
                draw_attention_points(env, terrain_attn.last_attention_weights)

        # process trajectory history
        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        trajectory_history = torch.concat((trajectory_history[:, 1:], obs[:, num_gait:].unsqueeze(1)), dim=1)


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
