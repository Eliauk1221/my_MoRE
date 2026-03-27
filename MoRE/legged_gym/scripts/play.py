from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from isaacgym import gymapi, gymutil
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit_resi, export_policy_as_jit_depth, task_registry
from legged_gym.utils.helpers import get_load_path
import torch
import numpy as np


def infer_model_config_from_checkpoint(checkpoint_path, num_actor_obs=57, his_latent_dim=64, 
                                        depth_dim=128, terrain_attn_dim=64):
    """
    从 checkpoint 中推断训练时的模型配置
    
    根据 actor 第一层权重的输入维度推断 use_terrain_attention 和 include_depth_in_actor
    
    Args:
        checkpoint_path: checkpoint 文件路径
        num_actor_obs: 观测维度 (默认 57)
        his_latent_dim: 历史编码维度 (默认 64)
        depth_dim: 深度特征维度 (默认 128)
        terrain_attn_dim: 地形注意力输出维度 (默认 64)
    
    Returns:
        dict: {'use_terrain_attention': bool, 'include_depth_in_actor': bool, ...}
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 获取 actor 第一层权重的输入维度
    actor_input_dim = checkpoint['model_state_dict']['actor.0.weight'].shape[1]
    
    # 计算基础维度 (obs + history)
    base_dim = num_actor_obs + his_latent_dim  # 57 + 64 = 121
    
    # 可能的组合及其维度：
    # attention=False, depth=False: base_dim = 121
    # attention=False, depth=True:  base_dim + depth_dim = 249
    # attention=True,  depth=False: base_dim + terrain_attn_dim = 185
    # attention=True,  depth=True:  base_dim + depth_dim + terrain_attn_dim = 313
    
    remaining_dim = actor_input_dim - base_dim
    
    # 根据剩余维度推断配置
    if remaining_dim == 0:
        # 121: attention=False, depth=False
        use_attention = False
        include_depth = False
    elif remaining_dim == depth_dim:
        # 249: attention=False, depth=True
        use_attention = False
        include_depth = True
    elif remaining_dim == terrain_attn_dim:
        # 185: attention=True, depth=False
        use_attention = True
        include_depth = False
    elif remaining_dim == depth_dim + terrain_attn_dim:
        # 313: attention=True, depth=True
        use_attention = True
        include_depth = True
    else:
        # 无法识别的维度，使用默认值并警告
        print(f"[Warning] Cannot infer config from actor_input_dim={actor_input_dim}, "
              f"remaining_dim={remaining_dim}. Using config file defaults.")
        return None
    
    # 检查是否有 q_norm（推断 terrain_attn_use_pre_ln，用于兼容旧模型）
    use_pre_ln = 'terrain_attention.q_norm.weight' in checkpoint['model_state_dict']
    
    # 推理时不需要 KL loss / scorer，设为 False
    use_attn_kl_loss = False
    
    print(f"[Config Inference] actor_input_dim={actor_input_dim} -> "
          f"use_terrain_attention={use_attention}, include_depth_in_actor={include_depth}, "
          f"terrain_attn_use_pre_ln={use_pre_ln}")
    
    return {
        'use_terrain_attention': use_attention,
        'include_depth_in_actor': include_depth,
        'use_attn_kl_loss': use_attn_kl_loss,
        'terrain_attn_use_pre_ln': use_pre_ln
    }


SPHERE_RADIUS = 0.025
SPHERE_SEGMENTS = 10
COLOR_LOW = (0.15, 0.25, 0.95)   # 柔和蓝
COLOR_HIGH = (0.95, 0.10, 0.10)  # 鲜亮红
COLOR_UNIFORM = (0.30, 0.55, 1.0) # 统一浅蓝


def _compute_world_points(env):
    """计算采样点的世界坐标（供注意力/均匀可视化共用）"""
    lookat_id = env.lookat_id if hasattr(env, 'lookat_id') else 0
    base_pos = env.root_states[lookat_id, :3].cpu().numpy()
    base_quat = env.root_states[lookat_id, 3:7].cpu().numpy()
    points_body_xy = env.height_points[lookat_id, :, :2].cpu().numpy()
    terrain_z = env.measured_heights[lookat_id].cpu().numpy()

    qx, qy, qz, qw = base_quat
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    world_x = base_pos[0] + points_body_xy[:, 0] * cos_yaw - points_body_xy[:, 1] * sin_yaw
    world_y = base_pos[1] + points_body_xy[:, 0] * sin_yaw + points_body_xy[:, 1] * cos_yaw
    world_z = terrain_z
    return lookat_id, world_x, world_y, world_z


def draw_attention_points(env, attention_weights):
    """根据注意力权重绘制彩色采样点：蓝色(低权重) → 红色(高权重)"""
    if env.viewer is None:
        return
    env.gym.clear_lines(env.viewer)

    lookat_id, world_x, world_y, world_z = _compute_world_points(env)

    weights = attention_weights[lookat_id].cpu().numpy()
    w_min, w_max = weights.min(), weights.max()
    if w_max - w_min > 1e-8:
        weights_norm = (weights - w_min) / (w_max - w_min)
    else:
        weights_norm = np.zeros_like(weights)

    for i in range(len(weights_norm)):
        w = weights_norm[i]
        r = COLOR_LOW[0] + (COLOR_HIGH[0] - COLOR_LOW[0]) * w
        g = COLOR_LOW[1] + (COLOR_HIGH[1] - COLOR_LOW[1]) * w
        b = COLOR_LOW[2] + (COLOR_HIGH[2] - COLOR_LOW[2]) * w
        sphere = gymutil.WireframeSphereGeometry(SPHERE_RADIUS, SPHERE_SEGMENTS, SPHERE_SEGMENTS, None, color=(r, g, b))
        pose = gymapi.Transform(gymapi.Vec3(world_x[i], world_y[i], world_z[i]), r=None)
        gymutil.draw_lines(sphere, env.gym, env.viewer, env.envs[lookat_id], pose)


def draw_uniform_points(env):
    """绘制所有采样点为统一浅蓝色（无注意力引导对比图用）"""
    if env.viewer is None:
        return
    env.gym.clear_lines(env.viewer)

    lookat_id, world_x, world_y, world_z = _compute_world_points(env)
    sphere = gymutil.WireframeSphereGeometry(SPHERE_RADIUS, SPHERE_SEGMENTS, SPHERE_SEGMENTS, None, color=COLOR_UNIFORM)

    for i in range(len(world_x)):
        pose = gymapi.Transform(gymapi.Vec3(world_x[i], world_y[i], world_z[i]), r=None)
        gymutil.draw_lines(sphere, env.gym, env.viewer, env.envs[lookat_id], pose)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 100
    env_cfg.env.num_envs = 1
    terrain_rows = getattr(args, "terrain_rows", 5)
    terrain_cols = getattr(args, "terrain_cols", 5)
    if terrain_rows < 1 or terrain_cols < 1:
        raise ValueError(
            f"terrain_rows and terrain_cols must be >= 1, got "
            f"terrain_rows={terrain_rows}, terrain_cols={terrain_cols}"
        )
    env_cfg.terrain.num_rows = terrain_rows
    env_cfg.terrain.num_cols = terrain_cols
    max_init_level = getattr(env_cfg.terrain, "max_init_terrain_level", terrain_rows - 1)
    if max_init_level is None:
        max_init_level = terrain_rows - 1
    env_cfg.terrain.max_init_terrain_level = min(int(max_init_level), terrain_rows - 1)
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

    default_terrain_dict = {
        "stepping_stones": 1,
        "parkour": 1,
        "pit": 1,
        "gap": 1,
        "stair": 1,
    }
    requested_terrain = getattr(args, "terrain_type", None)
    if requested_terrain is not None:
        requested_terrain = requested_terrain.strip().lower()
        if requested_terrain not in default_terrain_dict:
            supported = ", ".join(default_terrain_dict.keys())
            raise ValueError(
                f"Unknown terrain_type='{args.terrain_type}'. "
                f"Supported values: {supported}"
            )
        # Keep all terrain keys to avoid downstream assumptions on proportions length.
        env_cfg.terrain.terrain_dict = {
            name: int(name == requested_terrain) for name in default_terrain_dict
        }
        print(f"[Play] Using single terrain type: {requested_terrain}")
    else:
        env_cfg.terrain.terrain_dict = default_terrain_dict
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    
    # ===== 从 checkpoint 推断训练时的模型配置 =====
    # 在创建环境和 runner 之前，先读取 checkpoint 推断配置
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    resume_path = get_load_path(log_root, load_run=args.load_run, checkpoint=train_cfg.runner.checkpoint)
    
    # 从配置中获取维度参数（用于推断）
    num_actor_obs = env_cfg.env.num_observations
    his_latent_dim = getattr(train_cfg.policy, 'his_latent_dim', 64)
    depth_dim = 128  # DepthOnlyFCBackbone58x87 的输出维度
    terrain_attn_dim = getattr(train_cfg.policy, 'terrain_attn_output_dim', 64)
    
    inferred_config = infer_model_config_from_checkpoint(
        resume_path, 
        num_actor_obs=num_actor_obs,
        his_latent_dim=his_latent_dim,
        depth_dim=depth_dim,
        terrain_attn_dim=terrain_attn_dim
    )
    
    # 如果成功推断配置，覆盖配置文件的默认值
    if inferred_config is not None:
        # 覆盖 train_cfg.policy 中的配置
        train_cfg.policy.use_terrain_attention = inferred_config['use_terrain_attention']
        train_cfg.policy.include_depth_in_actor = inferred_config['include_depth_in_actor']
        train_cfg.policy.use_attn_kl_loss = inferred_config['use_attn_kl_loss']
        train_cfg.policy.terrain_attn_use_pre_ln = inferred_config['terrain_attn_use_pre_ln']
        
        if hasattr(env_cfg, 'terrain_attention'):
            env_cfg.terrain_attention.use_attention = inferred_config['use_terrain_attention']
            env_cfg.terrain_attention.include_depth_in_actor = inferred_config['include_depth_in_actor']
            env_cfg.terrain_attention.use_attn_kl_loss = inferred_config['use_attn_kl_loss']
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # # 改善地形渲染：提高整体亮度，侧向光增强障碍物轮廓的明暗对比
    # env.gym.set_light_parameters(env.sim, 0,
    #     gymapi.Vec3(0.9, 0.9, 0.9),
    #     gymapi.Vec3(0.55, 0.55, 0.6),
    #     gymapi.Vec3(1.0, 1.0, -1.0))
    # env.gym.set_light_parameters(env.sim, 1,
    #     gymapi.Vec3(0.4, 0.4, 0.45),
    #     gymapi.Vec3(0.0, 0.0, 0.0),
    #     gymapi.Vec3(-0.5, -1.0, -0.8))

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
    
    # ===== 检查是否使用地形注意力 =====
    use_terrain_attention = getattr(env, 'use_terrain_attention', False)
    actor_critic = ppo_runner.alg.actor_critic
    
    print(f"[Play] use_terrain_attention={use_terrain_attention}")

    for i in range(int(env.max_episode_length)):
        # get depth image
        if infos["depth"] is not None:
            depth_image = infos['depth']
        if env.cfg.depth.warp_camera or env.cfg.depth.use_camera:
            obs = (obs, depth_image)
        
        height_map = None
        terrain_xyz = None
        
        if use_terrain_attention and env.height_map is not None:
            height_map = env.height_map.clone().to(env.device)
            terrain_xyz = env.terrain_xyz.clone().to(env.device)

        if isinstance(obs, tuple):
            actions = policy(obs[0].detach(), trajectory_history.detach(), obs[1][:, :2, ...].detach(),
                           height_map=height_map, terrain_xyz=terrain_xyz)
        else:
            actions = policy(obs.detach(), trajectory_history,
                           height_map=height_map, terrain_xyz=terrain_xyz)
        obs, _, _, dones, infos, *_= env.step(actions.detach())
        
        # ===== 绘制注意力可视化 (按 T 键切换模式) =====
        viz_mode = getattr(env, 'attn_viz_mode', 'attention')
        if use_terrain_attention and viz_mode != "off":
            if viz_mode == "uniform":
                draw_uniform_points(env)
            elif viz_mode == "attention" and hasattr(actor_critic, 'terrain_attention'):
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
