from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit_resi, export_policy_as_jit_depth, task_registry
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.episode_length_s = 100
    # For visualization, keep a safe default (1) unless explicitly overridden via --num_envs.
    env_cfg.env.num_envs = args.num_envs if args.num_envs is not None else 1
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
    
    # 获取 actor_critic 模型引用，用于获取注意力权重
    actor_critic = ppo_runner.alg.actor_critic
    use_spatial_attention = hasattr(actor_critic, 'use_spatial_attention') and actor_critic.use_spatial_attention

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

    for i in range(int(env.max_episode_length)):
        # get depth image
        if infos["depth"] is not None:
            depth_image = infos['depth']
        if env.cfg.depth.warp_camera or env.cfg.depth.use_camera:
            obs = (obs, depth_image)

        # 准备高度点空间注意力所需的额外输入
        height_points = None
        safety_scores = None
        if use_spatial_attention:
            # height_points: 获取带高度的地形采样点 [num_envs, 17, 11, 3]
            if hasattr(env, 'get_height_points_with_heights'):
                height_points = env.get_height_points_with_heights()
            # safety_scores: 用于注意力安全偏置（若环境实现）
            if hasattr(env, 'compute_safety_scores'):
                safety_scores, _ = env.compute_safety_scores()

        if isinstance(obs, tuple):
            # 使用深度图的模型，支持基于高度点的空间感知注意力
            actions = policy(obs[0].detach(), trajectory_history.detach(), obs[1][:, :2, ...].detach(),
                           height_points=height_points, safety_scores=safety_scores)
        else:
            # 不使用深度图的模型
            actions = policy(obs.detach(), trajectory_history)
        
        # 获取并传递注意力权重给环境（用于可视化）
        if use_spatial_attention:
            attn_weights = actor_critic.get_attention_weights()
            if hasattr(env, 'set_attention_weights'):
                env.set_attention_weights(attn_weights)
        
        obs, _, _, dones, infos, *_= env.step(actions.detach())

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
