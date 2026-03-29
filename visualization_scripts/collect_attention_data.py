#!/usr/bin/env python3
"""
Collect attention weights from trained checkpoints for visualization.

Run this on a machine with GPU + IsaacGym. It loads a checkpoint, runs a few
steps on each terrain type, and saves attention weights + height maps to .npz.

Usage:
    python collect_attention_data.py --exp_id M1 --checkpoint /path/to/model_50000.pt
    python collect_attention_data.py --exp_id D1 --checkpoint /path/to/model_50000.pt

Output:
    figures/attn_data_{exp_id}.npz containing:
        height_maps: [N, 17, 11]   — height map per snapshot
        attn_weights: [N, 187]     — attention weights per snapshot
        prior_dists: [N, 187]      — prior distribution per snapshot (if available)
        terrain_types: [N]         — terrain type index per snapshot
        terrain_names: list        — terrain name strings

NOTE: This script requires IsaacGym and the training environment.
      If you cannot run this, you can also manually create .npz files from
      TensorBoard or WandB logged attention data.
"""

import sys
import os
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', 'MoRE')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'rsl_rl'))

import numpy as np


def _get_clean_train_args():
    """
    legged_gym.utils.get_args() 会读取 sys.argv。
    这里临时清空 argv，避免与本脚本参数（如 --checkpoint 路径）冲突。
    """
    from legged_gym.utils import get_args as get_train_args

    argv_backup = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        return get_train_args()
    finally:
        sys.argv = argv_backup


def collect(args):
    import isaacgym
    import torch
    from legged_gym.envs import task_registry
    from legged_gym.utils.experiment_profiles import apply_experiment_profile
    import copy

    # 先固定输出目录（按用户启动脚本时的当前目录解析）
    output_dir_abs = args.output_dir if os.path.isabs(args.output_dir) else os.path.abspath(args.output_dir)

    # 统一到项目根目录，避免配置里的相对路径（如 ./resources/...）解析失败
    os.chdir(project_root)

    train_args = _get_clean_train_args()
    train_args.task = 'g1_16dof_loco'
    train_args.num_envs = 64
    train_args.headless = True

    env_cfg, train_cfg = task_registry.get_cfgs(name=train_args.task)
    env_cfg = copy.deepcopy(env_cfg)
    train_cfg = copy.deepcopy(train_cfg)

    if args.exp_id:
        apply_experiment_profile(env_cfg, train_cfg, args.exp_id)

    if hasattr(env_cfg, 'amp_motion_files') and isinstance(env_cfg.amp_motion_files, str):
        if not os.path.isabs(env_cfg.amp_motion_files):
            env_cfg.amp_motion_files = os.path.join(
                project_root, env_cfg.amp_motion_files.lstrip('./'))

    env, env_cfg = task_registry.make_env(
        name=train_args.task, args=train_args, env_cfg=env_cfg)

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = os.path.dirname(args.checkpoint)
    ckpt_name = os.path.basename(args.checkpoint)
    train_cfg.runner.checkpoint = int(ckpt_name.replace('model_', '').replace('.pt', ''))

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=train_args.task, args=train_args, train_cfg=train_cfg)

    ppo_runner.alg.actor_critic.eval()

    height_maps_all = []
    attn_weights_all = []
    prior_dists_all = []
    terrain_types_all = []

    print(f"Collecting data for {args.exp_id}...")
    obs = env.get_observations()
    privileged_obs = env.get_privileged_observations()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    obs = obs.to(env.device)
    critic_obs = critic_obs.to(env.device)

    trajectory_history = torch.zeros(
        size=(env.num_envs, env.obs_history_len, env.num_obs), device=env.device)
    trajectory_history = torch.concat(
        (trajectory_history[:, 1:], obs.unsqueeze(1)), dim=1)

    infos = {}
    infos["depth"] = env.warp_depth_buffer.clone().to(env.device) if ppo_runner.use_depth else None

    for step in range(args.num_steps):
        with torch.inference_mode():
            history = trajectory_history

            if infos["depth"] is not None:
                depth_image = infos["depth"]
                obs_input = (obs, depth_image)
            else:
                obs_input = obs

            terrain_data = None
            if hasattr(env, 'height_map') and hasattr(env, 'terrain_xyz'):
                terrain_data = {
                    'height_map': env.height_map.clone().to(env.device),
                    'terrain_xyz': env.terrain_xyz.clone().to(env.device),
                }

            actions = ppo_runner.alg.act(obs_input, critic_obs, history, terrain_data=terrain_data)

        obs, privileged_obs, _, dones, infos, *_ = env.step(actions.detach())
        if ppo_runner.use_depth:
            if hasattr(env, "warp_depth_buffer") and env.warp_depth_buffer is not None:
                infos["depth"] = env.warp_depth_buffer.clone().to(env.device)
            elif hasattr(env, "depth_buffer") and env.depth_buffer is not None:
                infos["depth"] = env.depth_buffer.clone().to(env.device)
            else:
                raise RuntimeError("use_depth=True but neither warp_depth_buffer nor depth_buffer exists.")
        else:
            infos["depth"] = None
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs = obs.to(env.device)
        critic_obs = critic_obs.to(env.device)
        dones = dones.to(env.device)

        env_ids = dones.nonzero(as_tuple=False).flatten()
        trajectory_history[env_ids] = 0
        trajectory_history = torch.concat(
            (trajectory_history[:, 1:], obs.unsqueeze(1)), dim=1)

        if step >= args.warmup_steps and step % args.collect_interval == 0:
            hm = env.height_map.detach().cpu().numpy()
            height_maps_all.append(hm)

            actor_critic = ppo_runner.alg.actor_critic
            if hasattr(actor_critic, 'terrain_attention') and actor_critic.terrain_attention is not None:
                attn = actor_critic.terrain_attention.last_attention_weights
                if attn is not None:
                    attn_weights_all.append(attn.cpu().numpy())

            scorer = getattr(actor_critic, 'terrain_safety_scorer', None)
            if scorer is not None:
                with torch.no_grad():
                    prior = scorer(env.height_map, env.base_lin_vel)
                prior_dists_all.append(prior.cpu().numpy())

            if hasattr(env, 'terrain_types'):
                terrain_types_all.append(env.terrain_types.cpu().numpy())

            print(f"  Step {step}: collected {hm.shape[0]} snapshots")

    save_dict = {}
    if height_maps_all:
        save_dict['height_maps'] = np.concatenate(height_maps_all, axis=0)
    if attn_weights_all:
        save_dict['attn_weights'] = np.concatenate(attn_weights_all, axis=0)
    if prior_dists_all:
        save_dict['prior_dists'] = np.concatenate(prior_dists_all, axis=0)
    if terrain_types_all:
        save_dict['terrain_types'] = np.concatenate(terrain_types_all, axis=0)
    save_dict['terrain_names'] = np.array(
        ['stepping_stones', 'parkour', 'pit', 'gap', 'stair'])

    out_dir = output_dir_abs
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'attn_data_{args.exp_id}.npz')
    np.savez(out_path, **save_dict)
    print(f"Saved: {out_path}")
    print(f"  height_maps: {save_dict.get('height_maps', np.array([])).shape}")
    print(f"  attn_weights: {save_dict.get('attn_weights', np.array([])).shape}")
    print(f"  prior_dists: {save_dict.get('prior_dists', np.array([])).shape}")

    # IsaacGym 在 Python 解释器退出阶段偶发段错误（数据已保存）。
    # 默认使用硬退出避免“已完成但返回 139”的假失败。
    if not args.graceful_exit:
        print("Collection finished. Exiting via os._exit(0) to avoid IsaacGym teardown crash.")
        sys.stdout.flush()
        os._exit(0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_id', type=str, required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--num_steps', type=int, default=200)
    p.add_argument('--warmup_steps', type=int, default=50)
    p.add_argument('--collect_interval', type=int, default=10)
    p.add_argument('--output_dir', type=str, default='./figures')
    p.add_argument('--graceful_exit', action='store_true',
                   help='Use normal Python exit (may segfault on IsaacGym teardown)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    collect(args)
