import copy
import os

import isaacgym
import wandb

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, class_to_dict
from legged_gym.utils.experiment_profiles import (
    apply_experiment_profile,
    get_available_experiment_ids,
)


def train_with_profile(args):
    if args.exp_id is None:
        available = ", ".join(get_available_experiment_ids())
        raise ValueError(
            f"train_with_profile.py 需要显式指定 --exp_id。可选: {available}"
        )

    exp_id = args.exp_id.strip().upper()
    mode = "disabled" if args.no_wandb else "online"

    gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
    print(gpu_world_size)
    is_distributed = gpu_world_size > 1
    if is_distributed:
        gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        args.sim_device = f"cuda:{gpu_local_rank}"
        args.rl_device = f"cuda:{gpu_local_rank}"

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg = copy.deepcopy(env_cfg)
    train_cfg = copy.deepcopy(train_cfg)

    profile = apply_experiment_profile(env_cfg, train_cfg, exp_id)
    print(
        f"[ExperimentProfile] {profile['exp_id']} ({profile['description']}): "
        f"use_attention={profile['use_attention']}, "
        f"query_with_depth={profile['terrain_attn_query_with_depth']}, "
        f"depth_in_actor={profile['include_depth_in_actor']}, "
        f"use_attn_kl_loss={profile['use_attn_kl_loss']}, "
        f"lambda_kl={profile['attn_kl_coef']}, "
        f"w_support={profile['scorer_w_support']}, "
        f"w_margin={profile['scorer_w_margin']}, "
        f"w_edge={profile['scorer_w_edge']}, "
        f"w_forward={profile['scorer_w_forward']}"
    )

    # 默认自动附加实验 ID，避免多实验日志互相覆盖
    if args.run_name is None:
        args.run_name = profile["exp_id"].lower()
    else:
        args.run_name = f"{args.run_name}_{profile['exp_id'].lower()}"

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=args.log_root,
    )
    os.makedirs(ppo_runner.log_dir, exist_ok=True)

    wandb.init(
        project="hamp_terrain",
        group=f"{args.task}_{profile['exp_id']}",
        name=args.task + "_" + train_cfg.runner.run_name,
        config={
            "exp_profile": profile,
            "env": class_to_dict(env_cfg),
            "train": class_to_dict(train_cfg),
        },
        dir=ppo_runner.log_dir,
        mode=mode,
        sync_tensorboard=True,
    )

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()
    train_with_profile(args)
