from copy import deepcopy


# 统一实验配置（B1 与 M1 配置相同，因此不单独维护 B1）
#
# scorer 分量默认值（V2）:
#   w_support=0.3, w_margin=0.3, w_edge=0.5, w_forward=0.8

_SCORER_DEFAULTS = {
    "scorer_w_support": 0.3,
    "scorer_w_margin": 0.3,
    "scorer_w_edge": 0.5,
    "scorer_w_forward": 0.8,
}

EXPERIMENT_PROFILES = {
    "M1": {
        "description": "Ours full method",
        "use_attention": True,
        "include_depth_in_actor": False,
        "terrain_attn_query_with_history": False,
        "terrain_attn_query_with_depth": True,
        "use_attn_kl_loss": True,
        "attn_kl_coef": 0.1,
        **_SCORER_DEFAULTS,
    },
    "M2": {
        "description": "Depth encoding baseline (MoRE stage 1)",
        "use_attention": False,
        "include_depth_in_actor": True,
        "terrain_attn_query_with_history": False,
        "terrain_attn_query_with_depth": False,
        "use_attn_kl_loss": False,
        "attn_kl_coef": 0.1,
        **_SCORER_DEFAULTS,
    },
    "M3": {
        "description": "Attention baseline (He et al. style)",
        "use_attention": True,
        "include_depth_in_actor": False,
        "terrain_attn_query_with_history": False,
        "terrain_attn_query_with_depth": False,
        "use_attn_kl_loss": False,
        "attn_kl_coef": 0.1,
        **_SCORER_DEFAULTS,
    },
    "D1": {
        "description": "Ours without KL prior guidance",
        "use_attention": True,
        "include_depth_in_actor": False,
        "terrain_attn_query_with_history": False,
        "terrain_attn_query_with_depth": True,
        "use_attn_kl_loss": False,
        "attn_kl_coef": 0.1,
        **_SCORER_DEFAULTS,
    },
    "B2": {
        "description": "Depth as Actor input instead of Query",
        "use_attention": True,
        "include_depth_in_actor": True,
        "terrain_attn_query_with_history": False,
        "terrain_attn_query_with_depth": False,
        "use_attn_kl_loss": True,
        "attn_kl_coef": 0.1,
        **_SCORER_DEFAULTS,
    },
    "A2": {
        "description": "Remove forward_bias (w_forward=0)",
        "use_attention": True,
        "include_depth_in_actor": False,
        "terrain_attn_query_with_history": False,
        "terrain_attn_query_with_depth": True,
        "use_attn_kl_loss": True,
        "attn_kl_coef": 0.1,
        **_SCORER_DEFAULTS,
        "scorer_w_forward": 0.0,
    },
    "A3": {
        "description": "Remove edge_bonus (w_edge=0)",
        "use_attention": True,
        "include_depth_in_actor": False,
        "terrain_attn_query_with_history": False,
        "terrain_attn_query_with_depth": True,
        "use_attn_kl_loss": True,
        "attn_kl_coef": 0.1,
        **_SCORER_DEFAULTS,
        "scorer_w_edge": 0.0,
    },
}


def get_available_experiment_ids():
    return sorted(EXPERIMENT_PROFILES.keys())


def apply_experiment_profile(env_cfg, train_cfg, exp_id):
    exp_key = exp_id.strip().upper()
    if exp_key == "B1":
        raise ValueError("B1 与 M1 配置相同，请直接使用 --exp_id M1。")
    if exp_key not in EXPERIMENT_PROFILES:
        available = ", ".join(get_available_experiment_ids())
        raise ValueError(f"未知 exp_id='{exp_id}'，可选: {available}")

    if not hasattr(env_cfg, "terrain_attention"):
        raise AttributeError("env_cfg 缺少 terrain_attention 配置，无法应用实验开关。")
    if not hasattr(train_cfg, "policy"):
        raise AttributeError("train_cfg 缺少 policy 配置，无法应用实验开关。")

    profile = deepcopy(EXPERIMENT_PROFILES[exp_key])
    env_ta = env_cfg.terrain_attention
    policy = train_cfg.policy

    # env 侧开关（runner 使用）
    env_ta.use_attention = profile["use_attention"]
    env_ta.include_depth_in_actor = profile["include_depth_in_actor"]
    env_ta.terrain_attn_query_with_depth = profile["terrain_attn_query_with_depth"]
    env_ta.use_attn_kl_loss = profile["use_attn_kl_loss"]
    env_ta.attn_kl_coef = profile["attn_kl_coef"]
    env_ta.scorer_w_support = profile["scorer_w_support"]
    env_ta.scorer_w_margin = profile["scorer_w_margin"]
    env_ta.scorer_w_edge = profile["scorer_w_edge"]
    env_ta.scorer_w_forward = profile["scorer_w_forward"]

    # policy 侧开关（ActorCriticDepth 使用）
    policy.use_terrain_attention = profile["use_attention"]
    policy.include_depth_in_actor = profile["include_depth_in_actor"]
    policy.terrain_attn_query_with_history = profile["terrain_attn_query_with_history"]
    policy.terrain_attn_query_with_depth = profile["terrain_attn_query_with_depth"]
    policy.use_attn_kl_loss = profile["use_attn_kl_loss"]
    policy.scorer_w_support = profile["scorer_w_support"]
    policy.scorer_w_margin = profile["scorer_w_margin"]
    policy.scorer_w_edge = profile["scorer_w_edge"]
    policy.scorer_w_forward = profile["scorer_w_forward"]

    profile["exp_id"] = exp_key
    return profile
