# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# 本文件是空间感知注意力的通用入口
# 根据配置参数自动选择使用哪种实现方案：
#   - actor_critic_depth_image.py: 基于深度图像的方案
#   - actor_critic_depth_heightpoint.py: 基于地形高度点的方案
#
# 使用方式：
#   在配置中设置 attention_type = "image" 或 "heightpoint"

# 导出基础组件（两个方案通用）
from .actor_critic_depth_heightpoint import (
    DepthOnlyFCBackbone58x87,
    StackDepthEncoder,
    FootholdPredictor,
    get_activation,
)

# 导出两种 ActorCritic 实现
from .actor_critic_depth_image import ActorCriticDepth as ActorCriticDepthImage
from .actor_critic_depth_heightpoint import ActorCriticDepth as ActorCriticDepthHeightPoint

# 导出两种 SpatialAttentionEncoder 实现
from .actor_critic_depth_image import SpatialAttentionEncoder as SpatialAttentionEncoderImage
from .actor_critic_depth_heightpoint import HeightPointAttentionEncoder


def create_actor_critic(attention_type: str = "heightpoint", **kwargs):
    """
    工厂函数：根据 attention_type 创建对应的 ActorCritic 实例
    
    Args:
        attention_type: "image" 使用深度图像方案, "heightpoint" 使用高度点方案
        **kwargs: 传递给 ActorCritic 构造函数的参数
    
    Returns:
        ActorCriticDepth 实例
    """
    if attention_type == "image":
        print("=" * 60)
        print("  使用深度图像方案 (ActorCriticDepthImage)")
        print("=" * 60)
        return ActorCriticDepthImage(**kwargs)
    elif attention_type == "heightpoint":
        print("=" * 60)
        print("  使用高度点方案 (ActorCriticDepthHeightPoint)")
        print("=" * 60)
        return ActorCriticDepthHeightPoint(**kwargs)
    else:
        raise ValueError(f"Unknown attention_type: {attention_type}. Use 'image' or 'heightpoint'.")


# 默认导出高度点方案（当前开发的新方案）
# 这样保持向后兼容：直接 from actor_critic_depth import ActorCriticDepth 仍然可用
ActorCriticDepth = ActorCriticDepthHeightPoint
