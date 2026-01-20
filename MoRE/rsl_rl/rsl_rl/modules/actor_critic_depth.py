# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# 本文件是 ActorCriticDepth 的统一入口。
# 当前代码库仅保留“高度点空间注意力”方案（HeightPoint Attention）。
# 深度图像注意力方案已移除，以减少维护成本与混淆。

# 导出基础组件（两个方案通用）
from .actor_critic_depth_heightpoint import (
    DepthOnlyFCBackbone58x87,
    StackDepthEncoder,
    get_activation,
)

# 导出高度点 ActorCritic 实现
from .actor_critic_depth_heightpoint import ActorCriticDepth as ActorCriticDepthHeightPoint
from .actor_critic_depth_heightpoint import HeightPointAttentionEncoder

# 默认导出高度点方案
ActorCriticDepth = ActorCriticDepthHeightPoint
