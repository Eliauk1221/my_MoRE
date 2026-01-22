import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    """仅提取 yaw 分量并作用到 vec 上（严格的水平航向坐标系旋转）。

    说明：
    - 之前通过把 (x,y) 置零再 normalize 的方式近似“只保留 yaw”，但当 roll/pitch 不为 0 时，
      该方法会让 z,w 分量仍然耦合进 roll/pitch，导致所谓“yaw-only”仍随 roll/pitch 改变。
    - 这里改为严格从 quaternion 解析 yaw，再构造 yaw-only quaternion:
        q_yaw = [0, 0, sin(yaw/2), cos(yaw/2)]
      从而保证 Z 轴与重力方向对齐，不随 pitch/roll 旋转。
    """
    q = normalize(quat.clone().view(-1, 4))
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # yaw (Z 轴旋转): atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    q_yaw = torch.zeros_like(q)
    q_yaw[:, 2] = torch.sin(half)
    q_yaw[:, 3] = torch.cos(half)
    return quat_apply(q_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower