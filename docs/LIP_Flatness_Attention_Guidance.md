# 基于 LIP + 平坦度的注意力引导机制

## 1. 背景与问题

### 1.1 原有设计的问题

在原有的注意力机制设计中，存在以下问题：

1. **注意力缺乏直接监督信号**：注意力权重只能通过 Actor 的 PPO loss 间接反向传播来训练，导致注意力分布可能长期保持接近均匀分布。

2. **Query 信息量不足**：Query 由 `proprioception + Raibert 名义点` 构成，但缺乏地形安全性的先验知识。

3. **辅助任务效果有限**：FootholdPredictor 只监督预测的落足点是否正确，并没有直接告诉注意力"应该关注哪些点"。

### 1.2 核心思想

参考 PLANC 论文的设计思想，使用物理模型（线性倒立摆 LIP）为每个地形采样点计算"安全分数"，直接引导注意力关注安全的区域。

---

## 2. 设计方案

### 2.1 总体架构

采用 **"偏置 + 监督"** 的双重引导模式：

- **偏置**：在注意力计算时，将物理安全分数作为偏置加入，直接引导注意力
- **监督**：使用 KL 散度作为 loss，进一步强化学习

```
训练时：偏置 + 监督 (双重引导)
推理时：只用偏置 (物理保证)
```

### 2.2 安全分数计算

使用两个物理指标的组合：

| 指标 | 作用 | 计算方式 |
|------|------|----------|
| **LIP Score** | 动力学稳定性评估 | 基于 Orbital Energy |
| **Flatness Score** | 边缘/空隙检测 | 基于局部高度梯度 |

综合评分：
```
combined_score = lip_score × flatness_score
```

### 2.3 速度方向感知 (Heading Awareness)

**问题**：原始的 LIP Score 只评估能量，没有考虑速度方向。如果机器人向前跑，后方的点即使 LIP 能量再高，也不应该被关注。

**解决方案**：添加速度方向因子，只关注速度方向扇区内的点。

```
θ = arctan2(point_y - base_y, point_x - base_x) - arctan2(v_y, v_x)
heading_factor = ReLU(cos(θ))
final_score = combined_score × heading_factor
```

**物理含义**：
- `cos(θ) > 0`: 点在速度方向前方（±90°扇区内）→ 正常计分
- `cos(θ) ≤ 0`: 点在速度方向后方 → 分数归零，不关注

```
               速度方向 v
                  ↑
                  │
          ╭───────┼───────╮
        ╱      关注区域    ╲   cos(θ) > 0
       ╱      (前方扇区)    ╲
      ╱           │         ╲
     ╱            │          ╲
────┼─────────────●──────────┼────  cos(θ) = 0
     ╲      忽略区域（后方）  ╱      cos(θ) < 0
      ╲          │         ╱
       ╲         │        ╱
        ╲────────┼───────╱
```

### 2.4 课程式偏置衰减 (Curriculum-based β Decay)

**思想**：训练初期用强物理偏置快速引导，随着课程难度增加，逐渐减小偏置让网络自主学习。

```
β(level) = β_max × (1 - level / max_level) + β_min × (level / max_level)
```

| 训练阶段 | Curriculum Level | β 值 | 效果 |
|----------|------------------|------|------|
| 初期 | 0 | 2.0 | 强物理引导，快速学习基础 |
| 中期 | 5 | 1.0 | 中等引导，开始自主探索 |
| 后期 | 10 | 0.5~0 | 弱/无引导，完全自主决策 |

**类比**：
- 初期 = 骑自行车时扶着后座
- 后期 = 完全放手，让孩子自己骑

### 2.5 可部署性

所有计算所需的信息都是可部署的：

| 信息 | 仿真中 | 实机部署 |
|------|--------|----------|
| CoM 位置/速度 | `root_states` | IMU + 状态估计 |
| 采样点高度 | Raycast | 深度相机 + 点云 |

**这不是特权信息，可以在实机上部署！**

---

## 3. 核心模块

### 3.1 TerrainSafetyScorer

负责计算每个地形采样点的安全分数。

#### 输入
- `height_points [B, 17, 11, 3]`: 地形采样点 (x, y, z)
- `base_lin_vel [B, 3]`: 基座线速度 (body frame)
- `z_com`: CoM 高度

#### 输出
- `final_score [B, 187]`: 每个点的最终安全分数（含方向感知）
- `target_attention [B, 187]`: 目标注意力分布

#### LIP Score 计算

基于线性倒立摆的 Orbital Energy：

```python
z_eff = z_com - z_points          # 有效 CoM 高度
λ = sqrt(g / z_eff)               # LIP 自然频率
E = (v_x/λ)² + (v_y/λ)²           # Orbital Energy
lip_score = normalize(E)          # 归一化到 [0, 1]
```

**物理含义**：
- E > 0: 有足够动量跨过支撑点，可以继续前进（安全）
- E 越大: 动力学越稳定

#### Flatness Score 计算

基于局部高度梯度：

```python
# 计算高度梯度
grad_x = z[i+1, j] - z[i, j]
grad_y = z[i, j+1] - z[i, j]
gradient_mag = sqrt(grad_x² + grad_y²)

# 梯度越大，分数越低
flatness = exp(-gradient_mag / σ)
```

**物理含义**：
- 梯度大: 边缘或空隙边界 → 低分
- 梯度小: 平坦区域 → 高分

#### Heading Factor 计算

基于速度方向的扇区过滤：

```python
# 计算速度大小
vel_magnitude = sqrt(v_x² + v_y²)

# 低速时禁用方向过滤（避免原地站立时的不稳定）
if vel_magnitude < min_vel_threshold:
    heading_factor = 1.0  # 全部保留
else:
    # 计算点相对于机器人的方向角
    point_angle = atan2(point_y, point_x)  # body frame 下点的方向
    
    # 计算速度方向角
    vel_angle = atan2(v_y, v_x)            # body frame 下速度方向
    
    # 相对角度
    theta = point_angle - vel_angle
    
    # 方向因子：只保留前方扇区
    heading_factor = max(0, cos(theta))    # ReLU(cos(θ))
```

**物理含义**：
- `cos(θ) > 0`: 点在运动方向前方 → 保留分数
- `cos(θ) ≤ 0`: 点在运动方向后方 → 分数归零
- 低速时（< 0.1 m/s）禁用方向过滤，避免原地站立时注意力消失

#### 最终分数

```python
combined_score = lip_score × flatness_score
final_score = combined_score × heading_factor
```

### 3.2 注意力偏置机制

在 CrossAttention 中添加物理偏置：

```python
# 原始注意力分数
raw_attention = (Q × K^T) / √d

# 添加物理偏置 (使用含方向感知的最终分数)
biased_attention = raw_attention + β × final_score

# 最终注意力
pred_attention = softmax(biased_attention)
```

**参数 β（课程式衰减）**：
- 初期: β = 2.0，强物理引导
- 后期: β → 0.5 或 0，让网络自主学习
- 衰减公式: `β = β_max - (β_max - β_min) × (level / max_level)`

### 3.3 注意力监督 Loss

使用 KL 散度作为监督信号：

```python
# 目标分布 (使用含方向感知的最终分数)
target_attention = softmax(final_score / τ)

# 监督 loss
L_attn = KL_div(pred_attention, target_attention)
```

**参数 τ (温度)**：
- τ 小: 分布尖锐，只关注最安全的几个点
- τ 大: 分布平滑，关注更多的点
- 建议范围: 0.1 ~ 1.0

---

## 4. 数据流

### 4.1 训练时

```
1. 获取当前课程等级，计算动态 β:
   β = get_beta(curriculum_level, cfg)

2. 环境计算安全分数 (含方向感知):
   TerrainSafetyScorer(height_points, base_lin_vel)
   → lip_score, flatness_score, heading_factor
   → final_score = lip_score × flatness_score × heading_factor
   → target_attention = softmax(final_score / τ)

3. Actor 前向传播 (带动态偏置):
   raw_attention = Q × K^T / √d
   biased_attention = raw_attention + β × final_score
   pred_attention = softmax(biased_attention)
   attn_feature = Σ(pred_attention × values)
   actions = Actor([obs, his, depth, attn_feature])

4. 计算 Loss:
   L_attn = KL_div(pred_attention, target_attention)
   L_total = L_ppo + λ_v × L_value - λ_e × entropy + λ_attn × L_attn

5. 反向传播更新网络

注: 初期 β 大 → 强物理引导；后期 β 小 → 网络自主学习
```

### 4.2 推理时

```
1. 计算 β (可选择固定为较小值，如 0.5):
   β = β_inference  # 或使用训练结束时的值

2. 环境计算安全分数 (含方向感知):
   TerrainSafetyScorer(height_points, base_lin_vel)
   → final_score

3. Actor 前向传播 (带偏置):
   raw_attention = Q × K^T / √d
   biased_attention = raw_attention + β × final_score
   pred_attention = softmax(biased_attention)
   attn_feature = Σ(pred_attention × values)
   actions = Actor([obs, his, depth, attn_feature])

注意: 
- 推理时不需要计算 target_attention 和 loss
- 物理偏置 + 训练好的网络共同保证注意力的合理性
- 推理时 β 可保留一个小值 (如 0.5) 作为安全保障
```

---

## 5. 优势分析

### 5.1 为什么偏置 + 监督比纯监督更好？

| 方面 | 纯监督 | 偏置 + 监督 |
|------|--------|-------------|
| 初期学习 | 慢，注意力可能保持均匀 | 快，偏置直接引导 |
| 收敛速度 | 慢 | 快 |
| 推理时保证 | 完全依赖网络 | 有物理保证 |
| Sim2Real | 可能泛化不好 | 物理偏置增强鲁棒性 |

### 5.2 课程式衰减的优势

| 阶段 | 固定 β | 课程式衰减 β |
|------|--------|--------------|
| 初期 | 可能 β 太小学不动 | β 大，强引导快速入门 |
| 中期 | 可能 β 太大限制探索 | β 中等，平衡引导与探索 |
| 后期 | 可能过度依赖物理偏置 | β 小/零，网络完全自主 |

**核心思想**：让网络从"依赖物理先验"逐渐过渡到"自主决策"

### 5.3 速度方向感知的优势

| 情况 | 无方向感知 | 有方向感知 |
|------|------------|------------|
| 向前跑 | 后方安全点也被关注 → 注意力分散 | 只关注前方扇区 → 注意力集中 |
| 转弯 | 侧方点权重不合理 | 根据速度方向动态调整 |
| 原地站立 | 正常（此时禁用方向过滤） | 正常（速度太小时禁用） |

**类比**：
- 无方向感知 = 四面八方都看
- 有方向感知 = 只看前进方向，像汽车驾驶员一样

### 5.4 总体类比

- **纯监督** = 只告诉学生"这是正确答案"，让他自己悟
- **偏置 + 监督** = 手把手带着做 + 告诉正确答案
- **课程式衰减** = 初期扶着自行车，后期放手让孩子自己骑
- **速度方向感知** = 专注于前进方向，不分心看后面

---

## 6. 超参数配置

### 6.1 TerrainSafetyScorer 参数

```python
class TerrainSafetyScorerCfg:
    # LIP 参数
    z0 = 0.75                    # 名义 CoM 高度 (m)
    gravity = 9.81               # 重力加速度 (m/s²)
    
    # 平坦度参数
    sigma_flatness = 0.05        # 平坦度评分的梯度阈值
    
    # 方向感知参数
    use_heading_awareness = True # 是否启用速度方向感知
    min_vel_for_heading = 0.1    # 低于此速度时不使用方向过滤 (m/s)
    
    # 目标分布参数
    temperature = 0.1            # softmax 温度
```

### 6.2 注意力偏置参数（含课程式衰减）

```python
class AttentionCfg:
    use_safety_bias = True       # 是否使用安全偏置
    use_attention_loss = True    # 是否使用注意力监督 loss
    attention_loss_coef = 0.5    # 注意力 loss 权重 λ_attn
    
    # 课程式 β 衰减
    use_curriculum_decay = True  # 是否启用课程式衰减
    beta_max = 2.0               # 初期偏置强度
    beta_min = 0.0               # 后期偏置强度
    decay_curriculum_levels = 10 # 衰减所需的课程等级数
```

### 6.3 课程式衰减策略

```python
def get_beta(curriculum_level, cfg):
    """根据课程等级计算当前的偏置强度 β"""
    if not cfg.use_curriculum_decay:
        return cfg.beta_max
    
    # 线性衰减
    progress = min(curriculum_level / cfg.decay_curriculum_levels, 1.0)
    beta = cfg.beta_max - (cfg.beta_max - cfg.beta_min) * progress
    return beta

# 示例：
# Level 0  → β = 2.0 (强引导)
# Level 5  → β = 1.0 (中等引导)
# Level 10 → β = 0.0 (无引导，完全自主)
```

---

## 7. 实现文件

| 文件 | 修改内容 |
|------|----------|
| `legged_gym/envs/base/terrain_safety_scorer.py` | 新增：TerrainSafetyScorer 类 |
| `legged_gym/envs/g1_loco/g1_16dof_loco_env.py` | 修改：集成 TerrainSafetyScorer |
| `legged_gym/envs/g1_loco/g1_16dof_loco_config.py` | 修改：添加配置参数 |
| `rsl_rl/modules/actor_critic_depth_heightpoint.py` | 修改：添加安全偏置到注意力 |
| `rsl_rl/storage/rollout_storage_extra.py` | 修改：添加 target_attention buffer |
| `rsl_rl/algorithms/amp_ppo_multi.py` | 修改：添加 attention_loss 计算 |
| `rsl_rl/runners/amp_on_policy_runner_multi.py` | 修改：传递 target_attention |

---

## 8. 预期效果

### 8.1 注意力行为

| 指标 | 无引导 | 有引导 (初期) | 有引导 (后期) |
|------|--------|---------------|---------------|
| Entropy | 高 (均匀分布) | 中 (物理引导) | 低 (网络自主聚焦) |
| Peak Value | 低 | 中 | 高 |
| Sparsity | 低 | 中 | 高 |

### 8.2 预期训练曲线

```
Entropy ─────────────────────────────────────
         │  ╲
         │   ╲ ← 课程式衰减期 (β 逐渐减小)
         │    ╲ 
         │     ╲ ← 网络开始自主聚焦
         │      ───────── ← 稳定在较低水平
         └────────────────────────────────→ Level
              初期      中期      后期
```

### 8.3 具体效果

1. **注意力分布变化**：entropy 下降、peak_value 上升、sparsity 上升
2. **学习速度**：更快收敛到有效策略（尤其在训练初期）
3. **落脚精度**：机器人更准确地踩在安全区域
4. **方向一致性**：注意力集中在运动方向前方，减少无关区域的干扰
5. **自主决策能力**：后期网络能够脱离物理偏置独立做出正确判断
6. **Sim2Real**：物理偏置增强跨域泛化能力

---

## 9. 参考文献

- PLANC: Physics-guided RL for Agile Humanoid Locomotion on Constrained Footholds
- Linear Inverted Pendulum Model for Bipedal Locomotion
- Control Lyapunov Functions for Stability Analysis

