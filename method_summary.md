# Terrain-Aware Locomotion Policy with KL-Guided Attention

## 一、问题定义与整体架构

本系统面向 **G1 人形机器人（16自由度）** 的多地形运动控制策略，基于 **PPO (Proximal Policy Optimization)** 和 **Actor-Critic** 框架。核心创新在于引入了 **地形注意力机制（Terrain Attention Encoder）** 和 **KL 先验引导损失（KL Prior Guidance Loss）**，使策略网络能够自适应地感知并聚焦于关键地形区域，从而提升在 stepping stones、parkour、pit、gap、stair 等复杂地形上的运动能力。

整个策略网络采用 `ActorCriticDepth` 架构：

| 模块 | 输入维度 | 输出维度 | 说明 |
|------|---------|---------|------|
| **History Encoder** | 570 (10帧×57维) | 64 | 编码本体感知时序信息 |
| **Terrain Attention Encoder** | Height Map [17×11] + XYZ [187×3] + Query [57] | 64 | **新增**: 注意力加权地形特征 |
| **Actor MLP** | 185 (57+64+64) | 16 | 输出 16 个关节目标角度 |
| **Critic MLP** | ~367 (303特权观测+64) | 1 | 价值估计（使用特权信息） |

## 二、观测空间设计

### Actor 观测 $\mathbf{o}_t \in \mathbb{R}^{57}$

$$
\mathbf{o}_t = [\mathbf{cmd}(3),\ \boldsymbol{\omega}_{\text{base}}(3),\ \mathbf{g}_{\text{proj}}(3),\ \mathbf{q} - \mathbf{q}_{\text{default}}(16),\ \dot{\mathbf{q}}(16),\ \mathbf{a}_{t-1}(16)]
$$

- 速度指令 (3维)、基座角速度 (3维)、投影重力 (3维)
- 关节位置偏差 (16维)、关节速度 (16维)、上一步动作 (16维)

### Critic 特权观测 $\mathbf{o}_t^{\text{priv}} \in \mathbb{R}^{303}$

包括基座线速度、脚部位姿/速度 (12维)、接触力 (6维)、物理参数（质量、摩擦系数、质心偏移、PD增益等，38维）、以及 **187维高度测量值**（采样网格 17×11 = 187个点）。

### 关键设计

Actor 不直接访问高度测量值，而是通过 Terrain Attention Encoder 间接获取加工后的地形特征，迫使注意力机制学会有效的地形表征。Critic 直接访问完整的特权观测（包含原始高度值），遵循 **Asymmetric Actor-Critic** 范式。

## 三、Terrain Attention Encoder（地形注意力编码器）

在机器人前方以 17×11 的网格（1.6m × 1.0m，分辨率 0.1m）采样 187 个地形点，通过 **双路特征提取 + 多头交叉注意力** 生成地形特征向量。

### 3.1 地形数据预处理

每个采样点包含 $(x, y, z)$ 三维信息。为保证注意力机制中各维度尺度一致，采用统一的归一化策略：

$$
\tilde{x} = x / 0.8, \quad \tilde{y} = y / 0.5, \quad \tilde{z} = \text{clip}(z_{\text{raw}},\ -1,\ 1)
$$

其中 $z_{\text{raw}} = z_{\text{base}} - z_{\text{base\_height}} - z_{\text{terrain}}$。三个维度均映射到 $[-1, 1]$ 范围，避免某一维度主导注意力权重计算。

### 3.2 双路特征提取

**上路 — CNN 几何特征提取**:

将高度图 $\mathbf{H} \in \mathbb{R}^{17 \times 11}$ 视为单通道图像，通过两层 5×5 卷积提取局部几何特征：

$$
\mathbf{F}_{\text{cnn}} = \text{ReLU}(\text{Conv}_{5 \times 5}^{125}(\text{ReLU}(\text{Conv}_{5 \times 5}^{32}(\mathbf{H})))) \in \mathbb{R}^{B \times 187 \times 125}
$$

5×5 卷积核的感受野覆盖 0.5m×0.5m，足以捕捉台阶边缘、缝隙宽度等关键几何特征。

**下路 — 空间坐标直通**:

归一化后的 xyz 坐标 $\tilde{\mathbf{P}} \in \mathbb{R}^{B \times 187 \times 3}$ 直接作为位置编码使用，为注意力提供显式的空间位置信息。

**特征拼接**:

$$
\mathbf{KV} = [\mathbf{F}_{\text{cnn}} \| \tilde{\mathbf{P}}] \in \mathbb{R}^{B \times 187 \times 128}
$$

CNN 输出 125 维 + xyz 3 维 = 128 维（`hidden_dim`），确保拼接后维度与 Query 对齐。

### 3.3 Query 构建

Query 由本体感知观测投影得到：

$$
\mathbf{Q} = \text{Linear}_{57 \to 128}(\mathbf{o}_t) \in \mathbb{R}^{B \times 1 \times 128}
$$

当前配置中 `terrain_attn_query_with_history=False`，即 Query 仅包含当前观测（57维→128维），不包含历史编码特征。

### 3.4 Pre-LayerNorm

在注意力计算前，对 Q 和 KV 分别施加 LayerNorm：

$$
\hat{\mathbf{Q}} = \text{LayerNorm}_Q(\mathbf{Q}), \quad \hat{\mathbf{KV}} = \text{LayerNorm}_{KV}(\mathbf{KV})
$$

**设计动机**: CNN 上路经过 ReLU 激活后输出非负值（≥0），而下路 xyz 坐标在 $[-1, 1]$ 范围内，两者直接拼接会产生**尺度不对称**。Pre-LN 在注意力之前统一 Q 和 KV 的分布，稳定 softmax 的数值行为。

### 3.5 多头交叉注意力

$$
\text{Attention}(\hat{\mathbf{Q}},\ \hat{\mathbf{K}},\ \hat{\mathbf{V}}) = \text{softmax}\left(\frac{\hat{\mathbf{Q}} \hat{\mathbf{K}}^T}{\sqrt{d_k}}\right) \hat{\mathbf{V}}
$$

- 头数: 8，每头维度: 16
- 输出: $\mathbf{F}_{\text{attn}} \in \mathbb{R}^{B \times 1 \times 128}$
- **注意力权重**: $\boldsymbol{\alpha} \in \mathbb{R}^{B \times 187}$（多头平均后）

注意力权重同时保存两份：

1. `last_attention_weights`：**detached** 版本，用于日志记录和可视化
2. `last_attention_weights_raw`：**保留计算图**版本，供 KL loss 反向传播梯度

### 3.6 输出投影

$$
\mathbf{f}_{\text{terrain}} = \text{Linear}_{128 \to 64}(\mathbf{F}_{\text{attn}}) \in \mathbb{R}^{B \times 64}
$$

最终地形特征向量拼接到 Actor 输入中：

$$
\mathbf{x}_{\text{actor}} = [\mathbf{o}_t \| \mathbf{f}_{\text{his}} \| \mathbf{f}_{\text{terrain}}] \in \mathbb{R}^{B \times 185}
$$

## 四、KL 先验引导（KL Prior Guidance）

通过一个 **无可学习参数的物理先验打分器**（`TerrainSafetyScorer`）生成目标注意力分布，利用 KL 散度辅助损失引导注意力权重向安全区域聚焦。

### 4.1 TerrainSafetyScorer（地形安全先验）

纯基于物理规则的模块，**没有任何可学习参数**，输出 softmax 概率分布作为注意力的监督目标。

#### 组件 1 — 支撑面积分数 $S_{\text{support}}$

对每个采样点，在 3×3 邻域内进行局部最小二乘平面拟合。利用正则网格的对称性（$\Sigma x = \Sigma y = \Sigma xy = 0$），残差方差有解析公式：

$$
\sigma_{\text{res}}^2 = \frac{1}{9}\left[\sum z_i^2 - \frac{(\sum x_i z_i)^2}{6\Delta x^2} - \frac{(\sum y_i z_i)^2}{6\Delta y^2} - \frac{(\sum z_i)^2}{9}\right]
$$

三个求和项通过**预计算的固定卷积核**高效实现（`x_kernel`, `y_kernel`, `ones_kernel`），无需逐点循环。

$$
S_{\text{support}} = \exp\left(-\frac{\text{RMS}_{\text{res}}}{\sigma_{\text{scale}}}\right) \in [0, 1]
$$

- 平坦地面 / 均匀斜坡 → 残差 ≈ 0 → 高分
- 台阶边缘 / 缝隙 → 残差大 → 低分

#### 组件 2 — 边缘裕度分数 $S_{\text{margin}}$

1. 构建危险区域掩码：残差超阈值（0.02m）或深坑（> 0.3m）
2. 通过**形态学腐蚀**（3×3 min pooling，最多 3 步）计算到最近危险区域的 Chebyshev 距离
3. 使用 `replicate` padding 避免边界误判

$$
S_{\text{margin}} = \frac{\text{存活腐蚀轮数}}{\text{最大轮数}} \in [0, 1]
$$

- 远离边缘 → 高分；贴近边缘 → 低分

#### 组件 3 — 身后区域惩罚

根据机体线速度方向，通过点积半平面判断身后区域：

$$
\text{behind\_mask}_{i} = \mathbb{1}\left[\|\mathbf{v}_{xy}\| > v_{\text{thresh}} \ \wedge \ \frac{\mathbf{v}_{xy} \cdot \mathbf{p}_i}{\|\mathbf{v}_{xy}\|} < -d_{\text{thresh}}\right]
$$

#### 融合与输出

$$
\text{logits} = w_s \cdot S_{\text{support}} \cdot \mathbb{1}_{\text{safe}} + w_m \cdot S_{\text{margin}} + \delta_{\text{danger}} \cdot \mathbb{1}_{\text{danger}} + \delta_{\text{behind}} \cdot \mathbb{1}_{\text{behind}}
$$

$$
p_{\text{prior}} = \text{softmax}(\text{logits} / \tau)
$$

$$
p_{\text{smooth}} = (1 - \epsilon) \cdot p_{\text{prior}} + \epsilon \cdot \frac{1}{N}
$$

Label Smoothing（$\epsilon = 0.05$）确保所有点概率非零，改善 KL 梯度覆盖。

### 4.2 KL 散度辅助损失

$$
\mathcal{L}_{\text{KL}} = \text{KL}(p_{\text{prior}} \| p_{\text{attn}}) = \sum_{i=1}^{187} p_{\text{prior}}^{(i)} \log \frac{p_{\text{prior}}^{(i)}}{p_{\text{attn}}^{(i)}}
$$

**梯度流设计**:

- $p_{\text{prior}}$：**detach**，仅作为固定目标分布，不参与梯度计算
- $p_{\text{attn}}$：**保留计算图**，梯度回传至 CNN、Q 投影层、LayerNorm 等可学习参数

**Warmup Annealing**:

支持线性预热策略，KL 系数 $\lambda_{\text{kl}}$ 从 0 线性增长至目标值：

$$
\lambda_{\text{kl}}^{(t)} = \begin{cases}
0 & t < t_{\text{start}} \\
\lambda_{\text{kl}} \cdot \dfrac{t - t_{\text{start}}}{t_{\text{end}} - t_{\text{start}}} & t_{\text{start}} \leq t \leq t_{\text{end}} \\
\lambda_{\text{kl}} & t > t_{\text{end}}
\end{cases}
$$

当前配置为 `anneal_start=0, anneal_end=0`，即直接使用全量系数 $\lambda_{\text{kl}} = 0.1$。

## 五、训练目标

总损失函数：

$$
\mathcal{L} = \mathcal{L}_{\text{PPO}} + c_v \cdot \mathcal{L}_{\text{value}} - c_e \cdot H(\pi) + \lambda_{\text{kl}} \cdot \mathcal{L}_{\text{KL}}
$$

其中：

| 损失项 | 符号 | 值 | 说明 |
|-------|------|---|------|
| PPO Clipped Surrogate | $\mathcal{L}_{\text{PPO}}$ | clip=0.2 | 策略优化主损失 |
| Value Function Loss | $\mathcal{L}_{\text{value}}$ | $c_v = 0.5$ | Clipped 价值函数损失 |
| Entropy Bonus | $H(\pi)$ | $c_e = 0.01$ | 鼓励探索 |
| Terrain KL Prior | $\mathcal{L}_{\text{KL}}$ | $\lambda_{\text{kl}} = 0.1$ | 物理先验引导注意力 |

学习率采用 **Adaptive KL 调度**：监控策略更新前后的 KL 散度，自动调整学习率在 $[10^{-5}, 10^{-2}]$ 范围内（`desired_kl=0.01`）。

## 六、训练流程

### 1. Rollout 阶段（`inference_mode`，不记录梯度）

- 每步收集：obs、history、depth_image、height_map、terrain_xyz、base_lin_vel
- Actor 前向传播产生动作，Critic 估计价值
- 地形注意力模块输出注意力权重（用于后续 KL 计算）

### 2. 学习阶段

- GAE 计算优势估计（$\gamma = 0.998,\ \lambda = 0.95$）
- 8 个 mini-batch，每轮迭代内更新
- 总损失包含 PPO + Value + Entropy + KL Prior
- 梯度裁剪（`max_grad_norm = 1.0`）
- 记录注意力模块梯度范数，监控训练信号是否有效流入 attention 分支

### 3. 监控指标

- 注意力熵（entropy）、Top-10 权重占比、最大权重
- Per-head 注意力熵（多样性）
- Prior-Attention 余弦相似度（对齐程度）
- 按地形类型的 Traverse Rate 和 Success Rate

## 七、关键设计选择与消融配置

| 设计选择 | 当前配置 | 说明 |
|---------|---------|------|
| Depth 特征是否参与 Actor | `False` | 消融实验：排除 depth，仅依赖 terrain attention |
| Query 是否包含历史特征 | `False` | Query 仅用当前 obs（57维），不含 history |
| Pre-LayerNorm | `True` | 修正 CNN+xyz 尺度不对称 |
| KL 先验引导 | `True` | 物理先验引导注意力聚焦安全区域 |
| Label Smoothing | $\epsilon = 0.05$ | 确保非零梯度覆盖 |
| 温度参数 | $\tau = 1.0$ | Prior 分布锐度控制 |

## 八、环境与地形设置

| 配置项 | 值 |
|-------|---|
| 仿真器 | Isaac Gym (GPU 并行) |
| 并行环境数 | 4096 |
| 地形类型 | Stepping Stones, Parkour, Pit, Gap, Stair（各 20%） |
| 采样网格 | 17×11 = 187 点（前后 0.8m，左右 0.5m） |
| 课程学习 | 基于 terrain level 的渐进式难度提升 |
| 域随机化 | 摩擦系数、基座质量、链接质量、质心偏移、PD 增益、电机力矩、恢复系数、驱动偏移、动作延迟 |

## 九、核心代码文件索引

| 文件 | 内容 |
|------|------|
| `rsl_rl/modules/actor_critic_depth.py` | `TerrainAttentionEncoder` + `ActorCriticDepth` 主网络 |
| `rsl_rl/modules/terrain_safety_scorer.py` | `TerrainSafetyScorer` 物理先验打分器 |
| `rsl_rl/algorithms/amp_ppo_multi.py` | PPO 训练算法（含 KL loss 集成） |
| `rsl_rl/runners/amp_on_policy_runner_multi.py` | 训练主循环、KL annealing、日志记录 |
| `legged_gym/envs/g1_loco/g1_16dof_loco_env.py` | G1 环境（含地形数据更新逻辑） |
| `legged_gym/envs/g1_loco/g1_16dof_loco_config.py` | 全部超参数配置 |
| `rsl_rl/storage/rollout_storage_extra.py` | Rollout 数据存储（含地形注意力数据） |
