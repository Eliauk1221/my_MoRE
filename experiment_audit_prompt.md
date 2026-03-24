# Experiment Configuration Audit & Setup Prompt

## Background

I'm working on a humanoid robot locomotion project using RL (PPO + Actor-Critic). The core architecture is `ActorCriticDepth` with a `TerrainAttentionEncoder` and `TerrainSafetyScorer` for KL prior guidance. I need to set up multiple experiment configurations for a paper submission. Please help me audit the current codebase and prepare all necessary configs.

## Key Code Files to Check

- `rsl_rl/modules/actor_critic_depth.py` — `TerrainAttentionEncoder` + `ActorCriticDepth`
- `rsl_rl/modules/terrain_safety_scorer.py` — `TerrainSafetyScorer`
- `rsl_rl/algorithms/amp_ppo_multi.py` — PPO training (KL loss integration)
- `rsl_rl/runners/amp_on_policy_runner_multi.py` — Training loop, KL annealing, logging
- `legged_gym/envs/g1_loco/g1_16dof_loco_env.py` — G1 environment
- `legged_gym/envs/g1_loco/g1_16dof_loco_config.py` — All hyperparameters
- `rsl_rl/storage/rollout_storage_extra.py` — Rollout data storage

---

## Task 1: Audit Existing Switches and Flags

Please search the entire codebase for all boolean flags, switches, and configuration options related to the following features. For each one, tell me: (a) the exact variable name, (b) which file it's in, (c) its current default value, and (d) how it's used.

Features to check:

1. **Terrain Attention Encoder on/off** — Is there a flag to completely disable the terrain attention module and fall back to a non-attention baseline?
2. **Depth feature in Query** — Is there a flag like `terrain_attn_query_with_depth` or similar that controls whether depth features are included in the Query?
3. **Depth feature as Actor input** — Is there a flag that controls whether depth features are directly concatenated into Actor MLP input (as in the original MoRE framework)?
4. **KL prior guidance on/off** — Is there a flag like `use_terrain_kl_prior` or similar to enable/disable the KL divergence auxiliary loss?
5. **KL loss weight** — What is the variable name and current value for `lambda_kl`?
6. **Individual prior components** — Are there separate flags to enable/disable `S_support`, `S_margin`, and `behind_mask` independently in `TerrainSafetyScorer`?
8. **Pre-LayerNorm on/off** — Is there a flag to disable the LayerNorm before attention?

---

## Task 2: Verify Experiment Configurations

Based on the audit results, verify whether the following 8 experiment configurations can be achieved with existing flags. If any configuration CANNOT be achieved with current switches, clearly state what code modifications are needed.

### Comparison Experiments (vs. existing methods)

**M1 — Ours (full method):**
- Terrain attention: ON
- Depth feature in Query: YES
- Depth feature as Actor input: NO
- KL prior guidance: ON (all three components: S_support + S_margin + behind_mask)
- lambda_kl = 0.1

**M2 — Depth Encoding Baseline (MoRE Stage 1):**
- Terrain attention: OFF (completely disabled)
- Depth feature as Actor input: YES (original position, concatenated with obs + history feature into Actor MLP)
- KL prior guidance: OFF (not applicable)
- This represents the original MoRE first-stage policy without any attention mechanism

**M3 — Attention Baseline (He et al. style):**
- Terrain attention: ON
- Depth feature in Query: NO (Query uses only proprioceptive obs, 57-dim → 128-dim)
- Depth feature as Actor input: NO (depth not used at all, to match He et al.'s setting which doesn't use depth)
- KL prior guidance: OFF
- This represents the cross-attention approach from "Attention-based map encoding" without our KL prior

### Ablation Experiments

**D1 — Ours without KL (isolate KL contribution):**
- Same as M1, but KL prior guidance: OFF
- Everything else identical to M1

**B1 — Depth in Query (same as M1):**
- Same as M1

**B2 — Depth in Actor input (original position):**
- Terrain attention: ON
- Depth feature in Query: NO (Query uses only proprioceptive obs)
- Depth feature as Actor input: YES (concatenated into Actor MLP)
- KL prior guidance: ON
- This tests whether depth features are better used as Query vs as direct Actor input

**A2 — Remove S_support:**
- Same as M1, but S_support component disabled in TerrainSafetyScorer
- S_margin and behind_mask remain active

**A3 — Remove S_margin:**
- Same as M1, but S_margin component disabled in TerrainSafetyScorer
- S_support and behind_mask remain active

---

## Task 3: Verify Metrics and Logging

Check whether the following metrics are already being logged during training. For each one, tell me: (a) whether it exists, (b) the exact variable/key name used in logging, (c) which file computes it.

### Core Performance Metrics
1. **Success Rate** — per terrain type (alternating slopes, stairs, gaps) and overall average
2. **Traverse Rate** — maximum terrain difficulty level reached, per terrain type
3. **Tracking Error** — velocity command tracking error (linear velocity)
4. **Survival Time** — average episode length or time before falling

### Attention-Specific Metrics
5. **Attention Entropy** — entropy of the attention weight distribution (per-step average)
6. **Top-10 Attention Weight Sum** — sum of the 10 largest attention weights
7. **Max Attention Weight** — maximum single attention weight
8. **Per-Head Attention Entropy** — entropy computed per attention head (for diversity analysis)
9. **Prior-Attention Cosine Similarity** — cosine similarity between prior distribution and learned attention weights
10. **Attention Gradient Norm** — gradient norm flowing into the attention encoder

---

## Task 4: Verify Visualization Code

Check whether visualization code exists for the following. If it exists, tell me the file path and function name. If not, note that it needs to be created.

1. **Attention heatmap overlay** — visualizing 17×11 attention weights overlaid on the terrain heightmap (top-down view)
2. **Prior distribution visualization** — visualizing the TerrainSafetyScorer output distribution on the same terrain
3. **Side-by-side comparison** — showing attention heatmaps from two different models (e.g., M1 vs M3) on the same terrain snapshot
4. **Training curve plotting** — script to plot success rate / traverse rate vs training iterations for multiple experiments

---

## Task 5: Generate Config Files

After completing Tasks 1-4, generate a configuration snippet or script for each of the 8 experiments (M1, M2, M3, D1, B1, B2, A2, A3). Each config should be a minimal diff from the default config, showing only the flags that need to change. Format example:

```python
# === M2: Depth Encoding Baseline ===
# Changes from default config:
use_terrain_attention = False
depth_as_actor_input = True
use_terrain_kl_prior = False
```

If configs are managed via YAML, command-line args, or a dataclass, adapt the format accordingly.

---

## Important Notes

- Do NOT modify any existing code until I confirm the audit results.
- If a flag doesn't exist but is needed, propose the minimal code change to add it, including which file to modify and what the change looks like.
- Pay special attention to dimension mismatches — when toggling depth features between Query and Actor input, the input dimensions of Linear layers and Actor MLP will change. Flag whether automatic dimension adjustment is handled or needs manual specification.
- Check if there are any hardcoded dimension values (like `185` for Actor input dim) that would break when switching configurations.
