# Publication-Quality Visualization Scripts — Cursor Prompt

## Background

I'm preparing figures for a conference paper (ICANN, LNCS format). I need Python scripts that generate publication-quality plots directly usable in the paper (exported as PDF or high-resolution PNG). All scripts should be standalone — runnable independently from the training codebase, reading data from saved log files or checkpoints.

---

## Global Style Requirements (APPLY TO ALL SCRIPTS)

Every script must follow these academic figure standards:

```python
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# === GLOBAL STYLE CONFIG (put this at the top of every script) ===
plt.rcParams.update({
    # Font settings — use serif fonts for LNCS/Springer format
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,               # Base font size
    'axes.titlesize': 12,          # Subplot title
    'axes.labelsize': 11,          # Axis label (x, y)
    'xtick.labelsize': 10,         # Tick labels
    'ytick.labelsize': 10,
    'legend.fontsize': 9,          # Legend text
    'legend.title_fontsize': 10,

    # Use LaTeX-style math rendering (if LaTeX is available)
    'mathtext.fontset': 'stix',    # Matches Times New Roman math

    # Line and marker settings
    'lines.linewidth': 1.5,
    'lines.markersize': 5,

    # Axis settings
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,

    # Remove top and right spines for cleaner look
    'axes.spines.top': False,
    'axes.spines.right': False,

    # Figure DPI for export
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,

    # Legend frame
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.edgecolor': '0.8',
})

# === COLOR PALETTE (consistent across all figures) ===
# Use a colorblind-friendly palette
COLORS = {
    'ours':              '#D62728',   # Red — our method (always most prominent)
    'depth_baseline':    '#1F77B4',   # Blue — M2 depth encoding baseline
    'attn_baseline':     '#2CA02C',   # Green — M3 attention baseline (He et al.)
    'ablation_no_kl':    '#FF7F0E',   # Orange — D1 no KL
    'ablation_depth_actor': '#9467BD', # Purple — B2 depth in actor
    'ablation_no_support': '#8C564B', # Brown — A2
    'ablation_no_margin':  '#E377C2', # Pink — A3
}

# Method display names (for legends)
METHOD_NAMES = {
    'ours':              'Ours',
    'depth_baseline':    'Depth Encoding',
    'attn_baseline':     'Cross-Attn (w/o prior)',
    'ablation_no_kl':    'Ours w/o KL',
    'ablation_depth_actor': 'Depth as Actor Input',
    'ablation_no_support': 'Ours w/o $S_{\\mathrm{support}}$',
    'ablation_no_margin':  'Ours w/o $S_{\\mathrm{margin}}$',
}

# Line styles for distinguishing methods in grayscale printing
LINE_STYLES = {
    'ours':              '-',
    'depth_baseline':    '--',
    'attn_baseline':     '-.',
    'ablation_no_kl':    ':',
    'ablation_depth_actor': '--',
    'ablation_no_support': '-.',
    'ablation_no_margin':  ':',
}
```

### Additional Style Rules

1. **Figure sizes**: Use LNCS column width. Single-column figure: `figsize=(5.5, 3.5)`. Double-column/wide figure: `figsize=(5.5, 2.8)` per subplot row. Adjust height proportionally.
2. **Export format**: Always save as both PDF (for LaTeX) and PNG (300 DPI, for preview). Use `plt.savefig('figure_name.pdf')` and `plt.savefig('figure_name.png', dpi=300)`.
3. **No titles on final figures**: Subplot titles are OK for multi-panel figures, but no `suptitle` — the caption in the paper replaces it.
4. **Legend placement**: Prefer `legend(loc='lower right')` or outside the plot if too many entries. Never let the legend overlap data curves.
5. **Axis labels**: Always include units where applicable. Use LaTeX math notation: `$v_x$ tracking error (m/s)`.
6. **Smoothing for training curves**: Apply exponential moving average (EMA) with `alpha=0.9` or similar. Plot the raw data as a faint band (alpha=0.15) behind the smoothed line.
7. **Error bands**: If multiple seeds are available, plot mean ± std as a shaded band using `ax.fill_between()` with `alpha=0.2`.
8. **Consistent x-axis**: All training curves must use the same x-axis unit (training iterations or environment steps). State which one in the axis label.
9. **Terrain type names in English**: Use "Alternating Slopes", "Stairs", "Gaps" consistently across all figures and tables.

---

## Script 1: Training Curves Comparison

**Filename**: `plot_training_curves.py`

**Purpose**: Plot success rate (and optionally traverse rate) vs. training iterations for multiple methods on the same axes. This figure demonstrates: (a) our method achieves higher final performance, and (b) KL prior guidance accelerates convergence.

**Input data format**: The script should support reading from TensorBoard log directories OR from CSV files. Check what format our training logs use (likely TensorBoard via `SummaryWriter`). If TensorBoard, use `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` to extract scalar data. Provide a command-line interface or config dict at the top of the script to specify log directories for each method.

**Layout option A — single metric, single plot**:
- One figure, `figsize=(5.5, 3.5)`
- X-axis: Training iterations (or steps)
- Y-axis: Success Rate (%)
- One curve per method (M1, M2, M3 at minimum)
- Smoothed lines + faint raw data bands
- If multiple seeds: mean ± std shaded bands

**Layout option B — multiple metrics or per-terrain breakdown**:
- `fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.5))` for 3 terrain types side by side
- Each subplot: one terrain type (Alternating Slopes / Stairs / Gaps)
- Shared legend at bottom of figure using `fig.legend()`
- Shared y-axis label on leftmost subplot only

**Implementation details**:
```python
def smooth(data, alpha=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = data[0]
    for point in data:
        smoothed_val = alpha * last + (1 - alpha) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# For each method, plot:
# 1. Faint raw line: ax.plot(steps, raw_data, color=color, alpha=0.15, linewidth=0.8)
# 2. Smoothed line: ax.plot(steps, smoothed_data, color=color, linestyle=ls, label=name)
# 3. If multi-seed: ax.fill_between(steps, mean-std, mean+std, color=color, alpha=0.15)
```

**CLI usage example**:
```bash
python plot_training_curves.py \
    --logs ours=/path/to/M1/logs \
           depth_baseline=/path/to/M2/logs \
           attn_baseline=/path/to/M3/logs \
    --metric "success_rate" \
    --output training_curves.pdf
```

---

## Script 2: Attention Heatmap Visualization

**Filename**: `plot_attention_heatmap.py`

**Purpose**: Visualize 17×11 attention weights overlaid on terrain, comparing different methods side-by-side. This is the most visually impactful figure in the paper.

**Input data**: The script needs two things per snapshot:
1. Attention weights: shape `(187,)` or `(17, 11)` — from the model's `last_attention_weights`
2. Height map: shape `(17, 11)` — the terrain height values at the sampling grid

Check how the training code stores/logs attention weights. If not currently saved, the script should include a data collection function that loads a checkpoint, runs the model on a few environment steps, and extracts attention weights + height maps.

**Layout — side-by-side comparison (primary figure)**:
```
[Terrain Heightmap] [Method A Attention] [Method B Attention] [Prior Distribution]
```
- `fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.0))` for 4 panels
- Or `fig, axes = plt.subplots(2, 3, figsize=(5.5, 4.0))` for 2 terrain types × 3 views

**For each panel**:
1. Use `ax.imshow()` with the height map as the background (grayscale colormap, e.g., `cmap='terrain'` or `cmap='gray'`)
2. Overlay attention weights as a semi-transparent heatmap (`cmap='Reds'` or `cmap='hot'`, `alpha=0.65`)
3. Optionally mark the robot's position with a small marker (e.g., triangle or dot)
4. Add a small colorbar on the right side of each row (not per-panel, to save space)
5. X-axis: lateral direction (11 points, ±0.5m), Y-axis: forward direction (17 points, 0–1.6m)
6. Use physical coordinates (meters) as tick labels, not grid indices

**Colormap choice**:
- Height map background: `cmap='Greys_r'` (light=high, dark=low) — simple and clean
- Attention overlay: `cmap='Reds'` with `alpha=0.6` — red hotspots on gray background are visually clear
- Prior distribution: `cmap='Blues'` with `alpha=0.6` — visually distinct from attention (red vs blue)
- Alternative: use the same `cmap='YlOrRd'` for both attention and prior, and distinguish by column labels

**Panel labels**: Add `(a)`, `(b)`, `(c)`, `(d)` labels in the top-left corner of each subplot using `ax.text(0.05, 0.92, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold')`.

**Terrain-specific snapshots to generate**:
- Stairs: show attention focusing on upcoming step edges
- Gaps: show attention focusing on safe landing areas across the gap
- Alternating slopes: show attention tracking the slope transitions

---

## Script 3: Prior vs. Attention Distribution Comparison

**Filename**: `plot_prior_vs_attention.py`

**Purpose**: Show the TerrainSafetyScorer's prior distribution alongside the learned attention distribution on the same terrain snapshot. Demonstrates alignment between physics prior and learned attention.

**Layout**:
```
Row 1 (Stairs):  [Height Map] [Prior Distribution] [Learned Attention (Ours)] [Attention (w/o KL)]
Row 2 (Gaps):    [Height Map] [Prior Distribution] [Learned Attention (Ours)] [Attention (w/o KL)]
```
- `fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.5))`

**Data collection**: This script needs to:
1. Load a terrain snapshot (height map + terrain xyz)
2. Run `TerrainSafetyScorer` on it to get the prior distribution
3. Load checkpoint for M1 (ours) and D1 (w/o KL), run forward pass to get attention weights
4. Plot all four side by side

**Key visual element**: Under each attention panel, display the cosine similarity value between prior and attention as text: `cos_sim = 0.82`.

---

## Script 4: Quantitative Results Bar Chart (Optional but Recommended)

**Filename**: `plot_bar_comparison.py`

**Purpose**: Grouped bar chart showing success rate across terrain types for all comparison methods. Alternative to a pure table — sometimes reviewers prefer to see this visually.

**Layout**: 
- `figsize=(5.5, 3.0)`
- X-axis: Terrain types (3 groups: Alternating Slopes, Stairs, Gaps)
- Y-axis: Success Rate (%)
- Grouped bars: one bar per method per terrain, with method colors from the global palette
- Add value labels on top of each bar
- Error bars if multiple seeds available

```python
# Bar width and positioning
n_methods = 3  # M1, M2, M3
bar_width = 0.22
x = np.arange(len(terrain_types))

for i, (method_key, values) in enumerate(results.items()):
    offset = (i - n_methods/2 + 0.5) * bar_width
    bars = ax.bar(x + offset, values, bar_width,
                  label=METHOD_NAMES[method_key],
                  color=COLORS[method_key],
                  edgecolor='white', linewidth=0.5)
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
```

---

## Script 5: Ablation Results Comparison

**Filename**: `plot_ablation.py`

**Purpose**: Visualize ablation experiment results. Two sub-figures:

**Sub-figure A — Prior component ablation** (A2, A3 vs M1):
- Grouped bar chart, same style as Script 4
- 3 terrain types × 3 configs (Ours full, w/o S_support, w/o S_margin)

**Sub-figure B — Depth feature position** (B1 vs B2):
- Grouped bar chart
- 3 terrain types × 2 configs (Depth in Query vs Depth in Actor)

Layout: `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.8), gridspec_kw={'width_ratios': [3, 2]})`

---

## General Implementation Requirements

1. **Data loading**: Each script should have a clearly marked `# === DATA INPUT ===` section at the top where file paths are specified. Support both:
   - Direct specification: `LOG_DIR = "/path/to/experiment/logs"`
   - Command-line args: `argparse` for batch usage

2. **Modularity**: Put the global style config and color definitions in a shared `plot_utils.py` file. Import from there. This ensures visual consistency across all figures.

3. **Output**: Each script saves figures to a configurable output directory, default `./figures/`. Create the directory if it doesn't exist.

4. **Robustness**: Handle missing data gracefully (e.g., if only 2 of 3 methods have finished training, still plot what's available).

5. **Check existing code first**: Before writing these scripts from scratch, check if there are any existing plotting or visualization utilities in the codebase (e.g., in a `scripts/`, `tools/`, or `utils/` directory). If so, build on top of them rather than duplicating.

6. **Check data availability**: Verify what data is actually saved during training:
   - Are attention weights saved to TensorBoard or to separate files?
   - Are per-terrain-type success rates logged separately?
   - What is the exact TensorBoard tag name for each metric?
   - Are terrain height maps saved during evaluation?
   
   Report what data is available and what additional data collection code is needed.
