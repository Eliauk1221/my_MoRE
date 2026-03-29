"""
Shared style config, color palette, and utility functions for all visualization scripts.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Global rcParams — LNCS / Springer academic figure standard
# ============================================================================
RCPARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'legend.title_fontsize': 10,

    'mathtext.fontset': 'stix',

    'lines.linewidth': 1.5,
    'lines.markersize': 5,

    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,

    'axes.spines.top': False,
    'axes.spines.right': False,

    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,

    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.edgecolor': '0.8',
}


def apply_style():
    """Apply the global rcParams. Call at the top of every script."""
    plt.rcParams.update(RCPARAMS)


# ============================================================================
# Color palette — colorblind-friendly, consistent across all figures
# ============================================================================
COLORS = {
    'ours':                '#D62728',   # M1
    'depth_baseline':      '#1F77B4',   # M2
    'attn_baseline':       '#2CA02C',   # M3
    'ablation_no_kl':      '#FF7F0E',   # D1
    'ablation_depth_actor': '#9467BD',  # B2
    'ablation_no_forward': '#8C564B',   # A2
    'ablation_no_edge':    '#E377C2',   # A3
}

METHOD_NAMES = {
    'ours':                'Ours (M1)',
    'depth_baseline':      'Depth Encoding (M2)',
    'attn_baseline':       'Cross-Attn w/o prior (M3)',
    'ablation_no_kl':      'Ours w/o KL (D1)',
    'ablation_depth_actor': 'Depth as Actor Input (B2)',
    'ablation_no_forward': r'Ours w/o $f_{\mathrm{forward}}$ (A2)',
    'ablation_no_edge':    r'Ours w/o $f_{\mathrm{edge}}$ (A3)',
}

LINE_STYLES = {
    'ours':                '-',
    'depth_baseline':      '--',
    'attn_baseline':       '-.',
    'ablation_no_kl':      ':',
    'ablation_depth_actor': '--',
    'ablation_no_forward': '-.',
    'ablation_no_edge':    ':',
}

LOG_ROOT = '/home/nubot/ssd/my_MoRE/MoRE/logs/g1_16dof_loco'

LOG_DIRS = {
    'ours':                f'{LOG_ROOT}/Mar25_12-10-05_m1',
    'depth_baseline':      f'{LOG_ROOT}/Mar25_12-11-55_m2',
    'attn_baseline':       f'{LOG_ROOT}/Mar26_09-16-35_m3',
    'ablation_no_kl':      f'{LOG_ROOT}/Mar24_21-01-36_d1',
    'ablation_depth_actor': f'{LOG_ROOT}/Mar26_11-09-51_b2',
    'ablation_no_forward': f'{LOG_ROOT}/Mar25_12-19-09_a2',
    'ablation_no_edge':    f'{LOG_ROOT}/Mar25_12-20-26_a3',
}

TERRAIN_NAMES_DISPLAY = {
    'stepping_stones': 'Stepping Stones',
    'parkour':         'Tilted Ramp',
    'gap':             'Gap',
    'stair':           'Stair',
}

TERRAIN_ORDER = ['stepping_stones', 'parkour', 'gap', 'stair']

# Grid geometry used by the height scanner
MEASURED_POINTS_X = [
    -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
     0.0,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7, 0.8,
]
MEASURED_POINTS_Y = [
    -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
]
GRID_H, GRID_W = 17, 11


# ============================================================================
# Utility functions
# ============================================================================

def smooth(data: np.ndarray, alpha: float = 0.9) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = np.empty_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * smoothed[i - 1] + (1.0 - alpha) * data[i]
    return smoothed


def ensure_dir(path: str):
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_figure(fig, stem: str, output_dir: str = "./figures"):
    """Save *fig* as both PDF and 300-dpi PNG under *output_dir*."""
    ensure_dir(output_dir)
    fig.savefig(os.path.join(output_dir, f"{stem}.pdf"))
    fig.savefig(os.path.join(output_dir, f"{stem}.png"), dpi=300)
    print(f"Saved: {output_dir}/{stem}.pdf  &  {stem}.png")


def load_tb_scalar(log_dir: str, tag: str):
    """
    Read a single scalar tag from a TensorBoard event file.

    Returns (steps, values) as numpy arrays.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        available = ea.Tags().get("scalars", [])
        raise KeyError(
            f"Tag '{tag}' not found in {log_dir}.\n"
            f"Available scalar tags ({len(available)}): {available[:20]}..."
        )

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_tb_scalars_multi(log_dir: str, tags: list):
    """Load multiple scalar tags from one TensorBoard log dir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir)
    ea.Reload()

    result = {}
    for tag in tags:
        if tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            result[tag] = {
                'steps': np.array([e.step for e in events]),
                'values': np.array([e.value for e in events]),
            }
    return result


def add_panel_label(ax, label: str, x: float = 0.05, y: float = 0.92):
    """Add a bold (a)/(b)/… label in the top-left corner of *ax*."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight='bold',
        va='top',
    )


def get_extent():
    """Return imshow extent [y_min, y_max, x_min, x_max] for the 17×11 grid."""
    return [
        MEASURED_POINTS_Y[0], MEASURED_POINTS_Y[-1],
        MEASURED_POINTS_X[0], MEASURED_POINTS_X[-1],
    ]
