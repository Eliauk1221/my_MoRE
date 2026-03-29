#!/usr/bin/env python3
"""
Extract converged metrics from TensorBoard logs for paper tables.

For each experiment, reads the last 20% of training iterations and computes
mean / std / max / min for each metric. Outputs:
  1. Pretty-printed table to stdout
  2. CSV files ready for plot_bar_comparison.py and plot_ablation.py
  3. A combined LaTeX-friendly table

Usage:
    python extract_table_data.py
    python extract_table_data.py --tail_fraction 0.1   # use last 10% instead
"""

import argparse
import csv
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import (
    LOG_DIRS, TERRAIN_ORDER, TERRAIN_NAMES_DISPLAY, METHOD_NAMES,
    load_tb_scalar, smooth,
)

OUTPUT_DIR = './figures'

TERRAIN_METRICS = ['traverse_rate', 'survival_rate', 'success_rate',
                   'success@0.5', 'success@0.8']

GLOBAL_METRICS = ['Episode/lin_vel_tracking_error', 'Episode/rew_feet_slippage']

ATTENTION_METRICS = [
    'Attention/entropy_avg',
    'Attention/prior_attn_cosine',
    'Attention/front_region_weight',
    'Attention/top10_ratio',
    'Attention/max_weight',
]

COMPARISON_METHODS = ['ours', 'depth_baseline', 'attn_baseline']
ABLATION_METHODS_KL = ['ours', 'ablation_no_kl']
ABLATION_METHODS_PRIOR = ['ours', 'ablation_no_forward', 'ablation_no_edge']
ABLATION_METHODS_DEPTH = ['ours', 'ablation_depth_actor']
ALL_ABLATION_METHODS = ['ours', 'ablation_no_kl', 'ablation_depth_actor',
                        'ablation_no_forward', 'ablation_no_edge']


def extract_converged_value(log_dir, tag, tail_fraction=0.2):
    """
    Read a scalar tag and return stats from the last `tail_fraction` of data.
    Returns dict with mean/std/max/min, or None if tag not found.
    """
    try:
        steps, values = load_tb_scalar(log_dir, tag)
    except (KeyError, FileNotFoundError):
        return None

    n = len(values)
    if n == 0:
        return None

    tail_start = max(0, int(n * (1.0 - tail_fraction)))
    tail = values[tail_start:]

    smoothed = smooth(values, alpha=0.95)
    tail_smoothed = smoothed[tail_start:]

    return {
        'mean': float(np.mean(tail)),
        'std': float(np.std(tail)),
        'max': float(np.max(tail)),
        'min': float(np.min(tail)),
        'smoothed_mean': float(np.mean(tail_smoothed)),
        'smoothed_max': float(np.max(tail_smoothed)),
        'n_total': n,
        'n_tail': len(tail),
    }


def _load_all_from_one_dir(log_dir, tags_needed, tail_fraction=0.2):
    """Load all needed tags from one log dir in a single EventAccumulator pass."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    ea.Reload()
    available = set(ea.Tags().get('scalars', []))

    out = {}
    for tag in tags_needed:
        if tag not in available:
            out[tag] = None
            continue

        events = ea.Scalars(tag)
        values = np.array([e.value for e in events])
        n = len(values)
        if n == 0:
            out[tag] = None
            continue

        tail_start = max(0, int(n * (1.0 - tail_fraction)))
        tail = values[tail_start:]

        smoothed = smooth(values, alpha=0.95)
        tail_smoothed = smoothed[tail_start:]

        out[tag] = {
            'mean': float(np.mean(tail)),
            'std': float(np.std(tail)),
            'max': float(np.max(tail)),
            'min': float(np.min(tail)),
            'smoothed_mean': float(np.mean(tail_smoothed)),
            'smoothed_max': float(np.max(tail_smoothed)),
            'n_total': n,
            'n_tail': len(tail),
        }
    return out


def extract_all(methods, tail_fraction=0.2):
    """Extract all metrics for given methods. Returns nested dict."""
    results = {}

    terrain_tags = [f'Terrain/{t}/{m}' for t in TERRAIN_ORDER for m in TERRAIN_METRICS]
    all_tags = terrain_tags + GLOBAL_METRICS + ATTENTION_METRICS

    for method_key in methods:
        log_dir = LOG_DIRS.get(method_key)
        if log_dir is None:
            print(f"  [SKIP] {method_key}: no log directory configured")
            continue

        print(f"  Processing {method_key} ({METHOD_NAMES.get(method_key, method_key)})...",
              flush=True)
        raw = _load_all_from_one_dir(log_dir, all_tags, tail_fraction)

        for tag in terrain_tags:
            stats = raw.get(tag)
            if stats is not None:
                metric = tag.split('/')[-1]
                if metric in ('traverse_rate', 'survival_rate', 'success_rate',
                              'success@0.5', 'success@0.8'):
                    for k in ('mean', 'std', 'max', 'min', 'smoothed_mean', 'smoothed_max'):
                        stats[k] *= 100.0

        results[method_key] = raw

    return results


def print_comparison_table(results, methods, metric_name='success_rate'):
    """Print a formatted table to stdout."""
    header = f"{'Method':<30}"
    for t in TERRAIN_ORDER:
        header += f"  {TERRAIN_NAMES_DISPLAY.get(t, t):>16}"
    print(f"\n{'=' * 100}")
    print(f"  {metric_name}")
    print(f"{'=' * 100}")
    print(header)
    print('-' * 100)

    for mk in methods:
        if mk not in results:
            continue
        name = METHOD_NAMES.get(mk, mk)
        row = f"{name:<30}"
        for t in TERRAIN_ORDER:
            tag = f'Terrain/{t}/{metric_name}'
            stats = results[mk].get(tag)
            if stats is None:
                row += f"  {'N/A':>16}"
            else:
                row += f"  {stats['mean']:>7.2f}±{stats['std']:<6.2f}"
        print(row)

    print()


def print_global_metrics_table(results, methods):
    """Print global metrics (tracking error, slippage) table."""
    print(f"\n{'=' * 80}")
    print(f"  Global Metrics (last 20%)")
    print(f"{'=' * 80}")
    header = f"{'Method':<30}"
    for tag in GLOBAL_METRICS:
        short = tag.split('/')[-1]
        header += f"  {short:>20}"
    print(header)
    print('-' * 80)

    for mk in methods:
        if mk not in results:
            continue
        name = METHOD_NAMES.get(mk, mk)
        row = f"{name:<30}"
        for tag in GLOBAL_METRICS:
            stats = results[mk].get(tag)
            if stats is None:
                row += f"  {'N/A':>20}"
            else:
                row += f"  {stats['mean']:>9.4f}±{stats['std']:<8.4f}"
        print(row)
    print()


def print_attention_table(results, methods):
    """Print attention metrics table."""
    attn_methods = [m for m in methods if any(
        results.get(m, {}).get(tag) is not None for tag in ATTENTION_METRICS
    )]
    if not attn_methods:
        return

    print(f"\n{'=' * 100}")
    print(f"  Attention Metrics (last 20%)")
    print(f"{'=' * 100}")
    header = f"{'Method':<30}"
    for tag in ATTENTION_METRICS:
        short = tag.split('/')[-1]
        header += f"  {short:>16}"
    print(header)
    print('-' * 100)

    for mk in attn_methods:
        if mk not in results:
            continue
        name = METHOD_NAMES.get(mk, mk)
        row = f"{name:<30}"
        for tag in ATTENTION_METRICS:
            stats = results[mk].get(tag)
            if stats is None:
                row += f"  {'N/A':>16}"
            else:
                row += f"  {stats['mean']:>7.4f}±{stats['std']:<6.4f}"
        print(row)
    print()


def write_csv_for_metric(results, methods, metric_name, output_path):
    """Write a CSV file with one row per method, columns = terrains."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['method'] + TERRAIN_ORDER)
        for mk in methods:
            if mk not in results:
                continue
            row = [mk]
            for t in TERRAIN_ORDER:
                tag = f'Terrain/{t}/{metric_name}'
                stats = results[mk].get(tag)
                row.append(f"{stats['mean']:.2f}" if stats else '0.0')
            writer.writerow(row)
    print(f"  Saved: {output_path}")


def write_combined_csv(results, methods, output_path):
    """Write a single CSV with all metrics for all methods and terrains."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['method', 'terrain', 'metric', 'mean', 'std', 'max', 'min',
                  'smoothed_mean', 'smoothed_max']
        writer.writerow(header)

        for mk in methods:
            if mk not in results:
                continue
            for tag, stats in results[mk].items():
                if stats is None:
                    continue
                if 'Terrain/' in tag:
                    parts = tag.split('/')
                    terrain, metric = parts[1], parts[2]
                else:
                    terrain = 'global'
                    metric = tag

                writer.writerow([
                    mk, terrain, metric,
                    f"{stats['mean']:.4f}", f"{stats['std']:.4f}",
                    f"{stats['max']:.4f}", f"{stats['min']:.4f}",
                    f"{stats['smoothed_mean']:.4f}", f"{stats['smoothed_max']:.4f}",
                ])
    print(f"  Saved: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description='Extract converged metrics from TensorBoard.')
    p.add_argument('--tail_fraction', type=float, default=0.2,
                   help='Fraction of tail data to use (default: 0.2 = last 20%%)')
    p.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.output_dir
    tf = args.tail_fraction

    all_methods = list(dict.fromkeys(COMPARISON_METHODS + ALL_ABLATION_METHODS))

    print(f"\nExtracting converged metrics (last {tf*100:.0f}% of training)...")
    print(f"Log directories: {len(LOG_DIRS)}")
    results = extract_all(all_methods, tail_fraction=tf)

    print("\n" + "=" * 100)
    print("  COMPARISON TABLE (M1 vs M2 vs M3)")
    print("=" * 100)

    for metric in TERRAIN_METRICS:
        print_comparison_table(results, COMPARISON_METHODS, metric)

    print_global_metrics_table(results, COMPARISON_METHODS)

    print("\n" + "=" * 100)
    print("  ABLATION TABLE")
    print("=" * 100)

    for metric in TERRAIN_METRICS:
        print_comparison_table(results, ALL_ABLATION_METHODS, metric)

    print_attention_table(results, ALL_ABLATION_METHODS)

    print("\n--- Generating CSV files ---")

    for metric in TERRAIN_METRICS:
        write_csv_for_metric(results, COMPARISON_METHODS, metric,
                             os.path.join(out, f'comparison_{metric}.csv'))

    for metric in TERRAIN_METRICS:
        write_csv_for_metric(results, ABLATION_METHODS_KL, metric,
                             os.path.join(out, f'ablation_kl_{metric}.csv'))
        write_csv_for_metric(results, ABLATION_METHODS_PRIOR, metric,
                             os.path.join(out, f'ablation_prior_{metric}.csv'))
        write_csv_for_metric(results, ABLATION_METHODS_DEPTH, metric,
                             os.path.join(out, f'ablation_depth_{metric}.csv'))

    write_combined_csv(results, all_methods, os.path.join(out, 'all_metrics_combined.csv'))

    print("\nDone! CSV files are ready for plotting scripts.")
    print(f"  Use: python plot_bar_comparison.py --csv {out}/comparison_success_rate.csv")


if __name__ == '__main__':
    main()
