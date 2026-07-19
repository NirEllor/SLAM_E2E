"""
Multi-variant comparison plots for pipeline experiments.
Reuses existing visualization.py patterns (colors, GT-in-green, overlays).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def plot_multi_variant_trajectory(variants_dict, output_path=None, title="Multi-Variant Trajectory Comparison"):
    """
    Bird's-eye-view overlay of trajectories from multiple variants + ground truth.

    Args:
        variants_dict: {variant_label: {'pg_with_lc': {'result': Values}, ...}}
        output_path: where to save the PNG
        title: plot title
    """
    from utils.bundle_adjustment import pose_translation_np
    from utils.geometry import read_ground_truth_poses
    import gtsam

    fig, ax = plt.subplots(figsize=(12, 8))

    # Ground truth (green)
    gt_poses = read_ground_truth_poses()
    gt_positions = np.array([pose_translation_np(
        gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(*(-R.T @ t).flatten()))
    ) for R, t in gt_poses[:500]])  # Limit to first 500 for clarity
    ax.plot(gt_positions[:, 0], gt_positions[:, 2], 'g-', linewidth=2.5, label='Ground Truth', alpha=0.8)

    # Variants (distinct colors)
    colors = ['blue', 'orange', 'red', 'purple', 'brown']
    for (label, data), color in zip(sorted(variants_dict.items()), colors):
        try:
            # Try to use pre-extracted trajectory positions (from harness)
            positions = data.get('trajectory_positions')

            # Fallback: extract from GTSAM result if available
            if positions is None or len(positions) == 0:
                result = data.get('pg_with_lc', {}).get('result') or data.get('pg_no_lc', {}).get('result')
                if result is not None:
                    positions = []
                    for key in result.keys():
                        sym = gtsam.Symbol(key)
                        if sym.chr() == ord('c'):
                            pose = result.atPose3(key)
                            pos = pose_translation_np(pose)
                            positions.append(pos)
                    positions = np.array(positions) if positions else None

            if positions is not None and len(positions) > 0:
                ax.plot(positions[:, 0], positions[:, 2], '-', color=color, linewidth=2, label=label, alpha=0.8)
        except Exception as e:
            print(f"Error plotting {label}: {e}")

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()

    if output_path is None:
        output_path = PROJECT_ROOT / 'code' / 'project' / 'outputs' / 'compare_trajectory.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved trajectory plot to {output_path}")
    plt.close()


def plot_multi_variant_absolute_error(variants_dict, output_path=None, title="Multi-Variant Absolute Error"):
    """
    Overlay absolute location + angle errors vs frame for multiple variants.

    Args:
        variants_dict: {variant_label: {'absolute_errors_lc': dict with 'err_norm' and 'err_angle'}}
        output_path: where to save the PNG
        title: plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    colors = ['blue', 'orange', 'red', 'purple', 'brown']
    for (label, data), color in zip(sorted(variants_dict.items()), colors):
        try:
            errors = data.get('absolute_errors_lc', {})
            frame_ids = errors.get('frame_ids', [])
            err_norm = errors.get('err_norm', [])
            err_angle = errors.get('err_angle', [])

            if frame_ids and err_norm:
                ax1.plot(frame_ids, err_norm, '-', color=color, linewidth=1.5, label=label, alpha=0.8)
            if frame_ids and err_angle:
                ax2.plot(frame_ids, err_angle, '-', color=color, linewidth=1.5, label=label, alpha=0.8)
        except Exception as e:
            print(f"Error plotting {label}: {e}")

    ax1.set_ylabel('Location Error (m)', fontsize=11)
    ax1.set_title('Absolute Location Error', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Frame Index', fontsize=11)
    ax2.set_ylabel('Angle Error (deg)', fontsize=11)
    ax2.set_title('Absolute Angle Error', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    if output_path is None:
        output_path = PROJECT_ROOT / 'code' / 'project' / 'outputs' / 'compare_absolute_error.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved absolute error plot to {output_path}")
    plt.close()


def plot_multi_variant_scalar_comparison(variants_dict, metrics, output_path=None,
                                        title="Multi-Variant Scalar Metrics"):
    """
    Bar chart comparing scalar metrics (keyframe count, loop-closure count, runtime) across variants.

    Args:
        variants_dict: {variant_label: {...}} where data contains metrics
        metrics: list of metric keys to plot (e.g., ['keyframes', 'loop_count', 'runtime_sec'])
        output_path: where to save the PNG
        title: plot title
    """
    labels = sorted(variants_dict.keys())
    data_by_metric = {m: [] for m in metrics}

    for label in labels:
        data = variants_dict[label]
        for m in metrics:
            if m == 'keyframes':
                val = len(data.get('keyframes', []))
            elif m == 'loop_count':
                val = data.get('loop_count', 0)
            elif m == 'runtime_sec':
                val = data.get('runtime_sec', 0)
            else:
                val = data.get(m, 0)
            data_by_metric[m].append(val)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    colors_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for ax, metric, values in zip(axes, metrics, [data_by_metric[m] for m in metrics]):
        bars = ax.bar(labels, values, color=colors_palette[:len(labels)], alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}' if isinstance(val, float) else f'{int(val)}',
                   ha='center', va='bottom', fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        output_path = PROJECT_ROOT / 'code' / 'project' / 'outputs' / 'compare_scalar_metrics.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved scalar metrics plot to {output_path}")
    plt.close()


def plot_multi_variant_kitti_bar(variants_dict, segment_lengths=[100, 400, 800], output_path=None,
                                 title="Multi-Variant KITTI Segment Errors"):
    """
    Bar chart of KITTI-style average errors per segment length.

    Args:
        variants_dict: {variant_label: {'kitti_errors': {seg_len: {'location': %, 'angle': deg/m}}}}
        segment_lengths: which segments to include
        output_path: where to save the PNG
        title: plot title
    """
    labels = sorted(variants_dict.keys())
    colors_palette = ['#1f77b4', '#ff7f0e', '#2ca02c']  # One per segment length
    segment_colors = {100: colors_palette[0], 400: colors_palette[1], 800: colors_palette[2]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(labels))
    width = 0.25

    for i, seg_len in enumerate(segment_lengths):
        location_errors = []
        angle_errors = []

        for label in labels:
            data = variants_dict[label]
            kitti = data.get('kitti_errors', {})
            seg_data = kitti.get(seg_len, {})
            location_errors.append(seg_data.get('location', 0))
            angle_errors.append(seg_data.get('angle', 0))

        offset = (i - 1) * width
        ax1.bar(x + offset, location_errors, width, label=f'{seg_len}m', alpha=0.8)
        ax2.bar(x + offset, angle_errors, width, label=f'{seg_len}m', alpha=0.8)

    ax1.set_ylabel('Location Error (%)', fontsize=11)
    ax1.set_title('KITTI Location Error', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.set_ylabel('Angle Error (deg/m)', fontsize=11)
    ax2.set_title('KITTI Angle Error', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path is None:
        output_path = PROJECT_ROOT / 'code' / 'project' / 'outputs' / 'compare_kitti_errors.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved KITTI error plot to {output_path}")
    plt.close()
