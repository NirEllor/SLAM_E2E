"""
Comparison 2: Keyframe Density
Reuses baseline DB, varies distance_threshold in choose_keyframes.
Cheap experiment (no DB rebuild).
"""

import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'project'))
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from harness import build_variant_db, run_pipeline_variant, save_variant_result
from plotting import (plot_multi_variant_trajectory, plot_multi_variant_absolute_error,
                     plot_multi_variant_scalar_comparison, plot_multi_variant_kitti_bar)


def main():
    print("\n" + "="*80)
    print("COMPARISON 2: KEYFRAME DENSITY")
    print("="*80)

    # Load baseline DB (no rebuild, just reuse existing)
    db = build_variant_db(detector_type='akaze', cache_tag='baseline', num_frames=None)

    # Define variants: (label, distance_threshold)
    variants = [
        ('dense_kf', 1.25),
        ('baseline_kf', 2.5),
        ('sparse_kf', 5.0),
    ]

    results_dict = {}

    for label, dist_threshold in variants:
        print(f"\n--- Variant: {label} ---")
        print(f"  Distance threshold: {dist_threshold} m")

        # Run pipeline with these keyframe density params
        keyframe_kwargs = {
            'distance_threshold': dist_threshold,
            'min_gap': 5,
            'max_gap': 20
        }

        results = run_pipeline_variant(db, keyframe_kwargs=keyframe_kwargs, run_loop_closure=True)
        results_dict[label] = results

        # Save results
        save_variant_result(results, 'keyframe_density', label)

    # Generate comparison plots
    print("\n--- Generating Comparison Plots ---")
    output_dir = PROJECT_ROOT / 'code' / 'project' / 'experiments' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_multi_variant_trajectory(results_dict,
                                     output_path=output_dir / 'cmp_kf_density_trajectory.png',
                                     title='Keyframe Density: Trajectory Comparison')
    except Exception as e:
        print(f"Trajectory plot failed: {e}")

    try:
        plot_multi_variant_absolute_error(results_dict,
                                         output_path=output_dir / 'cmp_kf_density_absolute_error.png',
                                         title='Keyframe Density: Absolute Error')
    except Exception as e:
        print(f"Absolute error plot failed: {e}")

    try:
        plot_multi_variant_scalar_comparison(results_dict, ['keyframes', 'runtime_sec'],
                                            output_path=output_dir / 'cmp_kf_density_scalars.png',
                                            title='Keyframe Density: Scalar Metrics')
    except Exception as e:
        print(f"Scalar metrics plot failed: {e}")

    # Print summary stats
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    for label, results in sorted(results_dict.items()):
        print(f"\n{label}:")
        print(f"  Keyframes: {len(results['keyframes'])}")
        print(f"  Loop closures detected: {results['loop_count']}")
        print(f"  Runtime: {results['runtime_sec']:.1f} sec")
        if results['absolute_errors_lc']:
            mean_loc_err = np.mean(results['absolute_errors_lc'].get('err_norm', []))
            print(f"  Mean location error (PG+LC): {mean_loc_err:.4f} m")


if __name__ == '__main__':
    main()
