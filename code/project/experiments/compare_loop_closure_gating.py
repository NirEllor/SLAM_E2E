"""
Comparison 1: Loop-Closure Gating Strictness
Reuses baseline DB, varies mahalanobis_threshold and inlier_ratio_threshold.
Cheapest experiment (no DB rebuild).
"""

import os
import sys
from pathlib import Path

# Add to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'project'))
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from harness import build_variant_db, run_pipeline_variant, save_variant_result
from plotting import (plot_multi_variant_trajectory, plot_multi_variant_absolute_error,
                     plot_multi_variant_scalar_comparison, plot_multi_variant_kitti_bar)


def main():
    print("\n" + "="*80)
    print("COMPARISON 1: LOOP-CLOSURE GATING STRICTNESS")
    print("="*80)

    # Use full sequence
    print(f"Running on all available frames...")

    # Build a dedicated full-sequence DB for this experiment
    db = build_variant_db(detector_type='akaze', cache_tag='lc_test_full', num_frames=None)

    # Define variants: (label, mahalanobis_threshold, inlier_ratio_threshold)
    variants = [
        ('strict_lc', 300, 0.8),
        ('baseline_lc', 1000, 0.6),
        ('loose_lc', 3000, 0.4),
    ]

    results_dict = {}

    for label, mahalan_thresh, inlier_ratio in variants:
        print(f"\n--- Variant: {label} ---")
        print(f"  Mahalanobis threshold: {mahalan_thresh}")
        print(f"  Inlier ratio threshold: {inlier_ratio}")

        # Run pipeline with these loop-closure params
        loop_kwargs = {
            'mahalanobis_threshold': mahalan_thresh,
            'inlier_ratio_threshold': inlier_ratio
        }

        results = run_pipeline_variant(db, loop_kwargs=loop_kwargs, run_loop_closure=True)
        results_dict[label] = results

        # Save results
        save_variant_result(results, 'loop_closure_gating', label)

    # Generate comparison plots
    print("\n--- Generating Comparison Plots ---")
    output_dir = PROJECT_ROOT / 'code' / 'project' / 'experiments' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_multi_variant_trajectory(results_dict,
                                     output_path=output_dir / 'cmp_lc_gating_trajectory.png',
                                     title='Loop-Closure Gating: Trajectory Comparison')
    except Exception as e:
        print(f"Trajectory plot failed: {e}")

    try:
        plot_multi_variant_absolute_error(results_dict,
                                         output_path=output_dir / 'cmp_lc_gating_absolute_error.png',
                                         title='Loop-Closure Gating: Absolute Error')
    except Exception as e:
        print(f"Absolute error plot failed: {e}")

    try:
        plot_multi_variant_scalar_comparison(results_dict, ['keyframes', 'loop_count', 'runtime_sec'],
                                            output_path=output_dir / 'cmp_lc_gating_scalars.png',
                                            title='Loop-Closure Gating: Scalar Metrics')
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
    import numpy as np
    main()
