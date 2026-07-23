"""
Comparison 5: Feature Detectors
Builds separate TrackingDB per detector type (AKAZE/ORB/SIFT).
Most expensive experiment (requires 2 full DB rebuilds).
AKAZE reuses the baseline DB as the baseline variant.
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
                     plot_multi_variant_scalar_comparison)


def main():
    print("\n" + "="*80)
    print("COMPARISON 5: FEATURE DETECTORS")
    print("="*80)
    print("WARNING: This is the most expensive experiment.")
    print("         Building TrackingDB for SIFT and ORB (AKAZE reuses baseline).")
    print("         Running on full sequence.")
    print("="*80)

    # Define variants: (label, detector_type, cache_tag, orb_n_features)
    # AKAZE uses the existing baseline DB (no rebuild)
    variants = [
        ('akaze_baseline', 'akaze', 'baseline', None),
        ('orb_3000features', 'orb', 'orb_3000', 3000),
        ('sift_default', 'sift', 'sift_default', None),
    ]

    results_dict = {}

    for label, detector_type, cache_tag, orb_n_features in variants:
        print(f"\n--- Variant: {label} ---")
        print(f"  Detector: {detector_type}")

        if detector_type == 'akaze':
            print(f"  (Reusing baseline DB, no rebuild)")
        else:
            print(f"  (Building new DB — this may take several minutes)...")

        # ORB gets looser PnP threshold (4px) due to weaker correspondence quality
        pnp_threshold = 4 if detector_type == 'orb' else 2

        # Build or reuse variant DB
        db = build_variant_db(detector_type=detector_type, pnp_threshold=pnp_threshold,
                             pnp_iterations=50, pnp_max_no_improvement=12,
                             num_frames=None, cache_tag=cache_tag, orb_n_features=orb_n_features)

        print(f"  Running pipeline...")

        # Run pipeline (disable loop closure to avoid bottleneck)
        results = run_pipeline_variant(db, run_loop_closure=False)
        results_dict[label] = results

        # Save results
        save_variant_result(results, 'feature_detectors', label)

    # Generate comparison plots
    print("\n--- Generating Comparison Plots ---")
    output_dir = PROJECT_ROOT / 'code' / 'project' / 'experiments' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_multi_variant_trajectory(results_dict,
                                     output_path=output_dir / 'cmp_feature_detectors_trajectory.png',
                                     title='Feature Detectors: Trajectory Comparison')
    except Exception as e:
        print(f"Trajectory plot failed: {e}")

    try:
        plot_multi_variant_absolute_error(results_dict,
                                         output_path=output_dir / 'cmp_feature_detectors_absolute_error.png',
                                         title='Feature Detectors: Absolute Error')
    except Exception as e:
        print(f"Absolute error plot failed: {e}")

    try:
        plot_multi_variant_scalar_comparison(results_dict, ['keyframes', 'runtime_sec'],
                                            output_path=output_dir / 'cmp_feature_detectors_scalars.png',
                                            title='Feature Detectors: Scalar Metrics')
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
            mean_ang_err = np.mean(results['absolute_errors_lc'].get('err_angle', []))
            print(f"  Mean location error (PG+LC): {mean_loc_err:.4f} m")
            print(f"  Mean angle error (PG+LC): {mean_ang_err:.4f} deg")
            inlier_pct = np.mean(results['db'].inlier_percentages) if results['db'].inlier_percentages else 0
            print(f"  Mean PnP inlier percentage: {inlier_pct:.1f}%")


if __name__ == '__main__':
    main()
