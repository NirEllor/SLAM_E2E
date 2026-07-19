"""
Comparison 4: PnP-RANSAC Parameters
Builds separate TrackingDB per variant, varies pnp_threshold and pnp_iterations.
Expensive experiment (requires DB rebuild per variant).
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
    print("COMPARISON 4: PnP-RANSAC PARAMETERS")
    print("="*80)
    print("WARNING: This experiment requires rebuilding TrackingDB for each variant.")
    print("         Running on full sequence.")
    print("="*80)

    # Define variants: (label, cache_tag, pnp_threshold, pnp_iterations)
    variants = [
        ('strict_pnp', 'strict_pnp', 1.0, 100),
        ('baseline_pnp', 'baseline_pnp', 2.0, 50),
        ('loose_pnp', 'loose_pnp', 4.0, 25),
    ]

    results_dict = {}

    for label, cache_tag, pnp_threshold, pnp_iterations in variants:
        print(f"\n--- Variant: {label} ---")
        print(f"  PnP threshold: {pnp_threshold} px, iterations: {pnp_iterations}")
        print(f"  Building variant DB (this may take several minutes)...")

        # Build variant DB with custom PnP-RANSAC params
        db = build_variant_db(detector_type='akaze', pnp_threshold=pnp_threshold,
                             pnp_iterations=pnp_iterations, pnp_max_no_improvement=12,
                             num_frames=None, cache_tag=cache_tag)

        print(f"  Running pipeline...")

        # Run pipeline (disable loop closure to avoid 4+ hour bottleneck)
        results = run_pipeline_variant(db, run_loop_closure=False)
        results_dict[label] = results

        # Save results
        save_variant_result(results, 'pnp_ransac_params', label)

    # Generate comparison plots
    print("\n--- Generating Comparison Plots ---")
    output_dir = PROJECT_ROOT / 'code' / 'project' / 'experiments' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_multi_variant_trajectory(results_dict,
                                     output_path=output_dir / 'cmp_pnp_ransac_trajectory.png',
                                     title='PnP-RANSAC Parameters: Trajectory Comparison')
    except Exception as e:
        print(f"Trajectory plot failed: {e}")

    try:
        plot_multi_variant_absolute_error(results_dict,
                                         output_path=output_dir / 'cmp_pnp_ransac_absolute_error.png',
                                         title='PnP-RANSAC Parameters: Absolute Error')
    except Exception as e:
        print(f"Absolute error plot failed: {e}")

    try:
        plot_multi_variant_scalar_comparison(results_dict, ['keyframes', 'runtime_sec'],
                                            output_path=output_dir / 'cmp_pnp_ransac_scalars.png',
                                            title='PnP-RANSAC Parameters: Scalar Metrics')
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


if __name__ == '__main__':
    main()
