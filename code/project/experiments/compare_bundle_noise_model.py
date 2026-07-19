"""
Comparison 3: Bundle Adjustment Noise Model
Reuses baseline DB, varies prior_sigma and stereo_sigma.
Cheap experiment (no DB rebuild).
Extends the single-window prior-sigma sweep into full end-to-end pipeline impact.
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
    print("COMPARISON 3: BUNDLE ADJUSTMENT NOISE MODEL")
    print("="*80)

    # Load baseline DB (no rebuild, just reuse existing)
    db = build_variant_db(detector_type='akaze', cache_tag='baseline', num_frames=None)

    # Define variants: (label, prior_sigma, stereo_sigma, huber_k)
    # Test prior_sigma sweep from the existing 6.1 sensitivity analysis
    # Plus new stereo_sigma sweep
    variants = [
        # Prior sigma sweep (from existing 6.1 work)
        ('prior_1p0_stereo_1p0', 1.0, 1.0, 2.0),
        ('prior_0p05_stereo_1p0', 0.05, 1.0, 2.0),
        ('prior_1e-6_stereo_1p0', 1e-6, 1.0, 2.0),

        # Stereo sigma sweep (baseline prior)
        ('prior_1e-6_stereo_0p5', 1e-6, 0.5, 2.0),
        ('prior_1e-6_stereo_2p0', 1e-6, 2.0, 2.0),
    ]

    results_dict = {}

    for label, prior_sigma, stereo_sigma, huber_k in variants:
        print(f"\n--- Variant: {label} ---")
        print(f"  Prior sigma: {prior_sigma}, Stereo sigma: {stereo_sigma} px, Huber k: {huber_k}")

        # Run pipeline with these bundle noise model params
        bundle_kwargs = {
            'max_tracks_per_window': 150,
            'prior_sigma': prior_sigma,
            'stereo_sigma': stereo_sigma,
            'huber_k': huber_k
        }

        results = run_pipeline_variant(db, bundle_kwargs=bundle_kwargs, run_loop_closure=False)
        results_dict[label] = results

        # Save results
        save_variant_result(results, 'bundle_noise_model', label)

    # Generate comparison plots
    print("\n--- Generating Comparison Plots ---")
    output_dir = PROJECT_ROOT / 'code' / 'project' / 'experiments' / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_multi_variant_trajectory(results_dict,
                                     output_path=output_dir / 'cmp_bundle_noise_trajectory.png',
                                     title='Bundle Noise Model: Trajectory Comparison')
    except Exception as e:
        print(f"Trajectory plot failed: {e}")

    try:
        plot_multi_variant_absolute_error(results_dict,
                                         output_path=output_dir / 'cmp_bundle_noise_absolute_error.png',
                                         title='Bundle Noise Model: Absolute Error')
    except Exception as e:
        print(f"Absolute error plot failed: {e}")

    try:
        plot_multi_variant_scalar_comparison(results_dict, ['keyframes', 'runtime_sec'],
                                            output_path=output_dir / 'cmp_bundle_noise_scalars.png',
                                            title='Bundle Noise Model: Scalar Metrics')
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
