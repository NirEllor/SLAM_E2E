"""
Smoke test: Run a quick test on the loop-closure gating experiment with reduced frame count.
Verifies the harness works before committing to full-sequence runs.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'project'))
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from harness import build_variant_db, run_pipeline_variant, save_variant_result
from plotting import plot_multi_variant_absolute_error


def main():
    print("\n" + "="*80)
    print("SMOKE TEST: Loop-Closure Gating (reduced frame count)")
    print("="*80)

    # Use small frame count for quick test
    num_frames = 100
    print(f"\nBuilding baseline DB with {num_frames} frames...")

    try:
        db = build_variant_db(detector_type='akaze', cache_tag='baseline_smoke', num_frames=num_frames)
        print(f"✓ Baseline DB loaded: {db.frame_num()} frames")
    except Exception as e:
        print(f"✗ Failed to build DB: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run two quick variants
    variants = [
        ('strict_lc', 300, 0.8),
        ('baseline_lc', 1000, 0.6),
    ]

    results_dict = {}

    for label, mahalan_thresh, inlier_ratio in variants:
        print(f"\n--- Running variant: {label} ---")
        try:
            loop_kwargs = {
                'mahalanobis_threshold': mahalan_thresh,
                'inlier_ratio_threshold': inlier_ratio
            }
            results = run_pipeline_variant(db, loop_kwargs=loop_kwargs, run_loop_closure=True)
            results_dict[label] = results
            print(f"✓ {label} completed ({results['runtime_sec']:.1f} sec)")
        except Exception as e:
            print(f"✗ {label} failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Test plotting
    print(f"\n--- Testing plotting ---")
    try:
        output_dir = PROJECT_ROOT / 'code' / 'project' / 'outputs'
        plot_multi_variant_absolute_error(results_dict,
                                         output_path=output_dir / 'smoke_test_absolute_error.png',
                                         title='Smoke Test: Absolute Error')
        print("✓ Plotting succeeded")
    except Exception as e:
        print(f"✗ Plotting failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("SMOKE TEST PASSED ✓")
    print("="*80)
    print("\nReady to run full-sequence experiments.")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
