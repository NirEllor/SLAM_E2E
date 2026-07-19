"""
Master runner: Execute all 5 comparison experiments sequentially or selectively.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'project'))


def run_experiment(script_name, description):
    """Run a single comparison experiment script."""
    print(f"\n\n{'#'*80}")
    print(f"# RUNNING: {description}")
    print(f"{'#'*80}")

    script_path = Path(__file__).parent / script_name
    if not script_path.exists():
        print(f"ERROR: {script_path} not found!")
        return False

    try:
        # Import and run the script's main function
        spec = __import__('importlib.util').util.spec_from_file_location(script_name.replace('.py', ''), script_path)
        module = __import__('importlib.util').util.module_from_spec(spec)
        sys.modules[script_name.replace('.py', '')] = module
        spec.loader.exec_module(module)

        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"ERROR: No main() function in {script_name}")
            return False

        print(f"\n✓ {description} COMPLETED")
        return True

    except Exception as e:
        print(f"\n✗ {description} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*80)
    print("MASTER RUNNER: All Comparison Experiments")
    print("="*80)

    experiments = [
        # Cheap experiments (no DB rebuild)
        ('compare_loop_closure_gating.py', 'Comparison 1: Loop-Closure Gating (cheap)'),
        ('compare_keyframe_density.py', 'Comparison 2: Keyframe Density (cheap)'),
        ('compare_bundle_noise_model.py', 'Comparison 3: Bundle Noise Model (cheap)'),
        # Expensive experiments (requires DB rebuild)
        ('compare_pnp_ransac_params.py', 'Comparison 4: PnP-RANSAC Parameters (expensive)'),
        ('compare_feature_detectors.py', 'Comparison 5: Feature Detectors (expensive)'),
    ]

    print("\nExperiments to run:")
    for i, (script, desc) in enumerate(experiments, 1):
        print(f"  {i}. {desc}")

    print("\nStarting experiments...")
    start_time = time.time()

    results = {}
    for script, desc in experiments:
        results[desc] = run_experiment(script, desc)
        time.sleep(2)  # Brief pause between experiments

    # Print final summary
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for desc, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {desc}")

    print(f"\nTotal: {passed}/{total} experiments completed successfully")
    print(f"Total runtime: {(time.time() - start_time) / 60:.1f} minutes")

    if passed == total:
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY ✓")
        print("="*80)
        print("\nComparison plots saved to:")
        print(f"  {PROJECT_ROOT}/code/project/outputs/cmp_*.png")
        print("\nVariant results pickled to:")
        print(f"  {PROJECT_ROOT}/code/project/experiment_results/<topic>/<variant>.pkl")
        return True
    else:
        print("\nSome experiments failed. Check output above for details.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
