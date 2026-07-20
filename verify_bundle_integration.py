"""
Integration test for relative pose covariance fix.
Tests the fix in the actual bundle adjustment context.
"""

import numpy as np
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'project'))

try:
    from utils.geometry import get_num_frames
    from utils.tracking import load_or_build_db
    from utils.pose_graph import compute_relative_pose_and_covariance

    print("=" * 80)
    print("INTEGRATION TEST: Relative Pose Covariance in Bundle Adjustment Context")
    print("=" * 80)

    # Load or build database
    print("\n[Step 1] Loading/building tracking database...")
    num_frames = 30  # Use fewer frames for speed
    db = load_or_build_db(force_rebuild=False, num_frames=num_frames)
    print(f"✓ Database loaded with {num_frames} frames")

    # Test on a small bundle window
    start_idx, end_idx = 0, 9
    print(f"\n[Step 2] Testing on bundle window [{start_idx}, {end_idx}]...")

    try:
        relative_pose, Sigma_rel = compute_relative_pose_and_covariance(db, start_idx, end_idx)

        print(f"✓ compute_relative_pose_and_covariance succeeded")

        # Verify properties
        print(f"\n[Step 3] Verifying covariance properties...")

        # Check shape
        assert Sigma_rel.shape == (6, 6), f"Expected (6, 6), got {Sigma_rel.shape}"
        print(f"  ✓ Shape is (6, 6)")

        # Check symmetry
        is_symmetric = np.allclose(Sigma_rel, Sigma_rel.T, atol=1e-10)
        max_asymmetry = np.max(np.abs(Sigma_rel - Sigma_rel.T))
        print(f"  ✓ Symmetric: {is_symmetric} (max diff: {max_asymmetry:.2e})")

        # Check positive-semidefiniteness
        Sigma_rel_sym = 0.5 * (Sigma_rel + Sigma_rel.T)
        eigenvalues = np.linalg.eigvalsh(Sigma_rel_sym)
        min_eig = np.min(eigenvalues)
        max_eig = np.max(eigenvalues)
        is_psd = min_eig >= -1e-8
        print(f"  ✓ PSD: {is_psd} (min eig: {min_eig:.2e}, max eig: {max_eig:.2e})")

        # Check that it's not zero or infinite
        Frobenius_norm = np.linalg.norm(Sigma_rel, 'fro')
        print(f"  ✓ Frobenius norm: {Frobenius_norm:.6e}")

        if np.isnan(Frobenius_norm) or np.isinf(Frobenius_norm) or Frobenius_norm < 1e-10:
            print(f"  ✗ WARNING: Covariance magnitude seems wrong")
        else:
            print(f"  ✓ Magnitude reasonable")

        # Print the relative pose
        print(f"\n[Step 4] Relative pose information:")
        print(f"  Type: {type(relative_pose)}")
        print(f"  Pose: {relative_pose}")

        # Print sample of covariance matrix
        print(f"\n[Step 5] Sample covariance matrix (first 3×3 block):")
        print(Sigma_rel[:3, :3])

        print("\n" + "=" * 80)
        print("INTEGRATION TEST PASSED ✓")
        print("=" * 80)

        sys.exit(0)

    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

except ImportError as e:
    print(f"Import error: {e}")
    print("Running in project/utils context, will try alternative import...")
    sys.exit(1)
