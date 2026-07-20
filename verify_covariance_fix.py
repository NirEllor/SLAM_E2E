"""
Verification script for the relative pose covariance computation fix.
Tests:
1. Key ordering in jointMarginalCovariance
2. Jacobian matrix population in Pose3.between()
3. Symmetry and positive-semidefiniteness of resulting covariance
"""

import numpy as np
import gtsam
from gtsam import symbol
import sys

print("=" * 80)
print("VERIFICATION: Relative Pose Covariance Computation")
print("=" * 80)

# Test 1: Key ordering in jointMarginalCovariance
print("\n[TEST 1] Key Ordering in jointMarginalCovariance")
print("-" * 80)

# Create a simple test graph with two poses
graph = gtsam.NonlinearFactorGraph()
values = gtsam.Values()

# Add two poses
key_0 = symbol('c', 0)
key_k = symbol('c', 5)

pose_0 = gtsam.Pose3()
pose_k = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 0.5, 0.2))

values.insert(key_0, pose_0)
values.insert(key_k, pose_k)

# Add prior factor for first pose
prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
graph.add(gtsam.PriorFactorPose3(key_0, pose_0, prior_noise))

# Add between factor
between_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 0.1)
between_pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1.0, 0.5, 0.2))
graph.add(gtsam.BetweenFactorPose3(key_0, key_k, between_pose, between_noise))

# Get marginals
marginals = gtsam.Marginals(graph, values)

# Test jointMarginalCovariance with specific key ordering
key_vector = gtsam.KeyVector([key_0, key_k])
joint_cov_matrix = marginals.jointMarginalCovariance(key_vector).fullMatrix()

print(f"Joint covariance matrix shape: {joint_cov_matrix.shape}")
print(f"Expected shape: (12, 12)")
assert joint_cov_matrix.shape == (12, 12), "Joint covariance should be 12x12"

# Extract blocks
Sigma_00 = joint_cov_matrix[0:6, 0:6]
Sigma_0k = joint_cov_matrix[0:6, 6:12]
Sigma_k0 = joint_cov_matrix[6:12, 0:6]
Sigma_kk = joint_cov_matrix[6:12, 6:12]

print(f"\nBlock dimensions:")
print(f"  Sigma_00: {Sigma_00.shape}")
print(f"  Sigma_0k: {Sigma_0k.shape}")
print(f"  Sigma_k0: {Sigma_k0.shape}")
print(f"  Sigma_kk: {Sigma_kk.shape}")

# Verify block symmetry in joint covariance
print(f"\nVerifying block symmetry in joint covariance:")
print(f"  Sigma_0k == Sigma_k0^T: {np.allclose(Sigma_0k, Sigma_k0.T)}")
if not np.allclose(Sigma_0k, Sigma_k0.T):
    print(f"    Max difference: {np.max(np.abs(Sigma_0k - Sigma_k0.T))}")

# Test 2: Jacobian population in Pose3.between()
print("\n[TEST 2] Jacobian Population in Pose3.between()")
print("-" * 80)

# Jacobians MUST be Fortran-contiguous (column-major order) for GTSAM
H_0 = np.zeros((6, 6), dtype=np.float64, order='F')
H_k = np.zeros((6, 6), dtype=np.float64, order='F')

print(f"Before between():")
print(f"  H_0 shape: {H_0.shape}, Fortran-contiguous: {H_0.flags['F_CONTIGUOUS']}")
print(f"  H_k shape: {H_k.shape}, Fortran-contiguous: {H_k.flags['F_CONTIGUOUS']}")
print(f"  H_0 all zeros: {np.allclose(H_0, 0)}")
print(f"  H_k all zeros: {np.allclose(H_k, 0)}")

# Call between with Jacobians
relative_pose = pose_0.between(pose_k, H_0, H_k)

print(f"\nAfter between():")
print(f"  H_0 all zeros: {np.allclose(H_0, 0)}")
print(f"  H_k all zeros: {np.allclose(H_k, 0)}")
print(f"  H_0 shape: {H_0.shape}")
print(f"  H_k shape: {H_k.shape}")

# Check if Jacobians were actually populated
if np.allclose(H_0, 0) and np.allclose(H_k, 0):
    print("\n  WARNING: Both Jacobians are still zero!")
    print("  This suggests the between() call may not be populating Jacobians correctly.")
    print("  The Python GTSAM API may require a different method.")
else:
    print("\n  ✓ Jacobians populated successfully")
    print(f"  H_0 Frobenius norm: {np.linalg.norm(H_0, 'fro'):.6f}")
    print(f"  H_k Frobenius norm: {np.linalg.norm(H_k, 'fro'):.6f}")

# Test 3: Symmetry and positive-semidefiniteness
print("\n[TEST 3] Covariance Matrix Properties")
print("-" * 80)

# Compute relative covariance using the formula
Sigma_rel = (H_0 @ Sigma_00 @ H_0.T +
             H_k @ Sigma_kk @ H_k.T +
             H_0 @ Sigma_0k @ H_k.T +
             H_k @ Sigma_k0 @ H_0.T)

print(f"Sigma_rel shape: {Sigma_rel.shape}")
print(f"Sigma_rel Frobenius norm: {np.linalg.norm(Sigma_rel, 'fro'):.6f}")

# Check symmetry
is_symmetric = np.allclose(Sigma_rel, Sigma_rel.T)
max_asymmetry = np.max(np.abs(Sigma_rel - Sigma_rel.T))
print(f"\nSymmetry check:")
print(f"  Is symmetric: {is_symmetric}")
print(f"  Max asymmetry: {max_asymmetry:.2e}")

# Symmetrize to handle numerical errors
Sigma_rel_sym = 0.5 * (Sigma_rel + Sigma_rel.T)

# Check positive-semidefiniteness
eigenvalues = np.linalg.eigvalsh(Sigma_rel_sym)
min_eigenvalue = np.min(eigenvalues)
max_eigenvalue = np.max(eigenvalues)
condition_number = max_eigenvalue / (min_eigenvalue + 1e-12)

print(f"\nPositive-semidefiniteness check:")
print(f"  Min eigenvalue: {min_eigenvalue:.6e}")
print(f"  Max eigenvalue: {max_eigenvalue:.6e}")
print(f"  Condition number: {condition_number:.2e}")
print(f"  Is PSD (min_eig >= -1e-8): {min_eigenvalue >= -1e-8}")

# Summary
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

all_passed = True

if joint_cov_matrix.shape != (12, 12):
    print("✗ FAIL: Joint covariance matrix has wrong shape")
    all_passed = False
else:
    print("✓ PASS: Joint covariance matrix shape is correct")

if not np.allclose(Sigma_0k, Sigma_k0.T):
    print("✗ FAIL: Block covariance matrices not symmetric")
    all_passed = False
else:
    print("✓ PASS: Block covariance matrices are symmetric")

if np.allclose(H_0, 0) or np.allclose(H_k, 0):
    print("✗ FAIL: Jacobians not populated by between()")
    print("  ACTION NEEDED: May need to use a different GTSAM method for Jacobians")
    all_passed = False
else:
    print("✓ PASS: Jacobians populated by between()")

if not is_symmetric:
    print(f"⚠ WARN: Result covariance not perfectly symmetric (max diff: {max_asymmetry:.2e})")
    print("        This is acceptable if numerical error is small.")

if min_eigenvalue < -1e-8:
    print(f"✗ FAIL: Result covariance not positive-semidefinite (min eig: {min_eigenvalue:.2e})")
    all_passed = False
else:
    print("✓ PASS: Result covariance is positive-semidefinite")

print("\n" + "=" * 80)
if all_passed:
    print("OVERALL: Implementation verification PASSED ✓")
else:
    print("OVERALL: Implementation has issues that need addressing ✗")
print("=" * 80)

sys.exit(0 if all_passed else 1)