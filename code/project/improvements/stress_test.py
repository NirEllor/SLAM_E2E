#!/usr/bin/env python3
"""
Stress test: deliberately loosen loop-closure gates to inject false positives.

Tests whether baseline (plain BetweenFactorPose3), switchable constraints, and DCS
can gracefully degrade or suppress outlier loops that slip through tighter gates.
"""

import os
import time
import pickle
import numpy as np
import gtsam
from gtsam import symbol
from functools import partial
from pathlib import Path

# Import from the main pipeline (read-only reuse)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.pose_graph import (
    compute_all_relative_constraints, clean_pose_graph_edges,
    covariance_to_noise_model, compute_pose_graph_absolute_error,
    extract_trajectory_and_ids, read_ground_truth_poses
)
from utils.loop_closure import (
    detect_loop_closure_candidates, verify_loop_closures_consensus,
    estimate_verified_loop_relative_poses
)
from utils.tracking import load_or_build_db
from utils.bundle_adjustment import choose_keyframes


def build_baseline_graph_stress(cleaned_poses, cleaned_covs, loop_measurements, keyframes):
    """Baseline pose graph for stress test."""
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
    first_kf = keyframes[0]
    graph.add(gtsam.PriorFactorPose3(symbol('c', first_kf), gtsam.Pose3(), prior_noise))
    initial.insert(symbol('c', first_kf), gtsam.Pose3())

    sorted_edges = sorted(cleaned_poses.keys())
    current_pose = gtsam.Pose3()
    for start_kf, end_kf in sorted_edges:
        start_key = symbol('c', start_kf)
        end_key = symbol('c', end_kf)

        if not initial.exists(start_key):
            initial.insert(start_key, current_pose)
        current_pose = initial.atPose3(start_key)

        rel_pose = cleaned_poses[(start_kf, end_kf)]
        noise_model = covariance_to_noise_model(cleaned_covs[(start_kf, end_kf)])
        graph.add(gtsam.BetweenFactorPose3(start_key, end_key, rel_pose, noise_model))

        end_pose = current_pose.compose(rel_pose)
        if not initial.exists(end_key):
            initial.insert(end_key, end_pose)

    # Add loop closures (unweighted)
    for loop in loop_measurements:
        start_kf, end_kf = loop['start_kf'], loop['end_kf']
        start_key = symbol('c', start_kf)
        end_key = symbol('c', end_kf)
        rel_pose = loop['relative_pose']
        noise_model = covariance_to_noise_model(loop['relative_covariance'])
        graph.add(gtsam.BetweenFactorPose3(start_key, end_key, rel_pose, noise_model))

    return graph, initial


def switchable_constraint_error_stress(measurement_pair, this, values, jacobians):
    """Switched loop-closure error function (as in switchable_constraints.py)."""
    rel_pose, rel_cov = measurement_pair
    keys = this.keys()
    pose_i_key, pose_j_key, switch_key = keys[0], keys[1], keys[2]

    pose_i = values.atPose3(pose_i_key)
    pose_j = values.atPose3(pose_j_key)
    switch_val = values.atVector(switch_key)[0]

    H_i = np.zeros((6, 6), dtype=np.float64, order='F')
    H_j = np.zeros((6, 6), dtype=np.float64, order='F')
    between_pose = pose_i.between(pose_j, H_i, H_j)
    error_vec = gtsam.Pose3.Logmap(between_pose.between(rel_pose))
    switched_error = switch_val * error_vec

    if jacobians is not None:
        jacobians[0] = switch_val * H_i
        jacobians[1] = switch_val * H_j
        jacobians[2] = error_vec.reshape(6, 1)

    return switched_error


def build_switchable_graph_stress(cleaned_poses, cleaned_covs, loop_measurements, keyframes):
    """Switchable constraints graph for stress test."""
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
    first_kf = keyframes[0]
    graph.add(gtsam.PriorFactorPose3(symbol('c', first_kf), gtsam.Pose3(), prior_noise))
    initial.insert(symbol('c', first_kf), gtsam.Pose3())

    sorted_edges = sorted(cleaned_poses.keys())
    current_pose = gtsam.Pose3()
    for start_kf, end_kf in sorted_edges:
        start_key = symbol('c', start_kf)
        end_key = symbol('c', end_kf)

        if not initial.exists(start_key):
            initial.insert(start_key, current_pose)
        current_pose = initial.atPose3(start_key)

        rel_pose = cleaned_poses[(start_kf, end_kf)]
        noise_model = covariance_to_noise_model(cleaned_covs[(start_kf, end_kf)])
        graph.add(gtsam.BetweenFactorPose3(start_key, end_key, rel_pose, noise_model))

        end_pose = current_pose.compose(rel_pose)
        if not initial.exists(end_key):
            initial.insert(end_key, end_pose)

    # Add loop closures with switches
    switch_sigma = 0.3
    switch_prior_noise = gtsam.noiseModel.Isotropic.Sigma(1, switch_sigma)

    for loop_idx, loop in enumerate(loop_measurements):
        start_kf, end_kf = loop['start_kf'], loop['end_kf']
        start_key = symbol('c', start_kf)
        end_key = symbol('c', end_kf)
        switch_key = symbol('s', loop_idx)

        initial.insert(switch_key, np.array([1.0]))
        graph.add(gtsam.PriorFactorVector(switch_key, np.array([1.0]), switch_prior_noise))

        rel_pose = loop['relative_pose']
        rel_cov = loop['relative_covariance']
        measurement = (rel_pose, rel_cov)
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)

        switched_factor = gtsam.CustomFactor(
            measurement_noise,
            [start_key, end_key, switch_key],
            partial(switchable_constraint_error_stress, measurement)
        )
        graph.add(switched_factor)

    return graph, initial


def build_dcs_graph_stress(cleaned_poses, cleaned_covs, loop_measurements, keyframes):
    """DCS graph for stress test."""
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
    first_kf = keyframes[0]
    graph.add(gtsam.PriorFactorPose3(symbol('c', first_kf), gtsam.Pose3(), prior_noise))
    initial.insert(symbol('c', first_kf), gtsam.Pose3())

    sorted_edges = sorted(cleaned_poses.keys())
    current_pose = gtsam.Pose3()
    for start_kf, end_kf in sorted_edges:
        start_key = symbol('c', start_kf)
        end_key = symbol('c', end_kf)

        if not initial.exists(start_key):
            initial.insert(start_key, current_pose)
        current_pose = initial.atPose3(start_key)

        rel_pose = cleaned_poses[(start_kf, end_kf)]
        noise_model = covariance_to_noise_model(cleaned_covs[(start_kf, end_kf)])
        graph.add(gtsam.BetweenFactorPose3(start_key, end_key, rel_pose, noise_model))

        end_pose = current_pose.compose(rel_pose)
        if not initial.exists(end_key):
            initial.insert(end_key, end_pose)

    # Add loop closures with DCS
    dcs_threshold = 6.0
    dcs_kernel = gtsam.noiseModel.mEstimator.DCS.Create(dcs_threshold)

    for loop in loop_measurements:
        start_kf, end_kf = loop['start_kf'], loop['end_kf']
        start_key = symbol('c', start_kf)
        end_key = symbol('c', end_kf)
        rel_pose = loop['relative_pose']

        gaussian_noise = covariance_to_noise_model(loop['relative_covariance'])
        robust_noise = gtsam.noiseModel.Robust.Create(dcs_kernel, gaussian_noise)

        graph.add(gtsam.BetweenFactorPose3(start_key, end_key, rel_pose, robust_noise))

    return graph, initial


def optimize_pose_graph_lm(graph, initial):
    """Optimize pose graph using Levenberg-Marquardt optimizer."""
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    marginals = gtsam.Marginals(graph, result)
    return result, marginals


def run_stress_test_pipeline():
    """
    Stress test: deliberately loosen loop-closure gates, then compare baseline vs robust techniques.
    """
    print("\n" + "="*80)
    print("  STRESS TEST: DELIBERATELY-LOOSENED LOOP CLOSURES")
    print("="*80)

    # Load tracking database
    db = load_or_build_db(detector_type='akaze', num_frames=None)
    print(f"Loaded tracking database: {db.frame_num()} frames, {db.track_num()} tracks")

    # Select keyframes
    keyframes = choose_keyframes(db, distance_threshold=2.5, min_gap=20, max_gap=20)
    print(f"Selected {len(keyframes)} keyframes")

    # Compute relative pose constraints
    relative_poses, relative_covs = compute_all_relative_constraints(db, keyframes)
    cleaned_poses, cleaned_covs = clean_pose_graph_edges(relative_poses, relative_covs)
    print(f"Sequential edges after cleaning: {len(cleaned_poses)}")

    # Detect loop closures with LOOSENED gates (to admit potentially false positives)
    print("\n--- Detecting loop closures with LOOSENED gates ---")
    loose_mahalanobis = 5000.0  # 5× the default 1000.0 threshold
    loose_inlier_ratio = 0.3    # halved from default 0.6

    candidates, total_cand = detect_loop_closure_candidates(
        relative_poses, relative_covs, keyframes,
        mahalanobis_threshold=loose_mahalanobis
    )
    print(f"Candidates with loose Mahalanobis: {total_cand}")

    verified, total_ver = verify_loop_closures_consensus(db, candidates, inlier_ratio_threshold=loose_inlier_ratio)
    print(f"Verified with loose inlier ratio: {total_ver}")

    loop_measurements = estimate_verified_loop_relative_poses(db, verified)
    print(f"Loop measurements estimated: {len(loop_measurements)}")

    if len(loop_measurements) == 0:
        print("WARNING: No loop closures detected even with loosened gates. Stress test inconclusive.")
        return None

    # Build and optimize all three variants
    print("\n--- Baseline (no robust weighting) ---")
    t0 = time.time()
    baseline_graph, baseline_initial = build_baseline_graph_stress(
        cleaned_poses, cleaned_covs, loop_measurements, keyframes
    )
    baseline_result, baseline_marginals = optimize_pose_graph_lm(baseline_graph, baseline_initial)
    baseline_time = time.time() - t0
    print(f"Baseline optimization: {baseline_time:.2f} sec")

    baseline_ids, baseline_positions = extract_trajectory_and_ids(baseline_result)
    baseline_errors = compute_pose_graph_absolute_error(baseline_result, baseline_ids)

    print("\n--- Switchable Constraints ---")
    t0 = time.time()
    sc_graph, sc_initial = build_switchable_graph_stress(
        cleaned_poses, cleaned_covs, loop_measurements, keyframes
    )
    sc_result, sc_marginals = optimize_pose_graph_lm(sc_graph, sc_initial)
    sc_time = time.time() - t0
    print(f"Switchable constraints optimization: {sc_time:.2f} sec")

    sc_ids, sc_positions = extract_trajectory_and_ids(sc_result)
    sc_errors = compute_pose_graph_absolute_error(sc_result, sc_ids)

    # Extract switch values (diagnostic)
    switch_values = {}
    for loop_idx in range(len(loop_measurements)):
        switch_key = symbol('s', loop_idx)
        if sc_result.exists(switch_key):
            switch_values[loop_idx] = sc_result.atVector(switch_key)[0]

    print("\n--- Dynamic Covariance Scaling ---")
    t0 = time.time()
    dcs_graph, dcs_initial = build_dcs_graph_stress(
        cleaned_poses, cleaned_covs, loop_measurements, keyframes
    )
    dcs_result, dcs_marginals = optimize_pose_graph_lm(dcs_graph, dcs_initial)
    dcs_time = time.time() - t0
    print(f"DCS optimization: {dcs_time:.2f} sec")

    dcs_ids, dcs_positions = extract_trajectory_and_ids(dcs_result)
    dcs_errors = compute_pose_graph_absolute_error(dcs_result, dcs_ids)

    # Compute ground truth poses for reference
    gt_poses = read_ground_truth_poses()

    # Aggregate metrics
    results = {
        'baseline': {
            'trajectory_positions': baseline_positions,
            'keyframe_ids': baseline_ids,
            'absolute_errors': baseline_errors,
            'runtime_sec': baseline_time,
            'loop_count': len(loop_measurements),
            'gate_setting': 'LOOSE (maha=5000, inlier_ratio=0.3)',
        },
        'switchable_constraints': {
            'trajectory_positions': sc_positions,
            'keyframe_ids': sc_ids,
            'absolute_errors': sc_errors,
            'runtime_sec': sc_time,
            'loop_count': len(loop_measurements),
            'switch_values': switch_values,
            'gate_setting': 'LOOSE (maha=5000, inlier_ratio=0.3)',
        },
        'dcs': {
            'trajectory_positions': dcs_positions,
            'keyframe_ids': dcs_ids,
            'absolute_errors': dcs_errors,
            'runtime_sec': dcs_time,
            'loop_count': len(loop_measurements),
            'dcs_threshold': 6.0,
            'gate_setting': 'LOOSE (maha=5000, inlier_ratio=0.3)',
        }
    }

    # Print summary
    print("\n" + "-"*80)
    print("SUMMARY (STRESS TEST WITH LOOSENED GATES)")
    print("-"*80)
    for variant, data in results.items():
        errs = data['absolute_errors']
        if errs and errs.get('err_norm'):
            mean_loc_err = np.mean(errs['err_norm'])
            mean_ang_err = np.mean(errs['err_angle'])
        else:
            mean_loc_err, mean_ang_err = 0, 0
        print(f"{variant:30s} | Loc error: {mean_loc_err:8.4f} m | Ang error: {mean_ang_err:8.4f}°")
        print(f"{'':30s} | Runtime: {data['runtime_sec']:8.2f} sec | Loops: {data['loop_count']:3d}")

    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'stress_test.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_file}")

    # Generate trajectory comparison plot
    from experiments.plotting import plot_multi_variant_trajectory
    plot_output_dir = Path(__file__).parent / 'outputs'
    plot_output_dir.mkdir(exist_ok=True)
    plot_multi_variant_trajectory(
        results,
        output_path=plot_output_dir / 'stress_test_trajectory.png',
        title='Stress Test (Loosened Gates) — Baseline vs. SC vs. DCS — Trajectory'
    )

    return results


if __name__ == '__main__':
    results = run_stress_test_pipeline()
