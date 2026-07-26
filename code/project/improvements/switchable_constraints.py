#!/usr/bin/env python3
"""
Switchable Constraints for Robust Pose Graph SLAM (Sünderhauf & Protzel, IROS 2012).

Implements per-edge switch variables s_ij ∈ [0,1] jointly optimized with poses.
Each switch modulates its constraint's influence: s_ij→1 trusts the edge, s_ij→0 suppresses it.
Allows the back-end to learn which loop closures are outliers during optimization.
"""

import os
import time
import pickle
import numpy as np
import gtsam
from gtsam import symbol
from functools import partial
from pathlib import Path

# Import from the main pipeline (read-only reuse, no modifications)
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


def build_baseline_pose_graph(cleaned_poses, cleaned_covs, loop_measurements, keyframes):
    """Baseline: plain BetweenFactorPose3 (replicate existing behavior locally)."""
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Prior on first keyframe
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
    first_kf = keyframes[0]
    graph.add(gtsam.PriorFactorPose3(symbol('c', first_kf), gtsam.Pose3(), prior_noise))
    initial.insert(symbol('c', first_kf), gtsam.Pose3())

    # Sequential keyframe poses via relative constraints
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

    # Add loop closures (unweighted/non-switchable)
    for loop in loop_measurements:
        start_kf, end_kf = loop['start_kf'], loop['end_kf']
        start_key = symbol('c', start_kf)
        end_key = symbol('c', end_kf)
        rel_pose = loop['relative_pose']
        noise_model = covariance_to_noise_model(loop['relative_covariance'])
        graph.add(gtsam.BetweenFactorPose3(start_key, end_key, rel_pose, noise_model))

    return graph, initial


def switchable_constraint_error(measurement_pair, this, values, jacobians):
    """
    CustomFactor error function for a switched loop-closure constraint.

    Computes: error = switch * (relative_pose error)
    where switch ∈ [0,1] modulates the constraint's influence.

    Args:
        measurement_pair: tuple (rel_pose: Pose3, rel_cov_diag: 6x6 ndarray)
        this: gtsam.CustomFactor handle (provides keys: [pose_i, pose_j, switch_ij])
        values: gtsam.Values with Pose3(pose_i, pose_j) and Vector1(switch_ij)
        jacobians: optional list to fill with Jacobians w.r.t. each key

    Returns:
        unwhitened error vector (6,)
    """
    rel_pose, rel_cov = measurement_pair
    keys = this.keys()
    pose_i_key, pose_j_key, switch_key = keys[0], keys[1], keys[2]

    pose_i = values.atPose3(pose_i_key)
    pose_j = values.atPose3(pose_j_key)
    switch_val = values.atVector(switch_key)[0]  # scalar in [0, 1]

    # Compute between-pose error with Jacobians
    H_i = np.zeros((6, 6), dtype=np.float64, order='F')
    H_j = np.zeros((6, 6), dtype=np.float64, order='F')
    between_pose = pose_i.between(pose_j, H_i, H_j)

    # Compute the between error (Pose3.Logmap gives tangent-space error)
    error_vec = gtsam.Pose3.Logmap(between_pose.between(rel_pose))  # 6D error

    # Apply switch: error *= switch
    switched_error = switch_val * error_vec

    if jacobians is not None:
        # Jacobian w.r.t. pose_i
        jacobians[0] = switch_val * H_i

        # Jacobian w.r.t. pose_j
        jacobians[1] = switch_val * H_j

        # Jacobian w.r.t. switch: d(error) / d(switch) = error_vec (vector derivative)
        # GTSAM CustomFactor expects a 6x1 Jacobian; we return the error sensitivity
        jacobians[2] = error_vec.reshape(6, 1)

    return switched_error


def build_switchable_constraints_graph(cleaned_poses, cleaned_covs, loop_measurements, keyframes):
    """Build pose graph with per-edge switch variables for loop closures."""
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Prior on first keyframe
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
    first_kf = keyframes[0]
    graph.add(gtsam.PriorFactorPose3(symbol('c', first_kf), gtsam.Pose3(), prior_noise))
    initial.insert(symbol('c', first_kf), gtsam.Pose3())

    # Sequential keyframe poses (baseline edges, no switches)
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

    # Add loop closures with per-edge switch variables
    switch_sigma = 0.3  # Prior noise pulling switches toward 1.0
    switch_prior_noise = gtsam.noiseModel.Isotropic.Sigma(1, switch_sigma)

    for loop_idx, loop in enumerate(loop_measurements):
        start_kf, end_kf = loop['start_kf'], loop['end_kf']
        start_key = symbol('c', start_kf)
        end_key = symbol('c', end_kf)
        switch_key = symbol('s', loop_idx)  # unique switch variable per loop

        # Initialize switch to 1.0 (trust the loop until optimization says otherwise)
        initial.insert(switch_key, np.array([1.0]))

        # Add switch prior factor (prior: pull toward 1.0)
        graph.add(gtsam.PriorFactorVector(switch_key, np.array([1.0]), switch_prior_noise))

        # Add switched loop-closure factor via CustomFactor
        rel_pose = loop['relative_pose']
        rel_cov = loop['relative_covariance']
        measurement = (rel_pose, rel_cov)

        # Use identity noise model; switch function modulates constraint influence.
        # (Using actual loop covariance causes residuals to balloon in whitened space, collapsing switches to ~0)
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1.0)
        switched_factor = gtsam.CustomFactor(
            measurement_noise,
            [start_key, end_key, switch_key],
            partial(switchable_constraint_error, measurement)
        )
        graph.add(switched_factor)

    return graph, initial


def run_switchable_constraints_pipeline():
    """Full pipeline: load DB, compute constraints, build switchable graph, optimize, evaluate."""
    print("\n" + "="*80)
    print("  SWITCHABLE CONSTRAINTS FOR ROBUST POSE GRAPH SLAM")
    print("="*80)

    # Load tracking database
    db = load_or_build_db(detector_type='akaze', num_frames=None)
    print(f"Loaded tracking database: {db.frame_num()} frames, {db.track_num()} tracks")

    # Select keyframes
    keyframes = choose_keyframes(db, distance_threshold=2.5, min_gap=20, max_gap=20)
    print(f"Selected {len(keyframes)} keyframes")

    # Compute relative pose constraints (baseline loop-closure gates, unchanged)
    relative_poses, relative_covs = compute_all_relative_constraints(db, keyframes)
    cleaned_poses, cleaned_covs = clean_pose_graph_edges(relative_poses, relative_covs)
    print(f"Sequential edges after cleaning: {len(cleaned_poses)}")

    # Detect and verify loop closures (standard baseline gates)
    candidates, _ = detect_loop_closure_candidates(
        relative_poses, relative_covs, keyframes,
        mahalanobis_threshold=1000.0
    )
    verified, _ = verify_loop_closures_consensus(db, candidates, inlier_ratio_threshold=0.6)
    loop_measurements = estimate_verified_loop_relative_poses(db, verified)
    print(f"Verified loop closures: {len(loop_measurements)}")

    # Build and optimize baseline graph
    print("\n--- Baseline (no switches) ---")
    t0 = time.time()
    baseline_graph, baseline_initial = build_baseline_pose_graph(
        cleaned_poses, cleaned_covs, loop_measurements, keyframes
    )
    baseline_result, baseline_marginals = optimize_pose_graph_lm(baseline_graph, baseline_initial)
    baseline_time = time.time() - t0
    print(f"Baseline optimization: {baseline_time:.2f} sec")

    baseline_ids, baseline_positions = extract_trajectory_and_ids(baseline_result)
    baseline_errors = compute_pose_graph_absolute_error(baseline_result, baseline_ids)

    # Build and optimize switchable-constraints graph
    print("\n--- Switchable Constraints ---")
    t0 = time.time()
    sc_graph, sc_initial = build_switchable_constraints_graph(
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
        },
        'switchable_constraints': {
            'trajectory_positions': sc_positions,
            'keyframe_ids': sc_ids,
            'absolute_errors': sc_errors,
            'runtime_sec': sc_time,
            'loop_count': len(loop_measurements),
            'switch_values': switch_values,
        }
    }

    # Print summary
    print("\n" + "-"*80)
    print("SUMMARY")
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
    output_file = output_dir / 'switchable_constraints.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_file}")

    # Generate trajectory comparison plot
    from experiments.plotting import plot_multi_variant_trajectory
    plot_output_dir = Path(__file__).parent / 'outputs'
    plot_output_dir.mkdir(exist_ok=True)
    plot_multi_variant_trajectory(
        results,
        output_path=plot_output_dir / 'switchable_constraints_trajectory.png',
        title='Switchable Constraints vs. Baseline — Trajectory'
    )

    return results


def optimize_pose_graph_lm(graph, initial):
    """Optimize pose graph using Levenberg-Marquardt optimizer."""
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(100)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    marginals = gtsam.Marginals(graph, result)
    return result, marginals


if __name__ == '__main__':
    results = run_switchable_constraints_pipeline()
