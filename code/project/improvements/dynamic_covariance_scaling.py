#!/usr/bin/env python3
"""
Dynamic Covariance Scaling for Robust Map Optimization (Agarwal et al., ICRA 2013).

Implements per-iteration reweighting of constraint information matrices based on
their current chi-squared residuals. Outlier constraints are down-weighted continuously
during optimization via a native GTSAM robust kernel (mEstimator.DCS).
"""

import os
import time
import pickle
import numpy as np
import gtsam
from gtsam import symbol
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


def build_baseline_pose_graph_dcs(cleaned_poses, cleaned_covs, loop_measurements, keyframes):
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

    # Add loop closures (unweighted/non-robust)
    for loop in loop_measurements:
        start_kf, end_kf = loop['start_kf'], loop['end_kf']
        start_key = symbol('c', start_kf)
        end_key = symbol('c', end_kf)
        rel_pose = loop['relative_pose']
        noise_model = covariance_to_noise_model(loop['relative_covariance'])
        graph.add(gtsam.BetweenFactorPose3(start_key, end_key, rel_pose, noise_model))

    return graph, initial


DCS_THRESHOLD = 6.0  # chi-squared threshold for DCS robust kernel (≈ dof of Pose3 residual)


def build_dcs_pose_graph(cleaned_poses, cleaned_covs, loop_measurements, keyframes):
    """
    Build pose graph with Dynamic Covariance Scaling (DCS) on loop closures.

    Loop-closure edges use gtsam.noiseModel.Robust.Create(mEstimator.DCS.Create(c), gaussian_noise),
    where DCS is a native robust kernel that down-weights constraints based on their chi-squared residuals.
    """
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Prior on first keyframe
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
    first_kf = keyframes[0]
    graph.add(gtsam.PriorFactorPose3(symbol('c', first_kf), gtsam.Pose3(), prior_noise))
    initial.insert(symbol('c', first_kf), gtsam.Pose3())

    # Sequential keyframe poses (baseline edges, no robust wrapping)
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

    # Add loop closures with DCS robust wrapping
    dcs_kernel = gtsam.noiseModel.mEstimator.DCS.Create(DCS_THRESHOLD)

    for loop in loop_measurements:
        start_kf, end_kf = loop['start_kf'], loop['end_kf']
        start_key = symbol('c', start_kf)
        end_key = symbol('c', end_kf)
        rel_pose = loop['relative_pose']

        # Wrap DCS around the loop-closure covariance
        gaussian_noise = covariance_to_noise_model(loop['relative_covariance'])
        robust_noise = gtsam.noiseModel.Robust.Create(dcs_kernel, gaussian_noise)

        graph.add(gtsam.BetweenFactorPose3(start_key, end_key, rel_pose, robust_noise))

    return graph, initial


def run_dynamic_covariance_scaling_pipeline():
    """Full pipeline: load DB, compute constraints, build DCS graph, optimize, evaluate."""
    print("\n" + "="*80)
    print("  DYNAMIC COVARIANCE SCALING FOR ROBUST MAP OPTIMIZATION")
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

    # Detect and verify loop closures (standard baseline gates)
    candidates, _ = detect_loop_closure_candidates(
        relative_poses, relative_covs, keyframes,
        mahalanobis_threshold=1000.0
    )
    verified, _ = verify_loop_closures_consensus(db, candidates, inlier_ratio_threshold=0.6)
    loop_measurements = estimate_verified_loop_relative_poses(db, verified)
    print(f"Verified loop closures: {len(loop_measurements)}")

    # Build and optimize baseline graph
    print("\n--- Baseline (no robust weighting) ---")
    t0 = time.time()
    baseline_graph, baseline_initial = build_baseline_pose_graph_dcs(
        cleaned_poses, cleaned_covs, loop_measurements, keyframes
    )
    baseline_result, baseline_marginals = optimize_pose_graph_lm(baseline_graph, baseline_initial)
    baseline_time = time.time() - t0
    print(f"Baseline optimization: {baseline_time:.2f} sec")

    baseline_ids, baseline_positions = extract_trajectory_and_ids(baseline_result)
    baseline_errors = compute_pose_graph_absolute_error(baseline_result, baseline_ids)

    # Build and optimize DCS graph
    print("\n--- Dynamic Covariance Scaling (chi² threshold = 6.0) ---")
    t0 = time.time()
    dcs_graph, dcs_initial = build_dcs_pose_graph(
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
        },
        'dcs': {
            'trajectory_positions': dcs_positions,
            'keyframe_ids': dcs_ids,
            'absolute_errors': dcs_errors,
            'runtime_sec': dcs_time,
            'loop_count': len(loop_measurements),
            'dcs_threshold': DCS_THRESHOLD,
        }
    }

    # Print summary
    print("\n" + "-"*80)
    print(f"SUMMARY (DCS threshold = {DCS_THRESHOLD})")
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
    output_file = output_dir / 'dynamic_covariance_scaling.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_file}")

    # Generate trajectory comparison plot
    from experiments.plotting import plot_multi_variant_trajectory
    plot_output_dir = Path(__file__).parent / 'outputs'
    plot_output_dir.mkdir(exist_ok=True)
    plot_multi_variant_trajectory(
        results,
        output_path=plot_output_dir / 'dynamic_covariance_scaling_trajectory.png',
        title='Dynamic Covariance Scaling vs. Baseline — Trajectory'
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
    results = run_dynamic_covariance_scaling_pipeline()
