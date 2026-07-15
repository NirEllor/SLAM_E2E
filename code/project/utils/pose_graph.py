"""
utils/pose_graph.py
Pose graph construction, optimization, and covariance analysis utilities using GTSAM.
"""

############################################################################
# POSE GRAPH - Building and optimizing factor graph from keyframe constraints
############################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import gtsam
from gtsam import symbol
from gtsam.utils import plot as gtsam_plot

from .bundle_adjustment import pose_translation_np, solve_bundle_with_prior_sigma


def covariance_to_noise_model(cov, min_sigma=1e-6):
    """Converts 6x6 covariance matrix to regularized GTSAM Gaussian noise model."""
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (6, 6):
        raise ValueError(f"Expected 6x6 covariance, got {cov.shape}")
    cov = 0.5 * (cov + cov.T) + np.eye(6) * (min_sigma ** 2)
    return gtsam.noiseModel.Gaussian.Covariance(cov)


def find_component_start_keyframes(relative_poses):
    """Identifies starting keyframe nodes for disconnected sub-graphs."""
    edges = sorted(relative_poses.keys())
    if not edges:
        return []
    component_starts = [edges[0][0]]
    for (a, b), (c, d) in zip(edges[:-1], edges[1:]):
        if b != c:
            component_starts.append(c)
    return component_starts


def build_pose_graph(relative_poses, relative_covs):
    """Constructs GTSAM NonlinearFactorGraph composed of BetweenFactorPose3 constraints."""
    graph = gtsam.NonlinearFactorGraph()
    sorted_edges = sorted(relative_poses.keys())
    if not sorted_edges:
        raise RuntimeError("No relative pose constraints were provided.")

    strong_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
    for kf in find_component_start_keyframes(relative_poses):
        graph.add(gtsam.PriorFactorPose3(symbol("c", kf), gtsam.Pose3(), strong_prior_noise))

    for start_kf, end_kf in sorted_edges:
        noise_model = covariance_to_noise_model(relative_covs[(start_kf, end_kf)])
        graph.add(gtsam.BetweenFactorPose3(symbol("c", start_kf), symbol("c", end_kf), relative_poses[(start_kf, end_kf)], noise_model))

    return graph


def extract_trajectory_and_ids(values):
    """Generic helper to extract frame IDs and 3D positions sorted by keyframe index from GTSAM Values."""
    items = []
    for key in values.keys():
        sym = gtsam.Symbol(key)
        if sym.chr() == ord("c"):
            items.append((sym.index(), pose_translation_np(values.atPose3(key))))
    items.sort(key=lambda x: x[0])
    return [x[0] for x in items], np.array([x[1] for x in items]) if items else np.empty((0, 3))


def extract_pose_graph_positions(values):
    """Extracts sorted keyframe indices and translation vectors from GTSAM Values."""
    return extract_trajectory_and_ids(values)


def build_and_initialize_pose_graph(cleaned_poses, cleaned_covs):
    """Constructs GTSAM pose graph and computes initial trajectory estimates."""
    graph = build_pose_graph(cleaned_poses, cleaned_covs)
    initial = gtsam.Values()
    sorted_edges = sorted(cleaned_poses.keys())

    if not sorted_edges:
        raise RuntimeError("No valid relative poses left after cleaning.")

    first_kf = sorted_edges[0][0]
    current_global_pose = gtsam.Pose3()
    initial.insert(symbol("c", first_kf), current_global_pose)

    for start_kf, end_kf in sorted_edges:
        start_key, end_key = symbol("c", start_kf), symbol("c", end_kf)
        if not initial.exists(start_key):
            initial.insert(start_key, current_global_pose)
        else:
            current_global_pose = initial.atPose3(start_key)

        end_pose = current_global_pose.compose(cleaned_poses[(start_kf, end_kf)])
        if not initial.exists(end_key):
            initial.insert(end_key, end_pose)
            current_global_pose = end_pose

    return graph, initial


def optimize_pose_graph(graph, initial):
    """Solves nonlinear pose graph optimization using Levenberg-Marquardt optimizer."""
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    return result, gtsam.Marginals(graph, result)


def run_prior_sensitivity_sweep(db, c0_idx, ck_idx, output_dir):
    """Evaluates factor graph convergence sensitivity under varying anchor prior noise scales."""
    prior_configs = [
        ("Unit matrix (σ=1.0)", 1.0, "task_6_1_cov_prior_1p0.png"),
        ("I × 0.05  (σ=0.05)", 0.05, "task_6_1_cov_prior_0p05.png"),
        ("I × 1e-6  (σ=1e-6)", 1e-6, "task_6_1_cov_prior_1e-6.png"),
    ]
    for label, sigma, fname in prior_configs:
        print(f"  Prior noise σ = {sigma}  ({label}) ...", end="", flush=True)
        graph_s, result_s, wf_s = solve_bundle_with_prior_sigma(db, c0_idx, ck_idx, prior_sigma=sigma)
        marginal_s = gtsam.Marginals(graph_s, result_s)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"6.1 – Bundle {c0_idx}→{ck_idx}  |  prior noise: {label}\nFrame locations with marginal covariances", fontsize=10, fontweight='bold')

        for f_id in wf_s:
            key = symbol('c', f_id)
            try:
                gtsam_plot.plot_pose3_on_axes(ax, result_s.atPose3(key), axis_length=0.3, P=marginal_s.marginalCovariance(key))
            except Exception:
                gtsam_plot.plot_pose3_on_axes(ax, result_s.atPose3(key), axis_length=0.3)

        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        if sigma != 1.0:
            ax.set_xlim(-10.0, 10.0)
            ax.set_ylim(-10.0, 10.0)
        ax.view_init(elev=20, azim=-60)
        plt.tight_layout()
        path = os.path.join(output_dir, fname)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f" saved → {path}")


def compute_relative_pose_and_covariance(db, start_idx, end_idx):
    """Calculates marginal covariance and relative pose between keyframes using Schur Complements."""
    from .bundle_adjustment import solve_bundle_window

    br = solve_bundle_window(db, start_idx, end_idx)
    graph, result = br["graph"], br["result"]
    marginals = gtsam.Marginals(graph, result)

    key_0, key_k = symbol('c', start_idx), symbol('c', end_idx)
    joint_cov = marginals.jointMarginalCovariance(gtsam.KeyVector([key_0, key_k])).fullMatrix()

    Sigma_00, Sigma_0k = joint_cov[0:6, 0:6], joint_cov[0:6, 6:12]
    Sigma_k0, Sigma_kk = joint_cov[6:12, 0:6], joint_cov[6:12, 6:12]

    Sigma_conditional_global = Sigma_kk - Sigma_k0 @ np.linalg.inv(Sigma_00) @ Sigma_0k
    relative_pose = br["relative_pose"]
    Ad_k = relative_pose.AdjointMap()

    return relative_pose, Ad_k @ Sigma_conditional_global @ Ad_k.T


def compute_all_relative_constraints(db, keyframes):
    """Computes sequential keyframe-to-keyframe pose graph edge constraints and covariances."""
    bundle_windows = [(keyframes[i], keyframes[i + 1]) for i in range(len(keyframes) - 1)]
    relative_poses, relative_covs = {}, {}

    for sf, ef in bundle_windows:
        try:
            rel_pose, rel_cov = compute_relative_pose_and_covariance(db, sf, ef)
            relative_poses[(sf, ef)] = rel_pose
            relative_covs[(sf, ef)] = rel_cov
            print(f" Bundle ({sf:4d} -> {ef:4d}): OK")
        except Exception as e:
            print(f" Bundle ({sf:4d} -> {ef:4d}): FAILED ({e})")

    return relative_poses, relative_covs


def clean_pose_graph_edges(relative_poses, relative_covs):
    """Filters out unstable pose graph edges containing negative or ill-conditioned covariance diagonals."""
    cleaned_poses, cleaned_covs = {}, {}
    for edge, cov in relative_covs.items():
        if np.any(np.diag(cov) <= 0) or np.any(np.isnan(cov)):
            print(f"Skipping bad covariance edge: {edge}")
            continue
        cleaned_poses[edge] = relative_poses[edge]
        cleaned_covs[edge] = relative_covs[edge]
    return cleaned_poses, cleaned_covs


# Alias for backward compatibility with van_utils.py naming
run_and_plot_prior_sensitivity = run_prior_sensitivity_sweep


# Note: Visualization functions plot_pose_graph_trajectory, plot_pose_graph_with_covariances,
# and plot_pose_graph_2d_ellipses are in visualization.py as they handle rendering
