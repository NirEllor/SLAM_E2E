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

from .bundle_adjustment import pose_translation_np, solve_bundle_with_prior_sigma, get_c2w_pose
from .geometry import read_ground_truth_poses, relative_rotation_angle_deg


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
        ax.set_title(f"Graph 6.1c: Bundle {c0_idx}→{ck_idx} – prior noise: {label}\nFrame locations with marginal covariances\nrun_prior_sensitivity_sweep, pose_graph.py", fontsize=10, fontweight='bold')

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
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=-60)
        plt.tight_layout()
        path = os.path.join(output_dir, fname)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f" saved → {path}")


def compute_relative_pose_and_covariance(db, start_idx, end_idx, bundle_kwargs=None):
    """Calculates covariance of relative pose T_0k = T_0^{-1} T_k using joint marginal covariance and between() Jacobians."""
    from .bundle_adjustment import solve_bundle_window

    if bundle_kwargs is None:
        bundle_kwargs = {}
    br = solve_bundle_window(db, start_idx, end_idx, **bundle_kwargs)
    graph, result = br["graph"], br["result"]
    marginals = gtsam.Marginals(graph, result)

    key_0, key_k = symbol('c', start_idx), symbol('c', end_idx)
    pose_0 = result.atPose3(key_0)
    pose_k = result.atPose3(key_k)

    joint_cov = marginals.jointMarginalCovariance(gtsam.KeyVector([key_0, key_k])).fullMatrix()
    Sigma_00 = joint_cov[0:6, 0:6]
    Sigma_0k = joint_cov[0:6, 6:12]
    Sigma_k0 = joint_cov[6:12, 0:6]
    Sigma_kk = joint_cov[6:12, 6:12]

    H_0 = np.zeros((6, 6), dtype=np.float64, order='F')
    H_k = np.zeros((6, 6), dtype=np.float64, order='F')
    relative_pose = pose_0.between(pose_k, H_0, H_k)

    Sigma_rel = (H_0 @ Sigma_00 @ H_0.T +
                 H_k @ Sigma_kk @ H_k.T +
                 H_0 @ Sigma_0k @ H_k.T +
                 H_k @ Sigma_k0 @ H_0.T)

    return relative_pose, Sigma_rel


def compute_all_relative_constraints(db, keyframes, bundle_kwargs=None):
    """Computes sequential keyframe-to-keyframe pose graph edge constraints and covariances."""
    if bundle_kwargs is None:
        bundle_kwargs = {}
    bundle_windows = [(keyframes[i], keyframes[i + 1]) for i in range(len(keyframes) - 1)]
    relative_poses, relative_covs = {}, {}

    for sf, ef in bundle_windows:
        try:
            rel_pose, rel_cov = compute_relative_pose_and_covariance(db, sf, ef, bundle_kwargs=bundle_kwargs)
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


def compute_pose_graph_absolute_error(values, keyframe_ids=None):
    """Computes absolute PnP estimation error (X/Y/Z/norm + angle) for pose graph results vs ground truth."""
    from .bundle_adjustment import pose_translation_np
    gt_poses = read_ground_truth_poses()
    kf_ids, C_est, err_x, err_y, err_z, err_norm, err_angle = [], [], [], [], [], [], []

    for key in values.keys():
        sym = gtsam.Symbol(key)
        if sym.chr() == ord("c"):
            kf_id = sym.index()
            if kf_id >= len(gt_poses):
                continue
            pose_est = values.atPose3(key)
            C_e = pose_translation_np(pose_est)
            C_e = np.array(C_e).flatten()
            R_gt, t_gt = gt_poses[kf_id]
            from .geometry import camera_center
            C_g = camera_center(R_gt, t_gt)
            diff = C_e - C_g
            R_c2w_est = pose_est.rotation().matrix()
            R_c2w_gt = get_c2w_pose(R_gt, t_gt).rotation().matrix()
            kf_ids.append(kf_id)
            C_est.append(C_e)
            err_x.append(float(diff[0]))
            err_y.append(float(diff[1]))
            err_z.append(float(diff[2]))
            err_norm.append(float(np.linalg.norm(diff)))
            err_angle.append(relative_rotation_angle_deg(R_c2w_est, R_c2w_gt))

    sort_idx = np.argsort(kf_ids)
    return {
        "frame_ids": [kf_ids[i] for i in sort_idx],
        "positions": np.array([C_est[i] for i in sort_idx]),
        "err_x": [err_x[i] for i in sort_idx],
        "err_y": [err_y[i] for i in sort_idx],
        "err_z": [err_z[i] for i in sort_idx],
        "err_norm": [err_norm[i] for i in sort_idx],
        "err_angle": [err_angle[i] for i in sort_idx]
    }


def compute_bundle_relative_error_vs_gt(relative_poses):
    """Computes relative pose error (location/angle) for bundle-optimized edges vs ground truth."""
    gt_poses = read_ground_truth_poses()
    edge_ids, loc_errors, ang_errors = [], [], []

    for (sf, ef), rel_pose_est in sorted(relative_poses.items()):
        if sf >= len(gt_poses) or ef >= len(gt_poses):
            continue
        R_gt_sf, t_gt_sf = gt_poses[sf]
        R_gt_ef, t_gt_ef = gt_poses[ef]
        pose_sf_c2w = get_c2w_pose(R_gt_sf, t_gt_sf)
        pose_ef_c2w = get_c2w_pose(R_gt_ef, t_gt_ef)
        rel_pose_gt = pose_sf_c2w.between(pose_ef_c2w)
        rel_pose_err = rel_pose_est.inverse().compose(rel_pose_gt)
        R_err = rel_pose_err.rotation().matrix()
        t_err = np.array(rel_pose_err.translation()).flatten()
        edge_ids.append(len(edge_ids))
        loc_errors.append(float(np.linalg.norm(t_err)))
        ang_errors.append(relative_rotation_angle_deg(np.eye(3), R_err))

    return edge_ids, loc_errors, ang_errors


def compute_kitti_sequence_errors_keyframes(keyframe_ids, keyframe_poses_c2w, gt_poses, segment_length):
    """Computes KITTI-style errors for bundle-optimized keyframe poses (snapping segment ends to closest keyframe)."""
    from .geometry import camera_center
    loc_err_pct, ang_err_per_m = [], []

    for start_kf_idx in range(len(keyframe_ids)):
        start_kf = keyframe_ids[start_kf_idx]
        if start_kf + segment_length >= len(gt_poses):
            break
        end_target = start_kf + segment_length
        end_kf = min(keyframe_ids, key=lambda kf: abs(kf - end_target)) if keyframe_ids else end_target
        if end_kf <= start_kf or end_kf >= len(gt_poses):
            continue
        end_kf_idx = keyframe_ids.index(end_kf)

        if end_kf_idx not in keyframe_poses_c2w or start_kf_idx not in keyframe_poses_c2w:
            continue
        try:
            R_est_sf, t_est_sf = keyframe_poses_c2w[start_kf_idx]
            R_est_ef, t_est_ef = keyframe_poses_c2w[end_kf_idx]
            pose_est_sf = gtsam.Pose3(gtsam.Rot3(R_est_sf), gtsam.Point3(*t_est_sf.flatten()))
            pose_est_ef = gtsam.Pose3(gtsam.Rot3(R_est_ef), gtsam.Point3(*t_est_ef.flatten()))
            rel_pose_est = pose_est_sf.between(pose_est_ef)

            R_gt_sf, t_gt_sf = gt_poses[start_kf]
            R_gt_ef, t_gt_ef = gt_poses[end_kf]
            pose_gt_sf = get_c2w_pose(R_gt_sf, t_gt_sf)
            pose_gt_ef = get_c2w_pose(R_gt_ef, t_gt_ef)
            rel_pose_gt = pose_gt_sf.between(pose_gt_ef)
            rel_pose_err = rel_pose_est.inverse().compose(rel_pose_gt)

            R_err = rel_pose_err.rotation().matrix()
            t_err = np.array(rel_pose_err.translation()).flatten()
            loc_err = float(np.linalg.norm(t_err))
            ang_err = relative_rotation_angle_deg(np.eye(3), R_err)

            total_distance = sum(np.linalg.norm(camera_center(*gt_poses[i+1]) - camera_center(*gt_poses[i]))
                                  for i in range(start_kf, end_kf))
            if total_distance < 1e-6:
                continue
            loc_norm = 100.0 * loc_err / total_distance
            ang_norm = ang_err / total_distance
            # Only append if values are finite
            if np.isfinite(loc_norm) and np.isfinite(ang_norm):
                loc_err_pct.append(loc_norm)
                ang_err_per_m.append(ang_norm)
        except Exception:
            continue

    return loc_err_pct, ang_err_per_m


# Note: Visualization functions plot_pose_graph_trajectory, plot_pose_graph_with_covariances,
# and plot_pose_graph_2d_ellipses are in visualization.py as they handle rendering
