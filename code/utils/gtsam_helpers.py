"""
utils/gtsam_helpers.py
GTSAM factor graph building, calibration initialization, and factor solvers.
"""

import numpy as np
import gtsam
from gtsam import symbol
from .geometry import read_cameras, pose_translation_np


def init_gtsam_stereo_calibration():
    """Constructs GTSAM Cal3_S2Stereo calibration object from calib.txt matrices."""
    K_mat, _, m_right0 = read_cameras()
    fx, fy, cx, cy, skew = K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2], K_mat[0, 1]
    t_stereo = np.linalg.inv(K_mat) @ m_right0[:, 3]
    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, abs(t_stereo[0]))


def get_c2w_pose(R_w2c, t_w2c):
    """Converts World-to-Camera extrinsics (R_w2c, t_w2c) into a GTSAM Camera-to-World Pose3."""
    R_c2w = R_w2c.T
    t_c2w = (-R_w2c.T @ t_w2c).flatten()
    return gtsam.Pose3(gtsam.Rot3(R_c2w), gtsam.Point3(t_c2w[0], t_c2w[1], t_c2w[2]))


def get_gtsam_camera_pose(gt_poses, frame_id):
    """Fetches ground truth pose at frame_id as a GTSAM Pose3 object."""
    return get_c2w_pose(*gt_poses[frame_id])


def create_stereo_factor(obs, noise_model, pose_key, point_key, K_gtsam):
    """Constructs a GenericStereoFactor3D for factor graph optimization."""
    stereo_meas = gtsam.StereoPoint2(float(obs.x_left), float(obs.x_right), float(obs.y))
    return gtsam.GenericStereoFactor3D(stereo_meas, noise_model, pose_key, point_key, K_gtsam)


def compute_stereo_reprojection_error(pose, K_gtsam, point_3d, obs):
    """Computes absolute pixel L2 reprojection error for a single GTSAM stereo observation."""
    try:
        camera = gtsam.StereoCamera(pose, K_gtsam)
        proj = camera.project(point_3d)
        return np.sqrt((proj.uL() - obs.x_left) ** 2 + (proj.v() - obs.y) ** 2 + (proj.uR() - obs.x_right) ** 2)
    except RuntimeError:
        return np.nan


def compute_single_factor_error(pose, K_gtsam, point_3d, obs, pose_id=0, point_id=0):
    """Evaluates individual GenericStereoFactor3D scalar graph cost."""
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    pose_key, point_key = symbol('c', pose_id), symbol('q', point_id)
    factor = create_stereo_factor(obs, measurement_noise, pose_key, point_key, K_gtsam)

    values = gtsam.Values()
    values.insert(pose_key, pose)
    values.insert(point_key, point_3d)
    return factor.error(values)


def extract_optimized_geometry(result, window_frames, landmarks_in_window):
    """Extracts camera trajectories and 3D landmarks filtered by 3-STD statistical bounds."""
    cam_positions = np.array([result.atPose3(symbol('c', f_id)).translation() for f_id in window_frames])

    landmark_positions = []
    for t_id in landmarks_in_window:
        point_key = symbol('q', t_id)
        if result.exists(point_key):
            pt = result.atPoint3(point_key)
            landmark_positions.append([pt[0], pt[1], pt[2]])
    landmark_positions = np.array(landmark_positions)

    if len(landmark_positions) > 0:
        median = np.median(landmark_positions, axis=0)
        std = np.std(landmark_positions, axis=0) + 1e-9
        mask = np.all(np.abs(landmark_positions - median) < 3 * std, axis=1)
        lm_filtered = landmark_positions[mask]
    else:
        lm_filtered = landmark_positions

    return cam_positions, lm_filtered


def w2c_to_local_gtsam_pose(R_start, t_start, R_f, t_f):
    """Transforms World-to-Camera pose parameters into local window coordinate system Pose3."""
    R_rel = R_f @ R_start.T
    t_rel = t_f - R_f @ R_start.T @ t_start
    return gtsam.Pose3(gtsam.Rot3(R_rel.T), gtsam.Point3(*(-R_rel.T @ t_rel).flatten()))


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
    """Extracts frame IDs and 3D positions sorted by keyframe index from GTSAM Values."""
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


def _shortest_path_pose_graph(relative_poses, relative_covs):
    """Constructs a NetworkX graph weighted by the scalar trace of relative edge covariances."""
    import networkx as nx
    G = nx.Graph()
    for (start_kf, end_kf), cov in relative_covs.items():
        G.add_edge(start_kf, end_kf, weight=float(np.trace(cov)))
    return G


def _get_edge_pose_and_cov(u, v, relative_poses, relative_covs):
    """Retrieves edge constraints between keyframe nodes, automatically handling directional inversions."""
    if (u, v) in relative_poses:
        return relative_poses[(u, v)], relative_covs[(u, v)]

    if (v, u) in relative_poses:
        pose_uv = relative_poses[(v, u)].inverse()
        Ad_inv = pose_uv.AdjointMap()
        return pose_uv, Ad_inv @ relative_covs[(v, u)] @ Ad_inv.T

    raise KeyError(f"No pose-graph edge between keyframes {u} and {v}")