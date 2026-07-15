"""
utils/bundle_adjustment.py
GTSAM-based bundle adjustment, factor graph construction, and optimization utilities.
"""

############################################################################
# BUNDLE ADJUSTMENT - Joint optimization of camera poses and 3D landmarks
############################################################################

import os
import random
import numpy as np
import gtsam
from gtsam import symbol
from gtsam.utils import plot as gtsam_plot

from .geometry import read_cameras, read_ground_truth_poses, camera_center, triangulate_point_linear, valid_stereo_obs


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


def pose_translation_np(pose):
    """Extracts 3D translation vector from GTSAM Pose3 object into NumPy array (3,)."""
    return np.array(pose.translation()).reshape(3)


def choose_keyframes(db, distance_threshold=2.5, max_gap=20, min_gap=5):
    """Selects keyframe indices along sequence using path distance and frame gap criteria."""
    keyframes = [0]
    last_kf = 0
    accumulated_dist = 0.0

    for idx in range(1, db.frame_num()):
        p_prev = camera_center(*db.camera_poses[idx - 1])
        p_curr = camera_center(*db.camera_poses[idx])

        accumulated_dist += np.linalg.norm(p_curr - p_prev)
        frames_since_last = idx - last_kf

        if (frames_since_last >= min_gap and accumulated_dist >= distance_threshold) or frames_since_last >= max_gap:
            keyframes.append(idx)
            last_kf = idx
            accumulated_dist = 0.0

    if keyframes[-1] != db.frame_num() - 1:
        keyframes.append(db.frame_num() - 1)

    return keyframes


def build_and_solve_bundle_core(db, start_frame, end_frame, max_tracks_per_window=150, prior_sigma=1e-6):
    """Core bundle adjustment builder and solver function shared by standard and prior sensitivity tests."""
    K_gtsam = init_gtsam_stereo_calibration()
    _, P_left0, P_right0 = read_cameras()

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    measurement_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(2.0),
        gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    )

    window_frames = list(range(start_frame, end_frame + 1))
    pose_start_global = get_c2w_pose(*db.camera_poses[start_frame])

    for f_id in window_frames:
        pose_key = symbol("c", f_id)
        pose_f_global = get_c2w_pose(*db.camera_poses[f_id])
        initial_estimate.insert(pose_key, pose_start_global.between(pose_f_global))

        if f_id == start_frame:
            anchor_factor = gtsam.PriorFactorPose3(
                pose_key, gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * prior_sigma)
            )
            graph.add(anchor_factor)

    rng = random.Random(0)
    candidate_tracks = set()
    for f_id in window_frames:
        candidate_tracks.update(db.tracks(f_id))

    guaranteed_tracks = set()
    for f_id in window_frames:
        ts = list(db.tracks(f_id))
        if ts:
            guaranteed_tracks.update(rng.sample(ts, min(15, len(ts))))

    for a, b in zip(window_frames[:-1], window_frames[1:]):
        shared_tracks = list(set(db.tracks(a)) & set(db.tracks(b)))
        if shared_tracks:
            guaranteed_tracks.update(rng.sample(shared_tracks, min(15, len(shared_tracks))))

    remaining = max_tracks_per_window - len(guaranteed_tracks)
    extras = list(candidate_tracks - guaranteed_tracks)
    final_tracks_to_optimize = list(guaranteed_tracks) + (rng.sample(extras, min(remaining, len(extras))) if remaining > 0 else [])

    optimized_landmark_ids = []
    for track_id in final_tracks_to_optimize:
        track_frames = [f for f in db.frames(track_id) if f in window_frames]
        if len(track_frames) < 2:
            continue

        obs_init = db.observation(track_frames[0], track_id)
        if not valid_stereo_obs(obs_init, min_disp=1.0):
            continue

        X_cam = triangulate_point_linear(
            np.array([obs_init.x_left, obs_init.y]),
            np.array([obs_init.x_right, obs_init.y]),
            P_left0, P_right0
        )

        if not np.all(np.isfinite(X_cam)) or X_cam[2] <= 2.0 or X_cam[2] > 120.0:
            continue

        init_pose = initial_estimate.atPose3(symbol("c", track_frames[0]))
        X_local = init_pose.transformFrom(gtsam.Point3(float(X_cam[0]), float(X_cam[1]), float(X_cam[2])))
        point_key = symbol("q", track_id)

        temp_factors = []
        for f_id in track_frames:
            obs = db.observation(f_id, track_id)
            if valid_stereo_obs(obs, min_disp=1.0):
                temp_factors.append(create_stereo_factor(obs, measurement_noise, symbol("c", f_id), point_key, K_gtsam))

        if len(temp_factors) < 2:
            continue

        initial_estimate.insert(point_key, X_local)
        optimized_landmark_ids.append(track_id)
        for factor in temp_factors:
            graph.add(factor)

    initial_error = graph.error(initial_estimate)
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate).optimize()
    final_error = graph.error(result)

    return {
        "result": result, "graph": graph, "initial": initial_estimate,
        "relative_pose": result.atPose3(symbol("c", start_frame)).between(result.atPose3(symbol("c", end_frame))),
        "optimized_landmark_ids": optimized_landmark_ids,
        "anchor_factor": anchor_factor, "start_frame": start_frame, "end_frame": end_frame,
        "initial_error": initial_error, "final_error": final_error, "window_frames": window_frames
    }


def solve_bundle_window(db, start_frame, end_frame, max_tracks_per_window=150):
    """Executes Local Bundle Adjustment optimization over a sliding frame window."""
    bundle_res = build_and_solve_bundle_core(db, start_frame, end_frame, max_tracks_per_window, prior_sigma=1e-6)
    optimized_points_local = []
    for track_id in bundle_res["optimized_landmark_ids"]:
        point_key = symbol("q", track_id)
        if bundle_res["result"].exists(point_key):
            p = np.array(bundle_res["result"].atPoint3(point_key)).reshape(3)
            if np.all(np.isfinite(p)):
                optimized_points_local.append(p)

    bundle_res["optimized_points_local"] = np.array(optimized_points_local)
    return bundle_res


def solve_bundle_with_prior_sigma(db, start_frame, end_frame, prior_sigma):
    """Wrapper for bundle window optimization with custom prior noise variances."""
    res = build_and_solve_bundle_core(db, start_frame, end_frame, max_tracks_per_window=150, prior_sigma=prior_sigma)
    return res["graph"], res["result"], res["window_frames"]


def compute_window_projection_errors(db, bundle_res, K_gtsam):
    """Computes median initial and final projection error over all optimized landmarks in a bundle window."""
    initial_errors, final_errors = [], []
    for track_id in bundle_res["optimized_landmark_ids"]:
        point_key = symbol("q", track_id)
        for f_id in bundle_res["window_frames"]:
            obs = db.observation(f_id, track_id)
            if obs is None or not valid_stereo_obs(obs, min_disp=1.0):
                continue
            pose_key = symbol("c", f_id)
            if bundle_res["initial"].exists(point_key) and bundle_res["initial"].exists(pose_key):
                pt_init = bundle_res["initial"].atPoint3(point_key)
                pose_init = bundle_res["initial"].atPose3(pose_key)
                err_init = compute_stereo_reprojection_error(pose_init, K_gtsam, pt_init, obs)
                if np.isfinite(err_init):
                    initial_errors.append(err_init)
            if bundle_res["result"].exists(point_key) and bundle_res["result"].exists(pose_key):
                pt_final = bundle_res["result"].atPoint3(point_key)
                pose_final = bundle_res["result"].atPose3(pose_key)
                err_final = compute_stereo_reprojection_error(pose_final, K_gtsam, pt_final, obs)
                if np.isfinite(err_final):
                    final_errors.append(err_final)
    return (np.nanmedian(initial_errors) if initial_errors else np.nan,
            np.nanmedian(final_errors) if final_errors else np.nan)


def accumulate_window_projection_errors_by_distance(db, bundle_res, K_gtsam, errors_by_distance, max_distance=20):
    """Mutates errors_by_distance dict by accumulating projection errors binned by frame distance from window start."""
    start_frame = bundle_res["start_frame"]
    for track_id in bundle_res["optimized_landmark_ids"]:
        point_key = symbol("q", track_id)
        for f_id in bundle_res["window_frames"]:
            obs = db.observation(f_id, track_id)
            if obs is None or not valid_stereo_obs(obs, min_disp=1.0) or not bundle_res["result"].exists(point_key):
                continue
            distance = f_id - start_frame
            if distance > max_distance:
                continue
            pose_key = symbol("c", f_id)
            if bundle_res["result"].exists(pose_key):
                pt = bundle_res["result"].atPoint3(point_key)
                pose = bundle_res["result"].atPose3(pose_key)
                err = compute_stereo_reprojection_error(pose, K_gtsam, pt, obs)
                if np.isfinite(err):
                    if distance not in errors_by_distance:
                        errors_by_distance[distance] = []
                    errors_by_distance[distance].append(err)
