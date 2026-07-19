"""
utils/tracking.py
Feature tracking database construction, statistics, and analysis utilities.
"""

import os
import random
import pickle
import sys
from pathlib import Path
import numpy as np


# Custom unpickler to handle module name compatibility between homework and project
class CompatibleUnpickler(pickle.Unpickler):
    """Unpickler that remaps old homework module names to project structure."""
    def find_class(self, module, name):
        # Remap homework module references to shared tracking_database classes
        if 'tracking_database' in module:
            # Handle both Observation and TrackingDB from the shared tracking_database module
            try:
                # Try to import from the shared code-level tracking_database module
                sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                import tracking_database
                if name == 'TrackingDB':
                    return tracking_database.TrackingDB
                elif name == 'Observation':
                    return tracking_database.Observation
            except (ImportError, AttributeError):
                pass
        # Fall back to default behavior
        return super().find_class(module, name)
import cv2
import matplotlib.pyplot as plt

from .geometry import (
    read_images, get_num_frames, read_cameras, run_single_pair,
    build_pnp_correspondences, evaluate_supporters, compose_transform,
    read_ground_truth_poses, crop_around_point, project_point, triangulate_point_linear
)

# =============================================================================
# CONSTANTS & SETUP
# =============================================================================
# Navigate from project/utils/tracking.py up to VAN_ex/ root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'dataset' / 'dataset' / 'sequences' / '00'
# Database file is shared at code/tracking_db.pkl
DB_PKL_PATH = str(PROJECT_ROOT / 'code' / 'tracking_db.pkl')


def build_stereo_dict(frame_data):
    """Creates a map of query keypoint indices to stereo match objects."""
    return {m.queryIdx: m for m in frame_data["stereo_inliers"]}


def get_feature_observation(frame_data, feature_idx, stereo_dict):
    """Extracts left x, right x, and y pixel coordinates for a given feature index."""
    if feature_idx not in stereo_dict:
        return None
    stereo_match = stereo_dict[feature_idx]
    kp_left = frame_data["kp_left"][feature_idx]
    kp_right = frame_data["kp_right"][stereo_match.trainIdx]
    return {"x_left": kp_left.pt[0], "x_right": kp_right.pt[0], "y": kp_left.pt[1]}


def add_feature_to_db(db, frame_id, feature_idx, track_id, observation):
    """Inserts a feature observation entry into the tracking database object."""
    db.add_observation(
        frame_id=frame_id, feature_idx=feature_idx, track_id=track_id,
        x_left=observation["x_left"], x_right=observation["x_right"], y=observation["y"]
    )


def compute_tracking_statistics(db):
    """Calculates summary tracking statistics (track lengths, frame links) from the database."""
    track_lengths = [len(db.frames(t_id)) for t_id in db.track_to_frames if len(db.frames(t_id)) > 1]
    if not track_lengths:
        raise RuntimeError("No non-trivial tracks found.")

    frame_link_counts = [len(db.tracks(f_id)) for f_id in db.frame_to_tracks]

    return {
        "total_tracks": len(track_lengths),
        "num_frames": db.frame_num(),
        "mean_track_length": float(np.mean(track_lengths)),
        "max_track_length": int(np.max(track_lengths)),
        "min_track_length": int(np.min(track_lengths)),
        "mean_frame_links": float(np.mean(frame_link_counts)),
    }


def print_tracking_statistics(stats):
    """Prints descriptive tracking metrics to console."""
    print("\n--- Task 4.2: Tracking Statistics ---")
    print(f"Total number of tracks: {stats['total_tracks']}")
    print(f"Number of frames: {stats['num_frames']}")
    print(f"Mean track length: {stats['mean_track_length']:.2f}")
    print(f"Maximum track length: {stats['max_track_length']}")
    print(f"Minimum track length: {stats['min_track_length']}")
    print(f"Mean number of frame links: {stats['mean_frame_links']:.2f}")


def select_track_by_min_length(db, min_length=6):
    """Finds the longest feature track meeting a minimum length criterion."""
    valid_tracks = [t_id for t_id in db.track_to_frames if len(db.frames(t_id)) >= min_length]
    if not valid_tracks:
        raise RuntimeError(f"No track with length >= {min_length}")
    return max(valid_tracks, key=lambda t: len(db.frames(t)))


def compute_connectivity(db):
    """Computes shared outgoing feature track links between consecutive frames."""
    connectivity = []
    for frame_id in range(db.frame_num() - 1):
        curr_t = set(db.tracks(frame_id))
        next_t = set(db.tracks(frame_id + 1))
        connectivity.append(len(curr_t & next_t))
    return connectivity


def compute_track_lengths(db, min_length=2):
    """Returns array of track lengths above a specified minimum length threshold."""
    return [len(db.frames(t_id)) for t_id in db.track_to_frames if len(db.frames(t_id)) >= min_length]


def build_data(num_frames, detector_type='akaze', pnp_threshold=2, pnp_iterations=50, pnp_max_no_improvement=12):
    """Constructs the long-term TrackingDB object by matching features sequentially across frames.

    Args:
        num_frames: number of frames to process
        detector_type: 'akaze' (default), 'orb', or 'sift'
        pnp_threshold: reprojection error threshold (pixels) for RANSAC inlier cutoff
        pnp_iterations: max RANSAC iterations
        pnp_max_no_improvement: early stopping after N iterations with no improvement
    """
    # TrackingDB is in the shared code/ directory (works for both homework/ and project/)
    import sys
    from pathlib import Path
    code_root = Path(__file__).parent.parent.parent
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from tracking_database import TrackingDB

    db = TrackingDB()
    inlier_percentages = []
    matches_per_frame = []

    R_global, t_global = np.eye(3), np.zeros((3, 1))
    camera_poses = [(R_global.copy(), t_global.copy())]

    # Select matcher norm based on detector type
    if detector_type == 'sift':
        matcher_norm = cv2.NORM_L2
    else:  # 'akaze' or 'orb'
        matcher_norm = cv2.NORM_HAMMING

    prev_data = run_single_pair(idx=0, display=False, plot_3d=False, detector_type=detector_type)
    bf_matcher = cv2.BFMatcher(matcher_norm)

    for idx in range(1, num_frames):
        print(f"Processing Frame Sequence Node: {idx}/{num_frames - 1} (detector: {detector_type})")
        curr_data = run_single_pair(idx=idx, display=False, plot_3d=False, detector_type=detector_type)

        knn_matches = bf_matcher.knnMatch(prev_data["des_left"], curr_data["des_left"], k=2)
        temporal_matches = [m for m, n in knn_matches if m.distance < 0.7 * n.distance]
        matches_per_frame.append(len(temporal_matches))

        try:
            correspondences = build_pnp_correspondences(prev_data, curr_data, temporal_matches)
            if len(correspondences) < 4:
                raise RuntimeError("Not enough correspondences for PnP-RANSAC.")

            best_inliers, best_outliers, best_R, best_t = [], [], None, None
            k_matrix, _, _ = read_cameras()
            no_improvement, max_no_improvement = 0, pnp_max_no_improvement

            for _ in range(pnp_iterations):
                sample = random.sample(correspondences, 4)
                obj_pts = np.array([c["X"] for c in sample], dtype=np.float32)
                img_pts = np.array([c["obs_left1"] for c in sample], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, k_matrix, None, flags=cv2.SOLVEPNP_EPNP)
                if not success:
                    no_improvement += 1
                    continue

                R_candidate, _ = cv2.Rodrigues(rvec)
                inliers, outliers = evaluate_supporters(correspondences, prev_data, curr_data, R_candidate, tvec, threshold=pnp_threshold)

                if len(inliers) > len(best_inliers):
                    best_inliers, best_outliers, best_R, best_t = inliers, outliers, R_candidate, tvec
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= max_no_improvement:
                    break

            if best_R is None:
                raise RuntimeError("RANSAC failed to find a valid pose.")

            total = len(best_inliers) + len(best_outliers)
            inlier_percentages.append(100.0 * len(best_inliers) / total if total > 0 else 0.0)

            R_global, t_global = compose_transform(R_global, t_global, best_R, best_t)

        except RuntimeError as e:
            print(f"PnP-RANSAC failure at frame link {idx - 1}->{idx}: {e}")
            inlier_percentages.append(0.0)
            best_inliers = []

        ransac_temporal_matches = [c['temporal_match'] for c in best_inliers]
        db.update_tracks(
            idx, ransac_temporal_matches,
            build_stereo_dict(prev_data), build_stereo_dict(curr_data),
            prev_data, curr_data
        )

        camera_poses.append((R_global.copy(), t_global.copy()))
        prev_data = curr_data

    db.inlier_percentages = inlier_percentages
    db.matches_per_frame = matches_per_frame
    db.camera_poses = camera_poses
    return db


def load_or_build_db(force_rebuild=False, num_frames=None, detector_type='akaze', pnp_threshold=2,
                    pnp_iterations=50, pnp_max_no_improvement=12):
    """Loads pre-built TrackingDB pickle file or executes construction pipeline.

    Args:
        force_rebuild: whether to force rebuild from scratch
        num_frames: number of frames to process (default: all)
        detector_type: 'akaze' (default), 'orb', or 'sift'
        pnp_threshold: reprojection error threshold (pixels) for RANSAC
        pnp_iterations: max RANSAC iterations
        pnp_max_no_improvement: early stopping after N iterations
    """
    if num_frames is None:
        num_frames = get_num_frames()

    if os.path.exists(DB_PKL_PATH) and not force_rebuild:
        print("Loading TrackingDB from pickle...")
        try:
            with open(DB_PKL_PATH, "rb") as f:
                unpickler = CompatibleUnpickler(f)
                return unpickler.load()
        except (ModuleNotFoundError, AttributeError) as e:
            print(f"  [Note] Database pickle incompatible ({e}), rebuilding...")
            force_rebuild = True

    print("Building TrackingDB from scratch...")
    db = build_data(num_frames=num_frames, detector_type=detector_type, pnp_threshold=pnp_threshold,
                   pnp_iterations=pnp_iterations, pnp_max_no_improvement=pnp_max_no_improvement)
    with open(DB_PKL_PATH, "wb") as f:
        pickle.dump(db, f)
    return db


def select_long_track(db, min_length=10):
    """Randomly selects a track with at least min_length observations."""
    long_tracks = [t_id for t_id in db.track_to_frames if len(db.frames(t_id)) >= min_length]
    if not long_tracks:
        raise RuntimeError(f"No structural track found with an observation lifetime >= {min_length} frames.")
    return random.choice(long_tracks)


def prepare_stereo_projection_matrices(K, m_left0, m_right0, R_w2c, t_w2c):
    """Constructs stereo projection matrices for world-to-camera coordinates."""
    t_stereo = np.linalg.inv(K) @ m_right0[:, 3]
    P_left = K @ np.hstack([R_w2c, t_w2c])
    P_right = K @ np.hstack([R_w2c, t_w2c + R_w2c @ t_stereo.reshape(3, 1)])
    return P_left, P_right


def triangulate_track_reference_point(db, track_id, first_frame_id, K, m_left0, m_right0):
    """Triangulates the 3D reference world coordinate from the first observation of a track."""
    obs_first = db.observation(first_frame_id, track_id)
    gt_poses = read_ground_truth_poses()

    R_first_gt, t_first_gt = gt_poses[first_frame_id]
    R_first_w2c = R_first_gt.T
    t_first_w2c = -R_first_gt.T @ t_first_gt.reshape(3, 1)

    P_L_first, P_R_first = prepare_stereo_projection_matrices(K, m_left0, m_right0, R_first_w2c, t_first_w2c)

    p_left_first = np.array([obs_first.x_left, obs_first.y])
    p_right_first = np.array([obs_first.x_right, obs_first.y])

    return triangulate_point_linear(p_left_first, p_right_first, P_L_first, P_R_first)


def compute_track_reprojection_errors(db, track_id, frames, X_world, K, m_left0, m_right0):
    """Computes left/right reprojection errors for a 3D point across all frames in a track."""
    gt_poses = read_ground_truth_poses()
    left_errors = []
    right_errors = []

    for frame_id in frames:
        obs = db.observation(frame_id, track_id)
        R_curr_gt, t_curr_gt = gt_poses[frame_id]

        R_curr_w2c = R_curr_gt.T
        t_curr_w2c = -R_curr_gt.T @ t_curr_gt.reshape(3, 1)

        P_left, P_right = prepare_stereo_projection_matrices(K, m_left0, m_right0, R_curr_w2c, t_curr_w2c)

        proj_l = project_point(P_left, X_world)
        proj_r = project_point(P_right, X_world)

        obs_l = np.array([obs.x_left, obs.y])
        obs_r = np.array([obs.x_right, obs.y])

        err_l = np.linalg.norm(proj_l - obs_l)
        err_r = np.linalg.norm(proj_r - obs_r)

        left_errors.append(err_l)
        right_errors.append(err_r)

    return left_errors, right_errors


def triangulate_track_reference_point_from_pose(db, track_id, first_frame_id, K, m_left0, m_right0, R_first, t_first):
    """Triangulates the 3D reference world coordinate from first observation using provided pose."""
    obs_first = db.observation(first_frame_id, track_id)
    P_L_first, P_R_first = prepare_stereo_projection_matrices(K, m_left0, m_right0, R_first, t_first)
    p_left_first = np.array([obs_first.x_left, obs_first.y])
    p_right_first = np.array([obs_first.x_right, obs_first.y])
    return triangulate_point_linear(p_left_first, p_right_first, P_L_first, P_R_first)


def compute_track_reprojection_errors_from_poses(db, track_id, frames, X_world, K, m_left0, m_right0, camera_poses):
    """Computes left/right reprojection errors for a 3D point across all frames in a track using provided poses."""
    left_errors = []
    right_errors = []
    for frame_id in frames:
        obs = db.observation(frame_id, track_id)
        R_curr_w2c, t_curr_w2c = camera_poses[frame_id]
        P_left, P_right = prepare_stereo_projection_matrices(K, m_left0, m_right0, R_curr_w2c, t_curr_w2c)
        proj_l = project_point(P_left, X_world)
        proj_r = project_point(P_right, X_world)
        obs_l = np.array([obs.x_left, obs.y])
        obs_r = np.array([obs.x_right, obs.y])
        left_errors.append(np.linalg.norm(proj_l - obs_l))
        right_errors.append(np.linalg.norm(proj_r - obs_r))
    return left_errors, right_errors


def compute_pnp_projection_error_vs_distance(db, K, m_left0, m_right0, max_distance=40, min_track_length=5, sample_size=150, seed=0):
    """Computes median projection error vs distance-from-reference for a sampled subset of tracks using PnP poses."""
    rng = random.Random(seed)
    errors_by_distance = {}
    valid_tracks = [t_id for t_id in db.track_to_frames if len(db.frames(t_id)) >= min_track_length]
    sampled = rng.sample(valid_tracks, min(sample_size, len(valid_tracks)))

    for track_id in sampled:
        frames = sorted(db.frames(track_id))
        if not frames:
            continue
        first_frame = frames[0]
        R_first, t_first = db.camera_poses[first_frame]
        try:
            X_world = triangulate_track_reference_point_from_pose(db, track_id, first_frame, K, m_left0, m_right0, R_first, t_first)
            if not np.all(np.isfinite(X_world)):
                continue
            left_errors, right_errors = compute_track_reprojection_errors_from_poses(db, track_id, frames, X_world, K, m_left0, m_right0, db.camera_poses)
            for dist, (l_err, r_err) in enumerate(zip(left_errors, right_errors)):
                mean_err = (l_err + r_err) / 2.0
                if dist <= max_distance:
                    if dist not in errors_by_distance:
                        errors_by_distance[dist] = []
                    errors_by_distance[dist].append(mean_err)
        except Exception:
            continue

    distances = sorted(errors_by_distance.keys())
    median_errors = [np.nanmedian(errors_by_distance[d]) for d in distances]
    return distances, median_errors


def compute_absolute_pnp_error(db):
    """Computes absolute PnP estimation error (X/Y/Z/norm + angle) vs ground truth."""
    from .geometry import camera_center, relative_rotation_angle_deg
    gt_poses = read_ground_truth_poses()
    frame_ids, err_x, err_y, err_z, err_norm, err_angle = [], [], [], [], [], []

    for frame_id, (R_est, t_est) in enumerate(db.camera_poses):
        if frame_id >= len(gt_poses):
            break
        R_gt, t_gt = gt_poses[frame_id]
        C_est = camera_center(R_est, t_est)
        C_gt = camera_center(R_gt, t_gt)
        diff = C_est - C_gt
        frame_ids.append(frame_id)
        err_x.append(float(diff[0]))
        err_y.append(float(diff[1]))
        err_z.append(float(diff[2]))
        err_norm.append(float(np.linalg.norm(diff)))
        err_angle.append(relative_rotation_angle_deg(R_est, R_gt))

    return {
        "frame_ids": frame_ids,
        "err_x": err_x, "err_y": err_y, "err_z": err_z, "err_norm": err_norm,
        "err_angle": err_angle
    }


def compute_pnp_relative_error_vs_gt(db, edges):
    """Computes relative pose error (location/angle) for PnP-accumulated poses vs ground truth."""
    from .geometry import relative_pose_w2c, compute_relative_pose_error_deg_m
    gt_poses = read_ground_truth_poses()
    edge_ids, loc_errors, ang_errors = [], [], []

    for i, (sf, ef) in enumerate(sorted(edges)):
        if sf >= len(db.camera_poses) or ef >= len(db.camera_poses):
            continue
        if sf >= len(gt_poses) or ef >= len(gt_poses):
            continue
        R_est_sf, t_est_sf = db.camera_poses[sf]
        R_est_ef, t_est_ef = db.camera_poses[ef]
        R_gt_sf, t_gt_sf = gt_poses[sf]
        R_gt_ef, t_gt_ef = gt_poses[ef]

        R_est_rel, t_est_rel = relative_pose_w2c(R_est_sf, t_est_sf, R_est_ef, t_est_ef)
        R_gt_rel, t_gt_rel = relative_pose_w2c(R_gt_sf, t_gt_sf, R_gt_ef, t_gt_ef)
        loc_err, ang_err = compute_relative_pose_error_deg_m(R_est_rel, t_est_rel, R_gt_rel, t_gt_rel)

        edge_ids.append(i)
        loc_errors.append(loc_err)
        ang_errors.append(ang_err)

    return edge_ids, loc_errors, ang_errors
