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


def build_data(num_frames):
    """Constructs the long-term TrackingDB object by matching features sequentially across frames."""
    # TrackingDB is in the shared code/ directory (works for both homework/ and project/)
    import sys
    from pathlib import Path
    code_root = Path(__file__).parent.parent.parent
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    from tracking_database import TrackingDB

    db = TrackingDB()
    inlier_percentages = []

    R_global, t_global = np.eye(3), np.zeros((3, 1))
    camera_poses = [(R_global.copy(), t_global.copy())]

    prev_data = run_single_pair(idx=0, display=False, plot_3d=False)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    for idx in range(1, num_frames):
        print(f"Processing Frame Sequence Node: {idx}/{num_frames - 1}")
        curr_data = run_single_pair(idx=idx, display=False, plot_3d=False)

        knn_matches = bf_matcher.knnMatch(prev_data["des_left"], curr_data["des_left"], k=2)
        temporal_matches = [m for m, n in knn_matches if m.distance < 0.7 * n.distance]

        try:
            correspondences = build_pnp_correspondences(prev_data, curr_data, temporal_matches)
            if len(correspondences) < 4:
                raise RuntimeError("Not enough correspondences for PnP-RANSAC.")

            best_inliers, best_outliers, best_R, best_t = [], [], None, None
            k_matrix, _, _ = read_cameras()
            no_improvement, max_no_improvement = 0, 12

            for _ in range(50):
                sample = random.sample(correspondences, 4)
                obj_pts = np.array([c["X"] for c in sample], dtype=np.float32)
                img_pts = np.array([c["obs_left1"] for c in sample], dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, k_matrix, None, flags=cv2.SOLVEPNP_EPNP)
                if not success:
                    no_improvement += 1
                    continue

                R_candidate, _ = cv2.Rodrigues(rvec)
                inliers, outliers = evaluate_supporters(correspondences, prev_data, curr_data, R_candidate, tvec, threshold=2)

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
    db.camera_poses = camera_poses
    return db


def load_or_build_db(force_rebuild=False, num_frames=None):
    """Loads pre-built TrackingDB pickle file or executes construction pipeline."""
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
    db = build_data(num_frames=num_frames)
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
