import os
import random
import pickle
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import networkx as nx

import gtsam
from gtsam import symbol
from gtsam.utils import plot as gtsam_plot

from tracking_database_custom import TrackingDB

# =============================================================================
# CONSTANTS & SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'dataset' / 'dataset' / 'sequences' / '00'
DB_PKL_PATH = "tracking_db_ex5.pkl"


# =============================================================================
# EXERCISE 1 & COMMON UTILITIES
# =============================================================================

def read_images(idx):
    """
    Reads grayscale left and right images for a given frame index.
    """
    img_name = f'{idx:06d}.png'
    p1, p2 = DATA_PATH / 'image_0' / img_name, DATA_PATH / 'image_1' / img_name
    img1, img2 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE), cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not find images at {DATA_PATH}")
    return img1, img2


def get_orb_features(img, n_features=700):
    """
    Extracts keypoints and ORB descriptors from an image.
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    return orb.detectAndCompute(img, None)


def get_akaze_features(img):
    """
    Extracts keypoints and AKAZE descriptors from an image.
    """
    akaze = cv2.AKAZE_create(threshold=0.0001)
    return akaze.detectAndCompute(img, None)


def plot_stereo_side_by_side(img_l, img_r, title="Stereo Pair"):
    """
    Displays left and right stereo images side-by-side.
    """
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img_l, cmap='gray')
    plt.title(f"{title} - Left")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_r, cmap='gray')
    plt.title(f"{title} - Right")
    plt.axis('off')
    plt.tight_layout()


def draw_matches_custom(img1, kp1, img2, kp2, matches, title, num=20):
    """
    Draws a random subset of feature matches between two images.
    """
    if len(matches) > num:
        matches = random.sample(matches, num)

    img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.imshow(img_rgb)
    plt.axis('off')


def get_num_frames():
    """
    Returns the total number of stereo frames available in sequence '00'.
    """
    image_dir = DATA_PATH / 'image_0'
    return len(list(image_dir.glob("*.png")))


# =============================================================================
# EXERCISE 2: RECTIFICATION, TRIANGULATION & STEREO GEOMETRY
# =============================================================================

def compute_rectified_stereo_deviations(kp_left, kp_right, matches):
    """
    Computes absolute y-coordinate pixel discrepancies between matched stereo keypoints.
    """
    deviations = []
    for match in matches:
        _, y_left = kp_left[match.queryIdx].pt
        _, y_right = kp_right[match.trainIdx].pt
        deviations.append(abs(y_left - y_right))
    return np.array(deviations)


def plot_deviation_histogram(deviations):
    """
    Plots a histogram of y-deviation values for stereo matches.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(deviations, bins=50)
    plt.xlabel("Deviation from rectified stereo pattern")
    plt.ylabel("Number of matches")
    plt.title("Histogram of deviations from rectified stereo pattern")
    plt.tight_layout()


def print_large_deviation_percentage(deviations, threshold=2):
    """
    Prints percentage of matches exceeding the epipolar error threshold.
    """
    num_matches = len(deviations)
    num_bad_matches = np.sum(deviations > threshold)
    percentage = 100 * num_bad_matches / num_matches
    print(f"Matches with deviation > {threshold} pixels: {num_bad_matches}/{num_matches}")
    print(f"Percentage of matches with deviation > {threshold} pixels: {percentage:.2f}%")


def split_matches_by_rectified_pattern(kp_left, kp_right, matches, threshold=2):
    """
    Splits feature matches into inliers and outliers based on y-coordinate disparity tolerance.
    """
    inliers, outliers = [], []
    for match in matches:
        _, y_left = kp_left[match.queryIdx].pt
        _, y_right = kp_right[match.trainIdx].pt
        if abs(y_left - y_right) <= threshold:
            inliers.append(match)
        else:
            outliers.append(match)
    return inliers, outliers


def draw_inliers_outliers(img1, kp1, img2, kp2, inliers, outliers):
    """
    Overlays inlier (orange) and outlier (cyan) keypoint markers on stereo image pairs.
    """
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    for match in outliers:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        cv2.circle(img1_color, (int(pt1[0]), int(pt1[1])), 3, (0, 255, 255), -1)
        cv2.circle(img2_color, (int(pt2[0]), int(pt2[1])), 3, (0, 255, 255), -1)

    for match in inliers:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        cv2.circle(img1_color, (int(pt1[0]), int(pt1[1])), 3, (255, 165, 0), -1)
        cv2.circle(img2_color, (int(pt2[0]), int(pt2[1])), 3, (255, 165, 0), -1)

    plot_stereo_side_by_side(img1_color, img2_color, title="Inliers (orange) vs Outliers (cyan)")


def read_cameras():
    """
    Loads camera calibration matrices (K, P1, P2) from calib.txt.
    """
    calib_path = DATA_PATH / 'calib.txt'
    with open(calib_path) as f:
        l1 = f.readline().split()[1:]
        l2 = f.readline().split()[1:]

    p1 = np.array([float(i) for i in l1]).reshape(3, 4)
    p2 = np.array([float(i) for i in l2]).reshape(3, 4)
    k = p1[:, :3]

    return k, p1, p2


def get_matched_points(kp1, kp2, matches):
    """
    Extracts 2D coordinate arrays from feature matches.
    """
    points1 = [kp1[m.queryIdx].pt for m in matches]
    points2 = [kp2[m.trainIdx].pt for m in matches]
    return np.array(points1), np.array(points2)


def triangulate_point_linear(p1, p2, m1, m2):
    """
    Triangulates a single 3D point from 2D stereo observations using linear SVD.
    """
    x1, y1 = p1
    x2, y2 = p2

    A = np.array([
        x1 * m1[2, :] - m1[0, :],
        y1 * m1[2, :] - m1[1, :],
        x2 * m2[2, :] - m2[0, :],
        y2 * m2[2, :] - m2[1, :]
    ])

    _, _, vt = np.linalg.svd(A)
    X_homogeneous = vt[-1]
    if abs(X_homogeneous[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan])
    return X_homogeneous[:3] / X_homogeneous[3]


def triangulate_points_linear(points1, points2, m1, m2):
    """
    Triangulates multiple 3D points iteratively using the linear SVD method.
    """
    return np.array([triangulate_point_linear(p1, p2, m1, m2) for p1, p2 in zip(points1, points2)])


def triangulate_points_opencv(points1, points2, m1, m2):
    """
    Triangulates 3D points in batch using OpenCV's optimized cv2.triangulatePoints.
    """
    points_4d = cv2.triangulatePoints(m1, m2, points1.T, points2.T)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    return points_3d.T


def plot_3d_points(points_3d, title="3D Point Cloud"):
    """
    Plots a 3D scatter plot of a landmark point cloud.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=2)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()


def median_3d_distance(points_a, points_b):
    """
    Computes median Euclidean distance between two corresponding sets of 3D points.
    """
    return np.median(np.linalg.norm(points_a - points_b, axis=1))


# =============================================================================
# EXERCISE 3: TEMPORAL TRACKING & PNP ESTIMATION
# =============================================================================

def camera_center(R, t):
    """
    Calculates 3D world position of a camera center from extrinsics: C = -R^T * t.
    """
    return (-R.T @ t).flatten()


def plot_four_cameras(R, t, baseline=0.54):
    """
    Plots bird's-eye view 2D positions (X, Z) of four relative stereo cameras (2 time steps).
    """
    c_left_0 = np.array([0.0, 0.0])
    c_right_0 = np.array([baseline, 0.0])

    c_world_3d = camera_center(R, t)
    c_left_1 = np.array([c_world_3d[0], c_world_3d[2]])

    right_shift_world = R.T[:, 0] * baseline
    c_right_1 = c_left_1 + np.array([right_shift_world[0], right_shift_world[2]])

    plt.figure(figsize=(8, 6))
    plt.scatter(c_left_0[0], c_left_0[1], color='blue', marker='^', s=150, label='Left Camera (t0)')
    plt.scatter(c_right_0[0], c_right_0[1], color='cyan', marker='^', s=150, label='Right Camera (t0)')
    plt.scatter(c_left_1[0], c_left_1[1], color='red', marker='s', s=120, label='Left Camera (t1)')
    plt.scatter(c_right_1[0], c_right_1[1], color='orange', marker='s', s=120, label='Right Camera (t1)')

    plt.plot([c_left_0[0], c_right_0[0]], [c_left_0[1], c_right_0[1]], 'b--', alpha=0.5)
    plt.plot([c_left_1[0], c_right_1[0]], [c_left_1[1], c_right_1[1]], 'r--', alpha=0.5)

    plt.title("Relative Positions of Four Cameras (Bird's-Eye View)")
    plt.xlabel("X (Width / Lateral movement [m])")
    plt.ylabel("Z (Depth / Forward movement [m])")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.axis('equal')
    print("\n[Plot] Generated camera relative position chart successfully.")


def project_point(P, X):
    """
    Projects a single 3D point X using projection matrix P into pixel space (u, v).
    """
    X_h = np.append(X, 1.0)
    x_h = P @ X_h
    if abs(x_h[2]) < 1e-12:
        return np.array([np.nan, np.nan])
    return x_h[:2] / x_h[2]


def project_points_vectorized(P, X):
    """
    Projects batch N 3D points X into 2D pixel coordinates (N, 2).
    """
    X_h = np.hstack([X, np.ones((X.shape[0], 1))])
    x_h = (P @ X_h.T).T

    valid = np.abs(x_h[:, 2]) > 1e-12
    projected = np.full((X.shape[0], 2), np.nan)
    projected[valid] = x_h[valid, :2] / x_h[valid, 2:3]
    return projected


def draw_match_canvas(img0_gray, kp0, img1_gray, kp1, matches_green, matches_red, title):
    """
    Generic helper to visualize matches across two images (green=supporters/inliers, red=outliers).
    """
    img0 = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2RGB)
    img1 = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2RGB)
    w0 = img0.shape[1]
    canvas = np.hstack([img0, img1])

    def draw_line(match, color):
        q_idx = match.queryIdx if hasattr(match, 'queryIdx') else match['temporal_match'].queryIdx
        t_idx = match.trainIdx if hasattr(match, 'trainIdx') else match['temporal_match'].trainIdx
        pt0 = tuple(map(int, kp0[q_idx].pt))
        pt1_raw = kp1[t_idx].pt
        pt1 = (int(pt1_raw[0] + w0), int(pt1_raw[1]))
        cv2.circle(canvas, pt0, 3, color, -1)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.line(canvas, pt0, pt1, color, 1)

    for match in matches_red:
        draw_line(match, (255, 0, 0))
    for match in matches_green:
        draw_line(match, (0, 255, 0))

    plt.figure(figsize=(16, 8))
    plt.imshow(canvas)
    plt.title(title)
    plt.axis("off")


def draw_temporal_supporters(img_left0, kp_left0, img_left1, kp_left1, supporters, non_supporters, title="Supporters vs Non-supporters"):
    """
    Plots temporal tracking matches with green lines for supporters and red for outliers.
    """
    draw_match_canvas(img_left0, kp_left0, img_left1, kp_left1, supporters, non_supporters, title + " | green=supporters, red=outliers")


def draw_ransac_results(frame0_data, frame1_data, inliers, outliers):
    """
    Visualizes PnP-RANSAC temporally tracked inliers and outliers between consecutive frames.
    """
    draw_match_canvas(
        frame0_data['img_left'], frame0_data['kp_left'],
        frame1_data['img_left'], frame1_data['kp_left'],
        inliers, outliers, "3.5: RANSAC Inliers (green) vs Outliers (red)"
    )


def build_pnp_correspondences(frame0_data, frame1_data, good_temporal_matches):
    """
    Constructs 3D-2D quad-image observations across consecutive stereo frames.
    """
    left0_to_3d = {m.queryIdx: (idx, m) for idx, m in enumerate(frame0_data['stereo_inliers'])}
    left1_to_stereo = {m.queryIdx: m for m in frame1_data['stereo_inliers']}

    correspondences = []
    for temporal_match in good_temporal_matches:
        l0_idx, l1_idx = temporal_match.queryIdx, temporal_match.trainIdx
        if l0_idx not in left0_to_3d or l1_idx not in left1_to_stereo:
            continue

        point_3d_idx, stereo_match0 = left0_to_3d[l0_idx]
        stereo_match1 = left1_to_stereo[l1_idx]

        correspondences.append({
            'X': frame0_data['points_3d'][point_3d_idx],
            'left0_match': stereo_match0,
            'temporal_match': temporal_match,
            'right1_match': stereo_match1,
            'obs_left0': np.array(frame0_data['kp_left'][stereo_match0.queryIdx].pt),
            'obs_right0': np.array(frame0_data['kp_right'][stereo_match0.trainIdx].pt),
            'obs_left1': np.array(frame1_data['kp_left'][temporal_match.trainIdx].pt),
            'obs_right1': np.array(frame1_data['kp_right'][stereo_match1.trainIdx].pt)
        })
    return correspondences


def evaluate_supporters(correspondences, frame0_data, frame1_data, R, t, threshold=2):
    """
    Evaluates supporter status of 3D-2D correspondences by reprojection across 4 cameras.
    """
    k, P_left0, P_right0 = read_cameras()
    t_stereo = np.linalg.inv(k) @ P_right0[:, 3]

    P_left1 = k @ np.hstack([R, t.reshape(3, 1)])
    P_right1 = k @ np.hstack([R, (t.reshape(3) + t_stereo).reshape(3, 1)])

    X = np.array([c['X'] for c in correspondences], dtype=np.float64)

    err_left0 = np.linalg.norm(project_points_vectorized(P_left0, X) - np.array([c['obs_left0'] for c in correspondences]), axis=1)
    err_right0 = np.linalg.norm(project_points_vectorized(P_right0, X) - np.array([c['obs_right0'] for c in correspondences]), axis=1)
    err_left1 = np.linalg.norm(project_points_vectorized(P_left1, X) - np.array([c['obs_left1'] for c in correspondences]), axis=1)
    err_right1 = np.linalg.norm(project_points_vectorized(P_right1, X) - np.array([c['obs_right1'] for c in correspondences]), axis=1)

    supporter_mask = (err_left0 < threshold) & (err_right0 < threshold) & (err_left1 < threshold) & (err_right1 < threshold)

    inliers = [c for c, is_inlier in zip(correspondences, supporter_mask) if is_inlier]
    outliers = [c for c, is_inlier in zip(correspondences, supporter_mask) if not is_inlier]

    return inliers, outliers


def plot_transformed_clouds(frame0_data, frame1_data, R, t):
    """
    Visualizes point cloud alignment from Frame 0 transformed into Frame 1 coordinate system.
    """
    cloud0 = frame0_data['points_3d']
    cloud1 = frame1_data['points_3d']

    cloud0_transformed = (R @ cloud0.T + t.reshape(3, 1)).T

    mask0 = (cloud0_transformed[:, 2] > 0) & (cloud0_transformed[:, 2] < 80)
    mask1 = (cloud1[:, 2] > 0) & (cloud1[:, 2] < 80)

    plt.figure(figsize=(10, 8))
    plt.scatter(cloud0_transformed[mask0, 0], cloud0_transformed[mask0, 2], s=2, c='red', label='Pair0 after T')
    plt.scatter(cloud1[mask1, 0], cloud1[mask1, 2], s=2, c='blue', label='Pair1')

    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("3.5: Point Clouds Alignment")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)


def compose_transform(R1, t1, R2, t2):
    """
    Composes relative motion transformations: T_new = T2 * T1.
    """
    return R2 @ R1, R2 @ t1 + t2


def read_ground_truth_poses():
    """
    Reads KITTI benchmark ground truth camera matrices (3x4).
    """
    poses_path = PROJECT_ROOT / 'dataset' / 'dataset' / 'poses' / '00.txt'
    gt_poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            M = np.array(values).reshape(3, 4)
            gt_poses.append((M[:, :3], M[:, 3].reshape(3, 1)))
    return gt_poses


def plot_trajectory(est_positions, gt_positions):
    """
    Plots estimated vehicle trajectory against ground truth (X-Z Top View).
    """
    est_positions, gt_positions = np.array(est_positions), np.array(gt_positions)
    plt.figure(figsize=(10, 8))
    plt.plot(est_positions[:, 0], est_positions[:, 2], label='Estimated trajectory', linewidth=2)
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], label='Ground truth', linewidth=2)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title("3.6: Camera Trajectory (Top View)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)


# =============================================================================
# EXERCISE 4: LONG-TERM FEATURE TRACKING DATABASE
# =============================================================================

def build_stereo_dict(frame_data):
    """
    Creates a map of query keypoint indices to stereo match objects.
    """
    return {m.queryIdx: m for m in frame_data["stereo_inliers"]}


def get_feature_observation(frame_data, feature_idx, stereo_dict):
    """
    Extracts left x, right x, and y pixel coordinates for a given feature index.
    """
    if feature_idx not in stereo_dict:
        return None
    stereo_match = stereo_dict[feature_idx]
    kp_left = frame_data["kp_left"][feature_idx]
    kp_right = frame_data["kp_right"][stereo_match.trainIdx]
    return {"x_left": kp_left.pt[0], "x_right": kp_right.pt[0], "y": kp_left.pt[1]}


def add_feature_to_db(db, frame_id, feature_idx, track_id, observation):
    """
    Inserts a feature observation entry into the tracking database object.
    """
    db.add_observation(
        frame_id=frame_id, feature_idx=feature_idx, track_id=track_id,
        x_left=observation["x_left"], x_right=observation["x_right"], y=observation["y"]
    )


def get_or_create_track(db, frame_id, feature_idx):
    """
    Retrieves existing track ID or generates a new track entry for a feature point.
    """
    if db.has_feature(frame_id, feature_idx):
        return db.get_track_of_feature(frame_id, feature_idx)
    return db.create_track()


def compute_tracking_statistics(db):
    """
    Calculates summary tracking statistics (track lengths, frame links) from the database.
    """
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
    """
    Prints descriptive tracking metrics to console.
    """
    print("\n--- Task 4.2: Tracking Statistics ---")
    print(f"Total number of tracks: {stats['total_tracks']}")
    print(f"Number of frames: {stats['num_frames']}")
    print(f"Mean track length: {stats['mean_track_length']:.2f}")
    print(f"Maximum track length: {stats['max_track_length']}")
    print(f"Minimum track length: {stats['min_track_length']}")
    print(f"Mean number of frame links: {stats['mean_frame_links']:.2f}")


def debug_tracking_database(db, frame_id=10):
    """
    Prints diagnostic metadata about track associations at a given frame.
    """
    print("Frames:", db.frame_num())
    print("Tracks:", db.track_num())
    tracks_in_frame = db.tracks(frame_id)
    print("Tracks in frame:", tracks_in_frame)

    if tracks_in_frame:
        track_id = tracks_in_frame[0]
        print("Example track id:", track_id)
        print("Frames of example track:", db.frames(track_id))

    longest_track = max(db.track_to_frames, key=lambda t: len(db.frames(t)))
    print("Longest track id:", longest_track)
    print("Length:", len(db.frames(longest_track)))


def select_track_by_min_length(db, min_length=6):
    """
    Finds the longest feature track meeting a minimum length criterion.
    """
    valid_tracks = [t_id for t_id in db.track_to_frames if len(db.frames(t_id)) >= min_length]
    if not valid_tracks:
        raise RuntimeError(f"No track with length >= {min_length}")
    return max(valid_tracks, key=lambda t: len(db.frames(t)))


def crop_around_point(img, x, y, crop_size=20):
    """
    Crops a square image region around a 2D point coordinate.
    """
    h, w = img.shape[:2]
    half = crop_size // 2
    x_i, y_i = int(round(x)), int(round(y))

    x1, x2 = max(0, x_i - half), min(w, x_i + half)
    y1, y2 = max(0, y_i - half), min(h, y_i + half)

    return img[y1:y2, x1:x2], x - x1, y - y1


def plot_track_observations(db, track_id, crop_size=20):
    """
    Displays tracked feature patch crops across all frames where the track is present.
    """
    frames = db.frames(track_id)
    num_rows = len(frames)
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 2.2 * num_rows))
    if num_rows == 1:
        axes = np.array([axes])

    fig.suptitle(f"Track #{track_id}, length={len(frames)}", fontsize=14)

    for row, frame_id in enumerate(frames):
        obs = db.observation(frame_id, track_id)
        img_left, _ = read_images(frame_id)
        crop, lx, ly = crop_around_point(img_left, obs.x_left, obs.y, crop_size=crop_size)

        axes[row, 0].imshow(img_left, cmap="gray")
        axes[row, 0].scatter([obs.x_left], [obs.y], c="red", marker="x", s=40)
        axes[row, 0].set_title(f"Frame {frame_id}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(crop, cmap="gray")
        axes[row, 1].scatter([lx], [ly], c="red", marker="x", s=40)
        axes[row, 1].set_title(f"{crop_size}x{crop_size} crop")
        axes[row, 1].axis("off")

    plt.tight_layout()


def compute_connectivity(db):
    """
    Computes shared outgoing feature track links between consecutive frames.
    """
    connectivity = []
    for frame_id in range(db.frame_num() - 1):
        curr_t = set(db.tracks(frame_id))
        next_t = set(db.tracks(frame_id + 1))
        connectivity.append(len(curr_t & next_t))
    return connectivity


def plot_connectivity(connectivity):
    """
    Plots consecutive frame feature connectivity sequence over time.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(connectivity, linewidth=1)
    plt.axhline(np.mean(connectivity), color="green", linestyle="--", label=f"Mean={np.mean(connectivity):.1f}")
    plt.title("Connectivity")
    plt.xlabel("Frame")
    plt.ylabel("Outgoing Tracks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_inlier_percentage(inlier_percentages):
    """
    Plots the percentage of PnP RANSAC inliers per frame transition.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(inlier_percentages, linewidth=1)
    mean_val = np.mean(inlier_percentages)
    plt.axhline(mean_val, color="green", linestyle="--", label=f"Mean={mean_val:.2f}%")
    plt.title("Inlier Percentage Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Inlier Percentage (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def compute_track_lengths(db, min_length=2):
    """
    Returns array of track lengths above a specified minimum length threshold.
    """
    return [len(db.frames(t_id)) for t_id in db.track_to_frames if len(db.frames(t_id)) >= min_length]


def plot_track_length_histogram(db, min_length=2):
    """
    Plots a log-scale histogram of feature track lengths.
    """
    track_lengths = compute_track_lengths(db, min_length=min_length)
    plt.figure(figsize=(12, 5))
    bins = range(min(track_lengths), max(track_lengths) + 2)
    plt.hist(track_lengths, bins=bins, edgecolor="black")
    plt.yscale("log")
    plt.title("Track Length Histogram")
    plt.xlabel("Track Length")
    plt.ylabel("Track Count")
    plt.grid(True, axis="y")
    plt.tight_layout()


def project_stereo_point(K, R, t, t_stereo, X):
    """
    Projects a 3D point into left and right image plane coordinates.
    """
    P_left = K @ np.hstack([R, t.reshape(3, 1)])
    P_right = K @ np.hstack([R, (t.reshape(3) + t_stereo.reshape(3)).reshape(3, 1)])
    return project_point(P_left, X), project_point(P_right, X)


def run_single_pair(idx=0, display=False, plot_3d=True):
    """
    Executes feature detection, matching, epipolar filtering, and triangulation for a single frame.
    """
    if display:
        print(f"\n========== Processing Frame {idx} Pipeline ==========")

    img1, img2 = read_images(idx)
    kp1, des1 = get_akaze_features(img1)
    kp2, des2 = get_akaze_features(img2)

    assert len(kp1) >= 500 and len(kp2) >= 500, f"Insufficient feature count in frame {idx}!"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1, des2)

    if display:
        draw_matches_custom(img1, kp1, img2, kp2, matches, f"Frame {idx}: Raw Match Overlap Vectors")
        deviations = compute_rectified_stereo_deviations(kp1, kp2, matches)
        plot_deviation_histogram(deviations)
        print_large_deviation_percentage(deviations, threshold=2)

    inliers, outliers = split_matches_by_rectified_pattern(kp1, kp2, matches, threshold=2)

    if display:
        draw_inliers_outliers(img1, kp1, img2, kp2, inliers, outliers)

    k, m1, m2 = read_cameras()
    points1, points2 = get_matched_points(kp1, kp2, inliers)
    points_3d = triangulate_points_opencv(points1, points2, m1, m2)

    valid_depth_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 350)
    filtered_inliers = [inliers[i] for i in range(len(inliers)) if valid_depth_mask[i]]
    filtered_points_3d = points_3d[valid_depth_mask]

    if display or plot_3d:
        plot_3d_points(filtered_points_3d, title=f"Frame {idx}: Triangulated Landmark Points (<350m)")

    des_left_inliers = np.array([des1[m.queryIdx] for m in filtered_inliers]) if filtered_inliers else np.empty((0, des1.shape[1]))
    des_right_inliers = np.array([des2[m.trainIdx] for m in filtered_inliers]) if filtered_inliers else np.empty((0, des2.shape[1]))

    return {
        'img_left': img1, 'img_right': img2, 'kp_left': kp1, 'kp_right': kp2,
        'des_left': des1, 'des_right': des2, 'des_left_inliers': des_left_inliers,
        'des_right_inliers': des_right_inliers, 'stereo_inliers': filtered_inliers,
        'points_3d': filtered_points_3d
    }


# =============================================================================
# EXERCISE 5: GTSAM FACTOR GRAPHS & BUNDLE ADJUSTMENT
# =============================================================================

def init_gtsam_stereo_calibration():
    """
    Constructs GTSAM Cal3_S2Stereo calibration object from calib.txt matrices.
    """
    K_mat, _, m_right0 = read_cameras()
    fx, fy, cx, cy, skew = K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2], K_mat[0, 1]
    t_stereo = np.linalg.inv(K_mat) @ m_right0[:, 3]
    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, abs(t_stereo[0]))


def get_c2w_pose(R_w2c, t_w2c):
    """
    Converts World-to-Camera extrinsics (R_w2c, t_w2c) into a GTSAM Camera-to-World Pose3.
    """
    R_c2w = R_w2c.T
    t_c2w = (-R_w2c.T @ t_w2c).flatten()
    return gtsam.Pose3(gtsam.Rot3(R_c2w), gtsam.Point3(t_c2w[0], t_c2w[1], t_c2w[2]))


def get_gtsam_camera_pose(gt_poses, frame_id):
    """
    Fetches ground truth pose at frame_id as a GTSAM Pose3 object.
    """
    return get_c2w_pose(*gt_poses[frame_id])


def create_stereo_factor(obs, noise_model, pose_key, point_key, K_gtsam):
    """
    Constructs a GenericStereoFactor3D for factor graph optimization.
    """
    stereo_meas = gtsam.StereoPoint2(float(obs.x_left), float(obs.x_right), float(obs.y))
    return gtsam.GenericStereoFactor3D(stereo_meas, noise_model, pose_key, point_key, K_gtsam)


def compute_stereo_reprojection_error(pose, K_gtsam, point_3d, obs):
    """
    Computes absolute pixel L2 reprojection error for a single GTSAM stereo observation.
    """
    try:
        camera = gtsam.StereoCamera(pose, K_gtsam)
        proj = camera.project(point_3d)
        return np.sqrt((proj.uL() - obs.x_left) ** 2 + (proj.v() - obs.y) ** 2 + (proj.uR() - obs.x_right) ** 2)
    except RuntimeError:
        return np.nan


def compute_single_factor_error(pose, K_gtsam, point_3d, obs, pose_id=0, point_id=0):
    """
    Evaluates individual GenericStereoFactor3D scalar graph cost.
    """
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    pose_key, point_key = symbol('c', pose_id), symbol('q', point_id)
    factor = create_stereo_factor(obs, measurement_noise, pose_key, point_key, K_gtsam)

    values = gtsam.Values()
    values.insert(pose_key, pose)
    values.insert(point_key, point_3d)
    return factor.error(values)


def extract_optimized_geometry(result, window_frames, landmarks_in_window):
    """
    Extracts camera trajectories and 3D landmarks filtered by 3-STD statistical bounds.
    """
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


def draw_projection_validation_frames(frame_id, obs, proj_init, proj_final):
    """
    Draws measurement (Blue), pre-optimization (Red), and post-optimization (Green) pixel overlays.
    """
    img_left_gray, img_right_gray = read_images(frame_id)
    img_left = cv2.cvtColor(img_left_gray, cv2.COLOR_GRAY2BGR)
    img_right = cv2.cvtColor(img_right_gray, cv2.COLOR_GRAY2BGR)

    points_map = [
        ((int(obs.x_left), int(obs.y)), (int(proj_init.uL()), int(proj_init.v())), (int(proj_final.uL()), int(proj_final.v()))),
        ((int(obs.x_right), int(obs.y)), (int(proj_init.uR()), int(proj_init.v())), (int(proj_final.uR()), int(proj_final.v())))
    ]

    for img, (pt_meas, pt_before, pt_after) in zip([img_left, img_right], points_map):
        cv2.circle(img, pt_meas, radius=6, color=(255, 0, 0), thickness=-1)
        cv2.circle(img, pt_before, radius=6, color=(0, 0, 255), thickness=-1)
        cv2.circle(img, pt_after, radius=6, color=(0, 255, 0), thickness=-1)

    cv2.putText(img_left, "Blue: Meas | Red: Before | Green: After", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"task_5_3_worst_frame_{frame_id}_left.png"), img_left)
    cv2.imwrite(os.path.join(output_dir, f"task_5_3_worst_frame_{frame_id}_right.png"), img_right)


def w2c_to_local_gtsam_pose(R_start, t_start, R_f, t_f):
    """
    Transforms World-to-Camera pose parameters into local window coordinate system Pose3.
    """
    R_rel = R_f @ R_start.T
    t_rel = t_f - R_f @ R_start.T @ t_start
    return gtsam.Pose3(gtsam.Rot3(R_rel.T), gtsam.Point3(*(-R_rel.T @ t_rel).flatten()))


def valid_stereo_obs(obs, min_disp=1.0):
    """
    Validates stereo observations against negative or near-zero disparity degeneracies.
    """
    return (obs.x_left - obs.x_right) >= min_disp


def choose_keyframes(db, distance_threshold=2.5, max_gap=20, min_gap=5):
    """
    Selects keyframe indices along sequence using path distance and frame gap criteria.
    """
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


def pose_translation_np(pose):
    """
    Extracts 3D translation vector from GTSAM Pose3 object into NumPy array (3,).
    """
    return np.array(pose.translation()).reshape(3)


def build_and_solve_bundle_core(db, start_frame, end_frame, max_tracks_per_window=150, prior_sigma=1e-6):
    """
    Core bundle adjustment builder and solver function shared by standard and prior sensitivity tests.
    """
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
    """
    Executes Local Bundle Adjustment optimization over a sliding frame window.
    """
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
    """
    Wrapper for bundle window optimization with custom prior noise variances.
    """
    res = build_and_solve_bundle_core(db, start_frame, end_frame, max_tracks_per_window=150, prior_sigma=prior_sigma)
    return res["graph"], res["result"], res["window_frames"]


def plot_q5_4_results(keyframes, global_keyframe_poses, all_points_global, output_dir="./outputs"):
    """
    Plots optimized bundle keyframe positions against ground truth trajectories.
    """
    gt_poses = read_ground_truth_poses()
    valid_keyframes = [kf for kf in keyframes if kf in global_keyframe_poses]

    estimated_positions = np.array([pose_translation_np(global_keyframe_poses[kf]) for kf in valid_keyframes])
    gt_positions = np.array([camera_center(*gt_poses[kf]) for kf in valid_keyframes])
    points = np.array(all_points_global)

    plt.figure(figsize=(10, 8))
    if len(points) > 0:
        plt.scatter(points[:, 0], points[:, 2], s=1, alpha=0.25, label="Optimized 3D points")

    plt.plot(estimated_positions[:, 0], estimated_positions[:, 2], "bo-", markersize=3, label="Optimized keyframes")
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], "r--", linewidth=2, label="Ground truth keyframes")

    plt.title("5.4: Optimized Keyframe Trajectory vs Ground Truth")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "task_5_4_keyframes_vs_gt.png"), dpi=200)


def plot_keyframe_localization_error(keyframes, global_keyframe_poses, output_dir="./outputs"):
    """
    Plots Euclidean keyframe localization errors [m] relative to ground truth.
    """
    gt_poses = read_ground_truth_poses()
    errors, valid_keyframes = [], []

    for kf in keyframes:
        if kf not in global_keyframe_poses:
            continue
        est_pos = pose_translation_np(global_keyframe_poses[kf])
        gt_pos = camera_center(*gt_poses[kf])
        errors.append(np.linalg.norm(est_pos - gt_pos))
        valid_keyframes.append(kf)

    if not errors:
        print("No valid localization errors to plot.")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(valid_keyframes, errors, marker="o", linewidth=1)
    plt.title("5.4: Keyframe Localization Error")
    plt.xlabel("Frame")
    plt.ylabel("Localization Error [m]")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "task_5_4_keyframe_error.png"), dpi=200)
    print(f"Mean keyframe localization error: {np.mean(errors):.3f} m")
    print(f"Max keyframe localization error: {np.max(errors):.3f} m")


def build_data(num_frames=10):
    """
    Constructs the long-term TrackingDB object by matching features sequentially across frames.
    """
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
    """
    Loads pre-built TrackingDB pickle file or executes construction pipeline.
    """
    if num_frames is None:
        num_frames = get_num_frames()

    if os.path.exists(DB_PKL_PATH) and not force_rebuild:
        print("Loading TrackingDB from pickle...")
        with open(DB_PKL_PATH, "rb") as f:
            return pickle.load(f)

    print("Building TrackingDB from scratch...")
    db = build_data(num_frames=num_frames)
    with open(DB_PKL_PATH, "wb") as f:
        pickle.dump(db, f)
    return db


def debug_coordinate_system_alignment(keyframes, global_keyframe_poses):
    """
    Prints diagnostic alignment distances between estimated and ground truth frames.
    """
    gt_poses = read_ground_truth_poses()
    print("\n[Debug coordinate system alignment]")

    for kf in keyframes[:10]:
        if kf not in global_keyframe_poses:
            continue
        est_t = pose_translation_np(global_keyframe_poses[kf])
        gt_center = camera_center(*gt_poses[kf])
        print(f"KF {kf} err est_t vs gt_center: {np.linalg.norm(est_t - gt_center):.3f} m")


def debug_scale_drift(keyframes, global_keyframe_poses):
    """
    Calculates scale drift ratios between estimated and ground truth path step sizes.
    """
    gt_poses = read_ground_truth_poses()
    est_positions = [pose_translation_np(global_keyframe_poses[kf]) for kf in keyframes if kf in global_keyframe_poses]
    gt_positions = [camera_center(*gt_poses[kf]) for kf in keyframes if kf in global_keyframe_poses]

    est_steps = np.linalg.norm(np.diff(np.array(est_positions), axis=0), axis=1)
    gt_steps = np.linalg.norm(np.diff(np.array(gt_positions), axis=0), axis=1)

    print("\n[Debug scale drift]")
    print("Mean estimated step:", np.mean(est_steps))
    print("Mean GT step:", np.mean(gt_steps))
    print("Scale ratio est/gt:", np.sum(est_steps) / np.sum(gt_steps))


# =============================================================================
# GRAPH & BUNDLE PLOTTING UTILITIES
# =============================================================================

def reprojection_error_graph(frame_indices, left_reprojection_errors, right_reprojection_errors, output_dir):
    """
    Plots stereo reprojection errors across frame indices.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, left_reprojection_errors, label="Left camera reprojection error", color="#1f77b4")
    plt.plot(frame_indices, right_reprojection_errors, label="Right camera reprojection error", color="#ff7f0e")
    plt.title("Right camera reprojection error")
    plt.xlabel("Frame index")
    plt.ylabel("Reprojection error (L2 norm) - pixels")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, "reprojection_error_graph.png"), dpi=300)
    plt.close()


def factor_error_graph(frame_indices, factor_errors, output_dir):
    """
    Plots GTSAM factor cost graph evolution across frame indices.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, factor_errors, color="red", marker='s')
    plt.title("Factor error graph")
    plt.xlabel("Frame index")
    plt.ylabel("Factor Error")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "factor_error_graph.png"), dpi=300)
    plt.close()


def bundle1_3D(window_frames, cam_positions, result, axis_length, output_dir):
    """
    Renders 3D local bundle trajectory visualization with oriented pose axes.
    """
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')

    xs_plot, ys_plot, zs_plot = cam_positions[:, 2], cam_positions[:, 0], -cam_positions[:, 1]
    ax3d.plot(xs_plot, ys_plot, zs_plot, 'k--', linewidth=1.5, zorder=1)

    for i in range(len(window_frames)):
        pose = result.atPose3(symbol('c', window_frames[i]))
        R = pose.rotation().matrix()
        cx, cy, cz = xs_plot[i], ys_plot[i], zs_plot[i]

        ax3d.scatter(cx, cy, cz, color='black', s=15, zorder=2)
        ax3d.plot([cx, cx + axis_length * R[2, 0]], [cy, cy + axis_length * R[0, 0]], [cz, cz - axis_length * R[1, 0]], color='r', linewidth=1.5)
        ax3d.plot([cx, cx - axis_length * R[2, 1]], [cy, cy - axis_length * R[0, 1]], [cz, cz + axis_length * R[1, 1]], color='g', linewidth=1.5)
        ax3d.plot([cx, cx + axis_length * R[2, 2]], [cy, cy + axis_length * R[0, 2]], [cz, cz - axis_length * R[1, 2]], color='b', linewidth=1.5)

    ax3d.set_title("Local Window Bundle Adjustment: 3D Optimized Trajectory\n", fontsize=11, fontweight='bold')
    ax3d.set_xlabel("Z (Forward) [m]")
    ax3d.set_ylabel("X (Right) [m]")
    ax3d.set_zlabel("-Y (Up) [m]")
    ax3d.set_xlim([0, 6])
    ax3d.set_ylim([-2, 2])
    ax3d.set_zlim([-2, 2])
    ax3d.view_init(elev=14, azim=-72)
    plt.savefig(os.path.join(output_dir, "task_5_3_bundle1_3D.png"), dpi=200, bbox_inches='tight')
    plt.close()


def marginal_covariances(window_frames, graph, result, output_dir):
    """
    Plots GTSAM 3D trajectory with marginal covariance ellipsoids on camera poses.
    """
    fig_cov = plt.figure(figsize=(8, 6))
    ax_cov = fig_cov.add_subplot(111, projection='3d')
    ax_cov.set_title("Plot Trajectory\nGTSAM Factor Graph State with Marginal Covariances\n", fontsize=11, fontweight='bold')

    try:
        marginals = gtsam.Marginals(graph, result)
        for f_id in window_frames:
            pose_key = symbol('c', f_id)
            gtsam_plot.plot_pose3_on_axes(ax_cov, result.atPose3(pose_key), axis_length=0.4, P=marginals.marginalCovariance(pose_key))
    except Exception as e:
        print(f"[Warning] Covariance layout fallback: {e}")
        for f_id in window_frames:
            gtsam_plot.plot_pose3_on_axes(ax_cov, result.atPose3(symbol('c', f_id)), axis_length=0.4)

    ax_cov.set_xlabel("X axis")
    ax_cov.set_ylabel("Y axis")
    ax_cov.set_zlabel("Z axis")
    ax_cov.view_init(elev=20, azim=-35)
    plt.savefig(os.path.join(output_dir, "task_5_3_marginal_covariances.png"), dpi=200, bbox_inches='tight')
    plt.close()


def bundle1_2D_full(window_frames, initial_cam_positions, cam_positions, lm_filtered, output_dir):
    """
    Plots top-down (X-Z) overview of camera trajectories and triangulated landmarks.
    """
    fig2d_full, ax2d_full = plt.subplots(figsize=(8, 8))

    if len(lm_filtered) > 0:
        ax2d_full.scatter(lm_filtered[:, 0], lm_filtered[:, 2], s=2, c='orange', alpha=0.6, label=f'Landmarks ({len(lm_filtered)})')

    ax2d_full.plot(initial_cam_positions[:, 0], initial_cam_positions[:, 2], 'b-+', alpha=0.6, label='Initial Cameras (PnP)')
    ax2d_full.plot(cam_positions[:, 0], cam_positions[:, 2], 'r-^', markersize=6, linewidth=1.5, label='Optimized Cameras (BA)')

    for i, f_id in enumerate(window_frames):
        ax2d_full.annotate(str(f_id), (cam_positions[i, 0], cam_positions[i, 2]), textcoords="offset points", xytext=(4, 4), fontsize=7, color='darkred')

    ax2d_full.set_title("Top-Down View (X-Z) of Bundle Window\nAll Cameras & Landmarks", fontsize=11, fontweight='bold')
    ax2d_full.set_xlabel("X [m]")
    ax2d_full.set_ylabel("Z (Forward) [m]")
    ax2d_full.set_xlim([-50, 50])
    ax2d_full.set_ylim([-10, 100])
    ax2d_full.grid(True, alpha=0.5)
    ax2d_full.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, "task_5_3_bundle1_2D_full.png"), dpi=150, bbox_inches='tight')
    plt.close()


def bundle1_2D_zoomed(initial_cam_positions, cam_positions, output_dir):
    """
    Plots zoomed 2D top-down trajectory comparison of local bundle optimization.
    """
    fig2d_zoom, ax2d_zoom = plt.subplots(figsize=(6, 7))
    ax2d_zoom.plot(initial_cam_positions[:, 0], initial_cam_positions[:, 2], 'b-o', alpha=0.5, markersize=4, label='Initial (PnP)')
    ax2d_zoom.plot(cam_positions[:, 0], cam_positions[:, 2], 'r-o', markersize=4, label='Optimized (BA)')
    ax2d_zoom.set_title("Local Window Bundle Adjustment: 2D Bird's-Eye View Trajectory\n(Trajectory zoomed)")
    ax2d_zoom.set_xlabel("X Coordinate (East) [m]")
    ax2d_zoom.set_ylabel("Z Coordinate (North) [m]")
    ax2d_zoom.set_xlim([-2, 2])
    ax2d_zoom.set_ylim([-2, 7])
    ax2d_zoom.grid(True, alpha=0.3)
    ax2d_zoom.legend()
    plt.savefig(os.path.join(output_dir, "task_5_3_bundle1_2D_zoomed.png"), dpi=150)
    plt.close()


# =============================================================================
# EXERCISE 6: POSE GRAPH OPTIMIZATION & COVARIANCE PROPAGATION
# =============================================================================

def covariance_to_noise_model(cov, min_sigma=1e-6):
    """
    Converts 6x6 covariance matrix to regularized GTSAM Gaussian noise model.
    """
    cov = np.asarray(cov, dtype=np.float64)
    if cov.shape != (6, 6):
        raise ValueError(f"Expected 6x6 covariance, got {cov.shape}")
    cov = 0.5 * (cov + cov.T) + np.eye(6) * (min_sigma ** 2)
    return gtsam.noiseModel.Gaussian.Covariance(cov)


def find_component_start_keyframes(relative_poses):
    """
    Identifies starting keyframe nodes for disconnected sub-graphs.
    """
    edges = sorted(relative_poses.keys())
    if not edges:
        return []
    component_starts = [edges[0][0]]
    for (a, b), (c, d) in zip(edges[:-1], edges[1:]):
        if b != c:
            component_starts.append(c)
    return component_starts


def build_pose_graph(relative_poses, relative_covs):
    """
    Constructs GTSAM NonlinearFactorGraph composed of BetweenFactorPose3 constraints.
    """
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
    """
    Generic helper to extract frame IDs and 3D positions sorted by keyframe index from GTSAM Values.
    """
    items = []
    for key in values.keys():
        sym = gtsam.Symbol(key)
        if sym.chr() == ord("c"):
            items.append((sym.index(), pose_translation_np(values.atPose3(key))))
    items.sort(key=lambda x: x[0])
    return [x[0] for x in items], np.array([x[1] for x in items]) if items else np.empty((0, 3))


def extract_pose_graph_positions(values):
    """
    Extracts sorted keyframe indices and translation vectors from GTSAM Values.
    """
    return extract_trajectory_and_ids(values)


def plot_pose_graph_trajectory(values, title, output_path):
    """
    Plots top-down 2D trajectory of pose graph state.
    """
    _, positions = extract_pose_graph_positions(values)
    plt.figure(figsize=(10, 8))
    plt.plot(positions[:, 0], positions[:, 2], "bo-", markersize=3, linewidth=1.5)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


def plot_pose_graph_with_covariances(values, marginals, title, output_path, covariance_step=10, covariance_scale=1.0):
    """
    Visualizes 3D pose graph trajectory with marginal translation covariance ellipsoids.
    """
    frame_ids, positions = extract_pose_graph_positions(values)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], linestyle="--", marker=".", markersize=3, linewidth=1.0, label="Optimized keyframes")

    for frame_id in frame_ids[::covariance_step]:
        key = symbol("c", frame_id)
        if not values.exists(key):
            continue
        p = pose_translation_np(values.atPose3(key))

        try:
            cov6 = marginals.marginalCovariance(key)
            cov_xyz = 0.5 * (cov6[3:6, 3:6] + cov6[3:6, 3:6].T)

            eigvals, eigvecs = np.linalg.eigh(cov_xyz)
            radii = np.minimum(covariance_scale * np.sqrt(np.maximum(eigvals, 0.0)), 20.0)

            u, v = np.linspace(0, 2 * np.pi, 18), np.linspace(0, np.pi, 9)
            xs = radii[0] * np.outer(np.cos(u), np.sin(v))
            ys = radii[1] * np.outer(np.sin(u), np.sin(v))
            zs = radii[2] * np.outer(np.ones_like(u), np.cos(v))

            ellipsoid = np.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], axis=0)
            rotated = eigvecs @ ellipsoid

            ax.plot_wireframe(
                rotated[0, :].reshape(xs.shape) + p[0],
                rotated[1, :].reshape(ys.shape) + p[1],
                rotated[2, :].reshape(zs.shape) + p[2],
                linewidth=0.4, alpha=0.45
            )
        except Exception as e:
            print(f"[Warning] Could not plot covariance for frame {frame_id}: {e}")

    ax.set_title(title)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.grid(True)
    ax.legend()
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def debug_target_frame_geometry(db, start_frame=2840, end_frame=2860, target_frame=2860, max_tracks_per_window=150, output_dir="./outputs"):
    """
    Analyzes depth distribution, disparities, and pixel spread for target frames in BA windows.
    """
    os.makedirs(output_dir, exist_ok=True)
    K_mat, P_left0, P_right0 = read_cameras()
    fx = K_mat[0, 0]
    baseline = init_gtsam_stereo_calibration().baseline()

    window_frames = list(range(start_frame, end_frame + 1))
    candidate_tracks = set()
    for f_id in window_frames:
        candidate_tracks.update(db.tracks(f_id))

    guaranteed_tracks = set()
    rng = random.Random(0)
    for f_id in window_frames:
        tracks_in_frame = list(db.tracks(f_id))
        if tracks_in_frame:
            guaranteed_tracks.update(rng.sample(tracks_in_frame, min(15, len(tracks_in_frame))))

    remaining_slots = max_tracks_per_window - len(guaranteed_tracks)
    all_candidates = list(set(candidate_tracks) - guaranteed_tracks)
    final_tracks = list(guaranteed_tracks) + (rng.sample(all_candidates, min(remaining_slots, len(all_candidates))) if remaining_slots > 0 and all_candidates else [])

    rows = []
    for track_id in final_tracks:
        track_frames = [f for f in db.frames(track_id) if f in window_frames]
        if target_frame not in track_frames:
            continue

        obs_t = db.observation(target_frame, track_id)
        if not valid_stereo_obs(obs_t, min_disp=1.0):
            continue

        X_cam = triangulate_point_linear(
            np.array([obs_t.x_left, obs_t.y]),
            np.array([obs_t.x_right, obs_t.y]),
            P_left0, P_right0
        )
        disparity = obs_t.x_left - obs_t.x_right

        rows.append({
            "track_id": track_id, "num_frames_in_window": len(track_frames),
            "x": float(obs_t.x_left), "y": float(obs_t.y), "x_right": float(obs_t.x_right),
            "disparity": float(disparity),
            "depth_triangulated": float(X_cam[2]) if np.all(np.isfinite(X_cam)) else np.nan,
            "depth_formula": float(fx * baseline / disparity),
        })

    if not rows:
        print("No valid target-frame observations found.")
        return rows

    finite_depths = np.array([r["depth_triangulated"] for r in rows], dtype=float)
    finite_depths = finite_depths[np.isfinite(finite_depths)]

    print(f"\nDebug geometry for frame {target_frame} ({start_frame}->{end_frame})")
    print(f"Depth median: {np.median(finite_depths):.3f} m, max: {np.max(finite_depths):.3f} m")

    return rows


def run_and_plot_prior_sensitivity(db, c0_idx, ck_idx, output_dir):
    """
    Evaluates factor graph convergence sensitivity under varying anchor prior noise scales.
    """
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
    """
    Calculates marginal covariance and relative pose between keyframes using Schur Complements.
    """
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
    """
    Computes sequential keyframe-to-keyframe pose graph edge constraints and covariances.
    """
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
    """
    Filters out unstable pose graph edges containing negative or ill-conditioned covariance diagonals.
    """
    cleaned_poses, cleaned_covs = {}, {}
    for edge, cov in relative_covs.items():
        if np.any(np.diag(cov) <= 0) or np.any(np.isnan(cov)):
            print(f"Skipping bad covariance edge: {edge}")
            continue
        cleaned_poses[edge] = relative_poses[edge]
        cleaned_covs[edge] = relative_covs[edge]
    return cleaned_poses, cleaned_covs


def build_and_initialize_pose_graph(cleaned_poses, cleaned_covs):
    """
    Constructs GTSAM pose graph and computes initial trajectory estimates.
    """
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
    """
    Solves nonlinear pose graph optimization using Levenberg-Marquardt optimizer.
    """
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    return result, gtsam.Marginals(graph, result)


def plot_pose_graph_2d_ellipses(result, marginals, output_path, step=1, sigma_scale=20):
    """
    Renders top-down 2D keyframe trajectory overlaid with 2D marginal confidence ellipses.
    """
    frame_ids, positions = extract_pose_graph_positions(result)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(positions[:, 0], positions[:, 2], label="optimised trajectory", color="#1f77b4", linewidth=1.2)
    ax.scatter(positions[0, 0], positions[0, 2], color="black", marker="s", s=80, label="c_0", zorder=5)

    for frame_id in frame_ids[::step]:
        key = symbol("c", frame_id)
        if not result.exists(key):
            continue
        p = pose_translation_np(result.atPose3(key))

        try:
            cov6 = marginals.marginalCovariance(key)
            cov_xz = cov6[3:6, 3:6][np.ix_([0, 2], [0, 2])]

            eigvals, eigvecs = np.linalg.eigh(cov_xz)
            eigvals = np.maximum(eigvals, 0.0)

            width = 2 * sigma_scale * np.sqrt(eigvals[0])
            height = 2 * sigma_scale * np.sqrt(eigvals[1])
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

            ax.add_patch(Ellipse(
                xy=(p[0], p[2]), width=width, height=height, angle=angle,
                edgecolor="red", facecolor="none", alpha=0.4, linewidth=0.8
            ))
        except Exception:
            pass

    ax.set_title("Q6.2: keyframes with marginal covariances", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z — forward (m)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# =============================================================================
# EXERCISE 7: LOOP CLOSURE DETECTION & POSE GRAPH SLAM
# =============================================================================

def _shortest_path_pose_graph(relative_poses, relative_covs):
    """
    Constructs a NetworkX graph weighted by the scalar trace of relative edge covariances.
    """
    G = nx.Graph()
    for (start_kf, end_kf), cov in relative_covs.items():
        G.add_edge(start_kf, end_kf, weight=float(np.trace(cov)))
    return G


def _get_edge_pose_and_cov(u, v, relative_poses, relative_covs):
    """
    Retrieves edge constraints between keyframe nodes, automatically handling directional inversions.
    """
    if (u, v) in relative_poses:
        return relative_poses[(u, v)], relative_covs[(u, v)]

    if (v, u) in relative_poses:
        pose_uv = relative_poses[(v, u)].inverse()
        Ad_inv = pose_uv.AdjointMap()
        return pose_uv, Ad_inv @ relative_covs[(v, u)] @ Ad_inv.T

    raise KeyError(f"No pose-graph edge between keyframes {u} and {v}")


def detect_loop_closure_candidates(relative_poses, relative_covs, keyframes, mahalanobis_threshold):
    """
    Identifies candidate loop closures below a Mahalanobis distance threshold using Dijkstra shortest paths.
    """
    G = _shortest_path_pose_graph(relative_poses, relative_covs)
    loop_candidates = {}
    total_candidates_counter = 0

    for n_idx, c_n in enumerate(keyframes):
        if n_idx == 0 or c_n not in G:
            continue

        loop_candidates[c_n] = []
        for c_i in keyframes[:n_idx]:
            if c_i not in G:
                continue

            try:
                path = nx.shortest_path(G, source=c_n, target=c_i, weight="weight")
                suffix_pose = gtsam.Pose3()
                Sigma_rel = np.zeros((6, 6))

                for u, v in reversed(list(zip(path[:-1], path[1:]))):
                    edge_pose, edge_cov = _get_edge_pose_and_cov(u, v, relative_poses, relative_covs)
                    Ad_suffix_inv = suffix_pose.inverse().AdjointMap()
                    Sigma_rel += Ad_suffix_inv @ edge_cov @ Ad_suffix_inv.T
                    suffix_pose = edge_pose.compose(suffix_pose)

                xi_ni = gtsam.Pose3.Logmap(suffix_pose)
                Sigma_inv = np.linalg.inv(Sigma_rel + np.eye(6) * 1e-6)
                mahalanobis_dist = xi_ni.T @ Sigma_inv @ xi_ni

                if mahalanobis_dist < mahalanobis_threshold:
                    loop_candidates[c_n].append({
                        "candidate_kf": c_i, "mahalanobis_distance": mahalanobis_dist,
                        "relative_pose_estimate": suffix_pose, "relative_covariance": Sigma_rel, "path": path,
                    })
                    total_candidates_counter += 1
            except Exception:
                continue

    return loop_candidates, total_candidates_counter


def plot_loop_candidates(keyframes, candidates, optimized_values, output_dir="."):
    """
    Plots trajectory with candidate loop closure shortcut connections overlaid.
    """
    kf_positions = {
        kf: (optimized_values.atPose3(symbol('c', kf)).translation()[0], optimized_values.atPose3(symbol('c', kf)).translation()[2])
        for kf in keyframes if optimized_values.exists(symbol('c', kf))
    }

    positions_array = np.array([kf_positions[kf] for kf in keyframes if kf in kf_positions])

    plt.figure(figsize=(12, 9))
    plt.plot(positions_array[:, 0], positions_array[:, 1], color='gray', linestyle='-', alpha=0.5, label='Estimated Trajectory')
    plt.scatter(positions_array[:, 0], positions_array[:, 1], color='black', s=5, alpha=0.3)

    for c_n, cands in candidates.items():
        if not cands or c_n not in kf_positions:
            continue
        x_n, z_n = kf_positions[c_n]
        plt.scatter(x_n, z_n, color='green', s=30, zorder=3)

        for cand in cands:
            c_i = cand["candidate_kf"]
            if c_i in kf_positions:
                x_i, z_i = kf_positions[c_i]
                plt.scatter(x_i, z_i, color='magenta', s=15, zorder=3)
                plt.plot([x_n, x_i], [z_n, z_i], color='red', linestyle='--', alpha=0.6, linewidth=1.0)

    plt.title("Q7.1: Detected Loop Closure Candidates on ESTIMATED Trajectory", fontsize=12, fontweight='bold')
    plt.xlabel("X (Width) [m]")
    plt.ylabel("Z (Depth / Forward) [m]")
    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "../outputs/task_7_1_loop_candidates_trajectory.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def verify_loop_closures_consensus(db, loop_candidates, inlier_ratio_threshold, output_dir=None):
    """
    Verifies candidate loops by performing AKAZE matching and Fundamental matrix RANSAC filtering.
    """
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    verified_loops = {}
    total_verified_loops = 0
    MIN_ABSOLUTE_INLIERS = 20

    for c_n, cands in loop_candidates.items():
        if not cands:
            continue

        verified_loops[c_n] = []
        img_n, _ = read_images(c_n)
        kp_n, des_n = get_akaze_features(img_n)

        for cand in cands:
            c_i = cand["candidate_kf"]
            img_i, _ = read_images(c_i)
            kp_i, des_i = get_akaze_features(img_i)

            if des_n is None or des_i is None:
                continue

            knn_matches = bf_matcher.knnMatch(des_i, des_n, k=2)
            good_matches = [m for m, n in knn_matches if m.distance < 0.7 * n.distance]
            if len(good_matches) < 4:
                continue

            pts_i = np.array([kp_i[m.queryIdx].pt for m in good_matches], dtype=np.float32)
            pts_n = np.array([kp_n[m.trainIdx].pt for m in good_matches], dtype=np.float32)

            _, mask = cv2.findFundamentalMat(pts_i, pts_n, cv2.FM_RANSAC, 3.0, 0.99)
            if mask is None:
                continue

            inliers_count = int(np.sum(mask))
            inlier_ratio = inliers_count / len(good_matches)

            if inlier_ratio >= inlier_ratio_threshold and inliers_count >= MIN_ABSOLUTE_INLIERS:
                inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]
                verified_loops[c_n].append({"candidate_kf": c_i, "inliers_count": inliers_count, "inlier_matches": inlier_matches})
                total_verified_loops += 1

    return verified_loops, total_verified_loops


def estimate_loop_relative_pose_bundle(db, c_i, c_n, inlier_matches=None, max_landmarks=120, min_landmarks=8, output_dir="."):
    """
    Estimates relative pose and covariance across confirmed loop closure frames using 2-frame BA.
    """
    K_gtsam = init_gtsam_stereo_calibration()
    K_mat, _, _ = read_cameras()

    data_i = run_single_pair(c_i, display=False, plot_3d=False)
    data_n = run_single_pair(c_n, display=False, plot_3d=False)

    if inlier_matches is None:
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = bf_matcher.knnMatch(data_i["des_left"], data_n["des_left"], k=2)
        good = [m for m, nn in knn if m.distance < 0.7 * nn.distance]
        pts_i = np.array([data_i["kp_left"][m.queryIdx].pt for m in good], dtype=np.float32)
        pts_n = np.array([data_n["kp_left"][m.trainIdx].pt for m in good], dtype=np.float32)
        _, mask = cv2.findFundamentalMat(pts_i, pts_n, cv2.FM_RANSAC, 3.0, 0.99)
        inlier_matches = [good[k] for k in range(len(good)) if mask[k][0] == 1]

    i_q_to_pt = {m.queryIdx: idx for idx, m in enumerate(data_i["stereo_inliers"])}
    i_q_to_stereo = {m.queryIdx: m for m in data_i["stereo_inliers"]}
    n_q_to_stereo = {m.queryIdx: m for m in data_n["stereo_inliers"]}

    usable = []
    for m in inlier_matches:
        if m.queryIdx in i_q_to_pt and m.queryIdx in i_q_to_stereo and m.trainIdx in n_q_to_stereo:
            X_i = np.array(data_i["points_3d"][i_q_to_pt[m.queryIdx]], dtype=np.float64).reshape(3)
            if np.all(np.isfinite(X_i)) and 2.0 < X_i[2] < 120.0:
                usable.append((m, X_i))

    if len(usable) < min_landmarks:
        raise RuntimeError(f"Loop {c_i}->{c_n}: only {len(usable)} usable stereo landmarks")

    usable = sorted(usable, key=lambda item: item[0].distance)[:max_landmarks]

    obj_pts = np.array([X for _, X in usable], dtype=np.float32)
    img_pts = np.array([data_n["kp_left"][m.trainIdx].pt for m, _ in usable], dtype=np.float32)
    success, rvec, tvec, pnp_inliers = cv2.solvePnPRansac(obj_pts, img_pts, K_mat, None, iterationsCount=200, reprojectionError=3.0, confidence=0.99)

    if not success or pnp_inliers is None or len(pnp_inliers) < min_landmarks:
        raise RuntimeError(f"Loop {c_i}->{c_n}: PnP failed")

    R_ni, _ = cv2.Rodrigues(rvec)
    t_ni = tvec.reshape(3)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    key_i, key_n = symbol("c", int(c_i)), symbol("c", int(c_n))

    initial.insert(key_i, gtsam.Pose3())
    initial.insert(key_n, gtsam.Pose3(gtsam.Rot3(R_ni.T), gtsam.Point3(*(-R_ni.T @ t_ni).reshape(3))))
    graph.add(gtsam.PriorFactorPose3(key_i, gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)))

    meas_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(2.0), gtsam.noiseModel.Isotropic.Sigma(3, 1.0))
    kept_indices = set(int(x[0]) for x in pnp_inliers.reshape(-1, 1))

    for local_idx, (m, X_i) in enumerate(usable):
        if local_idx not in kept_indices:
            continue
        point_key = symbol("q", int(10_000_000 + c_i * 10_000 + c_n * 10 + local_idx))
        initial.insert(point_key, gtsam.Point3(float(X_i[0]), float(X_i[1]), float(X_i[2])))

        kp_li, kp_ri = data_i["kp_left"][m.queryIdx], data_i["kp_right"][i_q_to_stereo[m.queryIdx].trainIdx]
        kp_ln, kp_rn = data_n["kp_left"][m.trainIdx], data_n["kp_right"][n_q_to_stereo[m.trainIdx].trainIdx]

        graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(kp_li.pt[0], kp_ri.pt[0], kp_li.pt[1]), meas_noise, key_i, point_key, K_gtsam))
        graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(kp_ln.pt[0], kp_rn.pt[0], kp_ln.pt[1]), meas_noise, key_n, point_key, K_gtsam))

    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    marginals = gtsam.Marginals(graph, result)

    keys = gtsam.KeyVector()
    keys.append(key_i)
    keys.append(key_n)

    rel_cov = np.linalg.inv(marginals.jointMarginalInformation(keys).fullMatrix()[-6:, -6:])

    return {
        "start_kf": int(c_i), "end_kf": int(c_n),
        "relative_pose": result.atPose3(key_i).between(result.atPose3(key_n)),
        "relative_covariance": rel_cov, "graph": graph, "result": result
    }


def estimate_verified_loop_relative_poses(db, verified_loops, output_dir="."):
    """
    Executes relative pose estimation for all verified loop candidate pairs.
    """
    loop_measurements = []
    for c_n, loops in verified_loops.items():
        for loop in loops:
            try:
                m = estimate_loop_relative_pose_bundle(db, loop["candidate_kf"], c_n, inlier_matches=loop.get("inlier_matches"), output_dir=output_dir)
                loop_measurements.append(m)
            except Exception as e:
                print(f"[Q7.3] Loop {loop['candidate_kf']}->{c_n}: FAILED ({e})")
    return loop_measurements


def add_loop_closures_and_optimize(cleaned_poses, cleaned_covs, loop_measurements, output_dir=".", snapshot_count=4):
    """
    Adds loop closure constraints to full pose graph and resolves global SLAM optimization.
    """
    graph0, initial0 = build_and_initialize_pose_graph(cleaned_poses, cleaned_covs)
    no_loop_result, no_loop_marginals = optimize_pose_graph(graph0, initial0)

    final_graph, _ = build_and_initialize_pose_graph(cleaned_poses, cleaned_covs)
    relative_poses_lc, relative_covs_lc = dict(cleaned_poses), dict(cleaned_covs)
    current_initial, final_result, final_marginals = no_loop_result, no_loop_result, no_loop_marginals

    for meas in loop_measurements:
        edge = (meas["end_kf"], meas["start_kf"])
        meas_rel = meas["relative_pose"].inverse()
        loop_cov = 0.5 * (meas["relative_covariance"] + meas["relative_covariance"].T) + np.eye(6) * 1e-9

        final_graph.add(gtsam.BetweenFactorPose3(
            symbol("c", meas["end_kf"]), symbol("c", meas["start_kf"]),
            meas_rel, gtsam.noiseModel.Gaussian.Covariance(loop_cov)
        ))

        relative_poses_lc[edge] = meas_rel
        relative_covs_lc[edge] = loop_cov

        final_result, final_marginals = optimize_pose_graph(final_graph, current_initial)
        current_initial = final_result

    plot_pose_graph_2d_ellipses(final_result, final_marginals, os.path.join(output_dir, "task_7_5_final_loop_closed.png"), step=5)

    return {
        "no_loop_graph": graph0, "no_loop_result": no_loop_result, "no_loop_marginals": no_loop_marginals,
        "loop_graph": final_graph, "loop_result": final_result, "loop_marginals": final_marginals,
        "relative_poses_with_loops": relative_poses_lc, "relative_covs_with_loops": relative_covs_lc,
    }


def plot_pose_graph_vs_ground_truth(no_loop_values, loop_values, output_dir="."):
    """
    Plots optimized trajectories with and without loop closures against ground truth.
    """
    ids_no, pos_no = extract_pose_graph_positions(no_loop_values)
    _, pos_lc = extract_pose_graph_positions(loop_values)
    gt = read_ground_truth_poses()
    gt_pts = np.array([camera_center(*gt[f]) for f in ids_no if f < len(gt)])

    plt.figure(figsize=(10, 8))
    plt.plot(pos_no[:, 0], pos_no[:, 2], "o-", markersize=3, linewidth=1, label="without loop closures")
    plt.plot(pos_lc[:, 0], pos_lc[:, 2], "o-", markersize=3, linewidth=1, label="with loop closures")
    if len(gt_pts) == len(pos_no):
        plt.plot(gt_pts[:, 0], gt_pts[:, 2], "--", linewidth=2, label="ground truth")

    plt.title("Q7.5: Pose Graph vs Ground Truth")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_7_5_pose_graph_vs_ground_truth.png"), dpi=200)


def plot_absolute_location_error(no_loop_values, loop_values, output_dir="."):
    """
    Plots keyframe position errors relative to ground truth before and after loop closures.
    """
    ids_no, pos_no = extract_pose_graph_positions(no_loop_values)
    ids_lc, pos_lc = extract_pose_graph_positions(loop_values)
    gt = read_ground_truth_poses()

    gt_no = np.array([camera_center(*gt[f]) for f in ids_no if f < len(gt)])
    gt_lc = np.array([camera_center(*gt[f]) for f in ids_lc if f < len(gt)])

    plt.figure(figsize=(12, 5))
    plt.plot(ids_no[:len(gt_no)], np.linalg.norm(pos_no[:len(gt_no)] - gt_no, axis=1), marker="o", linewidth=1, label="without loop closures")
    plt.plot(ids_lc[:len(gt_lc)], np.linalg.norm(pos_lc[:len(gt_lc)] - gt_lc, axis=1), marker="o", linewidth=1, label="with loop closures")
    plt.title("Q7.5: Absolute Location Error")
    plt.xlabel("Keyframe")
    plt.ylabel("Position error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_7_5_absolute_location_error.png"), dpi=200)


def plot_location_uncertainty_size(no_loop_values, no_loop_marginals, loop_values, loop_marginals, output_dir="."):
    """
    Plots marginal position uncertainty area over keyframes before and after loop closures.
    """
    def _uncertainty_trace(values, marginals):
        ids, _ = extract_pose_graph_positions(values)
        sizes, good_ids = [], []
        for f in ids:
            try:
                cov6 = marginals.marginalCovariance(symbol("c", int(f)))
                cov_xz = cov6[np.ix_([3, 5], [3, 5])]
                sizes.append(float(np.sqrt(max(np.linalg.det(cov_xz), 0.0))))
                good_ids.append(f)
            except Exception:
                pass
        return good_ids, np.array(sizes)

    ids_no, unc_no = _uncertainty_trace(no_loop_values, no_loop_marginals)
    ids_lc, unc_lc = _uncertainty_trace(loop_values, loop_marginals)

    plt.figure(figsize=(12, 5))
    plt.plot(ids_no, unc_no, marker="o", linewidth=1, label="without loop closures")
    plt.plot(ids_lc, unc_lc, marker="o", linewidth=1, label="with loop closures")
    plt.title("Q7.5: Location Uncertainty Size")
    plt.xlabel("Keyframe")
    plt.ylabel(r"$\sqrt{\det(\Sigma_{xz})}$ [m²]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_7_5_location_uncertainty_size.png"), dpi=200)