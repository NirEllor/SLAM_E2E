import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'dataset' / 'dataset' / 'sequences' / '00'

#--------------------------------------ex1---------------------------------------------------------
def read_images(idx):
    img_name = f'{idx:06d}.png'
    p1, p2 = DATA_PATH / 'image_0' / img_name, DATA_PATH / 'image_1' / img_name
    img1, img2 = cv2.imread(str(p1), 0), cv2.imread(str(p2), 0)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not find images at {DATA_PATH}")
    return img1, img2

def get_orb_features(img, n_features=700):
    orb = cv2.ORB_create(nfeatures=n_features)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def plot_stereo_side_by_side(img_l, img_r, title="Stereo Pair"):
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1); plt.imshow(img_l, cmap='gray'); plt.title(f"{title} - Left"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(img_r, cmap='gray'); plt.title(f"{title} - Right"); plt.axis('off')
    plt.tight_layout()

def draw_matches_custom(img1, kp1, img2, kp2, matches, title, num=20):
    if len(matches) > num:
        matches = random.sample(matches, num)

    img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.imshow(img_rgb)
    plt.axis('off')

#--------------------------------------ex2---------------------------------------------------------

def compute_rectified_stereo_deviations(kp_left, kp_right, matches):
    deviations = []

    for match in matches:
        _, y_left = kp_left[match.queryIdx].pt
        _, y_right = kp_right[match.trainIdx].pt
        deviations.append(abs(y_left - y_right))

    return np.array(deviations)


def plot_deviation_histogram(deviations):
    plt.figure(figsize=(10, 6))
    plt.hist(deviations, bins=50)
    plt.xlabel("deviation from rectified stereo pattern")
    plt.ylabel("Number of matches")
    plt.title("Histogram of deviations from rectified stereo pattern")
    plt.tight_layout()


def print_large_deviation_percentage(deviations, threshold=2):
    num_matches = len(deviations)
    num_bad_matches = np.sum(deviations > threshold)

    percentage = 100 * num_bad_matches / num_matches

    print(f"Matches with deviation > {threshold} pixels: {num_bad_matches}/{num_matches}")
    print(f"Percentage of matches with deviation > {threshold} pixels: {percentage:.2f}%")


def split_matches_by_rectified_pattern(kp_left, kp_right, matches, threshold=2):
    inliers = []
    outliers = []

    for match in matches:
        _, y_left = kp_left[match.queryIdx].pt
        _, y_right = kp_right[match.trainIdx].pt

        if abs(y_left - y_right) <= threshold:
            inliers.append(match)
        else:
            outliers.append(match)

    return inliers, outliers


def draw_inliers_outliers(img1, kp1, img2, kp2, inliers, outliers):
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
    calib_path = DATA_PATH / 'calib.txt'

    with open(calib_path) as f:
        l1 = f.readline().split()[1:]
        l2 = f.readline().split()[1:]

    p1 = np.array([float(i) for i in l1]).reshape(3, 4)
    p2 = np.array([float(i) for i in l2]).reshape(3, 4)

    k = p1[:, :3]

    return k, p1, p2


def get_matched_points(kp1, kp2, matches):
    points1 = []
    points2 = []

    for match in matches:
        points1.append(kp1[match.queryIdx].pt)
        points2.append(kp2[match.trainIdx].pt)

    return np.array(points1), np.array(points2)


def triangulate_point_linear(p1, p2, m1, m2):
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
    X = X_homogeneous[:3] / X_homogeneous[3]

    return X

def triangulate_points_linear(points1, points2, m1, m2):
    points_3d = []

    for p1, p2 in zip(points1, points2):
        X = triangulate_point_linear(p1, p2, m1, m2)
        points_3d.append(X)

    return np.array(points_3d)

def triangulate_points_opencv(points1, points2, m1, m2):
    points1_t = points1.T
    points2_t = points2.T

    points_4d = cv2.triangulatePoints(m1, m2, points1_t, points2_t)

    points_3d = points_4d[:3, :] / points_4d[3, :]

    return points_3d.T

def plot_3d_points(points_3d, title="3D Point Cloud"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=2)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()


def median_3d_distance(points_a, points_b):
    distances = np.linalg.norm(points_a - points_b, axis=1)
    return np.median(distances)

#--------------------------------------ex3---------------------------------------------------------

def plot_four_cameras(R, t, baseline=0.54):
    """
    Plots the relative 2D positions (X, Z) of the four cameras from a bird's-eye view.
    """
    import numpy as np
    
    # 1. Position of Frame 0 cameras (Time t0)
    c_left_0 = np.array([0.0, 0.0])
    c_right_0 = np.array([baseline, 0.0])
    
    # 2. Compute position of left_1 using the formula from Q3: C = -R^T * t
    c_world_3d = -np.dot(R.T, t).flatten()
    c_left_1 = np.array([c_world_3d[0], c_world_3d[2]]) # Extract X (width) and Z (depth)
    
    # 3. Compute position of right_1 (shifted by baseline along the camera's local X-axis)
    right_shift_world = R.T[:, 0] * baseline
    c_right_1 = c_left_1 + np.array([right_shift_world[0], right_shift_world[2]])

    # 4. Plotting the cameras using matplotlib
    plt.figure(figsize=(8, 6))
    
    # Plot Frame 0 cameras as triangles (pointing up)
    plt.scatter(c_left_0[0], c_left_0[1], color='blue', marker='^', s=150, label='Left Camera (t0)')
    plt.scatter(c_right_0[0], c_right_0[1], color='cyan', marker='^', s=150, label='Right Camera (t0)')
    
    # Plot Frame 1 cameras as squares
    plt.scatter(c_left_1[0], c_left_1[1], color='red', marker='s', s=120, label='Left Camera (t1)')
    plt.scatter(c_right_1[0], c_right_1[1], color='orange', marker='s', s=120, label='Right Camera (t1)')
    
    # Draw dashed lines connecting the stereo baseline pairs
    plt.plot([c_left_0[0], c_right_0[0]], [c_left_0[1], c_right_0[1]], 'b--', alpha=0.5)
    plt.plot([c_left_1[0], c_right_1[0]], [c_left_1[1], c_right_1[1]], 'r--', alpha=0.5)

    # Graph formatting
    plt.title("Relative Positions of the Four Cameras (Bird's-Eye View)")
    plt.xlabel("X (Width / Lateral movement in meters)")
    plt.ylabel("Z (Depth / Forward movement in meters)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.axis('equal') # Maintain aspect ratio so distance scales are identical on both axes
    
    print("\n[Plot] Generated camera relative position chart successfully.")


def project_point(P, X):
    """
    Project a 3D point X using camera matrix P.
    X is shape (3,), P is shape (3,4).
    Returns pixel (u, v).
    """
    X_h = np.append(X, 1.0)
    x_h = P @ X_h

    if abs(x_h[2]) < 1e-12:
        return np.array([np.nan, np.nan])

    return x_h[:2] / x_h[2]

def draw_temporal_supporters(img_left0, kp_left0,
                             img_left1, kp_left1,
                             supporters, non_supporters,
                             title="3.4: Supporters vs Non-supporters"):
    """
    Draw temporal matches left0 -> left1.
    Supporters are green, non-supporters are red.
    """
    img0 = cv2.cvtColor(img_left0, cv2.COLOR_GRAY2RGB)
    img1 = cv2.cvtColor(img_left1, cv2.COLOR_GRAY2RGB)

    h0, w0 = img0.shape[:2]
    canvas = np.hstack([img0, img1])

    def draw_match(match, color):
        pt0 = tuple(map(int, kp_left0[match.queryIdx].pt))
        pt1_raw = kp_left1[match.trainIdx].pt
        pt1 = (int(pt1_raw[0] + w0), int(pt1_raw[1]))

        cv2.circle(canvas, pt0, 3, color, -1)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.line(canvas, pt0, pt1, color, 1)

    for match in non_supporters:
        draw_match(match, (255, 0, 0))  # red

    for match in supporters:
        draw_match(match, (0, 255, 0))  # green

    plt.figure(figsize=(16, 8))
    plt.imshow(canvas)
    plt.title(title + " | green=supporters, red=outliers")
    plt.axis("off")

def build_pnp_correspondences(frame0_data, frame1_data, good_temporal_matches):
    """
    Build all valid 3D <-> 2D correspondences that exist in all four images.
    """

    correspondences = []

    # left0 keypoint -> (3D index, stereo match in frame0)
    left0_to_3d = {
        m.queryIdx: (idx, m)
        for idx, m in enumerate(frame0_data['stereo_inliers'])
    }

    # left1 keypoint -> stereo match in frame1
    left1_to_stereo = {
        m.queryIdx: m
        for m in frame1_data['stereo_inliers']
    }

    for temporal_match in good_temporal_matches:

        left0_idx = temporal_match.queryIdx
        left1_idx = temporal_match.trainIdx

        # Must exist in all four images
        if left0_idx not in left0_to_3d:
            continue

        if left1_idx not in left1_to_stereo:
            continue

        point_3d_idx, stereo_match0 = left0_to_3d[left0_idx]
        stereo_match1 = left1_to_stereo[left1_idx]

        X = frame0_data['points_3d'][point_3d_idx]

        corr = {
            'X': X,

            'left0_match': stereo_match0,
            'temporal_match': temporal_match,
            'right1_match': stereo_match1,

            'obs_left0':
                np.array(frame0_data['kp_left'][stereo_match0.queryIdx].pt),

            'obs_right0':
                np.array(frame0_data['kp_right'][stereo_match0.trainIdx].pt),

            'obs_left1':
                np.array(frame1_data['kp_left'][temporal_match.trainIdx].pt),

            'obs_right1':
                np.array(frame1_data['kp_right'][stereo_match1.trainIdx].pt)
        }

        correspondences.append(corr)

    return correspondences

def project_points_vectorized(P, X):
    """
    Project many 3D points using projection matrix P.
    X shape: (N, 3)
    returns: (N, 2)
    """
    X_h = np.hstack([X, np.ones((X.shape[0], 1))])
    x_h = (P @ X_h.T).T

    valid = np.abs(x_h[:, 2]) > 1e-12

    projected = np.full((X.shape[0], 2), np.nan)
    projected[valid] = x_h[valid, :2] / x_h[valid, 2:3]

    return projected


def evaluate_supporters(correspondences,
                        frame0_data,
                        frame1_data,
                        R,
                        t,
                        threshold=2):
    """
    Vectorized supporter evaluation for a given pose.
    """

    k, P_left0, P_right0 = read_cameras()

    t_stereo = np.linalg.inv(k) @ P_right0[:, 3]

    P_left1 = k @ np.hstack([R, t.reshape(3, 1)])

    P_right1 = k @ np.hstack([
        R,
        (t.reshape(3) + t_stereo).reshape(3, 1)
    ])

    X = np.array([c['X'] for c in correspondences], dtype=np.float64)

    obs_left0 = np.array([c['obs_left0'] for c in correspondences])
    obs_right0 = np.array([c['obs_right0'] for c in correspondences])
    obs_left1 = np.array([c['obs_left1'] for c in correspondences])
    obs_right1 = np.array([c['obs_right1'] for c in correspondences])

    proj_left0 = project_points_vectorized(P_left0, X)
    proj_right0 = project_points_vectorized(P_right0, X)
    proj_left1 = project_points_vectorized(P_left1, X)
    proj_right1 = project_points_vectorized(P_right1, X)

    err_left0 = np.linalg.norm(proj_left0 - obs_left0, axis=1)
    err_right0 = np.linalg.norm(proj_right0 - obs_right0, axis=1)
    err_left1 = np.linalg.norm(proj_left1 - obs_left1, axis=1)
    err_right1 = np.linalg.norm(proj_right1 - obs_right1, axis=1)

    supporter_mask = (
        (err_left0 < threshold) &
        (err_right0 < threshold) &
        (err_left1 < threshold) &
        (err_right1 < threshold)
    )

    inliers = [
        c for c, is_inlier in zip(correspondences, supporter_mask)
        if is_inlier
    ]

    outliers = [
        c for c, is_inlier in zip(correspondences, supporter_mask)
        if not is_inlier
    ]

    return inliers, outliers




def draw_ransac_results(frame0_data,
                        frame1_data,
                        inliers,
                        outliers):

    img0 = cv2.cvtColor(
        frame0_data['img_left'],
        cv2.COLOR_GRAY2RGB
    )

    img1 = cv2.cvtColor(
        frame1_data['img_left'],
        cv2.COLOR_GRAY2RGB
    )

    h0, w0 = img0.shape[:2]

    canvas = np.hstack([img0, img1])

    def draw_corr(corr, color):

        match = corr['temporal_match']

        pt0 = tuple(map(
            int,
            frame0_data['kp_left'][match.queryIdx].pt
        ))

        pt1_raw = frame1_data['kp_left'][match.trainIdx].pt

        pt1 = (
            int(pt1_raw[0] + w0),
            int(pt1_raw[1])
        )

        cv2.circle(canvas, pt0, 3, color, -1)
        cv2.circle(canvas, pt1, 3, color, -1)

        cv2.line(canvas, pt0, pt1, color, 1)

    for corr in outliers:
        draw_corr(corr, (255, 0, 0))

    for corr in inliers:
        draw_corr(corr, (0, 255, 0))

    plt.figure(figsize=(16, 8))
    plt.imshow(canvas)
    plt.title("3.5: RANSAC Inliers (green) vs Outliers (red)")
    plt.axis("off")

def plot_transformed_clouds(frame0_data,
                            frame1_data,
                            R,
                            t):
    """
    Plot pair0 cloud transformed into left1 coordinates
    together with pair1 cloud.
    """

    cloud0 = frame0_data['points_3d']
    cloud1 = frame1_data['points_3d']

    # transform cloud0 into left1 coordinates
    cloud0_transformed = (
        R @ cloud0.T + t.reshape(3, 1)
    ).T

    # crop far points
    mask0 = (
        (cloud0_transformed[:, 2] > 0) &
        (cloud0_transformed[:, 2] < 80)
    )

    mask1 = (
        (cloud1[:, 2] > 0) &
        (cloud1[:, 2] < 80)
    )

    cloud0_transformed = cloud0_transformed[mask0]
    cloud1 = cloud1[mask1]

    plt.figure(figsize=(10, 8))

    # top-down view: X vs Z
    plt.scatter(
        cloud0_transformed[:, 0],
        cloud0_transformed[:, 2],
        s=2,
        c='red',
        label='Pair0 after T'
    )

    plt.scatter(
        cloud1[:, 0],
        cloud1[:, 2],
        s=2,
        c='blue',
        label='Pair1'
    )

    plt.xlabel("X")
    plt.ylabel("Z")

    plt.title("3.5: Point Clouds Alignment")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)

import time


def compose_transform(R1, t1, R2, t2):
    """
    Compose:
        T1: world -> i
        T2: i -> i+1

    Return:
        world -> i+1
    """

    R_new = R2 @ R1
    t_new = R2 @ t1 + t2

    return R_new, t_new


def camera_center(R, t):
    """
    Camera center in world coordinates:
        C = -R^T t
    """

    return (-R.T @ t).flatten()


def read_ground_truth_poses():
    """
    Read KITTI ground truth poses.
    """

    poses_path = PROJECT_ROOT / 'dataset' / 'dataset' / 'poses' / '00.txt'

    gt_poses = []

    with open(poses_path, 'r') as f:

        for line in f:

            values = list(map(float, line.strip().split()))

            M = np.array(values).reshape(3, 4)

            R = M[:, :3]
            t = M[:, 3].reshape(3, 1)

            gt_poses.append((R, t))

    return gt_poses


def plot_trajectory(est_positions, gt_positions):
    """
    Plot estimated and ground-truth trajectories from top view.
    """

    est_positions = np.array(est_positions)
    gt_positions = np.array(gt_positions)

    plt.figure(figsize=(10, 8))

    plt.plot(
        est_positions[:, 0],
        est_positions[:, 2],
        label='Estimated trajectory',
        linewidth=2
    )

    plt.plot(
        gt_positions[:, 0],
        gt_positions[:, 2],
        label='Ground truth',
        linewidth=2
    )

    plt.xlabel("X")
    plt.ylabel("Z")

    plt.title("3.6: Camera Trajectory (Top View)")

    plt.legend()
    plt.axis('equal')
    plt.grid(True)

def get_num_frames():
    image_dir = DATA_PATH / 'image_0'
    return len(list(image_dir.glob("*.png")))


# --------------------------------------ex4---------------------------------------------------------

def build_stereo_dict(frame_data):
    return {
        m.queryIdx: m
        for m in frame_data["stereo_inliers"]
    }


def get_feature_observation(frame_data,
                            feature_idx,
                            stereo_dict):

    if feature_idx not in stereo_dict:
        return None

    stereo_match = stereo_dict[feature_idx]

    kp_left = frame_data["kp_left"][feature_idx]

    kp_right = frame_data["kp_right"][
        stereo_match.trainIdx
    ]

    return {
        "x_left": kp_left.pt[0],
        "x_right": kp_right.pt[0],
        "y": kp_left.pt[1]
    }


def add_feature_to_db(db,
                      frame_id,
                      feature_idx,
                      track_id,
                      observation):

    db.add_observation(
        frame_id=frame_id,
        feature_idx=feature_idx,
        track_id=track_id,
        x_left=observation["x_left"],
        x_right=observation["x_right"],
        y=observation["y"]
    )


def get_or_create_track(db,
                        frame_id,
                        feature_idx):

    if db.has_feature(frame_id, feature_idx):

        return db.get_track_of_feature(
            frame_id,
            feature_idx
        )

    return db.create_track()


def compute_tracking_statistics(db):
    """
    4.2: Compute tracking statistics.
    Trivial tracks of length 1 are ignored.
    """

    track_lengths = [
        len(db.frames(track_id))
        for track_id in db.track_to_frames
        if len(db.frames(track_id)) > 1
    ]

    if len(track_lengths) == 0:
        raise RuntimeError("No non-trivial tracks found.")

    frame_link_counts = [
        len(db.tracks(frame_id))
        for frame_id in db.frame_to_tracks
    ]

    stats = {
        "total_tracks": len(track_lengths),
        "num_frames": db.frame_num(),
        "mean_track_length": np.mean(track_lengths),
        "max_track_length": np.max(track_lengths),
        "min_track_length": np.min(track_lengths),
        "mean_frame_links": np.mean(frame_link_counts),
    }

    return stats


def print_tracking_statistics(stats):
    print("\n--- Task 4.2: Tracking Statistics ---")
    print(f"Total number of tracks: {stats['total_tracks']}")
    print(f"Number of frames: {stats['num_frames']}")
    print(f"Mean track length: {stats['mean_track_length']:.2f}")
    print(f"Maximum track length: {stats['max_track_length']}")
    print(f"Minimum track length: {stats['min_track_length']}")
    print(f"Mean number of frame links: {stats['mean_frame_links']:.2f}")


def debug_tracking_database(db, frame_id=10):
    print("Frames:", db.frame_num())
    print("Tracks:", db.track_num())

    tracks_in_frame = db.tracks(frame_id)

    print("Tracks in frame:", tracks_in_frame)

    if len(tracks_in_frame) > 0:
        track_id = tracks_in_frame[0]
        print("Example track id:", track_id)
        print("Frames of example track:", db.frames(track_id))

    longest_track = max(
        db.track_to_frames,
        key=lambda t: len(db.frames(t))
    )

    print("Longest track id:", longest_track)
    print("Length:", len(db.frames(longest_track)))
    print("Frames:", db.frames(longest_track))