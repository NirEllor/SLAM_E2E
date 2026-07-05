import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
import gtsam
from gtsam import symbol
from gtsam.utils import plot as gtsam_plot

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'dataset' / 'dataset' / 'sequences' / '00'


# --------------------------------------ex1---------------------------------------------------------
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
    plt.subplot(1, 2, 1);
    plt.imshow(img_l, cmap='gray');
    plt.title(f"{title} - Left");
    plt.axis('off')
    plt.subplot(1, 2, 2);
    plt.imshow(img_r, cmap='gray');
    plt.title(f"{title} - Right");
    plt.axis('off')
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


# --------------------------------------ex2---------------------------------------------------------

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


# --------------------------------------ex3---------------------------------------------------------

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
    c_left_1 = np.array([c_world_3d[0], c_world_3d[2]])  # Extract X (width) and Z (depth)

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
    plt.axis('equal')  # Maintain aspect ratio so distance scales are identical on both axes

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


def select_track_by_min_length(db, min_length=6):
    valid_tracks = [
        track_id
        for track_id in db.track_to_frames
        if len(db.frames(track_id)) >= min_length
    ]

    if len(valid_tracks) == 0:
        raise RuntimeError(f"No track with length >= {min_length}")

    return max(
        valid_tracks,
        key=lambda t: len(db.frames(t))
    )


def crop_around_point(img, x, y, crop_size=20):
    h, w = img.shape[:2]
    half = crop_size // 2

    x = int(round(x))
    y = int(round(y))

    x1 = max(0, x - half)
    x2 = min(w, x + half)

    y1 = max(0, y - half)
    y2 = min(h, y + half)

    crop = img[y1:y2, x1:x2]

    local_x = x - x1
    local_y = y - y1

    return crop, local_x, local_y


def plot_track_observations(db, track_id, crop_size=20):
    frames = db.frames(track_id)

    num_rows = len(frames)

    fig, axes = plt.subplots(
        num_rows,
        2,
        figsize=(10, 2.2 * num_rows)
    )

    if num_rows == 1:
        axes = np.array([axes])

    fig.suptitle(
        f"Track #{track_id}, length={len(frames)}",
        fontsize=14
    )

    for row, frame_id in enumerate(frames):
        obs = db.observation(frame_id, track_id)

        img_left, _ = read_images(frame_id)

        x = obs.x_left
        y = obs.y

        crop, local_x, local_y = crop_around_point(
            img_left,
            x,
            y,
            crop_size=crop_size
        )

        # full image
        ax_img = axes[row, 0]
        ax_img.imshow(img_left, cmap="gray")
        ax_img.scatter([x], [y], c="red", marker="x", s=40)
        ax_img.set_title(f"Frame {frame_id}")
        ax_img.axis("off")

        # crop
        ax_crop = axes[row, 1]
        ax_crop.imshow(crop, cmap="gray")
        ax_crop.scatter([local_x], [local_y], c="red", marker="x", s=40)
        ax_crop.set_title(f"{crop_size}x{crop_size} crop")
        ax_crop.axis("off")

    plt.tight_layout()


def compute_connectivity(db):
    connectivity = []

    num_frames = db.frame_num()

    for frame_id in range(num_frames - 1):
        current_tracks = set(
            db.tracks(frame_id)
        )

        next_tracks = set(
            db.tracks(frame_id + 1)
        )

        outgoing_tracks = len(
            current_tracks & next_tracks
        )

        connectivity.append(
            outgoing_tracks
        )

    return connectivity


def plot_connectivity(connectivity):
    plt.figure(figsize=(12, 5))

    plt.plot(
        connectivity,
        linewidth=1
    )

    plt.axhline(
        np.mean(connectivity),
        color="green",
        linestyle="--",
        label=f"Mean={np.mean(connectivity):.1f}"
    )

    plt.title("Connectivity")
    plt.xlabel("Frame")
    plt.ylabel("Outgoing Tracks")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()


def plot_inlier_percentage(inlier_percentages):
    plt.figure(figsize=(12, 5))

    plt.plot(
        inlier_percentages,
        linewidth=1
    )

    mean_val = np.mean(inlier_percentages)

    plt.axhline(
        mean_val,
        color="green",
        linestyle="--",
        label=f"Mean={mean_val:.2f}%"
    )

    plt.title("Inlier Percentage Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Inlier Percentage (%)")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()


def compute_track_lengths(db, min_length=2):
    return [
        len(db.frames(track_id))
        for track_id in db.track_to_frames
        if len(db.frames(track_id)) >= min_length
    ]


def plot_track_length_histogram(db, min_length=2):
    track_lengths = compute_track_lengths(db, min_length=min_length)

    plt.figure(figsize=(12, 5))

    bins = range(
        min(track_lengths),
        max(track_lengths) + 2
    )

    plt.hist(
        track_lengths,
        bins=bins,
        edgecolor="black"
    )

    plt.yscale("log")

    plt.title("Track Length Histogram")
    plt.xlabel("Track Length")
    plt.ylabel("Track Count")

    plt.grid(True, axis="y")
    plt.tight_layout()


def project_stereo_point(K, R, t, t_stereo, X):
    """
    Projects a 3D point X into both left and right cameras of a given frame pose.
    X: shape (3,) in world coordinates
    Returns: (u_l, v_l), (u_r, v_r)
    """
    # Projection matrix for left camera: P_L = K * [R | t]
    P_left = K @ np.hstack([R, t.reshape(3, 1)])

    # Projection matrix for right camera: P_R = K * [R | t + t_stereo]
    P_right = K @ np.hstack([R, (t.reshape(3) + t_stereo.reshape(3)).reshape(3, 1)])

    # Project using the existing project_point function
    proj_l = project_point(P_left, X)
    proj_r = project_point(P_right, X)

    return proj_l, proj_r


def get_akaze_features(img):
    akaze = cv2.AKAZE_create(threshold=0.0001)
    kp, des = akaze.detectAndCompute(img, None)
    return kp, des


def run_single_pair(idx=0, display=False, plot_3d=True):
    """
    Runs the full computational pipeline for a single stereo pair frame node.
    Acts as a foundational backend tool without creating circular script references.
    """
    if display:
        print(f"\n========== Processing Frame {idx} Pipeline ==========")

    # 1. Read images and extract raw interest structures locally
    img1, img2 = read_images(idx)
    kp1, des1 = get_akaze_features(img1)
    kp2, des2 = get_akaze_features(img2)

    assert len(kp1) >= 500 and len(kp2) >= 500, f"Insufficient feature count in frame {idx}!"

    if display:
        img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
        img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))
        plot_stereo_side_by_side(img1_kp, img2_kp, f"Frame {idx}: Raw Structural Interest Points")

    # 2. Extract relative brute-force spatial mappings
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1, des2)

    if display:
        draw_matches_custom(img1, kp1, img2, kp2, matches, f"Frame {idx}: Raw Brute-Force Match Overlap Vectors")
        # Generate diagnostic tracking verification data directly via base tools
        deviations = compute_rectified_stereo_deviations(kp1, kp2, matches)
        plot_deviation_histogram(deviations)
        print_large_deviation_percentage(deviations, threshold=2)

    # 3. Enforce the early-stage geometric epipolar constraint validation mask
    inliers, outliers = split_matches_by_rectified_pattern(kp1, kp2, matches, threshold=2)

    if display:
        print(f"Frame {idx} Epipolar Check Metrics -> Inliers: {len(inliers)} | Outliers: {len(outliers)}")
        draw_inliers_outliers(img1, kp1, img2, kp2, inliers, outliers)

    # 4. Generate spatial structural coordinates via OpenCV triangulation channels
    k, m1, m2 = read_cameras()
    points1, points2 = get_matched_points(kp1, kp2, inliers)
    points_3d = triangulate_points_opencv(points1, points2, m1, m2)

    valid_depth_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 350)

    filtered_inliers = [inliers[i] for i in range(len(inliers)) if valid_depth_mask[i]]
    filtered_points_3d = points_3d[valid_depth_mask]

    # Generate the pristine visualization plot if requested
    if display or plot_3d:
        plot_3d_points(filtered_points_3d, title=f"Frame {idx}: Triangulated Landmark Points (<350m)")

    # Extract clean validated local feature matrix matrices
    des_left_inliers = np.array([des1[m.queryIdx] for m in filtered_inliers]) if filtered_inliers else np.empty(
        (0, des1.shape[1]))
    des_right_inliers = np.array([des2[m.trainIdx] for m in filtered_inliers]) if filtered_inliers else np.empty(
        (0, des2.shape[1]))

    # Package structural results safely for tracking sequence steps
    return {
        'img_left': img1,
        'img_right': img2,
        'kp_left': kp1,
        'kp_right': kp2,
        'des_left': des1,
        'des_right': des2,

        'des_left_inliers': des_left_inliers,
        'des_right_inliers': des_right_inliers,

        'stereo_inliers': filtered_inliers,
        'points_3d': filtered_points_3d
    }


#### ex5 ###


def init_gtsam_stereo_calibration():
    """Reads camera calibration matrices and constructs a GTSAM Cal3_S2Stereo object."""
    K_mat, _, m_right0 = read_cameras()
    fx, fy, cx, cy, skew = K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2], K_mat[0, 1]
    t_stereo = np.linalg.inv(K_mat) @ m_right0[:, 3]
    baseline = abs(t_stereo[0])
    return gtsam.Cal3_S2Stereo(fx, fy, skew, cx, cy, baseline)


def get_gtsam_camera_pose(gt_poses, frame_id):
    """Converts world-to-camera ground truth matrices into a GTSAM Pose3 object."""
    R_w2c, t_w2c = gt_poses[frame_id]
    R_c2w = R_w2c.T
    t_c2w = (-R_w2c.T @ t_w2c).flatten()
    return gtsam.Pose3(gtsam.Rot3(R_c2w), gtsam.Point3(t_c2w[0], t_c2w[1], t_c2w[2]))


def compute_stereo_reprojection_error(pose, K_gtsam, point_3d, obs):
    """Computes the L2 pixel reprojection error for a single stereo observation."""
    try:
        camera = gtsam.StereoCamera(pose, K_gtsam)
        proj = camera.project(point_3d)
        return np.sqrt((proj.uL() - obs.x_left) ** 2 + (proj.v() - obs.y) ** 2 + (proj.uR() - obs.x_right) ** 2)
    except RuntimeError:
        return np.nan


def compute_single_factor_error(pose, K_gtsam, point_3d, obs, pose_id=0, point_id=0):
    """Creates a temporary GenericStereoFactor3D and computes its scalar graph error."""
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    pose_key, point_key = symbol('c', pose_id), symbol('q', point_id)

    stereo_meas = gtsam.StereoPoint2(obs.x_left, obs.x_right, obs.y)
    factor = gtsam.GenericStereoFactor3D(stereo_meas, measurement_noise, pose_key, point_key, K_gtsam)

    values = gtsam.Values()
    values.insert(pose_key, pose)
    values.insert(point_key, point_3d)
    return factor.error(values)


def extract_optimized_geometry(result, window_frames, landmarks_in_window):
    """Extracts camera trajectories and applies a 3-STD statistical filter to 3D landmarks."""
    cam_positions = np.array([
        result.atPose3(symbol('c', f_id)).translation() for f_id in window_frames
    ])

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
    """Draws ground-truth measurements, pre-optimization, and post-optimization circles on images."""
    img_left_gray, img_right_gray = read_images(frame_id)
    img_left = cv2.cvtColor(img_left_gray, cv2.COLOR_GRAY2BGR)
    img_right = cv2.cvtColor(img_right_gray, cv2.COLOR_GRAY2BGR)

    meas_L = (int(obs.x_left), int(obs.y))
    meas_R = (int(obs.x_right), int(obs.y))
    proj_before_L = (int(proj_init.uL()), int(proj_init.v()))
    proj_before_R = (int(proj_init.uR()), int(proj_init.v()))
    proj_after_L = (int(proj_final.uL()), int(proj_final.v()))
    proj_after_R = (int(proj_final.uR()), int(proj_final.v()))

    for img, pt_meas, pt_before, pt_after in [(img_left, meas_L, proj_before_L, proj_after_L),
                                              (img_right, meas_R, proj_before_R, proj_after_R)]:
        cv2.circle(img, pt_meas, radius=6, color=(255, 0, 0), thickness=-1)  # Blue
        cv2.circle(img, pt_before, radius=6, color=(0, 0, 255), thickness=-1)  # Red
        cv2.circle(img, pt_after, radius=6, color=(0, 255, 0), thickness=-1)  # Green

    cv2.putText(img_left, "Blue: Meas | Red: Before | Green: After", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"task_5_3_worst_frame_{frame_id}_left.png"), img_left)
    cv2.imwrite(os.path.join(output_dir, f"task_5_3_worst_frame_{frame_id}_right.png"), img_right)


def w2c_to_local_gtsam_pose(R_start, t_start, R_f, t_f):
    """
    Transforms a world-to-camera (w2c) extrinsic pose into a local window coordinate system
    where the start frame is the origin, and returns it as a gtsam.Pose3 object (c2w).

    Parameters:
    - R_start, t_start: Extrinsics of the first frame in the bundle window (the local origin).
    - R_f, t_f: Extrinsics of the current frame to transform.
    """
    R_rel = R_f @ R_start.T
    t_rel = t_f - R_f @ R_start.T @ t_start
    R_gtsam = R_rel.T
    t_gtsam = -R_rel.T @ t_rel

    gtsam_rot = gtsam.Rot3(R_gtsam)
    gtsam_point = gtsam.Point3(float(t_gtsam[0]), float(t_gtsam[1]), float(t_gtsam[2]))

    return gtsam.Pose3(gtsam_rot, gtsam_point)


def valid_stereo_obs(obs, min_disp=1.0):
    """
    Validates a stereo observation by checking if the disparity is positive
    and above a minimal threshold, preventing degenerate triangulation.

    Parameters:
    - obs: An observation object containing x_left and x_right attributes.
    - min_disp: Minimum required disparity in pixels.
    """
    disparity = obs.x_left - obs.x_right

    return disparity >= min_disp


def choose_keyframes(db, distance_threshold=2.5, max_gap=20, min_gap=5):
    keyframes = [0]
    last_kf = 0
    accumulated_dist = 0.0

    for idx in range(1, db.frame_num()):
        R_prev, t_prev = db.camera_poses[idx - 1]
        R_curr, t_curr = db.camera_poses[idx]

        p_prev = camera_center(R_prev, t_prev)
        p_curr = camera_center(R_curr, t_curr)

        accumulated_dist += np.linalg.norm(p_curr - p_prev)
        frames_since_last = idx - last_kf

        if (
                frames_since_last >= min_gap
                and accumulated_dist >= distance_threshold
        ) or frames_since_last >= max_gap:
            keyframes.append(idx)
            last_kf = idx
            accumulated_dist = 0.0

    if keyframes[-1] != db.frame_num() - 1:
        keyframes.append(db.frame_num() - 1)

    return keyframes


def pose_translation_np(pose):
    return np.array(pose.translation()).reshape(3)


def solve_bundle_window(db, start_frame, end_frame, max_tracks_per_window=150):
    K_gtsam = init_gtsam_stereo_calibration()
    K_mat, P_left0, P_right0 = read_cameras()

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    base_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    measurement_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(2.0),
        base_noise
    )

    window_frames = list(range(start_frame, end_frame + 1))

    R_start_w2c, t_start_w2c = db.camera_poses[start_frame]
    pose_start_global = gtsam.Pose3(
        gtsam.Rot3(R_start_w2c.T),
        gtsam.Point3((-R_start_w2c.T @ t_start_w2c).flatten())
    )

    for f_id in window_frames:
        pose_key = symbol("c", f_id)
        R_f_w2c, t_f_w2c = db.camera_poses[f_id]

        # פוזה גלובלית במונחי GTSAM (Camera to World)
        pose_f_global = gtsam.Pose3(
            gtsam.Rot3(R_f_w2c.T),
            gtsam.Point3((-R_f_w2c.T @ t_f_w2c).flatten())
        )

        pose_local = pose_start_global.between(pose_f_global)
        initial_estimate.insert(pose_key, pose_local)

        if f_id == start_frame:
            anchor_factor = gtsam.PriorFactorPose3(
                pose_key,
                gtsam.Pose3(),
                gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
            )
            graph.add(anchor_factor)

    rng = random.Random(0)

    candidate_tracks = set()
    for f_id in window_frames:
        candidate_tracks.update(db.tracks(f_id))

    guaranteed_tracks = set()
    for f_id in window_frames:
        ts = list(db.tracks(f_id))
        guaranteed_tracks.update(rng.sample(ts, min(15, len(ts))))

    # Guarantee connectivity between every consecutive frame pair
    for a, b in zip(window_frames[:-1], window_frames[1:]):
        shared_tracks = list(set(db.tracks(a)) & set(db.tracks(b)))

        if len(shared_tracks) > 0:
            guaranteed_tracks.update(
                rng.sample(shared_tracks, min(15, len(shared_tracks)))
            )
        else:
            print(f"[Warning] No shared tracks between {a} and {b}")

    remaining = max_tracks_per_window - len(guaranteed_tracks)
    extras = list(candidate_tracks - guaranteed_tracks)

    final_tracks_to_optimize = list(guaranteed_tracks)

    if remaining > 0:
        final_tracks_to_optimize += rng.sample(
            extras,
            min(remaining, len(extras))
        )

    optimized_landmark_ids = []
    pose_factor_count = {f_id: 0 for f_id in window_frames}

    for track_id in final_tracks_to_optimize:
        track_frames = [f for f in db.frames(track_id) if f in window_frames]

        if len(track_frames) < 2:
            continue

        init_frame = track_frames[0]
        obs_init = db.observation(init_frame, track_id)

        if not valid_stereo_obs(obs_init, min_disp=1.0):
            continue

        X_cam = triangulate_point_linear(
            np.array([obs_init.x_left, obs_init.y]),
            np.array([obs_init.x_right, obs_init.y]),
            P_left0,
            P_right0
        )

        if not np.all(np.isfinite(X_cam)):
            continue

        if X_cam[2] <= 2.0 or X_cam[2] > 120.0:
            continue

        init_pose = initial_estimate.atPose3(symbol("c", init_frame))
        X_local = init_pose.transformFrom(
            gtsam.Point3(float(X_cam[0]), float(X_cam[1]), float(X_cam[2]))
        )

        point_key = symbol("q", track_id)
        temp_factors = []

        for f_id in track_frames:
            obs = db.observation(f_id, track_id)

            if not valid_stereo_obs(obs, min_disp=1.0):
                continue

            temp_factors.append(
                gtsam.GenericStereoFactor3D(
                    gtsam.StereoPoint2(
                        float(obs.x_left),
                        float(obs.x_right),
                        float(obs.y)
                    ),
                    measurement_noise,
                    symbol("c", f_id),
                    point_key,
                    K_gtsam
                )
            )

        if len(temp_factors) < 2:
            continue

        initial_estimate.insert(point_key, X_local)
        optimized_landmark_ids.append(track_id)

        for factor in temp_factors:
            graph.add(factor)

            for key in factor.keys():
                s = gtsam.Symbol(key)

                if chr(s.chr()) == "c":
                    f_id = s.index()

                    if f_id in pose_factor_count:
                        pose_factor_count[f_id] += 1

    bad_counts = {
        f_id: count
        for f_id, count in pose_factor_count.items()
        if f_id != start_frame and count == 0
    }

    if bad_counts:
        print(f"\nBundle {start_frame}->{end_frame}")
        print("Disconnected poses:", bad_counts)

    disconnected = list(bad_counts.keys())

    if disconnected:
        raise RuntimeError(
            f"Disconnected poses in bundle {start_frame}->{end_frame}: {disconnected}"
        )

    print(f"Graph size before optimization: {graph.size()}")
    print(f"Initial estimate size: {initial_estimate.size()}")

    initial_error = graph.error(initial_estimate)

    result = gtsam.LevenbergMarquardtOptimizer(
        graph,
        initial_estimate
    ).optimize()

    final_error = graph.error(result)

    print(
        f"Window {start_frame}->{end_frame}: "
        f"factors={graph.size()}, "
        f"error before={initial_error:.2f}, "
        f"after={final_error:.2f}"
    )

    pose_start = result.atPose3(symbol("c", start_frame))
    pose_end = result.atPose3(symbol("c", end_frame))

    relative_pose = pose_start.between(pose_end)

    optimized_points_local = []
    for track_id in optimized_landmark_ids:
        point_key = symbol("q", track_id)

        if result.exists(point_key):
            p = np.array(result.atPoint3(point_key)).reshape(3)
            if np.all(np.isfinite(p)):
                optimized_points_local.append(p)

    return {
        "result": result,
        "graph": graph,
        "initial": initial_estimate,
        "relative_pose": relative_pose,
        "optimized_points_local": np.array(optimized_points_local),
        "optimized_landmark_ids": optimized_landmark_ids,
        "anchor_factor": anchor_factor if 'anchor_factor' in locals() else graph.at(0),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "initial_error": initial_error,
        "final_error": final_error
    }


def plot_q5_4_results(keyframes, global_keyframe_poses, all_points_global, output_dir="./outputs"):
    gt_poses = read_ground_truth_poses()

    estimated_positions = []
    gt_positions = []

    valid_keyframes = [
        kf for kf in keyframes
        if kf in global_keyframe_poses
    ]

    for kf in valid_keyframes:
        estimated_positions.append(
            pose_translation_np(global_keyframe_poses[kf])
        )

        R_gt, t_gt = gt_poses[kf]
        gt_positions.append(
            camera_center(R_gt, t_gt)
        )

    estimated_positions = np.array(estimated_positions)
    gt_positions = np.array(gt_positions)
    points = np.array(all_points_global)

    plt.figure(figsize=(10, 8))

    if len(points) > 0:
        plt.scatter(
            points[:, 0],
            points[:, 2],
            s=1,
            alpha=0.25,
            label="Optimized 3D points"
        )

    plt.plot(
        estimated_positions[:, 0],
        estimated_positions[:, 2],
        "bo-",
        markersize=3,
        label="Optimized keyframes"
    )

    plt.plot(
        gt_positions[:, 0],
        gt_positions[:, 2],
        "r--",
        linewidth=2,
        label="Ground truth keyframes"
    )

    plt.title("5.4: Optimized Keyframe Trajectory vs Ground Truth")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        os.path.join(output_dir, "task_5_4_keyframes_vs_gt.png"),
        dpi=200
    )


def plot_keyframe_localization_error(keyframes, global_keyframe_poses, output_dir="./outputs"):
    gt_poses = read_ground_truth_poses()

    errors = []
    valid_keyframes = []

    # print("\n[Debug localization comparison]")

    for kf in keyframes:
        if kf not in global_keyframe_poses:
            continue

        est_pos = pose_translation_np(
            global_keyframe_poses[kf]
        )

        R_gt, t_gt = gt_poses[kf]
        gt_pos = camera_center(R_gt, t_gt)

        err = np.linalg.norm(est_pos - gt_pos)

        errors.append(err)
        valid_keyframes.append(kf)

        # print(f"KF {kf}")
        # print(f"  est_pos:  {est_pos}")
        # print(f"  gt_pos:   {gt_pos}")
        # print(f"  error:    {err:.3f} m")

    if len(errors) == 0:
        print("No valid localization errors to plot.")
        return

    plt.figure(figsize=(12, 5))

    plt.plot(
        valid_keyframes,
        errors,
        marker="o",
        linewidth=1
    )

    plt.title("5.4: Keyframe Localization Error")
    plt.xlabel("Frame")
    plt.ylabel("Localization Error [m]")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "task_5_4_keyframe_error.png"
    )

    plt.savefig(output_path, dpi=200)

    print(
        f"Mean keyframe localization error: {np.mean(errors):.3f} m"
    )
    print(
        f"Max keyframe localization error: {np.max(errors):.3f} m"
    )


import pickle

DB_PKL_PATH = "tracking_db_ex5.pkl"
from tracking_database_custom import TrackingDB


def build_data(num_frames=10):
    """
    4.1: Building the long-term tracking database.
    Also stores global PnP camera poses for Exercise 5.
    Pose convention:
        X_cam = R_global @ X_left0 + t_global
    """

    db = TrackingDB()
    inlier_percentages = []

    # Global pose of frame 0 in left0 coordinates
    R_global = np.eye(3)
    t_global = np.zeros((3, 1))

    camera_poses = [(R_global.copy(), t_global.copy())]

    prev_data = run_single_pair(
        idx=0,
        display=False,
        plot_3d=False
    )

    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    for idx in range(1, num_frames):
        print(f"Processing Frame Sequence Node: {idx}/{num_frames - 1}")

        curr_data = run_single_pair(
            idx=idx,
            display=False,
            plot_3d=False
        )

        knn_matches = bf_matcher.knnMatch(
            prev_data["des_left"],
            curr_data["des_left"],
            k=2
        )

        temporal_matches = [
            m for m, n in knn_matches
            if m.distance < 0.7 * n.distance
        ]
        if idx == 2853:
            print("Temporal matches:", len(temporal_matches))

        try:
            correspondences = build_pnp_correspondences(
                prev_data,
                curr_data,
                temporal_matches
            )
            if idx == 2853:
                print("PnP correspondences:", len(correspondences))

            if len(correspondences) < 4:
                raise RuntimeError("Not enough correspondences for PnP-RANSAC.")

            best_inliers = []
            best_outliers = []
            best_R = None
            best_t = None

            k_matrix, _, _ = read_cameras()

            no_improvement = 0
            max_no_improvement = 12

            for _ in range(50):
                sample = random.sample(correspondences, 4)

                obj_pts = np.array(
                    [c["X"] for c in sample],
                    dtype=np.float32
                )

                img_pts = np.array(
                    [c["obs_left1"] for c in sample],
                    dtype=np.float32
                )

                success, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    img_pts,
                    k_matrix,
                    None,
                    flags=cv2.SOLVEPNP_EPNP
                )

                if not success:
                    no_improvement += 1
                    continue

                R_candidate, _ = cv2.Rodrigues(rvec)

                inliers, outliers = evaluate_supporters(
                    correspondences,
                    prev_data,
                    curr_data,
                    R_candidate,
                    tvec,
                    threshold=2
                )

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    if idx == 2853:
                        print("RANSAC inliers:", len(best_inliers))
                    best_outliers = outliers
                    best_R = R_candidate
                    best_t = tvec
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= max_no_improvement:
                    break

            if best_R is None:
                raise RuntimeError("RANSAC failed to find a valid pose.")

            total = len(best_inliers) + len(best_outliers)

            if total > 0:
                inlier_percentages.append(
                    100.0 * len(best_inliers) / total
                )
            else:
                inlier_percentages.append(0.0)

            # Compose global pose:
            R_global, t_global = compose_transform(
                R_global,
                t_global,
                best_R,
                best_t
            )

        except RuntimeError as e:
            print(f"PnP-RANSAC failure at frame link {idx - 1}->{idx}: {e}")
            inlier_percentages.append(0.0)
            R_global = R_global.copy()
            t_global = t_global.copy()
            best_inliers = []

        ransac_temporal_matches = [c['temporal_match'] for c in best_inliers]

        prev_stereo = build_stereo_dict(prev_data)
        curr_stereo = build_stereo_dict(curr_data)

        db.update_tracks(
            idx,
            ransac_temporal_matches,
            prev_stereo,
            curr_stereo,
            prev_data,
            curr_data
        )

        camera_poses.append((R_global.copy(), t_global.copy()))

        prev_data = curr_data

    db.inlier_percentages = inlier_percentages
    db.camera_poses = camera_poses

    return db


def load_or_build_db(force_rebuild=False, num_frames=None):
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
    gt_poses = read_ground_truth_poses()

    print("\n[Debug coordinate system alignment]")

    for kf in keyframes[:10]:
        if kf not in global_keyframe_poses:
            continue

        est_pose = global_keyframe_poses[kf]

        est_t = pose_translation_np(est_pose)
        est_inv_t = pose_translation_np(est_pose.inverse())

        R_gt, t_gt = gt_poses[kf]

        gt_t = t_gt.flatten()
        gt_center = camera_center(R_gt, t_gt)

        print(f"\nKF {kf}")
        print("est_t:      ", est_t)
        print("est_inv_t:  ", est_inv_t)
        print("gt_t:       ", gt_t)
        print("gt_center:  ", gt_center)

        print("err est_t vs gt_center:     ", np.linalg.norm(est_t - gt_center))
        print("err est_inv_t vs gt_center: ", np.linalg.norm(est_inv_t - gt_center))
        print("err est_t vs gt_t:          ", np.linalg.norm(est_t - gt_t))
        print("err est_inv_t vs gt_t:      ", np.linalg.norm(est_inv_t - gt_t))


def debug_scale_drift(keyframes, global_keyframe_poses):
    gt_poses = read_ground_truth_poses()

    est_positions = []
    gt_positions = []

    for kf in keyframes:
        if kf not in global_keyframe_poses:
            continue

        est_positions.append(
            pose_translation_np(global_keyframe_poses[kf])
        )

        R_gt, t_gt = gt_poses[kf]
        gt_positions.append(
            camera_center(R_gt, t_gt)
        )

    est_positions = np.array(est_positions)
    gt_positions = np.array(gt_positions)

    est_steps = np.linalg.norm(
        np.diff(est_positions, axis=0),
        axis=1
    )

    gt_steps = np.linalg.norm(
        np.diff(gt_positions, axis=0),
        axis=1
    )

    print("\n[Debug scale drift]")
    print("mean estimated step:", np.mean(est_steps))
    print("mean GT step:", np.mean(gt_steps))
    print("scale ratio est/gt:", np.sum(est_steps) / np.sum(gt_steps))


# =============================================================================
# GRAPH PLOTTING FUNCTIONS
# =============================================================================

def reprojection_error_graph(frame_indices, left_reprojection_errors, right_reprojection_errors, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, left_reprojection_errors, label="Left camera reprojection error", color="#1f77b4")
    plt.plot(frame_indices, right_reprojection_errors, label="Right camera reprojection error", color="#ff7f0e")
    plt.title("Right camera reprojection error")  # כותרת הגרף כפי שמופיעה בדוגמה
    plt.xlabel("Frame index")
    plt.ylabel("Reprojection error (L2 norm) - pixels")
    plt.grid(True)
    plt.legend(loc="upper right")
    reproj_path = os.path.join(output_dir, "reprojection_error_graph.png")
    plt.savefig(reproj_path, dpi=300)
    plt.close()


def factor_error_graph(frame_indices, factor_errors, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, factor_errors, color="red", marker='s')
    plt.title("Factor error graph")
    plt.xlabel("Frame index")
    plt.ylabel("Factor Error")
    plt.grid(True)
    factor_path = os.path.join(output_dir, "factor_error_graph.png")
    plt.savefig(factor_path, dpi=300)
    plt.close()


def bundle1_3D(window_frames, cam_positions, result, axis_length, output_dir):
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')

    xs_plot = cam_positions[:, 2]
    ys_plot = cam_positions[:, 0]
    zs_plot = -cam_positions[:, 1]

    ax3d.plot(xs_plot, ys_plot, zs_plot, 'k--', linewidth=1.5, zorder=1)

    for i in range(len(window_frames)):
        pose = result.atPose3(symbol('c', window_frames[i]))
        R = pose.rotation().matrix()
        cx, cy, cz = xs_plot[i], ys_plot[i], zs_plot[i]

        ax3d.scatter(cx, cy, cz, color='black', s=15, zorder=2)

        ax_right = R[:, 0]
        ax_down = R[:, 1]
        ax_forward = R[:, 2]

        ax3d.plot([cx, cx + axis_length * ax_right[2]],
                  [cy, cy + axis_length * ax_right[0]],
                  [cz, cz - axis_length * ax_right[1]], color='r', linewidth=1.5)
        ax3d.plot([cx, cx - axis_length * ax_down[2]],
                  [cy, cy - axis_length * ax_down[0]],
                  [cz, cz + axis_length * ax_down[1]], color='g', linewidth=1.5)
        ax3d.plot([cx, cx + axis_length * ax_forward[2]],
                  [cy, cy + axis_length * ax_forward[0]],
                  [cz, cz - axis_length * ax_forward[1]], color='b', linewidth=1.5)

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
    fig_cov = plt.figure(figsize=(8, 6))
    ax_cov = fig_cov.add_subplot(111, projection='3d')
    ax_cov.set_title("Plot Trajectory\nGTSAM Factor Graph State with Marginal Covariances\n", fontsize=11,
                     fontweight='bold')

    try:
        marginals = gtsam.Marginals(graph, result)
        for f_id in window_frames:
            pose_key = symbol('c', f_id)
            pose = result.atPose3(pose_key)
            cov = marginals.marginalCovariance(pose_key)
            gtsam_plot.plot_pose3_on_axes(ax_cov, pose, axis_length=0.4, P=cov)
    except Exception as e:
        print(f"[Warning] Covariance layout fallback: {e}")
        for f_id in window_frames:
            pose = result.atPose3(symbol('c', f_id))
            gtsam_plot.plot_pose3_on_axes(ax_cov, pose, axis_length=0.4)

    ax_cov.set_xlabel("X axis")
    ax_cov.set_ylabel("Y axis")
    ax_cov.set_zlabel("Z axis")
    ax_cov.view_init(elev=20, azim=-35)
    plt.savefig(os.path.join(output_dir, "task_5_3_marginal_covariances.png"), dpi=200, bbox_inches='tight')
    plt.close()


def bundle1_2D_full(window_frames, initial_cam_positions, cam_positions, lm_filtered, output_dir):
    fig2d_full, ax2d_full = plt.subplots(figsize=(8, 8))

    if len(lm_filtered) > 0:
        ax2d_full.scatter(lm_filtered[:, 0], lm_filtered[:, 2],
                          s=2, c='orange', alpha=0.6, label=f'Landmarks ({len(lm_filtered)})')

    ax2d_full.plot(initial_cam_positions[:, 0], initial_cam_positions[:, 2],
                   'b-+', alpha=0.6, label='Initial Cameras (PnP)')
    ax2d_full.plot(cam_positions[:, 0], cam_positions[:, 2],
                   'r-^', markersize=6, linewidth=1.5, label='Optimized Cameras (BA)')

    for i, f_id in enumerate(window_frames):
        ax2d_full.annotate(str(f_id),
                           (cam_positions[i, 0], cam_positions[i, 2]),
                           textcoords="offset points", xytext=(4, 4),
                           fontsize=7, color='darkred')

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
    fig2d_zoom, ax2d_zoom = plt.subplots(figsize=(6, 7))
    ax2d_zoom.plot(initial_cam_positions[:, 0], initial_cam_positions[:, 2],
                   'b-o', alpha=0.5, markersize=4, label='Initial (PnP)')
    ax2d_zoom.plot(cam_positions[:, 0], cam_positions[:, 2],
                   'r-o', markersize=4, label='Optimized (BA)')
    ax2d_zoom.set_title("Local Window Bundle Adjustment: 2D Bird's-Eye View Trajectory\n(Trajectory zoomed)")
    ax2d_zoom.set_xlabel("X Coordinate (East) [m]")
    ax2d_zoom.set_ylabel("Z Coordinate (North) [m]")
    ax2d_zoom.set_xlim([-2, 2])
    ax2d_zoom.set_ylim([-2, 7])
    ax2d_zoom.grid(True, alpha=0.3)
    ax2d_zoom.legend()
    plt.savefig(os.path.join(output_dir, "task_5_3_bundle1_2D_zoomed.png"), dpi=150)
    plt.close()


def covariance_to_noise_model(cov, min_sigma=1e-6):
    """
    Convert a 6x6 covariance matrix into a GTSAM Gaussian noise model.
    Adds small diagonal regularization for numerical stability.
    """
    cov = np.asarray(cov, dtype=np.float64)

    if cov.shape != (6, 6):
        raise ValueError(f"Expected 6x6 covariance, got {cov.shape}")

    cov = 0.5 * (cov + cov.T)
    cov = cov + np.eye(6) * (min_sigma ** 2)

    return gtsam.noiseModel.Gaussian.Covariance(cov)


def build_pose_graph_initial_estimate(relative_poses):
    initial = gtsam.Values()

    sorted_edges = sorted(relative_poses.keys())
    if len(sorted_edges) == 0:
        raise RuntimeError("No relative poses were provided.")

    current_pose = gtsam.Pose3()
    first_kf = sorted_edges[0][0]
    initial.insert(symbol("c", first_kf), current_pose)

    for start_kf, end_kf in sorted_edges:
        start_key = symbol("c", start_kf)
        end_key = symbol("c", end_kf)

        if not initial.exists(start_key):
            # New disconnected component.
            initial.insert(start_key, gtsam.Pose3())

        start_pose = initial.atPose3(start_key)
        rel_pose = relative_poses[(start_kf, end_kf)]
        end_pose = start_pose.compose(rel_pose)

        if not initial.exists(end_key):
            initial.insert(end_key, end_pose)

    return initial


def build_pose_graph(relative_poses, relative_covs):
    graph = gtsam.NonlinearFactorGraph()

    sorted_edges = sorted(relative_poses.keys())
    if len(sorted_edges) == 0:
        raise RuntimeError("No relative pose constraints were provided.")

    component_starts = find_component_start_keyframes(relative_poses)

    strong_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.ones(6) * 1e-6
    )

    # Anchor every disconnected component.
    # This does NOT connect components; it only makes marginals well-defined.
    for kf in component_starts:
        graph.add(
            gtsam.PriorFactorPose3(
                symbol("c", kf),
                gtsam.Pose3(),
                strong_prior_noise
            )
        )

    for start_kf, end_kf in sorted_edges:
        rel_pose = relative_poses[(start_kf, end_kf)]
        rel_cov = relative_covs[(start_kf, end_kf)]

        noise_model = covariance_to_noise_model(rel_cov)

        graph.add(
            gtsam.BetweenFactorPose3(
                symbol("c", start_kf),
                symbol("c", end_kf),
                rel_pose,
                noise_model
            )
        )

    return graph


def extract_pose_graph_positions(values):
    """
    Extract sorted pose keys and translations from GTSAM Values.
    """
    keys = values.keys()
    pose_items = []

    for key in keys:
        sym = gtsam.Symbol(key)
        if sym.chr() == ord("c"):
            pose = values.atPose3(key)
            pose_items.append((sym.index(), pose_translation_np(pose)))

    pose_items.sort(key=lambda x: x[0])

    frame_ids = [x[0] for x in pose_items]
    positions = np.array([x[1] for x in pose_items])

    return frame_ids, positions


def plot_pose_graph_trajectory(values, title, output_path):
    frame_ids, positions = extract_pose_graph_positions(values)

    plt.figure(figsize=(10, 8))

    plt.plot(
        positions[:, 0],
        positions[:, 2],
        "bo-",
        markersize=3,
        linewidth=1.5
    )

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=200)


def plot_pose_graph_with_covariances(values,
                                     marginals,
                                     title,
                                     output_path,
                                     covariance_step=10,
                                     covariance_scale=1.0):
    """
    Plot all optimized pose locations and visualize final marginal covariances.

    We plot the full keyframe trajectory as a dashed 3D curve.
    To keep the plot readable, covariance is shown only every few keyframes,
    using the translation covariance block Σ_xyz.
    """

    frame_ids, positions = extract_pose_graph_positions(values)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Full optimized keyframe trajectory
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        linestyle="--",
        marker=".",
        markersize=3,
        linewidth=1.0,
        label="Optimized keyframes"
    )

    # Plot covariance markers every few poses
    for frame_id in frame_ids[::covariance_step]:
        key = symbol("c", frame_id)

        if not values.exists(key):
            continue

        pose = values.atPose3(key)
        p = pose_translation_np(pose)

        try:
            cov6 = marginals.marginalCovariance(key)

            # Use only translation covariance, not full Pose3 covariance
            cov_xyz = cov6[3:6, 3:6]
            cov_xyz = 0.5 * (cov_xyz + cov_xyz.T)

            eigvals, eigvecs = np.linalg.eigh(cov_xyz)
            eigvals = np.maximum(eigvals, 0.0)

            # 1-sigma ellipsoid radii, scaled for visualization
            radii = covariance_scale * np.sqrt(eigvals)

            # Avoid huge unreadable ellipsoids
            radii = np.minimum(radii, 20.0)

            u = np.linspace(0, 2 * np.pi, 18)
            v = np.linspace(0, np.pi, 9)

            xs = radii[0] * np.outer(np.cos(u), np.sin(v))
            ys = radii[1] * np.outer(np.sin(u), np.sin(v))
            zs = radii[2] * np.outer(np.ones_like(u), np.cos(v))

            ellipsoid = np.stack(
                [xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)],
                axis=0
            )

            rotated = eigvecs @ ellipsoid

            X = rotated[0, :].reshape(xs.shape) + p[0]
            Y = rotated[1, :].reshape(ys.shape) + p[1]
            Z = rotated[2, :].reshape(zs.shape) + p[2]

            ax.plot_wireframe(
                X,
                Y,
                Z,
                linewidth=0.4,
                alpha=0.45
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

    print(f"Saved: {output_path}")


def solve_bundle_with_prior_sigma(db, start_frame, end_frame, prior_sigma):
    """
    Re-solves a bundle window identically to lib.solve_bundle_window but
    allows overriding the prior-factor noise sigma on the anchor pose.
    Returns (graph, result, window_frames).
    """
    import random
    K_gtsam = init_gtsam_stereo_calibration()
    K_mat, P_left0, P_right0 = read_cameras()

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    base_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    measurement_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(2.0),
        base_noise
    )

    window_frames = list(range(start_frame, end_frame + 1))

    R_start_w2c, t_start_w2c = db.camera_poses[start_frame]
    pose_start_global = gtsam.Pose3(
        gtsam.Rot3(R_start_w2c.T),
        gtsam.Point3((-R_start_w2c.T @ t_start_w2c).flatten())
    )

    for f_id in window_frames:
        pose_key = symbol("c", f_id)
        R_f, t_f = db.camera_poses[f_id]
        pose_f_global = gtsam.Pose3(
            gtsam.Rot3(R_f.T),
            gtsam.Point3((-R_f.T @ t_f).flatten())
        )
        pose_local = pose_start_global.between(pose_f_global)
        initial_estimate.insert(pose_key, pose_local)

        if f_id == start_frame:
            # --- custom prior noise here ---
            prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.ones(6) * prior_sigma
            )
            graph.add(gtsam.PriorFactorPose3(pose_key, gtsam.Pose3(), prior_noise))

    # Populate landmarks (same logic as solve_bundle_window)
    candidate_tracks = set()
    for f_id in window_frames:
        candidate_tracks.update(db.tracks(f_id))

    guaranteed_tracks = set()
    for f_id in window_frames:
        ts = list(db.tracks(f_id))
        guaranteed_tracks.update(random.sample(ts, min(15, len(ts))))

    remaining = 150 - len(guaranteed_tracks)
    extras = list(set(candidate_tracks) - guaranteed_tracks)
    final_tracks = list(guaranteed_tracks) + (
        random.sample(extras, min(remaining, len(extras))) if remaining > 0 else []
    )

    for track_id in final_tracks:
        track_frames = [f for f in db.frames(track_id) if f in window_frames]
        if len(track_frames) < 2:
            continue
        init_frame = track_frames[0]
        obs_init = db.observation(init_frame, track_id)
        if not valid_stereo_obs(obs_init, min_disp=1.0):
            continue
        X_cam = triangulate_point_linear(
            np.array([obs_init.x_left, obs_init.y]),
            np.array([obs_init.x_right, obs_init.y]),
            P_left0, P_right0
        )
        if not np.all(np.isfinite(X_cam)) or X_cam[2] <= 2.0 or X_cam[2] > 120.0:
            continue

        init_pose = initial_estimate.atPose3(symbol("c", init_frame))
        X_local = init_pose.transformFrom(
            gtsam.Point3(float(X_cam[0]), float(X_cam[1]), float(X_cam[2]))
        )
        point_key = symbol("q", track_id)
        temp_factors = []
        for f_id in track_frames:
            obs = db.observation(f_id, track_id)
            if valid_stereo_obs(obs, min_disp=1.0):
                temp_factors.append(
                    gtsam.GenericStereoFactor3D(
                        gtsam.StereoPoint2(float(obs.x_left), float(obs.x_right), float(obs.y)),
                        measurement_noise,
                        symbol("c", f_id), point_key, K_gtsam
                    )
                )
        if len(temp_factors) < 2:
            continue
        initial_estimate.insert(point_key, X_local)
        for fac in temp_factors:
            graph.add(fac)

    result = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate).optimize()
    return graph, result, window_frames


def find_component_start_keyframes(relative_poses):
    edges = sorted(relative_poses.keys())

    if len(edges) == 0:
        return []

    component_starts = [edges[0][0]]

    for (a, b), (c, d) in zip(edges[:-1], edges[1:]):
        if b != c:
            component_starts.append(c)

    return component_starts


def debug_target_frame_geometry(db, start_frame=2840, end_frame=2860,
                                target_frame=2860,
                                max_tracks_per_window=150,
                                output_dir="./outputs"):
    os.makedirs(output_dir, exist_ok=True)

    K_mat, P_left0, P_right0 = read_cameras()
    fx = K_mat[0, 0]
    baseline = init_gtsam_stereo_calibration().baseline()

    window_frames = list(range(start_frame, end_frame + 1))

    # Same selection logic as solve_bundle_window
    candidate_tracks = set()
    for f_id in window_frames:
        candidate_tracks.update(db.tracks(f_id))
    candidate_tracks = list(candidate_tracks)

    guaranteed_tracks = set()
    rng = random.Random(0)

    for f_id in window_frames:
        tracks_in_frame = list(db.tracks(f_id))
        if len(tracks_in_frame) > 0:
            sampled = rng.sample(tracks_in_frame, min(15, len(tracks_in_frame)))
            guaranteed_tracks.update(sampled)

    remaining_slots = max_tracks_per_window - len(guaranteed_tracks)
    all_candidates = list(set(candidate_tracks) - guaranteed_tracks)

    if remaining_slots > 0 and len(all_candidates) > 0:
        additional_tracks = rng.sample(all_candidates, min(remaining_slots, len(all_candidates)))
        final_tracks = list(guaranteed_tracks) + additional_tracks
    else:
        final_tracks = list(guaranteed_tracks)

    rows = []

    for track_id in final_tracks:
        track_frames = [f for f in db.frames(track_id) if f in window_frames]

        if target_frame not in track_frames:
            continue

        valid_frames = []
        for f in track_frames:
            obs = db.observation(f, track_id)
            if valid_stereo_obs(obs, min_disp=1.0):
                valid_frames.append(f)

        obs_t = db.observation(target_frame, track_id)

        if not valid_stereo_obs(obs_t, min_disp=1.0):
            continue

        p_left = np.array([obs_t.x_left, obs_t.y])
        p_right = np.array([obs_t.x_right, obs_t.y])

        X_cam = triangulate_point_linear(
            p_left,
            p_right,
            P_left0,
            P_right0
        )

        disparity = obs_t.x_left - obs_t.x_right
        stereo_depth_formula = fx * baseline / disparity

        rows.append({
            "track_id": track_id,
            "num_frames_in_window": len(track_frames),
            "num_valid_obs_in_window": len(valid_frames),
            "x": float(obs_t.x_left),
            "y": float(obs_t.y),
            "x_right": float(obs_t.x_right),
            "disparity": float(disparity),
            "depth_triangulated": float(X_cam[2]) if np.all(np.isfinite(X_cam)) else np.nan,
            "depth_formula": float(stereo_depth_formula),
            "valid_frames": valid_frames,
        })

    if len(rows) == 0:
        print("No valid target-frame observations found.")
        return rows

    depths = np.array([r["depth_triangulated"] for r in rows], dtype=float)
    xs = np.array([r["x"] for r in rows], dtype=float)
    ys = np.array([r["y"] for r in rows], dtype=float)
    valid_counts = np.array([r["num_valid_obs_in_window"] for r in rows], dtype=float)
    disparities = np.array([r["disparity"] for r in rows], dtype=float)

    finite_depths = depths[np.isfinite(depths)]

    print("\n==============================")
    print(f"Debug geometry for frame {target_frame}")
    print(f"Window: {start_frame}->{end_frame}")
    print("==============================")
    print(f"Tracks selected for BA: {len(final_tracks)}")
    print(f"Tracks reaching target frame and valid: {len(rows)}")

    print("\nDepth statistics:")
    print(f"  min    = {np.min(finite_depths):.3f}")
    print(f"  median = {np.median(finite_depths):.3f}")
    print(f"  mean   = {np.mean(finite_depths):.3f}")
    print(f"  max    = {np.max(finite_depths):.3f}")
    print(f"  >80m   = {np.sum(finite_depths > 80)}")
    print(f"  >120m  = {np.sum(finite_depths > 120)}")

    print("\nDisparity statistics:")
    print(f"  min    = {np.min(disparities):.3f}")
    print(f"  median = {np.median(disparities):.3f}")
    print(f"  mean   = {np.mean(disparities):.3f}")
    print(f"  max    = {np.max(disparities):.3f}")

    print("\nImage spread:")
    print(f"  x range = [{np.min(xs):.1f}, {np.max(xs):.1f}], std={np.std(xs):.1f}")
    print(f"  y range = [{np.min(ys):.1f}, {np.max(ys):.1f}], std={np.std(ys):.1f}")

    print("\nValid observations per track:")
    print(f"  min    = {np.min(valid_counts):.0f}")
    print(f"  median = {np.median(valid_counts):.0f}")
    print(f"  mean   = {np.mean(valid_counts):.2f}")
    print(f"  max    = {np.max(valid_counts):.0f}")

    print("\nWorst shallow/low-disparity tracks:")
    rows_sorted = sorted(rows, key=lambda r: r["disparity"])
    for r in rows_sorted[:10]:
        print(
            f"track={r['track_id']}, "
            f"valid_obs={r['num_valid_obs_in_window']}, "
            f"disp={r['disparity']:.3f}, "
            f"depth={r['depth_triangulated']:.3f}, "
            f"pixel=({r['x']:.1f},{r['y']:.1f}), "
            f"frames={r['valid_frames']}"
        )

    # Plot pixel spread on image
    img_left, _ = read_images(target_frame)
    plt.figure(figsize=(12, 5))
    plt.imshow(img_left, cmap="gray")
    plt.scatter(xs, ys, s=20)
    plt.title(f"Frame {target_frame}: selected BA observations")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"debug_frame_{target_frame}_image_spread.png"), dpi=200)
    plt.close()

    # Plot depth histogram
    plt.figure(figsize=(8, 5))
    plt.hist(finite_depths, bins=30)
    plt.title(f"Frame {target_frame}: depth histogram")
    plt.xlabel("Triangulated depth Z [m]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"debug_frame_{target_frame}_depth_hist.png"), dpi=200)
    plt.close()

    # Plot valid observations histogram
    plt.figure(figsize=(8, 5))
    plt.hist(valid_counts, bins=range(1, int(np.max(valid_counts)) + 2))
    plt.title(f"Frame {target_frame}: valid observations per track")
    plt.xlabel("Valid observations in window")
    plt.ylabel("Track count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"debug_frame_{target_frame}_valid_obs_hist.png"), dpi=200)
    plt.close()

    return rows


# =============================================================================
# EXERCISE 6 BACKEND MODULES
# =============================================================================
from matplotlib.patches import Ellipse


def run_and_plot_prior_sensitivity(db, c0_idx, ck_idx, output_dir):
    """Executes bundle window optimizations over varying noise scales and saves 3D plots."""
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
        ax.set_title(f"6.1 – Bundle {c0_idx}→{ck_idx}  |  prior noise: {label}\n"
                     "Frame locations with marginal covariances", fontsize=10, fontweight='bold')

        for f_id in wf_s:
            key = gtsam.symbol('c', f_id)
            pose = result_s.atPose3(key)
            try:
                cov = marginal_s.marginalCovariance(key)
                gtsam_plot.plot_pose3_on_axes(ax, pose, axis_length=0.3, P=cov)
            except Exception:
                gtsam_plot.plot_pose3_on_axes(ax, pose, axis_length=0.3)

        ax.set_xlabel("X axis");
        ax.set_ylabel("Y axis");
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
    """Computes relative transformation and conditional covariance via Schur Complement."""
    br = solve_bundle_window(db, start_idx, end_idx)
    graph, result = br["graph"], br["result"]
    marginals = gtsam.Marginals(graph, result)

    key_0 = gtsam.symbol('c', start_idx)
    key_k = gtsam.symbol('c', end_idx)

    joint_cov = marginals.jointMarginalCovariance(gtsam.KeyVector([key_0, key_k])).fullMatrix()
    Sigma_00 = joint_cov[0:6, 0:6]
    Sigma_0k = joint_cov[0:6, 6:12]
    Sigma_k0 = joint_cov[6:12, 0:6]
    Sigma_kk = joint_cov[6:12, 6:12]

    Sigma_conditional_global = Sigma_kk - Sigma_k0 @ np.linalg.inv(Sigma_00) @ Sigma_0k
    relative_pose = br["relative_pose"]
    Ad_k = relative_pose.AdjointMap()
    Sigma_rel = Ad_k @ Sigma_conditional_global @ Ad_k.T
    return relative_pose, Sigma_rel


def compute_all_relative_constraints(db, keyframes):
    """Loops over sequential keyframes to gather valid pose graph constraints."""
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

    print(f"\nSuccessfully computed {len(relative_poses)}/{len(bundle_windows)} relative pose constraints.")
    print(f"\n{'Pair':<15} {'|t| [m]':>10}  {'det(Σ_rel)':>14}")
    print("-" * 43)
    for (sf, ef), rp in relative_poses.items():
        t_norm = float(np.linalg.norm(rp.translation()))
        det = float(np.linalg.det(relative_covs[(sf, ef)]))
        print(f"({sf:4d},{ef:4d})   {t_norm:10.4f}   {det:14.6e}")

    return relative_poses, relative_covs


def clean_pose_graph_edges(relative_poses, relative_covs):
    """Filters out constraints containing uninitialized or invalid covariance values."""
    cleaned_poses, cleaned_covs = {}, {}
    for edge in sorted(relative_poses.keys()):
        cov = relative_covs[edge]
        if np.any(np.diag(cov) <= 0) or np.any(np.isnan(cov)):
            print(f"Skipping bad covariance edge: {edge}")
            continue
        cleaned_poses[edge] = relative_poses[edge]
        cleaned_covs[edge] = relative_covs[edge]
    return cleaned_poses, cleaned_covs


def build_and_initialize_pose_graph(cleaned_poses, cleaned_covs):
    """Constructs the factor graph and provides robust chain initialization across gaps."""
    graph = build_pose_graph(cleaned_poses, cleaned_covs)
    initial = gtsam.Values()
    sorted_edges = sorted(cleaned_poses.keys())

    if len(sorted_edges) == 0:
        raise RuntimeError("No valid relative poses left after cleaning.")

    first_kf = sorted_edges[0][0]
    current_global_pose = gtsam.Pose3()
    initial.insert(gtsam.symbol("c", first_kf), current_global_pose)

    for start_kf, end_kf in sorted_edges:
        start_key = gtsam.symbol("c", start_kf)
        end_key = gtsam.symbol("c", end_kf)

        if not initial.exists(start_key):
            initial.insert(start_key, current_global_pose)
        else:
            current_global_pose = initial.atPose3(start_key)

        rel_pose = cleaned_poses[(start_kf, end_kf)]
        end_pose = current_global_pose.compose(rel_pose)

        if not initial.exists(end_key):
            initial.insert(end_key, end_pose)
            current_global_pose = end_pose

    # Fix safety isolates
    missing = []
    for i in range(graph.size()):
        factor = graph.at(i)
        for key in factor.keys():
            if not initial.exists(key):
                missing.append(gtsam.Symbol(key).index())
    if missing:
        print("Warning - Fixed missing keys that would have collapsed to 0:", sorted(set(missing)))
        for m_key in missing:
            initial.insert(gtsam.symbol("c", m_key), gtsam.Pose3())

    return graph, initial


def optimize_pose_graph(graph, initial):
    """Optimizes the pose graph factor collection via Levenberg-Marquardt."""
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()
    marginals = gtsam.Marginals(graph, result)
    return result, marginals


def plot_pose_graph_2d_ellipses(result, marginals, output_path, step=1, sigma_scale=20):
    """Generates a top-down birds-eye-view plotting 20-sigma confidence ellipses."""
    frame_ids, positions = extract_pose_graph_positions(result)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(positions[:, 0], positions[:, 2], label="optimised trajectory", color="#1f77b4", linewidth=1.2)
    ax.scatter(positions[0, 0], positions[0, 2], color="black", marker="s", s=80, label="c_0", zorder=5)

    for frame_id in frame_ids[::step]:
        key = gtsam.symbol("c", frame_id)
        if not result.exists(key):
            continue
        pose = result.atPose3(key)
        p = pose_translation_np(pose)

        try:
            cov6 = marginals.marginalCovariance(key)
            cov_trans = cov6[3:6, 3:6]
            cov_xz = cov_trans[np.ix_([0, 2], [0, 2])]

            eigvals, eigvecs = np.linalg.eigh(cov_xz)
            eigvals = np.maximum(eigvals, 0.0)

            width = 2 * sigma_scale * np.sqrt(eigvals[0])
            height = 2 * sigma_scale * np.sqrt(eigvals[1])
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

            ellipse = Ellipse(
                xy=(p[0], p[2]), width=width, height=height, angle=angle,
                edgecolor="red", facecolor="none", alpha=0.4, linewidth=0.8
            )
            ax.add_patch(ellipse)
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
    print(f"Saved 2D Covariance Plot → {output_path}")

    # =============================================================================


# EXERCISE 7 BACKEND MODULES
# =============================================================================
import networkx as nx


def _shortest_path_pose_graph(relative_poses, relative_covs):
    """Builds a NetworkX graph weighted by the trace of relative covariances."""
    G = nx.Graph()
    for (start_kf, end_kf), cov in relative_covs.items():
        weight = float(np.trace(cov))
        G.add_edge(start_kf, end_kf, weight=weight)
    return G


def _get_edge_pose_and_cov(u, v, relative_poses, relative_covs):
    """Retrieves relative pose/cov, handles path inversion via Adjoint propagation."""
    if (u, v) in relative_poses:
        return relative_poses[(u, v)], relative_covs[(u, v)]

    if (v, u) in relative_poses:
        pose_vu = relative_poses[(v, u)]
        cov_vu = relative_covs[(v, u)]

        pose_uv = pose_vu.inverse()
        Ad_inv = pose_uv.AdjointMap()
        cov_uv = Ad_inv @ cov_vu @ Ad_inv.T
        return pose_uv, cov_uv

    raise KeyError(f"No pose-graph edge between keyframes {u} and {v}")


def detect_loop_closure_candidates(relative_poses, relative_covs, keyframes, mahalanobis_threshold):
    """Finds historical matches within a covariance uncertainty threshold using Dijkstra search."""
    G = _shortest_path_pose_graph(relative_poses, relative_covs)
    loop_candidates = {}
    total_candidates_counter = 0

    for n_idx, c_n in enumerate(keyframes):
        if n_idx == 0:
            continue

        loop_candidates[c_n] = []
        if c_n not in G:
            continue

        print(f"Processing Keyframe c_{c_n} ({n_idx}/{len(keyframes) - 1})...")

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

                delta_c_ni = suffix_pose
                xi_ni = gtsam.Pose3.Logmap(delta_c_ni)
                Sigma_inv = np.linalg.inv(Sigma_rel + np.eye(6) * 1e-6)
                mahalanobis_dist = xi_ni.T @ Sigma_inv @ xi_ni

                if mahalanobis_dist < mahalanobis_threshold:
                    loop_candidates[c_n].append({
                        "candidate_kf": c_i,
                        "mahalanobis_distance": mahalanobis_dist,
                        "relative_pose_estimate": delta_c_ni,
                        "relative_covariance": Sigma_rel,
                        "path": path,
                    })
                    total_candidates_counter += 1

            except nx.NetworkXNoPath:
                continue
            except Exception:
                continue

        num_found = len(loop_candidates[c_n])
        if num_found > 0:
            print(f"  --> Found {num_found} loop closure candidates for c_{c_n}!")
            for cand in loop_candidates[c_n]:
                print(
                    f"    * Candidate c_{cand['candidate_kf']}: Mahalanobis Dist = {cand['mahalanobis_distance']:.4f} "
                    f"(path length {len(cand['path']) - 1})")

    return loop_candidates, total_candidates_counter


def plot_loop_candidates(keyframes, candidates, optimized_values, output_dir="."):
    """Plots the estimated pose trajectory overlaying active loop-closure shortcuts."""
    kf_positions = {}
    for kf in keyframes:
        key = gtsam.symbol('c', kf)
        if not optimized_values.exists(key):
            continue
        pose = optimized_values.atPose3(key)
        pos_3d = pose.translation()
        kf_positions[kf] = (pos_3d[0], pos_3d[2])

    plotted_keyframes = [kf for kf in keyframes if kf in kf_positions]
    positions_array = np.array([kf_positions[kf] for kf in plotted_keyframes])

    plt.figure(figsize=(12, 9))
    plt.plot(positions_array[:, 0], positions_array[:, 1], color='gray', linestyle='-', alpha=0.5,
             label='Estimated Trajectory')
    plt.scatter(positions_array[:, 0], positions_array[:, 1], color='black', s=5, alpha=0.3)

    has_drawn_candidate_label = False
    has_drawn_link_label = False

    for c_n, cands in candidates.items():
        if not cands or c_n not in kf_positions:
            continue

        x_n, z_n = kf_positions[c_n]
        plt.scatter(x_n, z_n, color='green', s=30, zorder=3,
                    label='Current Frame ($c_n$)' if not has_drawn_candidate_label else "")
        has_drawn_candidate_label = True

        for cand in cands:
            c_i = cand["candidate_kf"]
            if c_i not in kf_positions:
                continue
            x_i, z_i = kf_positions[c_i]

            plt.scatter(x_i, z_i, color='magenta', s=15, zorder=3)
            plt.plot([x_n, x_i], [z_n, z_i], color='red', linestyle='--', alpha=0.6, linewidth=1.0,
                     label='Loop Closure Constraint' if not has_drawn_link_label else "")
            has_drawn_link_label = True

    plt.title("Q7.1: Detected Loop Closure Candidates on the ESTIMATED Trajectory (Covariance-Weighted)", fontsize=12,
              fontweight='bold')
    plt.xlabel("X (Width) [m]")
    plt.ylabel("Z (Depth / Forward) [m]")
    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "task_7_1_loop_candidates_trajectory.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\n[Plot] Saved loop candidates visualization to: {output_path}")


def verify_loop_closures_consensus(db, loop_candidates, inlier_threshold, output_dir):
    """Performs AKAZE matching and Fundamental RANSAC verification to confirm loops."""
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    verified_loops = {}
    total_verified_loops = 0

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
            print(
                f"[Q7.2] Testing Link c_{c_n} -> c_{c_i}: Geometric Inliers = {inliers_count} (Required: {inlier_threshold})")

            if inliers_count >= inlier_threshold:
                print(f"  --> [SUCCESS] Loop Verified! Inliers: {inliers_count}")

                inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]
                outlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 0]

                verified_loops[c_n].append({
                    "candidate_kf": c_i,
                    "inliers_count": inliers_count,
                    "inlier_matches": inlier_matches
                })
                total_verified_loops += 1

                img_i_color = cv2.cvtColor(img_i, cv2.COLOR_GRAY2RGB)
                img_n_color = cv2.cvtColor(img_n, cv2.COLOR_GRAY2RGB)

                for match in outlier_matches:
                    pt_i = tuple(map(int, kp_i[match.queryIdx].pt))
                    pt_n = tuple(map(int, kp_n[match.trainIdx].pt))
                    cv2.circle(img_i_color, pt_i, 3, (0, 255, 255), -1)
                    cv2.circle(img_n_color, pt_n, 3, (0, 255, 255), -1)

                for match in inlier_matches:
                    pt_i = tuple(map(int, kp_i[match.queryIdx].pt))
                    pt_n = tuple(map(int, kp_n[match.trainIdx].pt))
                    cv2.circle(img_i_color, pt_i, 3, (255, 165, 0), -1)
                    cv2.circle(img_n_color, pt_n, 3, (255, 165, 0), -1)

                fig, axes = plt.subplots(2, 1, figsize=(15, 10))
                axes[0].imshow(img_i_color)
                axes[0].set_title(f"Image 1: Inliers (orange), Outliers (cyan) | Frame c_{c_i}")
                axes[0].axis('off')

                axes[1].imshow(img_n_color)
                axes[1].set_title(
                    f"Image 2: Inliers (orange), Outliers (cyan) | Frame c_{c_n} (Total Inliers: {inliers_count})")
                axes[1].axis('off')

                plt.tight_layout()
                os.makedirs(output_dir, exist_ok=True)
                out_img_path = os.path.join(output_dir, f"task_7_2_loop_{c_i}_{c_n}.png")
                plt.savefig(out_img_path, dpi=200, bbox_inches='tight')
                plt.close()

    return verified_loops, total_verified_loops


def _left_stereo_query_to_point_index(frame_data):
    """Map left keypoint index -> triangulated 3D point index for a run_single_pair result."""
    return {m.queryIdx: idx for idx, m in enumerate(frame_data["stereo_inliers"])}


def _left_stereo_query_to_match(frame_data):
    """Map left keypoint index -> stereo match for a run_single_pair result."""
    return {m.queryIdx: m for m in frame_data["stereo_inliers"]}


def _safe_stereo_factor(obs, noise_model, pose_key, point_key, K_gtsam):
    return gtsam.GenericStereoFactor3D(
        gtsam.StereoPoint2(float(obs[0]), float(obs[1]), float(obs[2])),
        noise_model,
        pose_key,
        point_key,
        K_gtsam,
    )


def _stereo_obs_from_feature(frame_data, left_feature_idx, stereo_match):
    """Return GTSAM stereo observation tuple (uL, uR, v)."""
    kp_l = frame_data["kp_left"][left_feature_idx]
    kp_r = frame_data["kp_right"][stereo_match.trainIdx]
    return (kp_l.pt[0], kp_r.pt[0], kp_l.pt[1])


def estimate_loop_relative_pose_bundle(db, c_i, c_n, inlier_matches=None,
                                       max_landmarks=120,
                                       min_landmarks=8,
                                       output_dir="."):
    """
    Q7.3: Estimate a loop-closure relative pose c_i -> c_n with a small 2-frame stereo BA.

    Returned relative_pose convention matches the pose graph convention used in exercise 6:
        relative_pose = pose_i.between(pose_n)
    where both poses are GTSAM camera-to-world Pose3 objects.
    Since pose_i is fixed to identity in this local bundle, the marginal covariance of pose_n
    is the required conditional covariance of the relative measurement.
    """
    K_gtsam = init_gtsam_stereo_calibration()
    K_mat, _, _ = read_cameras()

    data_i = run_single_pair(c_i, display=False, plot_3d=False)
    data_n = run_single_pair(c_n, display=False, plot_3d=False)

    # If matches were not supplied, recompute exactly the same visual verification stage.
    if inlier_matches is None:
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = bf_matcher.knnMatch(data_i["des_left"], data_n["des_left"], k=2)
        good = [m for m, nn in knn if m.distance < 0.7 * nn.distance]
        if len(good) < 4:
            raise RuntimeError(f"Loop {c_i}->{c_n}: not enough AKAZE matches")
        pts_i = np.array([data_i["kp_left"][m.queryIdx].pt for m in good], dtype=np.float32)
        pts_n = np.array([data_n["kp_left"][m.trainIdx].pt for m in good], dtype=np.float32)
        _, mask = cv2.findFundamentalMat(pts_i, pts_n, cv2.FM_RANSAC, 3.0, 0.99)
        if mask is None:
            raise RuntimeError(f"Loop {c_i}->{c_n}: fundamental matrix failed")
        inlier_matches = [good[k] for k in range(len(good)) if mask[k][0] == 1]

    i_q_to_pt = _left_stereo_query_to_point_index(data_i)
    i_q_to_stereo = _left_stereo_query_to_match(data_i)
    n_q_to_stereo = _left_stereo_query_to_match(data_n)

    usable = []
    for m in inlier_matches:
        if m.queryIdx not in i_q_to_pt:
            continue
        if m.queryIdx not in i_q_to_stereo:
            continue
        if m.trainIdx not in n_q_to_stereo:
            continue
        X_i = np.array(data_i["points_3d"][i_q_to_pt[m.queryIdx]], dtype=np.float64).reshape(3)
        if np.all(np.isfinite(X_i)) and 2.0 < X_i[2] < 120.0:
            usable.append((m, X_i))

    if len(usable) < min_landmarks:
        raise RuntimeError(f"Loop {c_i}->{c_n}: only {len(usable)} usable stereo landmarks")

    usable = sorted(usable, key=lambda item: item[0].distance)[:max_landmarks]

    # PnP gives X_n = R_ni * X_i + t_ni. For GTSAM c2w pose of n in i coordinates,
    # use the inverse: X_i = R_ni^T * X_n - R_ni^T t_ni.
    obj_pts = np.array([X for _, X in usable], dtype=np.float32)
    img_pts = np.array([data_n["kp_left"][m.trainIdx].pt for m, _ in usable], dtype=np.float32)
    success, rvec, tvec, pnp_inliers = cv2.solvePnPRansac(
        obj_pts, img_pts, K_mat, None,
        iterationsCount=200,
        reprojectionError=3.0,
        confidence=0.99,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or pnp_inliers is None or len(pnp_inliers) < min_landmarks:
        raise RuntimeError(f"Loop {c_i}->{c_n}: PnP failed")

    R_ni, _ = cv2.Rodrigues(rvec)
    t_ni = tvec.reshape(3)
    init_pose_n = gtsam.Pose3(
        gtsam.Rot3(R_ni.T),
        gtsam.Point3(*(-R_ni.T @ t_ni).reshape(3)),
    )

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    key_i = symbol("c", int(c_i))
    key_n = symbol("c", int(c_n))
    initial.insert(key_i, gtsam.Pose3())
    initial.insert(key_n, init_pose_n)

    # Fix first loop frame. The second pose covariance is therefore conditional on the first.
    graph.add(gtsam.PriorFactorPose3(
        key_i,
        gtsam.Pose3(),
        gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6),
    ))

    meas_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(2.0),
        gtsam.noiseModel.Isotropic.Sigma(3, 1.0),
    )

    kept_indices = set(int(x[0]) for x in pnp_inliers.reshape(-1, 1))
    factor_landmarks = 0
    for local_idx, (m, X_i) in enumerate(usable):
        if local_idx not in kept_indices:
            continue
        point_key = symbol("q", int(10_000_000 + c_i * 10_000 + c_n * 10 + local_idx))
        initial.insert(point_key, gtsam.Point3(float(X_i[0]), float(X_i[1]), float(X_i[2])))

        obs_i = _stereo_obs_from_feature(data_i, m.queryIdx, i_q_to_stereo[m.queryIdx])
        obs_n = _stereo_obs_from_feature(data_n, m.trainIdx, n_q_to_stereo[m.trainIdx])
        graph.add(_safe_stereo_factor(obs_i, meas_noise, key_i, point_key, K_gtsam))
        graph.add(_safe_stereo_factor(obs_n, meas_noise, key_n, point_key, K_gtsam))
        factor_landmarks += 1

    if factor_landmarks < min_landmarks:
        raise RuntimeError(f"Loop {c_i}->{c_n}: only {factor_landmarks} BA landmarks after PnP filtering")

    initial_error = graph.error(initial)
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    final_error = graph.error(result)

    pose_i = result.atPose3(key_i)
    pose_n = result.atPose3(key_n)
    relative_pose = pose_i.between(pose_n)

    marginals = gtsam.Marginals(graph, result)
    rel_cov = marginals.marginalCovariance(key_n)
    rel_cov = 0.5 * (rel_cov + rel_cov.T) + np.eye(6) * 1e-9

    return {
        "start_kf": int(c_i),
        "end_kf": int(c_n),
        "relative_pose": relative_pose,
        "relative_covariance": rel_cov,
        "num_ba_landmarks": factor_landmarks,
        "initial_error": float(initial_error),
        "final_error": float(final_error),
        "graph": graph,
        "initial": initial,
        "result": result,
    }


def estimate_verified_loop_relative_poses(db, verified_loops, output_dir="."):
    """Run Q7.3 for every verified loop from Q7.2."""
    loop_measurements = []
    for c_n, loops in verified_loops.items():
        for loop in loops:
            c_i = loop["candidate_kf"]
            try:
                m = estimate_loop_relative_pose_bundle(
                    db, c_i, c_n,
                    inlier_matches=loop.get("inlier_matches"),
                    output_dir=output_dir,
                )
                loop_measurements.append(m)
                print(f"[Q7.3] Loop {c_i}->{c_n}: BA landmarks={m['num_ba_landmarks']}, "
                      f"error {m['initial_error']:.2f}->{m['final_error']:.2f}")
            except Exception as e:
                print(f"[Q7.3] Loop {c_i}->{c_n}: FAILED ({e})")
    return loop_measurements


def add_loop_closures_and_optimize(cleaned_poses, cleaned_covs, loop_measurements,
                                   output_dir=".", snapshot_count=4):
    """Q7.4: add loop BetweenFactorPose3 constraints one-by-one and re-optimize."""

    relative_poses_lc = dict(cleaned_poses)
    relative_covs_lc = dict(cleaned_covs)

    graph0, initial0 = build_and_initialize_pose_graph(cleaned_poses, cleaned_covs)
    no_loop_result, no_loop_marginals = optimize_pose_graph(graph0, initial0)

    snapshots = [("before_loop_closures", no_loop_result, no_loop_marginals, 0)]

    total = len(loop_measurements)
    snapshot_after = set()
    if total > 0:
        snapshot_after = set(
            np.unique(
                np.linspace(
                    1,
                    total,
                    min(snapshot_count - 1, total),
                    dtype=int
                )
            )
        )

    final_graph = graph0
    final_result = no_loop_result
    final_marginals = no_loop_marginals

    added_count = 0

    for meas in loop_measurements:

        start_kf = meas["start_kf"]
        end_kf = meas["end_kf"]
        edge = (start_kf, end_kf)

        pose_i = no_loop_result.atPose3(symbol("c", start_kf))
        pose_j = no_loop_result.atPose3(symbol("c", end_kf))

        pred_rel = pose_i.between(pose_j)
        meas_rel = meas["relative_pose"]

        pred_t = np.array(pred_rel.translation()).reshape(3)
        meas_t = np.array(meas_rel.translation()).reshape(3)

        diff_direct = np.linalg.norm(pred_t - meas_t)

        rot_err = pred_rel.rotation().between(meas_rel.rotation()).rpy()
        rot_err_norm = np.linalg.norm(rot_err)

        print(
            f"[Q7.4 debug] {start_kf}->{end_kf} "
            f"direct={diff_direct:.2f} "
            f"rot={rot_err_norm:.3f} "
            f"landmarks={meas['num_ba_landmarks']}"
        )

        loop_pose = meas_rel

        if meas["num_ba_landmarks"] < 40:
            print("  rejected: too few landmarks")
            continue

        if diff_direct > 12.0:
            print("  rejected: translation mismatch")
            continue

        if rot_err_norm > 0.35:
            print("  rejected: rotation mismatch")
            continue

        loop_cov = np.array(meas["relative_covariance"])
        loop_cov = 0.5 * (loop_cov + loop_cov.T)
        loop_cov *= 100.0
        loop_cov += np.eye(6) * 1e-3

        relative_poses_lc[edge] = loop_pose
        relative_covs_lc[edge] = loop_cov

        added_count += 1

        final_graph, initial = build_and_initialize_pose_graph(
            relative_poses_lc,
            relative_covs_lc
        )

        final_result, final_marginals = optimize_pose_graph(
            final_graph,
            initial
        )

        if added_count in snapshot_after:
            snapshots.append(
                (
                    f"after_{added_count}_loop_closures",
                    final_result,
                    final_marginals,
                    added_count
                )
            )
    if snapshots[-1][3] != added_count:
        snapshots.append(
            (
                f"after_{added_count}_loop_closures",
                final_result,
                final_marginals,
                added_count
            )
        )
    for name, values, marginals, idx in snapshots:
        plot_pose_graph_2d_ellipses(
            values,
            marginals,
            output_path=os.path.join(
                output_dir,
                f"task_7_5_snapshot_{idx:02d}_{name}.png"
            ),
            step=5,
            sigma_scale=20,
        )


    return {
        "no_loop_graph": graph0,
        "no_loop_result": no_loop_result,
        "no_loop_marginals": no_loop_marginals,
        "loop_graph": final_graph,
        "loop_result": final_result,
        "loop_marginals": final_marginals,
        "relative_poses_with_loops": relative_poses_lc,
        "relative_covs_with_loops": relative_covs_lc,
        "snapshots": snapshots,
        "num_added_loop_closures": added_count,
    }


def _gt_positions_for_frame_ids(frame_ids):
    gt = read_ground_truth_poses()
    pts = []
    valid = []
    for f in frame_ids:
        if f < len(gt):
            R, t = gt[f]
            pts.append(camera_center(R, t))
            valid.append(f)
    return valid, np.array(pts)


def plot_pose_graph_vs_ground_truth(no_loop_values, loop_values, output_dir="."):
    ids_no, pos_no = extract_pose_graph_positions(no_loop_values)
    ids_lc, pos_lc = extract_pose_graph_positions(loop_values)
    _, gt_no = _gt_positions_for_frame_ids(ids_no)

    plt.figure(figsize=(10, 8))
    plt.plot(pos_no[:, 0], pos_no[:, 2], "o-", markersize=3, linewidth=1, label="without loop closures")
    plt.plot(pos_lc[:, 0], pos_lc[:, 2], "o-", markersize=3, linewidth=1, label="with loop closures")
    if len(gt_no) == len(pos_no):
        plt.plot(gt_no[:, 0], gt_no[:, 2], "--", linewidth=2, label="ground truth")
    plt.title("Q7.5: Pose Graph vs Ground Truth")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "task_7_5_pose_graph_vs_ground_truth.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved → {path}")


def plot_absolute_location_error(no_loop_values, loop_values, output_dir="."):
    ids_no, pos_no = extract_pose_graph_positions(no_loop_values)
    ids_lc, pos_lc = extract_pose_graph_positions(loop_values)
    _, gt_no = _gt_positions_for_frame_ids(ids_no)
    _, gt_lc = _gt_positions_for_frame_ids(ids_lc)

    err_no = np.linalg.norm(pos_no[:len(gt_no)] - gt_no, axis=1)
    err_lc = np.linalg.norm(pos_lc[:len(gt_lc)] - gt_lc, axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(ids_no[:len(err_no)], err_no, marker="o", linewidth=1, label="without loop closures")
    plt.plot(ids_lc[:len(err_lc)], err_lc, marker="o", linewidth=1, label="with loop closures")
    plt.title("Q7.5: Absolute Location Error")
    plt.xlabel("Keyframe")
    plt.ylabel("Position error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "task_7_5_absolute_location_error.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved → {path}")


def _uncertainty_trace(values, marginals):
    ids, _ = extract_pose_graph_positions(values)
    sizes = []
    good_ids = []
    for f in ids:
        key = symbol("c", int(f))
        try:
            cov6 = marginals.marginalCovariance(key)
            cov_xz = cov6[np.ix_([3, 5], [3, 5])]
            sizes.append(float(np.sqrt(max(np.linalg.det(cov_xz), 0.0))))
            good_ids.append(f)
        except Exception:
            pass
    return good_ids, np.array(sizes)


def plot_location_uncertainty_size(no_loop_values, no_loop_marginals,
                                   loop_values, loop_marginals,
                                   output_dir="."):
    """Uncertainty size = sqrt(det(Sigma_xz)), i.e. 1-sigma covariance ellipse area / pi."""
    ids_no, unc_no = _uncertainty_trace(no_loop_values, no_loop_marginals)
    ids_lc, unc_lc = _uncertainty_trace(loop_values, loop_marginals)

    plt.figure(figsize=(12, 5))
    plt.plot(ids_no, unc_no, marker="o", linewidth=1, label="without loop closures")
    plt.plot(ids_lc, unc_lc, marker="o", linewidth=1, label="with loop closures")
    plt.title("Q7.5: Location Uncertainty Size")
    plt.xlabel("Keyframe")
    plt.ylabel(r"$\sqrt{det(\Sigma_{xz})}$ [m²]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "task_7_5_location_uncertainty_size.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Saved → {path}")
