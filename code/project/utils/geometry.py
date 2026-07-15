"""
utils/geometry.py
Spatial transformations, camera projection, feature extraction, and triangulation utilities.
"""

from pathlib import Path
import numpy as np
import cv2

# =============================================================================
# CONSTANTS & SETUP
# =============================================================================
# Navigate from project/utils/geometry.py up to VAN_ex/ root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'dataset' / 'dataset' / 'sequences' / '00'


def read_images(idx):
    """Reads grayscale left and right images for a given frame index."""
    img_name = f'{idx:06d}.png'
    p1, p2 = DATA_PATH / 'image_0' / img_name, DATA_PATH / 'image_1' / img_name
    img1, img2 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE), cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not find images at {DATA_PATH}")
    return img1, img2


def get_num_frames():
    """Returns the total number of stereo frames available in sequence '00'."""
    image_dir = DATA_PATH / 'image_0'
    return len(list(image_dir.glob("*.png")))


def read_cameras():
    """Loads camera calibration matrices (K, P1, P2) from calib.txt."""
    calib_path = DATA_PATH / 'calib.txt'
    with open(calib_path) as f:
        l1 = f.readline().split()[1:]
        l2 = f.readline().split()[1:]

    p1 = np.array([float(i) for i in l1]).reshape(3, 4)
    p2 = np.array([float(i) for i in l2]).reshape(3, 4)
    k = p1[:, :3]
    return k, p1, p2


def read_ground_truth_poses():
    """Reads KITTI benchmark ground truth camera matrices (3x4)."""
    poses_path = PROJECT_ROOT / 'dataset' / 'dataset' / 'poses' / '00.txt'
    gt_poses = []
    with open(poses_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            M = np.array(values).reshape(3, 4)
            gt_poses.append((M[:, :3], M[:, 3].reshape(3, 1)))
    return gt_poses


def get_orb_features(img, n_features=700):
    """Extracts keypoints and ORB descriptors from an image."""
    orb = cv2.ORB_create(nfeatures=n_features)
    return orb.detectAndCompute(img, None)


def get_akaze_features(img):
    """Extracts keypoints and AKAZE descriptors from an image."""
    akaze = cv2.AKAZE_create(threshold=0.0001)
    return akaze.detectAndCompute(img, None)


def camera_center(R, t):
    """Calculates 3D world position of a camera center from extrinsics: C = -R^T * t."""
    return (-R.T @ t).flatten()


def compose_transform(R1, t1, R2, t2):
    """Composes relative motion transformations: T_new = T2 * T1."""
    return R2 @ R1, R2 @ t1 + t2


def rotation_angle_deg(R):
    """Computes the magnitude of a rotation matrix in degrees via the Rodrigues vector norm."""
    rvec, _ = cv2.Rodrigues(R)
    return float(np.linalg.norm(rvec) * 180.0 / np.pi)


def relative_rotation_angle_deg(R_a, R_b):
    """Computes the angular difference in degrees between two rotation matrices of the same convention."""
    return rotation_angle_deg(R_a.T @ R_b)


def relative_pose_w2c(R_a, t_a, R_b, t_b):
    """Computes the relative W2C transform mapping frame a's camera frame into frame b's, by inverting a and composing with b."""
    return compose_transform(R_a.T, -R_a.T @ t_a, R_b, t_b)


def compute_relative_pose_error_deg_m(R_est, t_est, R_gt, t_gt):
    """Computes the SE(3) residual (location error in meters, angle error in degrees) between an estimated and ground-truth relative pose."""
    R_err, t_err = compose_transform(R_est.T, -R_est.T @ t_est, R_gt, t_gt)
    return float(np.linalg.norm(t_err)), rotation_angle_deg(R_err)


def compute_kitti_sequence_errors(poses, gt_poses, segment_length):
    """Computes KITTI-style normalized relative location/angle error over all overlapping segments of a fixed frame length."""
    n = min(len(poses), len(gt_poses))
    loc_err_pct, ang_err_per_m = [], []
    for start in range(0, n - segment_length):
        end = start + segment_length
        try:
            R_est_rel, t_est_rel = relative_pose_w2c(*poses[start], *poses[end])
            R_gt_rel, t_gt_rel = relative_pose_w2c(*gt_poses[start], *gt_poses[end])
            loc_err, ang_err = compute_relative_pose_error_deg_m(R_est_rel, t_est_rel, R_gt_rel, t_gt_rel)
            total_distance = sum(np.linalg.norm(camera_center(*gt_poses[i+1]) - camera_center(*gt_poses[i]))
                                  for i in range(start, end))
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


def project_point(P, X):
    """Projects a single 3D point X using projection matrix P into pixel space (u, v)."""
    X_h = np.append(X, 1.0)
    x_h = P @ X_h
    if abs(x_h[2]) < 1e-12:
        return np.array([np.nan, np.nan])
    return x_h[:2] / x_h[2]


def project_points_vectorized(P, X):
    """Projects batch N 3D points X into 2D pixel coordinates (N, 2)."""
    X_h = np.hstack([X, np.ones((X.shape[0], 1))])
    x_h = (P @ X_h.T).T

    valid = np.abs(x_h[:, 2]) > 1e-12
    projected = np.full((X.shape[0], 2), np.nan)
    projected[valid] = x_h[valid, :2] / x_h[valid, 2:3]
    return projected


def compute_rectified_stereo_deviations(kp_left, kp_right, matches):
    """Computes absolute y-coordinate pixel discrepancies between matched stereo keypoints."""
    deviations = []
    for match in matches:
        _, y_left = kp_left[match.queryIdx].pt
        _, y_right = kp_right[match.trainIdx].pt
        deviations.append(abs(y_left - y_right))
    return np.array(deviations)


def split_matches_by_rectified_pattern(kp_left, kp_right, matches, threshold=2):
    """Splits feature matches into inliers and outliers based on y-coordinate disparity tolerance."""
    inliers, outliers = [], []
    for match in matches:
        _, y_left = kp_left[match.queryIdx].pt
        _, y_right = kp_right[match.trainIdx].pt
        if abs(y_left - y_right) <= threshold:
            inliers.append(match)
        else:
            outliers.append(match)
    return inliers, outliers


def get_matched_points(kp1, kp2, matches):
    """Extracts 2D coordinate arrays from feature matches."""
    points1 = [kp1[m.queryIdx].pt for m in matches]
    points2 = [kp2[m.trainIdx].pt for m in matches]
    return np.array(points1), np.array(points2)


def triangulate_point_linear(p1, p2, m1, m2):
    """Triangulates a single 3D point from 2D stereo observations using linear SVD."""
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
    """Triangulates multiple 3D points iteratively using the linear SVD method."""
    return np.array([triangulate_point_linear(p1, p2, m1, m2) for p1, p2 in zip(points1, points2)])


def triangulate_points_opencv(points1, points2, m1, m2):
    """Triangulates 3D points in batch using OpenCV's optimized cv2.triangulatePoints."""
    points_4d = cv2.triangulatePoints(m1, m2, points1.T, points2.T)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    return points_3d.T


def median_3d_distance(points_a, points_b):
    """Computes median Euclidean distance between two corresponding sets of 3D points."""
    return np.median(np.linalg.norm(points_a - points_b, axis=1))


def crop_around_point(img, x, y, crop_size=20):
    """Crops a square image region around a 2D point coordinate."""
    h, w = img.shape[:2]
    half = crop_size // 2
    x_i, y_i = int(round(x)), int(round(y))

    x1, x2 = max(0, x_i - half), min(w, x_i + half)
    y1, y2 = max(0, y_i - half), min(h, y_i + half)

    return img[y1:y2, x1:x2], x - x1, y - y1


def valid_stereo_obs(obs, min_disp=1.0):
    """Validates stereo observations against negative or near-zero disparity degeneracies."""
    return (obs.x_left - obs.x_right) >= min_disp


def run_custom_pnp_ransac(frame0_data, frame1_data, good_temporal_matches, iterations=50, threshold=2,
                          early_stop_ratio=0.78, max_no_improvement=12, pnp_method=cv2.SOLVEPNP_EPNP):
    """Custom PnP-RANSAC with dynamic stopping: samples 4 correspondences, solves PnP, evaluates support."""
    import random
    k, _, _ = read_cameras()
    correspondences = build_pnp_correspondences(frame0_data, frame1_data, good_temporal_matches)

    if len(correspondences) < 4:
        raise RuntimeError("Not enough correspondences for PnP-RANSAC.")

    best_inliers, best_R, best_t, no_improvement = [], None, None, 0

    for i in range(iterations):
        sample = random.sample(correspondences, 4)
        obj_pts = np.array([c['X'] for c in sample], dtype=np.float32)
        img_pts = np.array([c['obs_left1'] for c in sample], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, k, None, flags=pnp_method)
        if not success:
            no_improvement += 1
            continue

        R, _ = cv2.Rodrigues(rvec)
        inliers, _ = evaluate_supporters(correspondences, frame0_data, frame1_data, R, tvec, threshold)

        if len(inliers) > len(best_inliers):
            best_inliers, best_R, best_t, no_improvement = inliers, R, tvec, 0
            if len(best_inliers) > early_stop_ratio * len(correspondences):
                break
        else:
            no_improvement += 1

        if no_improvement >= max_no_improvement:
            break

    if best_R is None or len(best_inliers) < 4:
        raise RuntimeError("RANSAC failed to find a valid pose.")

    obj_pts = np.array([c['X'] for c in best_inliers], dtype=np.float32)
    img_pts = np.array([c['obs_left1'] for c in best_inliers], dtype=np.float32)
    rvec_guess, _ = cv2.Rodrigues(best_R)

    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, k, None, rvec=rvec_guess, tvec=best_t,
                                        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        refined_R, _ = cv2.Rodrigues(rvec)
        final_inliers, final_outliers = evaluate_supporters(correspondences, frame0_data, frame1_data,
                                                             refined_R, tvec, threshold)
        if len(final_inliers) >= 0.8 * len(best_inliers):
            return refined_R, tvec, final_inliers, final_outliers

    final_inliers, final_outliers = evaluate_supporters(correspondences, frame0_data, frame1_data,
                                                         best_R, best_t, threshold)
    return best_R, best_t, final_inliers, final_outliers


def find_four_view_supporters(frame0_data, frame1_data, good_temporal_matches, R, t, threshold=2):
    """Finds temporal matches whose 3D point reprojects acceptably into all four images."""
    k, m_left0, m_right0 = read_cameras()
    t_stereo = np.linalg.inv(k) @ m_right0[:, 3]
    P_left1 = k @ np.hstack([R, t.reshape(3, 1)])
    P_right1 = k @ np.hstack([R, (t.reshape(3) + t_stereo).reshape(3, 1)])

    left0_to_3d = {m.queryIdx: (idx, m) for idx, m in enumerate(frame0_data['stereo_inliers'])}
    left1_to_stereo = {m.queryIdx: m for m in frame1_data['stereo_inliers']}

    supporters, non_supporters = [], []
    for temporal_match in good_temporal_matches:
        if temporal_match.queryIdx not in left0_to_3d or temporal_match.trainIdx not in left1_to_stereo:
            continue

        point_3d_idx, stereo_match0 = left0_to_3d[temporal_match.queryIdx]
        stereo_match1 = left1_to_stereo[temporal_match.trainIdx]
        X = frame0_data['points_3d'][point_3d_idx]

        obs_left0 = np.array(frame0_data['kp_left'][stereo_match0.queryIdx].pt)
        obs_right0 = np.array(frame0_data['kp_right'][stereo_match0.trainIdx].pt)
        obs_left1 = np.array(frame1_data['kp_left'][temporal_match.trainIdx].pt)
        obs_right1 = np.array(frame1_data['kp_right'][stereo_match1.trainIdx].pt)

        proj_left0 = project_point(m_left0, X)
        proj_right0 = project_point(m_right0, X)
        proj_left1 = project_point(P_left1, X)
        proj_right1 = project_point(P_right1, X)

        errors = [
            np.linalg.norm(proj_left0 - obs_left0),
            np.linalg.norm(proj_right0 - obs_right0),
            np.linalg.norm(proj_left1 - obs_left1),
            np.linalg.norm(proj_right1 - obs_right1),
        ]

        (supporters if all(e < threshold for e in errors) else non_supporters).append(temporal_match)

    return supporters, non_supporters


def estimate_next_pose_with_rejection(prev_data, curr_data, R_global, t_global, step_size_limit=3.0, min_inliers=30):
    """One iteration of trajectory tracking: temporal match → RANSAC → step-size/inlier rejection → compose."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(prev_data['des_left'], curr_data['des_left'], k=2)
    good_temporal_matches = [m for m, n in knn_matches if m.distance < 0.7 * n.distance]

    if not good_temporal_matches:
        return None, None, None, None

    try:
        R_rel, t_rel, inliers, outliers = run_custom_pnp_ransac(prev_data, curr_data, good_temporal_matches,
                                                                  iterations=50, threshold=2)
    except RuntimeError:
        return None, None, None, None

    if len(inliers) < min_inliers:
        return None, None, None, None

    C_prev = camera_center(R_global, t_global)
    R_candidate, t_candidate = compose_transform(R_global, t_global, R_rel, t_rel)
    C_candidate = camera_center(R_candidate, t_candidate)
    step_size = np.linalg.norm(C_candidate - C_prev)

    if step_size > step_size_limit:
        return None, None, None, None

    return R_candidate, t_candidate, inliers, outliers


def run_single_pair(idx=0, display=False, plot_3d=True):
    """Executes feature detection, matching, epipolar filtering, and triangulation for a single frame."""
    if display:
        print(f"\n========== Processing Frame {idx} Pipeline ==========")

    img1, img2 = read_images(idx)
    kp1, des1 = get_akaze_features(img1)
    kp2, des2 = get_akaze_features(img2)

    assert len(kp1) >= 500 and len(kp2) >= 500, f"Insufficient feature count in frame {idx}!"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1, des2)

    inliers, outliers = split_matches_by_rectified_pattern(kp1, kp2, matches, threshold=2)

    k, m1, m2 = read_cameras()
    points1, points2 = get_matched_points(kp1, kp2, inliers)
    points_3d = triangulate_points_opencv(points1, points2, m1, m2)

    valid_depth_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 350)
    filtered_inliers = [inliers[i] for i in range(len(inliers)) if valid_depth_mask[i]]
    filtered_points_3d = points_3d[valid_depth_mask]

    des_left_inliers = np.array([des1[m.queryIdx] for m in filtered_inliers]) if filtered_inliers else np.empty((0, des1.shape[1]))
    des_right_inliers = np.array([des2[m.trainIdx] for m in filtered_inliers]) if filtered_inliers else np.empty((0, des2.shape[1]))

    return {
        'img_left': img1, 'img_right': img2, 'kp_left': kp1, 'kp_right': kp2,
        'des_left': des1, 'des_right': des2, 'des_left_inliers': des_left_inliers,
        'des_right_inliers': des_right_inliers, 'stereo_inliers': filtered_inliers,
        'points_3d': filtered_points_3d
    }


def build_pnp_correspondences(frame0_data, frame1_data, good_temporal_matches):
    """Constructs 3D-2D quad-image observations across consecutive stereo frames."""
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
    """Evaluates supporter status of 3D-2D correspondences by reprojection across 4 cameras."""
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


def estimate_relative_pose_ransac(correspondences, prev_data, curr_data):
    """Estimates relative pose using PnP-RANSAC with early stopping on converged inliers."""
    k_matrix, _, _ = read_cameras()
    best_inliers, best_outliers, best_R, best_t = [], [], None, None
    no_improvement, max_no_improvement = 0, 12

    for _ in range(50):
        import random
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

    return best_inliers, best_outliers, best_R, best_t