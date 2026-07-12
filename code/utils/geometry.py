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
PROJECT_ROOT = Path(__file__).parent.parent.parent
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


def project_stereo_point(K, R, t, t_stereo, X):
    """Projects a 3D point into left and right image plane coordinates."""
    P_left = K @ np.hstack([R, t.reshape(3, 1)])
    P_right = K @ np.hstack([R, (t.reshape(3) + t_stereo.reshape(3)).reshape(3, 1)])
    return project_point(P_left, X), project_point(P_right, X)


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