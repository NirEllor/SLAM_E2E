# ex3.py
import time

from ex2 import *
import cv2
import numpy as np

NUM_FRAMES = lib.get_num_frames()


#--------------------------------------ex3---------------------------------------------------------

def q3_1(idx=0):
    """
    3.1: Create a point cloud for the next stereo pair (frame idx + 1).
    Uses the updated run_single_pair function in silent mode to extract features,
    filter outliers, and triangulate. Then, plots the resulting 3D cloud.
    """
    print(f"\n--- Task 3.1: Generating Data for Next Stereo Pair (Frame {idx + 1}) ---")

    # 1. Run the updated single pair pipeline for the next frame without intermediate plots
    frame1_data = lib.run_single_pair(idx=idx + 1, display=False)

    # 2. Plot only the 3D point cloud required for task 3.1
    lib.plot_3d_points(frame1_data['points_3d'], title=f"3.1: Point Cloud for Frame {idx + 1}")

    # 3. Return the full dictionary containing all extracted data for frame 1
    return frame1_data

def q3_2(frame0_data, frame1_data, draw=False):
    """
    3.2: Match features between the two left images (left_0 and left_1).
    Uses BF-KNN matcher + Lowe ratio test.
    """

    print("\n--- Task 3.2: Matching Features Between Left_0 and Left_1 ---")

    img_left0 = frame0_data['img_left']
    img_left1 = frame1_data['img_left']

    kp_left0 = frame0_data['kp_left']
    kp_left1 = frame1_data['kp_left']

    des_left0 = frame0_data['des_left']
    des_left1 = frame1_data['des_left']

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    knn_matches = bf.knnMatch(des_left0, des_left1, k=2)

    good_temporal_matches = []

    for m, n in knn_matches:
        if m.distance < 0.7 * n.distance:
            good_temporal_matches.append(m)

    print(
        f"Found {len(good_temporal_matches)} validated temporal matches "
        f"between left_0 and left_1."
    )

    if draw:
        lib.draw_matches_custom(
            img_left0,
            kp_left0,
            img_left1,
            kp_left1,
            good_temporal_matches,
            title="3.2: Matches between Left_0 and Left_1 (Temporal Tracking)"
        )

    return good_temporal_matches
def q3_3(frame0_data, frame1_data, good_temporal_matches):
    """
    3.3: Estimate the camera motion between frame 0 and frame 1 using PnP.
    Selects 4 random corresponding points between 3D points of frame 0 and
    2D pixels of left_1, applies cv2.solvePnP, and extracts [R|t].
    """
    print("\n--- Task 3.3: PnP Motion Estimation with 4 Points ---")

    # 1. Get the camera intrinsic matrix K
    k, _, _ = lib.read_cameras()

    # 2. Map temporal matches to link 3D points (from t0) with 2D pixels (in left_1)
    # The 'stereo_inliers' in frame 0 correspond to the indices of 'points_3d'
    # We need to match the indices of the keypoints tracked over time
    pts_3d_list = []
    pts_2d_list = []

    # Create a mapping from queryIdx (kp in left_0) to its corresponding 3D point index
    # Since points_3d aligns with the stereo_inliers of frame 0:
    left0_pt_to_3d_idx = {m.queryIdx: idx for idx, m in enumerate(frame0_data['stereo_inliers'])}

    # Cross-reference with temporal matches (left_0 -> left_1)
    for match in good_temporal_matches:
        if match.queryIdx in left0_pt_to_3d_idx:
            # Get the 3D point from frame 0
            three_d_idx = left0_pt_to_3d_idx[match.queryIdx]
            p3d = frame0_data['points_3d'][three_d_idx]

            # Get the 2D pixel coordinate from left_1
            p2d = frame1_data['kp_left'][match.trainIdx].pt

            pts_3d_list.append(p3d)
            pts_2d_list.append(p2d)

    # Convert lists to NumPy arrays as required by OpenCV
    pts_3d_all = np.array(pts_3d_list, dtype=np.float32)
    pts_2d_all = np.array(pts_2d_list, dtype=np.float32)

    print(f"Total matching consensus points available for PnP: {len(pts_3d_all)}")
    assert len(pts_3d_all) >= 4, "Not enough matching points found across all four images!"

    # 3. Select exactly 4 corresponding points as requested by the specification
    # We can pick 4 random indices or the first 4 indices
    indices = [0, 1, 2, 3]
    object_points = pts_3d_all[indices]
    image_points = pts_2d_all[indices]

    print("\nSelected 4 Object Points (3D from t0):")
    print(object_points)
    print("Selected 4 Image Points (2D from left_1):")
    print(image_points)

    # 4. Apply the PnP algorithm using OpenCV
    # distCoeffs is None because our dataset images are already rectified and distortion-free
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=k,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        raise RuntimeError("PnP solver failed to find a valid pose matrix estimation.")

    # 5. Convert the rotation vector (rvec) to a 3x3 rotation matrix (R)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec

    print("\n--- Extrinsic Camera Matrix Results [R|t] ---")
    print("Rotation Matrix (R, 3x3):")
    print(R)
    print("Translation Vector (t, 3x1):")
    print(t)

    return R, t

def q3_4(frame0_data, frame1_data, good_temporal_matches, R, t, threshold=2):
    """
    3.4: Find supporters of the transformation T using reprojection error
    in all four images: left0, right0, left1, right1.
    """
    print("\n--- Task 3.4: Supporters by 4-view reprojection consistency ---")

    k, m_left0, m_right0 = lib.read_cameras()

    # Stereo translation from left camera to right camera.
    # m_right0 = K [I | t_stereo]
    t_stereo = np.linalg.inv(k) @ m_right0[:, 3]

    # Projection matrices for frame 1
    P_left1 = k @ np.hstack([R, t.reshape(3, 1)])
    P_right1 = k @ np.hstack([R, (t.reshape(3) + t_stereo).reshape(3, 1)])

    # Map left0 keypoint index -> (3D index, stereo match)
    left0_to_3d = {
        m.queryIdx: (idx, m)
        for idx, m in enumerate(frame0_data['stereo_inliers'])
    }

    # Map left1 keypoint index -> stereo match in frame1
    left1_to_stereo = {
        m.queryIdx: m
        for m in frame1_data['stereo_inliers']
    }

    supporters = []
    non_supporters = []

    for temporal_match in good_temporal_matches:
        left0_idx = temporal_match.queryIdx
        left1_idx = temporal_match.trainIdx

        # Need the point to exist in all four images
        if left0_idx not in left0_to_3d:
            continue
        if left1_idx not in left1_to_stereo:
            continue

        point_3d_idx, stereo_match0 = left0_to_3d[left0_idx]
        stereo_match1 = left1_to_stereo[left1_idx]

        X = frame0_data['points_3d'][point_3d_idx]

        # Observed pixel locations
        obs_left0 = np.array(frame0_data['kp_left'][stereo_match0.queryIdx].pt)
        obs_right0 = np.array(frame0_data['kp_right'][stereo_match0.trainIdx].pt)
        obs_left1 = np.array(frame1_data['kp_left'][temporal_match.trainIdx].pt)
        obs_right1 = np.array(frame1_data['kp_right'][stereo_match1.trainIdx].pt)

        # Reproject the same 3D point to all four cameras
        proj_left0 = lib.project_point(m_left0, X)
        proj_right0 = lib.project_point(m_right0, X)
        proj_left1 = lib.project_point(P_left1, X)
        proj_right1 = lib.project_point(P_right1, X)

        errors = [
            np.linalg.norm(proj_left0 - obs_left0),
            np.linalg.norm(proj_right0 - obs_right0),
            np.linalg.norm(proj_left1 - obs_left1),
            np.linalg.norm(proj_right1 - obs_right1),
        ]

        if all(e < threshold for e in errors):
            supporters.append(temporal_match)
        else:
            non_supporters.append(temporal_match)

    print(f"Supporters: {len(supporters)}")
    print(f"Non-supporters: {len(non_supporters)}")

    lib.draw_temporal_supporters(
        frame0_data['img_left'],
        frame0_data['kp_left'],
        frame1_data['img_left'],
        frame1_data['kp_left'],
        supporters,
        non_supporters
    )

    return supporters, non_supporters

def q3_5(frame0_data,
         frame1_data,
         good_temporal_matches,
         iterations=50,
         threshold=2,
         early_stop_ratio=0.78,
         max_no_improvement=12,
         pnp_method=cv2.SOLVEPNP_EPNP):
    """
    3.5: Custom PnP-RANSAC with dynamic stopping.
    """

    print("\n--- Task 3.5: Custom PnP-RANSAC ---")

    k, _, _ = lib.read_cameras()

    correspondences = lib.build_pnp_correspondences(
        frame0_data,
        frame1_data,
        good_temporal_matches
    )

    print(f"Total correspondences: {len(correspondences)}")

    if len(correspondences) < 4:
        raise RuntimeError("Not enough correspondences for PnP-RANSAC.")

    best_inliers = []
    best_R = None
    best_t = None
    no_improvement = 0

    for i in range(iterations):

        sample = random.sample(correspondences, 4)

        object_points = np.array(
            [c['X'] for c in sample],
            dtype=np.float32
        )

        image_points = np.array(
            [c['obs_left1'] for c in sample],
            dtype=np.float32
        )

        success, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=k,
            distCoeffs=None,
            flags=pnp_method
        )

        if not success:
            no_improvement += 1
            continue

        R, _ = cv2.Rodrigues(rvec)
        t = tvec

        inliers, outliers = lib.evaluate_supporters(
            correspondences,
            frame0_data,
            frame1_data,
            R,
            t,
            threshold=threshold
        )

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R = R
            best_t = t
            no_improvement = 0

            if len(best_inliers) > early_stop_ratio * len(correspondences):
                print(f"Early stopping at iteration {i}")
                break
        else:
            no_improvement += 1

        if no_improvement >= max_no_improvement:
            print(f"Dynamic stopping at iteration {i}: no significant improvement.")
            break

    if best_R is None or len(best_inliers) < 4:
        raise RuntimeError("RANSAC failed to find a valid pose.")

    print(f"Best inliers found: {len(best_inliers)}")

    object_points = np.array(
        [c['X'] for c in best_inliers],
        dtype=np.float32
    )

    image_points = np.array(
        [c['obs_left1'] for c in best_inliers],
        dtype=np.float32
    )

    rvec_guess, _ = cv2.Rodrigues(best_R)

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=k,
        distCoeffs=None,
        rvec=rvec_guess,
        tvec=best_t,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        refined_R, _ = cv2.Rodrigues(rvec)
        refined_t = tvec

        final_inliers, final_outliers = lib.evaluate_supporters(
            correspondences,
            frame0_data,
            frame1_data,
            refined_R,
            refined_t,
            threshold=threshold
        )

        if len(final_inliers) >= 0.8 * len(best_inliers):
            print(f"Final refined inliers: {len(final_inliers)}")
            print(f"Final refined outliers: {len(final_outliers)}")
            return refined_R, refined_t, final_inliers, final_outliers

    final_inliers, final_outliers = lib.evaluate_supporters(
        correspondences,
        frame0_data,
        frame1_data,
        best_R,
        best_t,
        threshold=threshold
    )

    print("Refinement rejected; using best RANSAC pose.")
    print(f"Final inliers: {len(final_inliers)}")
    print(f"Final outliers: {len(final_outliers)}")

    return best_R, best_t, final_inliers, final_outliers

def q3_6(num_frames=20):
    print("\n==================== Task 3.6 ====================")

    start_time = time.time()

    R_global = np.eye(3)
    t_global = np.zeros((3, 1))

    estimated_positions = [lib.camera_center(R_global, t_global)]

    gt_poses = lib.read_ground_truth_poses()
    gt_positions = [
        lib.camera_center(R_gt, t_gt)
        for R_gt, t_gt in gt_poses[:num_frames]
    ]

    # compute first frame only once
    prev_data = lib.run_single_pair(
        idx=0,
        display=False,
        plot_3d=False
    )

    for idx in range(num_frames - 1):
        print(f"\n========== Frame {idx} -> {idx + 1} ==========")

        # compute only next frame
        next_data = lib.run_single_pair(
            idx=idx + 1,
            display=False,
            plot_3d=False
        )

        good_temporal_matches = q3_2(
            prev_data,
            next_data,
            draw=False
        )

        try:
            R_rel, t_rel, inliers, outliers = q3_5(
                prev_data,
                next_data,
                good_temporal_matches,
                iterations=50,
                threshold=2
            )
        except RuntimeError as e:
            print(f"Pose estimation failed: {e}")
            prev_data = next_data
            estimated_positions.append(estimated_positions[-1])
            continue

        if len(inliers) < 30:
            print("Pose rejected due to too few inliers; skipping this frame.")
            prev_data = next_data
            estimated_positions.append(estimated_positions[-1])
            continue

        C_prev = lib.camera_center(R_global, t_global)

        R_candidate, t_candidate = lib.compose_transform(
            R_global,
            t_global,
            R_rel,
            t_rel
        )

        C_candidate = lib.camera_center(R_candidate, t_candidate)

        step_size = np.linalg.norm(C_candidate - C_prev)

        if step_size > 3.0:
            print(f"Pose rejected due to unreasonable step size: step={step_size:.2f}")
            prev_data = next_data
            estimated_positions.append(estimated_positions[-1])
            continue

        R_global, t_global = R_candidate, t_candidate

        C = lib.camera_center(R_global, t_global)
        estimated_positions.append(C)

        print(f"Estimated camera position: {C}")
        print(f"Inliers: {len(inliers)}, Outliers: {len(outliers)}")

        # reuse
        prev_data = next_data

    total_time = time.time() - start_time
    print(f"\nTracking time: {total_time:.2f} seconds")

    lib.plot_trajectory(estimated_positions, gt_positions)

    return estimated_positions, gt_positions

def q3(idx=0):

    print(f"\n==================== Starting Ex 3 Pipeline (Frame {idx} -> {idx+1}) ====================")

    # 3.1
    frame0_data = lib.run_single_pair(idx=idx, display=True)

    # 3.1
    frame1_data = q3_1(idx=idx)

    # 3.2
    good_temporal_matches = q3_2(
        frame0_data,
        frame1_data
    )

    # 3.3
    R, t = q3_3(
        frame0_data,
        frame1_data,
        good_temporal_matches
    )

    # camera visualization
    lib.plot_four_cameras(R, t, baseline=0.54)

    # 3.4
    q3_4(
        frame0_data,
        frame1_data,
        good_temporal_matches,
        R,
        t,
        threshold=2
    )

    # 3.5
    R_ransac, t_ransac, inliers, outliers = q3_5(
        frame0_data,
        frame1_data,
        good_temporal_matches
    )

    lib.draw_ransac_results(
        frame0_data,
        frame1_data,
        inliers,
        outliers
    )

    lib.plot_transformed_clouds(
        frame0_data,
        frame1_data,
        R_ransac,
        t_ransac
    )

    # 3.6
    estimated_positions, gt_positions = q3_6(
        num_frames=NUM_FRAMES
    )

    return {
        'frame0_data': frame0_data,
        'frame1_data': frame1_data,
        'good_temporal_matches': good_temporal_matches,
        'R': R,
        't': t,
        'R_ransac': R_ransac,
        't_ransac': t_ransac,
        'inliers': inliers,
        'outliers': outliers,
        'trajectory_estimated': estimated_positions,
        'trajectory_gt': gt_positions
    }

def benchmark_tracking_configs(idx=0):
    """
    High-level utility function to evaluate alternative tracking configuration parameters,
    detector types, and variant PnP calculation methods. Moved to van_utils library layer.
    """
    configs = [
        ("ORB", 700, cv2.SOLVEPNP_EPNP),
        ("ORB", 1000, cv2.SOLVEPNP_EPNP),
        ("ORB", 1500, cv2.SOLVEPNP_EPNP),
        ("AKAZE", 0, cv2.SOLVEPNP_EPNP),
        ("ORB", 1000, cv2.SOLVEPNP_P3P),
        ("ORB", 1000, cv2.SOLVEPNP_AP3P),
    ]

    for detector_type, n_features, pnp_method in configs:
        print("\n" + "="*40)
        print(f"Config Setup -> Detector: {detector_type} | Limit: {n_features} | Method ID: {pnp_method}")

        frame0_data = lib.run_single_pair(idx=idx, display=False, plot_3d=False)
        frame1_data = lib.run_single_pair(idx=idx + 1, display=False, plot_3d=False)

        matches = q3_2(frame0_data, frame1_data, draw=False)

        try:
            R, t, inliers, outliers = q3_5(
                frame0_data, frame1_data, matches,
                iterations=50, threshold=2
            )
            print(f"Resulting Temporal Intersections: {len(matches)}")
            print(f"Verified Consensus Inliers: {len(inliers)}")
            if matches:
                print(f"Inlier Verification Success Ratio: {len(inliers) / len(matches):.3f}")
        except RuntimeError as e:
            print(f"Calculation pass aborted: {e}")
def main():
    q3()

if __name__ == '__main__':
    main()