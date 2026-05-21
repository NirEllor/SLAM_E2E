import time

import cv2
import matplotlib.pyplot as plt
import random
import van_utils as lib
import numpy as np

NUM_FRAMES = 3300

#--------------------------------------ex1---------------------------------------------------------
def q1_1(idx=0, plot = True):
    """1.1: Detect & Present Keypoints."""
    img1, img2 = lib.read_images(idx)
    kp1, des1 = lib.get_orb_features(img1)
    kp2, des2 = lib.get_orb_features(img2)

    assert len(kp1) >= 500 and len(kp2) >= 500, "Insufficient keypoints!"

    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))
    lib.plot_stereo_side_by_side(img1_kp, img2_kp, "ORB Keypoints")

    return img1, img2, kp1, kp2, des1, des2


def q1_2(des1):
    """1.2: Calculate & Print Descriptors."""
    print("\n--- Task 1.2: Descriptor Info ---")
    if des1 is not None and len(des1) >= 2:
        print("Image 1 - Descriptor 0:", des1[0])
        print("Image 1 - Descriptor 1:", des1[1])


def q1_3(img1, img2, kp1, kp2, des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1, des2)

    lib.draw_matches_custom(img1, kp1, img2, kp2, matches, "1.3: 20 Random Raw Matches")
    return matches


def q1_4(img1, img2, kp1, kp2, des1, des2):
    """1.4: Significance (Ratio) Test."""
    ratio_value = 0.7
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    rejected_matches = []

    for m, n in knn_matches:
        if m.distance < ratio_value * n.distance:
            good_matches.append(m)
        else:
            rejected_matches.append(m)

    # 1. Output 20 resulting matches
    lib.draw_matches_custom(img1, kp1, img2, kp2, good_matches, f"1.4: Good Matches (Ratio={ratio_value})")

    # 2. Statistics
    print("\n--- Task 1.4: Significance Test ---")
    print(f"Ratio Value Used: {ratio_value}")
    print(f"Matches Discarded: {len(rejected_matches)}")

    # 3. Present a failed match (rejected but visually correct)
    if rejected_matches:
        # Pick one rejected match to display
        fail = random.choice(rejected_matches)
        img1_fail = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2_fail = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        pt1 = tuple(map(int, kp1[fail.queryIdx].pt))
        pt2 = tuple(map(int, kp2[fail.trainIdx].pt))

        cv2.circle(img1_fail, pt1, 10, (255, 0, 0), -1)
        cv2.circle(img2_fail, pt2, 10, (255, 0, 0), -1)
        lib.plot_stereo_side_by_side(img1_fail, img2_fail, "1.4: Failed Significance Test Match (Rejected Match Example"
                                                           ")")


#--------------------------------------ex2---------------------------------------------------------
def q2_1(kp1, kp2, matches):
    """2.1: Analyze deviations from the rectified stereo pattern."""
    print("\n--- Task 2.1: Rectified Stereo Pattern ---")

    deviations = lib.compute_rectified_stereo_deviations(kp1, kp2, matches)

    lib.plot_deviation_histogram(deviations)
    lib.print_large_deviation_percentage(deviations, threshold=2)

    return deviations


def q2_2(img1, img2, kp1, kp2, matches, threshold=2):
    """2.2: Reject matches using the rectified stereo pattern."""
    print("\n--- Task 2.2: Reject Matches by Rectified Stereo Constraint ---")

    inliers, outliers = lib.split_matches_by_rectified_pattern(
        kp1, kp2, matches, threshold=threshold
    )

    print(f"Inliers: {len(inliers)}")
    print(f"Outliers: {len(outliers)}")
    print(f"Discarded matches: {len(outliers)}")

    lib.draw_inliers_outliers(img1, kp1, img2, kp2, inliers, outliers)

    return inliers, outliers

def q2_3(kp1, kp2, matches):
    """2.3: Linear least-squares triangulation and OpenCV comparison."""
    print("\n--- Task 2.3: Triangulation ---")

    k, m1, m2 = lib.read_cameras()

    points1, points2 = lib.get_matched_points(kp1, kp2, matches)

    points_3d_linear = lib.triangulate_points_linear(points1, points2, m1, m2)

    lib.plot_3d_points(points_3d_linear, title="2.3: Linear Least-Squares Triangulation")

    points_3d_cv = lib.triangulate_points_opencv(points1, points2, m1, m2)

    lib.plot_3d_points(points_3d_cv, title="2.3: OpenCV triangulatePoints")

    median_dist = lib.median_3d_distance(points_3d_linear, points_3d_cv)

    print(f"Median distance between linear and OpenCV 3D points: {median_dist}")

    return points_3d_linear, points_3d_cv


def run_single_pair(idx=0, display=False, plot_3d=True):
    """
    Runs the full pipeline for a single stereo pair (Ex1 & Ex2).
    Supports silent running, with an option to plot ONLY the final 3D point cloud.
    """
    if display:
        print(f"\n========== Processing Frame {idx} (with Visualizations) ==========")
    
    # 1. Read images and extract features
    img1, img2 = lib.read_images(idx)
    kp1, des1 = lib.get_orb_features(img1)
    kp2, des2 = lib.get_orb_features(img2)
    
    assert len(kp1) >= 500 and len(kp2) >= 500, f"Insufficient keypoints in frame {idx}!"
    
    # Optional display for keypoints (only if global display is True)
    if display:
        img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
        img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))
        lib.plot_stereo_side_by_side(img1_kp, img2_kp, f"Frame {idx}: ORB Keypoints")
        q1_2(des1)
    
    # 2. Initial brute-force matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1, des2)
    
    if display:
        lib.draw_matches_custom(img1, kp1, img2, kp2, matches, f"Frame {idx}: Raw Matches")
        q1_4(img1, img2, kp1, kp2, des1, des2) # Significance test visualization
        q2_1(kp1, kp2, matches)                # Deviation histogram
    
    # 3. Filter matches using the rectified stereo pattern
    inliers, outliers = lib.split_matches_by_rectified_pattern(
        kp1, kp2, matches, threshold=2
    )
    
    if display:
        print(f"Frame {idx} - Inliers: {len(inliers)}, Outliers: {len(outliers)}")
        lib.draw_inliers_outliers(img1, kp1, img2, kp2, inliers, outliers)
    
    # 4. Triangulation to build the 3D point cloud
    k, m1, m2 = lib.read_cameras()
    points1, points2 = lib.get_matched_points(kp1, kp2, inliers)
    points_3d = lib.triangulate_points_opencv(points1, points2, m1, m2)
    
    # <<=== התיקון כאן: מציג את הגרף הסופי אם display=True או אם plot_3d=True ===>>
    if display or plot_3d:
        # פילטר קטן כדי לנקות את נקודות האינסוף שראינו קודם ולא להרוס את הגרף
        valid_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 100)
        clean_points_3d = points_3d[valid_mask]
        
        lib.plot_3d_points(clean_points_3d, title=f"Frame {idx}: Final 3D Point Cloud")
    
    # Return everything so task 3 has full access to images, keypoints, and 3D data
    return {
        'img_left': img1,
        'img_right': img2,
        'kp_left': kp1,
        'kp_right': kp2,
        'des_left': des1,
        'des_right': des2,
        'stereo_inliers': inliers,
        'points_3d': points_3d
    }
#--------------------------------------ex3---------------------------------------------------------

def q3_1(idx=0):
    """
    3.1: Create a point cloud for the next stereo pair (frame idx + 1).
    Uses the updated run_single_pair function in silent mode to extract features,
    filter outliers, and triangulate. Then, plots the resulting 3D cloud.
    """
    print(f"\n--- Task 3.1: Generating Data for Next Stereo Pair (Frame {idx + 1}) ---")
    
    # 1. Run the updated single pair pipeline for the next frame without intermediate plots
    frame1_data = run_single_pair(idx=idx + 1, display=False)
    
    # 2. Plot only the 3D point cloud required for task 3.1
    lib.plot_3d_points(frame1_data['points_3d'], title=f"3.1: Point Cloud for Frame {idx + 1}")
    
    # 3. Return the full dictionary containing all extracted data for frame 1
    return frame1_data

def q3_2(frame0_data, frame1_data):
    """
    3.2: Match features between the two left images (left_0 and left_1).
    Uses a brute-force KNN matcher and filters the results using a 
    Significance (Ratio) Test. Since the camera has moved over time, 
    the images are not rectified, so the vertical stereo constraint cannot be used.
    """
    print("\n--- Task 3.2: Matching Features Between Left_0 and Left_1 ---")
    
    # Extract images, keypoints, and descriptors from the frame data dictionaries
    img_left0 = frame0_data['img_left']
    img_left1 = frame1_data['img_left']
    
    kp_left0 = frame0_data['kp_left']
    kp_left1 = frame1_data['kp_left']
    
    des_left0 = frame0_data['des_left']
    des_left1 = frame1_data['des_left']
    
    # 1. Initialize Brute-Force Matcher for Hamming distance (suitable for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # 2. Find the top 2 best matches for each descriptor to apply the ratio test
    knn_matches = bf.knnMatch(des_left0, des_left1, k=2)
    
    # 3. Apply Lowe's Significance (Ratio) Test with a threshold of 0.7
    good_temporal_matches = []
    for m, n in knn_matches:
        if m.distance < 0.7 * n.distance:
            good_temporal_matches.append(m)
            
    print(f"Found {len(good_temporal_matches)} validated temporal matches between left_0 and left_1.")
    
    # 4. Draw 20 random matches to visualize the temporal motion tracking (red arrows/lines)
    lib.draw_matches_custom(
        img_left0, kp_left0, 
        img_left1, kp_left1, 
        good_temporal_matches, 
        title="3.2: Matches between Left_0 and Left_1 (Temporal Tracking)"
    )
    
    # Return the filtered temporal matches for downstream PnP and RANSAC optimization
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
         iterations=500,
         threshold=2):
    """
    Custom RANSAC with PnP as inner model.
    """

    print("\n--- Task 3.5: Custom PnP-RANSAC ---")

    k, _, _ = lib.read_cameras()

    correspondences = lib.build_pnp_correspondences(
        frame0_data,
        frame1_data,
        good_temporal_matches
    )

    print(f"Total correspondences: {len(correspondences)}")

    best_inliers = []
    best_R = None
    best_t = None

    for i in range(iterations):

        # Random minimal sample of 4 correspondences
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
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success:
            continue

        R, _ = cv2.Rodrigues(rvec)
        t = tvec

        inliers, outliers = lib.evaluate_supporters(
            correspondences,
            frame0_data,
            frame1_data,
            R,
            t,
            threshold
        )

        if len(inliers) > len(best_inliers):

            best_inliers = inliers
            best_R = R
            best_t = t

    print(f"Best inliers found: {len(best_inliers)}")

    # =========================
    # Refinement step
    # =========================

    object_points = np.array(
        [c['X'] for c in best_inliers],
        dtype=np.float32
    )

    image_points = np.array(
        [c['obs_left1'] for c in best_inliers],
        dtype=np.float32
    )

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=k,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success:
        raise RuntimeError("Refinement PnP failed.")

    refined_R, _ = cv2.Rodrigues(rvec)
    refined_t = tvec

    final_inliers, final_outliers = lib.evaluate_supporters(
        correspondences,
        frame0_data,
        frame1_data,
        refined_R,
        refined_t,
        threshold
    )

    print(f"Final refined inliers: {len(final_inliers)}")
    print(f"Final refined outliers: {len(final_outliers)}")

    return refined_R, refined_t, final_inliers, final_outliers

def q3_6(num_frames=NUM_FRAMES):
    """
    Full visual odometry pipeline over the whole movie.
    """

    print("\n==================== Task 3.6 ====================")

    start_time = time.time()

    # =========================
    # Initial global pose
    # left0 coordinates
    # =========================

    R_global = np.eye(3)
    t_global = np.zeros((3, 1))

    estimated_positions = [
        lib.camera_center(R_global, t_global)
    ]

    # Ground truth
    gt_poses = lib.read_ground_truth_poses()

    gt_positions = []

    for R_gt, t_gt in gt_poses[:num_frames]:

        gt_positions.append(
            lib.camera_center(R_gt, t_gt)
        )

    # =========================
    # Process movie
    # =========================

    for idx in range(num_frames - 1):

        print(f"\n========== Frame {idx} -> {idx+1} ==========")

        # frame i
        frame0_data = run_single_pair(
            idx=idx,
            display=False,
            plot_3d=False
        )

        # frame i+1
        frame1_data = run_single_pair(
            idx=idx + 1,
            display=False,
            plot_3d=False
        )

        # temporal tracking
        good_temporal_matches = q3_2(
            frame0_data,
            frame1_data
        )

        # relative pose using custom RANSAC
        R_rel, t_rel, inliers, outliers = q3_5(
            frame0_data,
            frame1_data,
            good_temporal_matches,
            iterations=300,
            threshold=2
        )

        # =========================
        # Compose into global pose
        # =========================

        R_global, t_global = lib.compose_transform(
            R_global,
            t_global,
            R_rel,
            t_rel
        )

        # camera location
        C = lib.camera_center(R_global, t_global)

        estimated_positions.append(C)

        print(f"Estimated camera position: {C}")

    end_time = time.time()

    total_time = end_time - start_time

    print(f"\nTracking time: {total_time:.2f} seconds")

    # =========================
    # Plot trajectory
    # =========================

    lib.plot_trajectory(
        estimated_positions,
        gt_positions
    )

    return estimated_positions, gt_positions

def q3(idx=0):

    print(f"\n==================== Starting Ex 3 Pipeline (Frame {idx} -> {idx+1}) ====================")

    # 3.1
    frame0_data = run_single_pair(idx=idx, display=True)

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
    supporters, non_supporters = q3_4(
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



def main():

    q3()

    plt.show()


if __name__ == '__main__':
    main()