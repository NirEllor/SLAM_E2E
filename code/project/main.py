import os
import random
import cv2
import gtsam
from gtsam import symbol

# Import from utils package
import numpy as np
from utils.geometry import (read_images, get_orb_features, read_cameras, run_single_pair,
                           get_num_frames, build_pnp_correspondences,
                           camera_center, read_ground_truth_poses,
                           split_matches_by_rectified_pattern, compute_rectified_stereo_deviations,
                           triangulate_points_linear, triangulate_points_opencv, median_3d_distance,
                           get_matched_points, run_custom_pnp_ransac, find_four_view_supporters,
                           estimate_next_pose_with_rejection, compute_kitti_sequence_errors)
from utils.tracking import (load_or_build_db, compute_tracking_statistics, print_tracking_statistics,
                           select_track_by_min_length, compute_connectivity,
                           select_long_track, triangulate_track_reference_point,
                           compute_track_reprojection_errors, compute_absolute_pnp_error,
                           compute_pnp_projection_error_vs_distance)
from utils.bundle_adjustment import (init_gtsam_stereo_calibration, get_gtsam_camera_pose,
                                     pose_translation_np, choose_keyframes, solve_bundle_window,
                                     compute_single_factor_error, extract_optimized_geometry,
                                     compute_window_projection_errors,
                                     accumulate_window_projection_errors_by_distance)
from utils.pose_graph import (compute_relative_pose_and_covariance, compute_all_relative_constraints,
                              clean_pose_graph_edges, build_and_initialize_pose_graph, optimize_pose_graph,
                              run_prior_sensitivity_sweep, extract_trajectory_and_ids,
                              compute_pose_graph_absolute_error, compute_bundle_relative_error_vs_gt,
                              compute_kitti_sequence_errors_keyframes)
from utils.loop_closure import (detect_loop_closure_candidates, verify_loop_closures_consensus,
                                estimate_verified_loop_relative_poses, add_loop_closures_and_optimize)
from utils.visualization import (plot_stereo_side_by_side, draw_matches_custom,
                                 plot_matches_per_frame, plot_absolute_location_error_components,
                                 plot_absolute_angle_error, plot_bundle_optimization_error,
                                 plot_bundle_projection_error, plot_projection_error_vs_distance,
                                 plot_relative_error_comparison, plot_full_trajectory_comparison,
                                 plot_kitti_sequence_error, plot_loop_closure_match_stats,
                                 plot_angle_uncertainty_size)

# Output directory for all stages
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_and_visualize_orb_features(frame_index=0, plot=True):
    """Detects ORB keypoints on a stereo image pair and displays them side-by-side."""
    img_left, img_right = read_images(frame_index)
    kp_left, des_left = get_orb_features(img_left)
    kp_right, des_right = get_orb_features(img_right)

    assert len(kp_left) >= 500 and len(kp_right) >= 500, "Insufficient keypoints detected!"

    if plot:
        img_left_kp = cv2.drawKeypoints(img_left, kp_left, None, color=(0, 255, 0))
        img_right_kp = cv2.drawKeypoints(img_right, kp_right, None, color=(0, 255, 0))
        plot_stereo_side_by_side(img_left_kp, img_right_kp, "ORB Keypoints", graph_id="1.1", func_name="detect_and_visualize_orb_features")

    return img_left, img_right, kp_left, kp_right, des_left, des_right


def inspect_feature_descriptors(des_left):
    """Prints raw feature descriptor values to console for inspection."""
    print("\n--- Task 1.2: Descriptor Inspection ---")
    if des_left is not None and len(des_left) >= 2:
        print("Left Image - Descriptor 0:", des_left[0])
        print("Left Image - Descriptor 1:", des_left[1])


def compute_raw_bruteforce_matches(img_left, img_right, kp_left, kp_right, des_left, des_right):
    """Computes raw Hamming brute-force matches between left and right descriptors."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des_left, des_right)

    draw_matches_custom(img_left, kp_left, img_right, kp_right, matches, "Graph 1.3: Random Raw Matches\ndraw_matches_custom, visualization.py")
    return matches


def apply_lowe_ratio_test(img_left, img_right, kp_left, kp_right, des_left, des_right, ratio_threshold=0.7):
    """Filters ambiguous matches using Lowe's ratio test on k-Nearest Neighbors matches."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des_left, des_right, k=2)

    good_matches, rejected_matches = [], []
    for m, n in knn_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
        else:
            rejected_matches.append(m)

    # Visualize filtered matches
    draw_matches_custom(img_left, kp_left, img_right, kp_right, good_matches, f"Graph 1.4a: Good Matches (Ratio={ratio_threshold})\ndraw_matches_custom, visualization.py")

    # Print summary metrics
    print("\n--- Task 1.4: Lowe's Ratio Test Summary ---")
    print(f"Ratio Threshold: {ratio_threshold}")
    print(f"Total Valid Matches: {len(good_matches)}")
    print(f"Total Discarded Matches: {len(rejected_matches)}")

    # Show a rejected match sample if available
    if rejected_matches:
        fail = random.choice(rejected_matches)
        draw_matches_custom(
            img_left, kp_left, img_right, kp_right, [fail],
            "Graph 1.4b: Failed Ratio Test Match Example\ndraw_matches_custom, visualization.py", num=1
        )


def run_feature_detection_and_matching_pipeline(frame_index=0):
    """Executes the complete Exercise 1 pipeline sequentially using descriptive function calls."""
    print("==================================================")
    print("  EXERCISE 1: FEATURE DETECTION & MATCHING       ")
    print("==================================================")

    # 1. Detect keypoints
    img_left, img_right, kp_left, kp_right, des_left, des_right = detect_and_visualize_orb_features(frame_index)

    # 2. Inspect descriptor structure
    inspect_feature_descriptors(des_left)

    # 3. Compute raw matches
    compute_raw_bruteforce_matches(img_left, img_right, kp_left, kp_right, des_left, des_right)

    # 4. Filter matches with Lowe's Ratio Test
    apply_lowe_ratio_test(img_left, img_right, kp_left, kp_right, des_left, des_right)


def analyze_rectified_stereo_deviations(idx=0):
    """Task 2.1: Computes and visualizes y-axis deviation distribution."""
    print("\n--- Task 2.1: Stereo Rectification Analysis ---")
    from utils.visualization import plot_deviation_histogram, print_large_deviation_percentage

    img_left, img_right = read_images(idx)
    kp_left, des_left = get_orb_features(img_left)
    kp_right, des_right = get_orb_features(img_right)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des_left, des_right)

    deviations = compute_rectified_stereo_deviations(kp_left, kp_right, matches)
    plot_deviation_histogram(deviations)
    print_large_deviation_percentage(deviations, threshold=2)
    return kp_left, kp_right, matches


def filter_matches_by_rectified_pattern(kp_left, kp_right, matches, threshold=2):
    """Task 2.2: Splits matches into inliers/outliers by rectified-stereo y-deviation."""
    print("\n--- Task 2.2: Filtered Stereo Matches ---")
    from utils.visualization import draw_inliers_outliers

    img_left, img_right = read_images(0)
    inliers, outliers = split_matches_by_rectified_pattern(kp_left, kp_right, matches, threshold)

    print(f"Inliers: {len(inliers)} | Outliers: {len(outliers)}")
    draw_inliers_outliers(img_left, kp_left, img_right, kp_right, inliers, outliers)
    return inliers


############################################################################
# TRIANGULATION - Reconstructing 3D points from stereo observations
############################################################################
def triangulate_and_compare_methods(idx=0):
    """Task 2.3: Triangulates 3D points using linear SVD and OpenCV; compares results."""
    print("\n--- Task 2.3: Triangulation Methods Comparison ---")
    from utils.visualization import plot_3d_points

    img_left, img_right = read_images(idx)
    kp_left, des_left = get_orb_features(img_left)
    kp_right, des_right = get_orb_features(img_right)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des_left, des_right)

    inliers, _ = split_matches_by_rectified_pattern(kp_left, kp_right, matches, threshold=2)
    k, m1, m2 = read_cameras()
    points_left, points_right = get_matched_points(kp_left, kp_right, inliers)

    points_3d_linear = triangulate_points_linear(points_left, points_right, m1, m2)
    points_3d_opencv = triangulate_points_opencv(points_left, points_right, m1, m2)

    valid_mask = (points_3d_linear[:, 2] > 0) & (points_3d_linear[:, 2] < 350)
    filtered_3d_linear = points_3d_linear[valid_mask]
    filtered_3d_opencv = points_3d_opencv[valid_mask]

    plot_3d_points(filtered_3d_linear, title="Graph 2.3a: Linear SVD Triangulation")
    plot_3d_points(filtered_3d_opencv, title="Graph 2.3b: OpenCV Triangulation")

    dist = median_3d_distance(filtered_3d_linear, filtered_3d_opencv)
    print(f"Median 3D distance between methods: {dist:.4f} m")


def run_stereo_rectification_pipeline(frame_index=0):
    """Exercise 2: Stereo rectification and triangulation pipeline."""
    kp_left, kp_right, matches = analyze_rectified_stereo_deviations(frame_index)
    inliers = filter_matches_by_rectified_pattern(kp_left, kp_right, matches)
    triangulate_and_compare_methods(frame_index)


def generate_next_frame_point_cloud(idx=0):
    """Task 3.1: Generates and plots 3D point cloud for the next stereo frame."""
    print(f"\n--- Task 3.1: Next Frame Point Cloud (Frame {idx + 1}) ---")
    from utils.visualization import plot_3d_points

    frame_data = run_single_pair(idx=idx + 1, display=False)
    plot_3d_points(frame_data['points_3d'], title=f"Graph 3.1: Frame {idx + 1} Landmarks")
    return frame_data


def match_temporal_features_left_images(frame0_data, frame1_data):
    """Task 3.2: Matches features between left images of consecutive frames."""
    print("\n--- Task 3.2: Temporal Feature Matching ---")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(frame0_data['des_left'], frame1_data['des_left'], k=2)
    good_matches = [m for m, n in knn_matches if m.distance < 0.7 * n.distance]
    print(f"Found {len(good_matches)} validated temporal matches")
    return good_matches


############################################################################
# PNP - Perspective-n-Point pose estimation for camera localization
############################################################################
def estimate_motion_with_four_point_pnp(frame0_data, frame1_data, good_temporal_matches):
    """Task 3.3: Estimates camera motion using 4-point PnP with exact points."""
    print("\n--- Task 3.3: PnP Motion Estimation ---")
    k, _, _ = read_cameras()
    correspondences = build_pnp_correspondences(frame0_data, frame1_data, good_temporal_matches)

    object_points = np.array([correspondences[i]['X'] for i in range(min(4, len(correspondences)))], dtype=np.float32)
    image_points = np.array([correspondences[i]['obs_left1'] for i in range(min(4, len(correspondences)))], dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(object_points, image_points, k, None, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        raise RuntimeError("PnP solver failed")

    R, _ = cv2.Rodrigues(rvec)
    print(f"Estimated pose: R shape {R.shape}, t shape {tvec.shape}")
    return R, tvec


def find_four_view_reprojection_supporters(frame0_data, frame1_data, good_temporal_matches, R, t):
    """Task 3.4: Finds matches that reproject consistently into all four cameras."""
    print("\n--- Task 3.4: Four-View Reprojection Supporters ---")
    from utils.visualization import draw_temporal_supporters

    supporters, non_supporters = find_four_view_supporters(frame0_data, frame1_data, good_temporal_matches, R, t)
    print(f"Supporters: {len(supporters)} | Non-supporters: {len(non_supporters)}")
    draw_temporal_supporters(frame0_data['img_left'], frame0_data['kp_left'],
                            frame1_data['img_left'], frame1_data['kp_left'],
                            supporters, non_supporters)
    return supporters, non_supporters


############################################################################
# RANSAC - Robust pose estimation using consensus-based outlier rejection
############################################################################
def estimate_motion_with_pnp_ransac(frame0_data, frame1_data, good_temporal_matches):
    """Task 3.5: Estimates pose via custom PnP-RANSAC with early/dynamic stopping."""
    print("\n--- Task 3.5: Custom PnP-RANSAC ---")
    from utils.visualization import draw_ransac_results

    R, t, inliers, outliers = run_custom_pnp_ransac(frame0_data, frame1_data, good_temporal_matches,
                                                     iterations=50, threshold=2)
    print(f"Best inliers: {len(inliers)}")
    draw_ransac_results(frame0_data, frame1_data, inliers, outliers)
    return R, t, inliers, outliers


############################################################################
# PNP TRAJECTORY - Temporal pose estimation and trajectory tracking
############################################################################
def track_trajectory_over_sequence(num_frames=20):
    """Task 3.6: Tracks camera trajectory over sequence using frame-by-frame RANSAC."""
    print("\n--- Task 3.6: Trajectory Estimation ---")
    from utils.visualization import plot_trajectory

    R_global, t_global = np.eye(3), np.zeros((3, 1))
    estimated_positions = [camera_center(R_global, t_global)]
    gt_poses = read_ground_truth_poses()
    gt_positions = [camera_center(R_gt, t_gt) for R_gt, t_gt in gt_poses[:num_frames]]

    prev_data = run_single_pair(idx=0, display=False, plot_3d=False)

    for idx in range(num_frames - 1):
        print(f"Frame {idx} -> {idx + 1}")
        curr_data = run_single_pair(idx=idx + 1, display=False, plot_3d=False)

        result = estimate_next_pose_with_rejection(prev_data, curr_data, R_global, t_global)
        if result[0] is None:
            estimated_positions.append(estimated_positions[-1])
        else:
            R_global, t_global, _, _ = result
            C = camera_center(R_global, t_global)
            estimated_positions.append(C)
            print(f"Camera position: {C}")

        prev_data = curr_data

    plot_trajectory(estimated_positions, gt_positions)
    return estimated_positions, gt_positions


def run_temporal_tracking_and_pnp_pipeline(idx=0):
    """Exercise 3: Temporal feature tracking and PnP-based motion estimation."""
    from utils.visualization import plot_four_cameras, plot_transformed_clouds

    frame0_data = run_single_pair(idx=idx, display=True, plot_3d=True)
    frame1_data = generate_next_frame_point_cloud(idx)

    good_matches = match_temporal_features_left_images(frame0_data, frame1_data)

    R, t = estimate_motion_with_four_point_pnp(frame0_data, frame1_data, good_matches)
    plot_four_cameras(R, t, baseline=0.54)

    supporters, non_supporters = find_four_view_reprojection_supporters(frame0_data, frame1_data, good_matches, R, t)

    R_ransac, t_ransac, inliers, outliers = estimate_motion_with_pnp_ransac(frame0_data, frame1_data, good_matches)

    plot_transformed_clouds(frame0_data, frame1_data, R_ransac, t_ransac)

    track_trajectory_over_sequence(num_frames=get_num_frames())


############################################################################
# DATABASE DEFINITION - Core data structures for feature track storage
# ADDING FRAMES TO DATABASE - Temporal tracking integration
############################################################################
def build_tracking_database(num_frames=None):
    """Task 4.1: Builds the long-term TrackingDB by matching features sequentially across frames."""
    print("\n--- Task 4.1: Building Tracking Database ---")
    if num_frames is None:
        num_frames = get_num_frames()
    return load_or_build_db(force_rebuild=False, num_frames=num_frames)


def report_tracking_statistics(db):
    """Task 4.2: Computes and prints descriptive tracking statistics."""
    stats = compute_tracking_statistics(db)
    print_tracking_statistics(stats)
    return db


def visualize_sample_track_patches(db):
    """Task 4.3: Visualizes localized crop windows of a specific multi-frame feature track."""
    from utils.visualization import plot_track_observations

    track_id = select_track_by_min_length(db, min_length=6)

    print(f"\n--- Task 4.3: Track Patch Visualization ---")
    print(f"Selected Track ID: {track_id}")
    print(f"Track Continuity Lifespan: {len(db.frames(track_id))} frames")
    print(f"Spanned Frame Nodes: {db.frames(track_id)}")

    plot_track_observations(db, track_id, crop_size=20)


def analyze_track_connectivity(db):
    """Task 4.4: Computes and charts track connectivity linkages between successive frames."""
    from utils.visualization import plot_connectivity

    connectivity = compute_connectivity(db)
    print(f"\n--- Task 4.4: Track Connectivity Analysis ---")
    print(f"Mean Track Continuity Connectivity Coefficient: {np.mean(connectivity):.2f}")
    plot_connectivity(connectivity)


def analyze_ransac_inlier_percentage(db):
    """Task 4.5: Charts the proportion of verification inliers across the timeline."""
    from utils.visualization import plot_inlier_percentage

    print(f"\n--- Task 4.5: RANSAC Inlier Proportion Analysis ---")
    mean_inliers = np.mean(db.inlier_percentages)
    print(f"Mean RANSAC Inlier Ratio: {mean_inliers:.2f}%")
    plot_inlier_percentage(db.inlier_percentages)


def plot_track_length_distribution(db):
    """Task 4.6: Generates a track lifespan frequency distribution histogram."""
    from utils.visualization import plot_track_length_histogram

    print("\n--- Task 4.6: Track Length Distribution ---")
    plot_track_length_histogram(db, min_length=2)


def analyze_track_reprojection_error(db):
    """Task 4.7: Analyzes spatial error baseline drift metrics over long duration paths."""
    from utils.visualization import plot_track_reprojection_error_analysis

    print("\n--- Task 4.7: Reprojection Error Analysis ---")

    track_id = select_long_track(db, min_length=10)
    frames = db.frames(track_id)
    print(f"Selected Validation Track ID: {track_id} | Lifespan: {len(frames)} | Nodes: {frames}")

    K, m_left0, m_right0 = read_cameras()

    first_frame_id = frames[0]
    X_world = triangulate_track_reference_point(db, track_id, first_frame_id, K, m_left0, m_right0)

    left_errors, right_errors = compute_track_reprojection_errors(db, track_id, frames, X_world, K, m_left0, m_right0)

    plot_track_reprojection_error_analysis(frames, left_errors, right_errors)


def analyze_matches_per_frame(db):
    """Task 4.8: Visualizes the number of matches per frame."""
    if not hasattr(db, "matches_per_frame"):
        print("\n--- Task 4.8: Matches per Frame ---")
        print("Skipped: matches_per_frame not in database (rebuild required)")
        return
    print("\n--- Task 4.8: Matches per Frame ---")
    plot_matches_per_frame(db.matches_per_frame, output_dir=OUTPUT_DIR)


def run_tracking_database_pipeline(num_frames=None):
    """Exercise 4: Tracking database construction and analysis."""
    print("==================================================")
    print("  EXERCISE 4: TRACKING DATABASE & STATISTICS      ")
    print("==================================================")

    db = build_tracking_database(num_frames=num_frames)
    report_tracking_statistics(db)
    visualize_sample_track_patches(db)
    analyze_track_connectivity(db)
    analyze_ransac_inlier_percentage(db)
    plot_track_length_distribution(db)
    analyze_track_reprojection_error(db)
    analyze_matches_per_frame(db)

    return db


def analyze_single_track_error_gtsam(db):
    """Task 5.1: Single track error analysis with GTSAM stereo camera model."""
    import gtsam
    from utils.visualization import reprojection_error_graph, factor_error_graph

    print("\n--- Task 5.1: Single Track Error Analysis with GTSAM ---")

    exact_tracks = [t_id for t_id in db.track_to_frames if len(db.frames(t_id)) == 15]
    if not exact_tracks:
        raise RuntimeError("No track found with length == 15 frames.")

    track_id = random.choice(exact_tracks)
    frames = db.frames(track_id)

    K_gtsam = init_gtsam_stereo_calibration()
    gt_poses = read_ground_truth_poses()

    poses_dict = {f_id: get_gtsam_camera_pose(gt_poses, frame_id=f_id) for f_id in frames}

    last_frame_id = frames[-1]
    obs_last = db.observation(last_frame_id, track_id)
    camera_last = gtsam.StereoCamera(poses_dict[last_frame_id], K_gtsam)
    X_world_gtsam = camera_last.backproject(
        gtsam.StereoPoint2(obs_last.x_left, obs_last.x_right, obs_last.y)
    )

    left_reprojection_errors = []
    right_reprojection_errors = []
    factor_errors = []

    frame_indices = list(range(len(frames)))

    for f_id in frames:
        obs = db.observation(f_id, track_id)
        camera = gtsam.StereoCamera(poses_dict[f_id], K_gtsam)

        projected_stereo_point = camera.project(X_world_gtsam)

        err_left = np.sqrt((obs.x_left - projected_stereo_point.uL())**2 + (obs.y - projected_stereo_point.v())**2)
        err_right = np.sqrt((obs.x_right - projected_stereo_point.uR())**2 + (obs.y - projected_stereo_point.v())**2)

        left_reprojection_errors.append(err_left)
        right_reprojection_errors.append(err_right)

        factor_errors.append(compute_single_factor_error(poses_dict[f_id], K_gtsam, X_world_gtsam, obs))

    reprojection_error_graph(frame_indices, left_reprojection_errors, right_reprojection_errors, OUTPUT_DIR)
    factor_error_graph(frame_indices, factor_errors, OUTPUT_DIR)


############################################################################
# BUNDLE ADJUSTMENT - Joint optimization of camera poses and 3D landmarks
############################################################################
def run_first_bundle_window(db):
    """Task 5.3: First bundle adjustment window optimization (10 frames)."""
    from utils.visualization import bundle1_3D, marginal_covariances, bundle1_2D_full, bundle1_2D_zoomed

    print("\n================================================================================")
    print("TASK 5.3: FIRST BUNDLE ADJUSTMENT WINDOW (10 FRAMES)")
    print("================================================================================")

    TARGET_FRAMES_COUNT = 10
    start_frame = 0
    end_frame = start_frame + TARGET_FRAMES_COUNT - 1
    window_frames = list(range(start_frame, end_frame + 1))

    print(f"Bundle Window spans from Frame {start_frame} to Frame {end_frame} (Total {len(window_frames)} frames)")

    landmarks_in_window = set()
    for f_id in window_frames:
        landmarks_in_window.update(db.tracks(f_id))
    print(f"Number of landmarks participating in this bundle: {len(landmarks_in_window)}")

    bundle_res = solve_bundle_window(db, start_frame, end_frame, max_tracks_per_window=150)

    result = bundle_res["result"]
    graph = bundle_res["graph"]
    cam_positions, lm_filtered = extract_optimized_geometry(result, window_frames, landmarks_in_window)

    initial_cam_positions = np.array([pose_translation_np(bundle_res["initial"].atPose3(symbol('c', f_id))) for f_id in window_frames])

    axis_length = 0.5
    bundle1_3D(window_frames, cam_positions, result, axis_length, OUTPUT_DIR)
    marginal_covariances(window_frames, graph, result, OUTPUT_DIR)
    bundle1_2D_full(window_frames, initial_cam_positions, cam_positions, lm_filtered, OUTPUT_DIR)
    bundle1_2D_zoomed(initial_cam_positions, cam_positions, OUTPUT_DIR)

    print(f"Initial error: {bundle_res['initial_error']:.4f}")
    print(f"Final error: {bundle_res['final_error']:.4f}")


def run_full_sliding_bundle_adjustment(db):
    """Task 5.4: Full sliding window bundle adjustment over all keyframes."""
    from utils.visualization import plot_q5_4_results, plot_keyframe_localization_error

    print("\n--- Task 5.4: Full Sliding Window Bundle Adjustment ---")

    keyframes = choose_keyframes(db, distance_threshold=2.5, max_gap=20, min_gap=5)
    print(f"Selected {len(keyframes)} keyframes: {keyframes}")

    global_keyframe_poses = {}
    current_global_pose = gtsam.Pose3()
    global_keyframe_poses[keyframes[0]] = current_global_pose

    all_points_global = []
    window_start_ids = []
    mean_initial_errors, mean_final_errors = [], []
    median_initial_errors, median_final_errors = [], []
    K_gtsam = init_gtsam_stereo_calibration()
    errors_by_distance = {}

    for i in range(len(keyframes) - 1):
        start_kf, end_kf = keyframes[i], keyframes[i + 1]
        print(f"Solving bundle window: {start_kf} -> {end_kf}")

        try:
            bundle_res = solve_bundle_window(db, start_kf, end_kf, max_tracks_per_window=150)
            result = bundle_res["result"]

            for track_id in bundle_res["optimized_landmark_ids"]:
                point_key = symbol('q', track_id)
                if result.exists(point_key):
                    pt_gtsam = result.atPoint3(point_key)
                    all_points_global.append([pt_gtsam[0], pt_gtsam[1], pt_gtsam[2]])

            relative_pose = bundle_res["relative_pose"]
            pose_end_global = current_global_pose.compose(relative_pose)

            global_keyframe_poses[end_kf] = pose_end_global
            current_global_pose = pose_end_global

            window_start_ids.append(start_kf)
            n = bundle_res["graph"].size()
            mean_initial_errors.append(bundle_res["initial_error"] / n)
            mean_final_errors.append(bundle_res["final_error"] / n)

            med_init, med_final = compute_window_projection_errors(db, bundle_res, K_gtsam)
            median_initial_errors.append(med_init)
            median_final_errors.append(med_final)

            accumulate_window_projection_errors_by_distance(db, bundle_res, K_gtsam, errors_by_distance, max_distance=20)

        except Exception as e:
            print(f"  Error in bundle window {start_kf}->{end_kf}: {e}")

    plot_q5_4_results(keyframes, global_keyframe_poses, all_points_global, output_dir=OUTPUT_DIR)
    plot_keyframe_localization_error(keyframes, global_keyframe_poses, output_dir=OUTPUT_DIR)

    if window_start_ids and mean_initial_errors and mean_final_errors:
        plot_bundle_optimization_error(window_start_ids, mean_initial_errors, mean_final_errors, output_dir=OUTPUT_DIR)

    if window_start_ids and len(median_initial_errors) == len(window_start_ids):
        valid_indices = [i for i in range(len(median_initial_errors)) if np.isfinite(median_initial_errors[i]) and np.isfinite(median_final_errors[i])]
        if valid_indices:
            window_ids_valid = [window_start_ids[i] for i in valid_indices]
            med_init_valid = [median_initial_errors[i] for i in valid_indices]
            med_final_valid = [median_final_errors[i] for i in valid_indices]
            plot_bundle_projection_error(window_ids_valid, med_init_valid, med_final_valid, output_dir=OUTPUT_DIR)

    dist_list = sorted(errors_by_distance.keys())
    if dist_list:
        median_errs = [np.nanmedian(errors_by_distance[d]) for d in dist_list]
        valid_dists = [d for d, err in zip(dist_list, median_errs) if np.isfinite(err)]
        if valid_dists:
            valid_errs = [np.nanmedian(errors_by_distance[d]) for d in valid_dists]
            plot_projection_error_vs_distance(
                valid_dists, valid_errs,
                "Task 5.4: Bundle Projection Error vs Distance",
                os.path.join(OUTPUT_DIR, "task_5_4_projection_error_vs_distance.png")
            )

    return {
        "global_keyframe_poses": global_keyframe_poses,
        "window_start_ids": window_start_ids,
        "mean_initial_errors": mean_initial_errors,
        "mean_final_errors": mean_final_errors,
        "median_initial_errors": median_initial_errors,
        "median_final_errors": median_final_errors,
        "errors_by_distance": errors_by_distance
    }


def run_bundle_adjustment_pipeline(db):
    """Exercise 5: GTSAM bundle adjustment and keyframe optimization."""
    print("==================================================")
    print("  EXERCISE 5: BUNDLE ADJUSTMENT & OPTIMIZATION    ")
    print("==================================================")

    analyze_single_track_error_gtsam(db)
    run_first_bundle_window(db)
    bundle_stats = run_full_sliding_bundle_adjustment(db)
    return bundle_stats


############################################################################
# RELATIVE TRANSFORMATION & COVARIANCE - Extracting pose uncertainty from bundle
############################################################################
def extract_relative_pose_constraints(db):
    """Task 6.1: Extract relative pose constraints between consecutive keyframes."""
    from utils.bundle_adjustment import choose_keyframes

    print("\n" + "=" * 80)
    print("TASK 6.1: RELATIVE POSE CONSTRAINTS FROM BUNDLE ADJUSTMENT")
    print("=" * 80)

    keyframes = choose_keyframes(db, distance_threshold=2.5, min_gap=20, max_gap=20)
    c0_idx, ck_idx = keyframes[0], keyframes[1]
    print(f"Total keyframes : {len(keyframes)}")
    print(f"First two       : c0={c0_idx}, c_k={ck_idx}")

    print(f"\n--- Plotting covariance sensitivity for bundle {c0_idx}→{ck_idx} ---")
    run_prior_sensitivity_sweep(db, c0_idx, ck_idx, OUTPUT_DIR)

    # Extract first relative pose and conditional covariance
    print(f"\n--- Relative pose & covariance for c0={c0_idx} → c_k={ck_idx} ---")
    relative_pose, Sigma_rel = compute_relative_pose_and_covariance(db, c0_idx, ck_idx)
    print(f"\nRelative Pose between keyframes c{c0_idx} and c{ck_idx}:")
    print(relative_pose)
    print(f"\nRelative pose covariance (T_0{ck_idx} = T_0^(-1) T_{ck_idx}):")
    print(np.array2string(Sigma_rel, precision=8, suppress_small=True))

    # Process all consecutive pairs
    print("\n--- Processing all consecutive keyframe pairs ---")
    relative_poses, relative_covs = compute_all_relative_constraints(db, keyframes)

    print("\n--- Relative Error Analysis (Bundle vs PnP vs Ground Truth) ---")
    bundle_edge_ids, bundle_loc_errors, bundle_ang_errors = compute_bundle_relative_error_vs_gt(relative_poses)
    pnp_edge_ids, pnp_loc_errors, pnp_ang_errors = [], [], []
    try:
        from utils.tracking import compute_pnp_relative_error_vs_gt
        pnp_edge_ids, pnp_loc_errors, pnp_ang_errors = compute_pnp_relative_error_vs_gt(db, sorted(relative_poses.keys()))
    except Exception:
        pass

    if bundle_edge_ids:
        plot_relative_error_comparison(
            bundle_edge_ids, bundle_loc_errors, pnp_loc_errors if pnp_loc_errors else bundle_loc_errors,
            "Location Error (m)", "Graph 6.1a: Relative Location Error",
            os.path.join(OUTPUT_DIR, "task_6_1_relative_location_error.png")
        )
        plot_relative_error_comparison(
            bundle_edge_ids, bundle_ang_errors, pnp_ang_errors if pnp_ang_errors else bundle_ang_errors,
            "Angle Error (deg)", "Graph 6.1b: Relative Angle Error",
            os.path.join(OUTPUT_DIR, "task_6_1_relative_angle_error.png")
        )
        print(f"Bundle relative location error (mean): {np.mean(bundle_loc_errors):.4f} m")
        print(f"Bundle relative angle error (mean): {np.mean(bundle_ang_errors):.4f} deg")

    return relative_poses, relative_covs


############################################################################
# POSE GRAPH - Building and optimizing factor graph from keyframe constraints
############################################################################
def optimize_pose_graph_from_constraints(db, relative_poses, relative_covs):
    """Task 6.2: Optimize pose graph from relative pose constraints."""
    from utils.visualization import plot_pose_graph_trajectory, plot_pose_graph_with_covariances, plot_pose_graph_2d_ellipses

    print("\n" + "=" * 80)
    print("TASK 6.2: POSE GRAPH OPTIMIZATION")
    print("=" * 80)

    # Clean bad edges
    cleaned_poses, cleaned_covs = clean_pose_graph_edges(relative_poses, relative_covs)

    # Build graph and robustly initialize estimates
    graph, initial = build_and_initialize_pose_graph(cleaned_poses, cleaned_covs)
    print(f"Pose graph factors: {graph.size()}")
    print(f"Initial poses: {initial.size()}")
    print(f"Pose graph error BEFORE optimization: {graph.error(initial):.8e}")

    # Plot initial state
    plot_pose_graph_trajectory(
        initial,
        title="Graph 6.2a: Initial Pose Graph Trajectory (Fixed Gaps)",
        output_path=os.path.join(OUTPUT_DIR, "task_6_2_initial_pose_graph.png")
    )

    # Optimize
    result, marginals = optimize_pose_graph(graph, initial)
    print(f"Pose graph error AFTER optimization: {graph.error(result):.8e}")

    # Save final visual artifacts
    plot_pose_graph_trajectory(
        result,
        title="Graph 6.2b: Optimized Pose Graph Trajectory",
        output_path=os.path.join(OUTPUT_DIR, "task_6_2_optimized_pose_graph.png")
    )
    plot_pose_graph_with_covariances(
        result,
        marginals,
        title="Graph 6.2c: Optimized Pose Graph With Final Marginal Covariances",
        output_path=os.path.join(OUTPUT_DIR, "task_6_2_pose_graph_covariances.png"),
        covariance_step=5,
    )
    plot_pose_graph_2d_ellipses(
        result,
        marginals,
        output_path=os.path.join(OUTPUT_DIR, "task_6_2_pose_graph_2d_covariances.png")
    )

    return result


def run_pose_graph_optimization_pipeline(db):
    """Exercise 6: Pose graph construction and optimization."""
    print("==================================================")
    print("  EXERCISE 6: POSE GRAPH OPTIMIZATION             ")
    print("==================================================")

    relative_poses, relative_covs = extract_relative_pose_constraints(db)
    optimize_pose_graph_from_constraints(db, relative_poses, relative_covs)

    return relative_poses, relative_covs


############################################################################
# LOOP CLOSURE DETECTION - Finding candidate loop-closure constraints
############################################################################
def detect_loop_candidates(db, relative_poses, relative_covs, keyframes, mahalanobis_threshold=1000.0):
    """Task 7.1: Detect loop closure candidates using Mahalanobis distance filtering."""

    print("\n" + "=" * 80)
    print("TASK 7.1: DETECT LOOP CLOSURE CANDIDATES (SHORTEST-PATH COVARIANCE VERSION)")
    print("=" * 80)

    candidates, total_found = detect_loop_closure_candidates(
        relative_poses, relative_covs, keyframes, mahalanobis_threshold
    )

    print("\n" + "=" * 80)
    print(f"Total Loop Closure Candidates Found: {total_found}")
    print("=" * 80)

    return candidates


############################################################################
# LOOP CLOSURE VERIFICATION - Consensus-based validation of loop-closure matches
############################################################################
def verify_loop_closures(db, loop_candidates, inlier_ratio_threshold=0.6):
    """Task 7.2: Verify loop closures using visual consensus (RANSAC fundamental matrix)."""
    print("\n" + "=" * 80)
    print("TASK 7.2: CONSENSUS MATCHING (VISUAL VERIFICATION)")
    print("=" * 80)

    verified_loops, total_verified = verify_loop_closures_consensus(
        db, loop_candidates, inlier_ratio_threshold, OUTPUT_DIR
    )

    print("\n" + "=" * 80)
    print(f"SUMMARY 7.2: Total Visually Verified Loops: {total_verified}")
    print("=" * 80)

    return verified_loops


############################################################################
# LOOP CLOSURE POSE ESTIMATION - Factor creation from verified loop pairs
############################################################################
def estimate_loop_closure_poses(db, verified_loops):
    """Task 7.3: Estimate relative poses for verified loop closures using bundle adjustment."""
    print("\n" + "=" * 80)
    print("TASK 7.3: RELATIVE POSE ESTIMATION FROM SMALL BUNDLE")
    print("=" * 80)

    loop_measurements = estimate_verified_loop_relative_poses(db, verified_loops, output_dir=OUTPUT_DIR)
    print(f"SUMMARY 7.3: Successful relative loop measurements: {len(loop_measurements)}")
    return loop_measurements


############################################################################
# LOOP CLOSURE INTEGRATION - Adding loop constraints and re-optimizing pose graph
############################################################################
def update_pose_graph_with_closures(cleaned_poses, cleaned_covs, loop_measurements):
    """Task 7.4: Update pose graph with loop closure constraints and re-optimize."""
    print("\n" + "=" * 80)
    print("TASK 7.4: UPDATE POSE GRAPH WITH LOOP CLOSURES")
    print("=" * 80)

    pg_results = add_loop_closures_and_optimize(
        cleaned_poses, cleaned_covs, loop_measurements, output_dir=OUTPUT_DIR
    )

    print(
        f"SUMMARY 7.4: Added {pg_results['num_added_loop_closures']} loop closure factors"
    )

    return pg_results


def report_loop_closure_results(pg_results, loop_measurements):
    """Task 7.5: Generate final reports and visualizations comparing pre/post loop closure."""
    from utils.visualization import plot_pose_graph_vs_ground_truth, plot_absolute_location_error, plot_location_uncertainty_size

    print("\n" + "=" * 80)
    print("TASK 7.5: FINAL PLOTS AND REPORT NUMBERS")
    print("=" * 80)

    print(f"Visually verified / estimated loop measurements: {len(loop_measurements)}")
    print(f"Loop closure factors added to pose graph: {pg_results['num_added_loop_closures']}")

    plot_pose_graph_vs_ground_truth(
        pg_results["no_loop_result"],
        pg_results["loop_result"],
        output_dir=OUTPUT_DIR,
    )
    plot_absolute_location_error(
        pg_results["no_loop_result"],
        pg_results["loop_result"],
        output_dir=OUTPUT_DIR,
    )
    plot_location_uncertainty_size(
        pg_results["no_loop_result"],
        pg_results["no_loop_marginals"],
        pg_results["loop_result"],
        pg_results["loop_marginals"],
        output_dir=OUTPUT_DIR,
    )

    no_loop_abs_err = compute_pose_graph_absolute_error(pg_results["no_loop_result"])
    loop_abs_err = compute_pose_graph_absolute_error(pg_results["loop_result"])

    if no_loop_abs_err["frame_ids"]:
        plot_absolute_location_error_components(
            no_loop_abs_err["frame_ids"], no_loop_abs_err["err_x"], no_loop_abs_err["err_y"],
            no_loop_abs_err["err_z"], no_loop_abs_err["err_norm"],
            "Graph 7.5d: Pose Graph Absolute Location Error (No Loop Closure)",
            os.path.join(OUTPUT_DIR, "task_7_5_absolute_location_error_no_lc.png")
        )
        plot_absolute_angle_error(
            no_loop_abs_err["frame_ids"], no_loop_abs_err["err_angle"],
            "Graph 7.5f: Pose Graph Absolute Angle Error (No Loop Closure)",
            os.path.join(OUTPUT_DIR, "task_7_5_absolute_angle_error_no_lc.png")
        )

    if loop_abs_err["frame_ids"]:
        plot_absolute_location_error_components(
            loop_abs_err["frame_ids"], loop_abs_err["err_x"], loop_abs_err["err_y"],
            loop_abs_err["err_z"], loop_abs_err["err_norm"],
            "Graph 7.5e: Pose Graph Absolute Location Error (With Loop Closure)",
            os.path.join(OUTPUT_DIR, "task_7_5_absolute_location_error_lc.png")
        )
        plot_absolute_angle_error(
            loop_abs_err["frame_ids"], loop_abs_err["err_angle"],
            "Graph 7.5g: Pose Graph Absolute Angle Error (With Loop Closure)",
            os.path.join(OUTPUT_DIR, "task_7_5_absolute_angle_error_lc.png")
        )

    if loop_measurements:
        plot_loop_closure_match_stats(loop_measurements, output_dir=OUTPUT_DIR)

    plot_angle_uncertainty_size(
        pg_results["no_loop_result"], pg_results["no_loop_marginals"],
        pg_results["loop_result"], pg_results["loop_marginals"],
        output_dir=OUTPUT_DIR
    )

    print("\nSnapshot choice for report:")
    print("1. before adding loop closures")
    print("2-4. after evenly spaced loop-closure additions, including the final graph")
    print("Uncertainty size measure: sqrt(det(Sigma_xz)), i.e. 1-sigma location ellipse area divided by pi.")


def run_loop_closure_pipeline(db, relative_poses, relative_covs):
    """Exercise 7: Loop closure detection, verification, and pose graph refinement."""
    print("==================================================")
    print("  EXERCISE 7: LOOP CLOSURE DETECTION & SLAM       ")
    print("==================================================")

    from utils.bundle_adjustment import choose_keyframes

    keyframes = choose_keyframes(db, distance_threshold=2.5, min_gap=20, max_gap=20)
    cleaned_poses, cleaned_covs = clean_pose_graph_edges(relative_poses, relative_covs)

    mahalanobis_threshold = 1000.0
    candidates = detect_loop_candidates(db, relative_poses, relative_covs, keyframes, mahalanobis_threshold)
    verified_loops = verify_loop_closures(db, candidates, inlier_ratio_threshold=0.6)
    loop_measurements = estimate_loop_closure_poses(db, verified_loops)
    pg_results = update_pose_graph_with_closures(cleaned_poses, cleaned_covs, loop_measurements)
    report_loop_closure_results(pg_results, loop_measurements)
    return pg_results


def analyze_absolute_pnp_error(db):
    """Performance Analysis: Absolute PnP trajectory error vs ground truth."""
    print("\n--- Performance Analysis: Absolute PnP Error ---")
    pnp_err = compute_absolute_pnp_error(db)
    plot_absolute_location_error_components(
        pnp_err["frame_ids"], pnp_err["err_x"], pnp_err["err_y"],
        pnp_err["err_z"], pnp_err["err_norm"],
        "Graph PA.1a: Absolute PnP Location Error",
        os.path.join(OUTPUT_DIR, "task_pa_absolute_location_error_pnp.png")
    )
    plot_absolute_angle_error(
        pnp_err["frame_ids"], pnp_err["err_angle"],
        "Graph PA.1b: Absolute PnP Angle Error",
        os.path.join(OUTPUT_DIR, "task_pa_absolute_angle_error_pnp.png")
    )


def analyze_pnp_projection_error_vs_distance(db):
    """Performance Analysis: PnP projection error vs distance from reference frame."""
    print("\n--- Performance Analysis: PnP Projection Error vs Distance ---")
    K, m_left0, m_right0 = read_cameras()
    distances, median_errors = compute_pnp_projection_error_vs_distance(
        db, K, m_left0, m_right0, max_distance=40, min_track_length=5, sample_size=150, seed=0
    )
    if distances:
        plot_projection_error_vs_distance(
            distances, median_errors,
            "Graph PA.2: PnP Projection Error vs Distance",
            os.path.join(OUTPUT_DIR, "task_pa_pnp_projection_error_vs_distance.png")
        )


def analyze_kitti_sequence_errors(db, keyframes, bundle_keyframe_poses):
    """Performance Analysis: KITTI-style error over sub-sections (100/400/800 frames)."""
    print("\n--- Performance Analysis: KITTI Sequence Errors ---")
    segment_lengths = [100, 400, 800]

    pnp_loc_results = {}
    pnp_ang_results = {}
    for seg_len in segment_lengths:
        loc_pct, ang_m = compute_kitti_sequence_errors(db.camera_poses, read_ground_truth_poses(), seg_len)
        pnp_loc_results[seg_len] = loc_pct
        pnp_ang_results[seg_len] = ang_m
        if loc_pct and ang_m:
            print(f"  PnP segment {seg_len}: {len(loc_pct)} segments, location error {np.mean(loc_pct):.2f}%, angle error {np.mean(ang_m):.4f} deg/m")
        else:
            print(f"  PnP segment {seg_len}: no valid segments (loc={len(loc_pct)}, ang={len(ang_m)})")

    bundle_loc_results = {}
    bundle_ang_results = {}
    for seg_len in segment_lengths:
        try:
            bundle_poses_dict = {}
            for i, kf in enumerate(keyframes):
                if kf in bundle_keyframe_poses:
                    pose_gtsam = bundle_keyframe_poses[kf]
                    # Convert GTSAM Pose3 (C2W) to (R, t) tuple
                    R_c2w = np.array(pose_gtsam.rotation().matrix())
                    t_c2w = np.array(pose_gtsam.translation()).reshape(3, 1)
                    bundle_poses_dict[i] = (R_c2w, t_c2w)

            if bundle_poses_dict:
                loc_pct, ang_m = compute_kitti_sequence_errors_keyframes(
                    keyframes, bundle_poses_dict, read_ground_truth_poses(), seg_len
                )
                bundle_loc_results[seg_len] = loc_pct
                bundle_ang_results[seg_len] = ang_m
                if loc_pct and ang_m:
                    print(f"  Bundle segment {seg_len}: {len(loc_pct)} segments, location error {np.mean(loc_pct):.2f}%, angle error {np.mean(ang_m):.4f} deg/m")
                else:
                    print(f"  Bundle segment {seg_len}: {len(bundle_poses_dict)} poses, but no valid segments (loc={len(loc_pct)}, ang={len(ang_m)})")
            else:
                print(f"  Bundle segment {seg_len}: no bundle poses available ({len(keyframes)} keyframes, {len(bundle_keyframe_poses)} in poses dict)")
        except Exception as e:
            import traceback
            print(f"  Bundle segment {seg_len}: error ({type(e).__name__}: {e})")
            traceback.print_exc()

    # Only plot if we have non-empty data
    pnp_loc_has_data = any(len(v) > 0 for v in pnp_loc_results.values())
    pnp_ang_has_data = any(len(v) > 0 for v in pnp_ang_results.values())
    bundle_loc_has_data = any(len(v) > 0 for v in bundle_loc_results.values())
    bundle_ang_has_data = any(len(v) > 0 for v in bundle_ang_results.values())

    if pnp_loc_has_data:
        plot_kitti_sequence_error(
            pnp_loc_results, "Location Error (%)",
            "Graph PA.3a: KITTI PnP Location Error by Segment",
            os.path.join(OUTPUT_DIR, "task_pa_kitti_pnp_location_error.png")
        )
    if pnp_ang_has_data:
        plot_kitti_sequence_error(
            pnp_ang_results, "Angle Error (deg/m)",
            "Graph PA.3b: KITTI PnP Angle Error by Segment",
            os.path.join(OUTPUT_DIR, "task_pa_kitti_pnp_angle_error.png")
        )
    if bundle_loc_has_data:
        plot_kitti_sequence_error(
            bundle_loc_results, "Location Error (%)",
            "Graph PA.3c: KITTI Bundle Location Error by Segment",
            os.path.join(OUTPUT_DIR, "task_pa_kitti_bundle_location_error.png")
        )
    if bundle_ang_has_data:
        plot_kitti_sequence_error(
            bundle_ang_results, "Angle Error (deg/m)",
            "Graph PA.3d: KITTI Bundle Angle Error by Segment",
            os.path.join(OUTPUT_DIR, "task_pa_kitti_bundle_angle_error.png")
        )


def compare_full_pipeline_trajectories(db, bundle_keyframe_poses, pg_results):
    """Performance Analysis: Overlay PnP, Bundle, Pose-Graph+LC, and GT trajectories."""
    print("\n--- Performance Analysis: Full Trajectory Comparison ---")
    from utils.geometry import camera_center

    pnp_positions = np.array([camera_center(*pose) for pose in db.camera_poses])
    gt_positions = np.array([camera_center(*pose) for pose in read_ground_truth_poses()[:len(db.camera_poses)]])

    bundle_ids, bundle_pos = extract_trajectory_and_ids(pg_results["no_loop_result"])
    if len(bundle_pos) == 0:
        bundle_pos = np.array([[0, 0, 0]])
    pg_ids, pg_pos = extract_trajectory_and_ids(pg_results["loop_result"])
    if len(pg_pos) == 0:
        pg_pos = np.array([[0, 0, 0]])

    plot_full_trajectory_comparison(
        pnp_positions, bundle_ids, bundle_pos, pg_ids, pg_pos, gt_positions, output_dir=OUTPUT_DIR,
        title="Graph PA.4: Full Trajectory Comparison (Bird's Eye)"
    )


def run_performance_analysis_pipeline(db, bundle_keyframe_poses, pg_results):
    """Performance Analysis: cross-cutting metrics that don't map to a single existing exercise stage."""
    print("==================================================")
    print("  PERFORMANCE ANALYSIS STAGE                     ")
    print("==================================================")

    keyframes = choose_keyframes(db, distance_threshold=2.5, min_gap=20, max_gap=20)
    analyze_absolute_pnp_error(db)
    analyze_pnp_projection_error_vs_distance(db)
    analyze_kitti_sequence_errors(db, keyframes, bundle_keyframe_poses)
    compare_full_pipeline_trajectories(db, bundle_keyframe_poses, pg_results)


def main():
    print("==================================================")
    print("  VISUAL AERIAL NAVIGATION (VAN) - FULL PIPELINE  ")
    print("==================================================\n")

    # --- Exercise 1: Feature Extraction & Matching ---
    print("\n[Stage 1/7] Running Feature Detection and Matching...")
    # run_feature_detection_and_matching_pipeline(frame_index=0)

    # --- Exercise 2: Stereo Rectification & Triangulation ---
    print("\n[Stage 2/7] Running Stereo Rectification and Triangulation...")
    # run_stereo_rectification_pipeline(frame_index=0)

    # --- Exercise 3: Temporal Tracking & PnP Estimation ---
    print("\n[Stage 3/7] Running Temporal Tracking and PnP Motion Estimation...")
    # run_temporal_tracking_and_pnp_pipeline(idx=0)

    # --- Exercise 4: Tracking Database & Statistics ---
    print("\n[Stage 4/7] Running Tracking Database and Analysis...")
    db = run_tracking_database_pipeline(num_frames=get_num_frames())

    # --- Exercise 5: Bundle Adjustment & Optimization ---
    print("\n[Stage 5/7] Running Bundle Adjustment and Keyframe Optimization...")
    bundle_stats = run_bundle_adjustment_pipeline(db)

    # --- Exercise 6: Pose Graph Optimization ---
    print("\n[Stage 6/7] Running Pose Graph Optimization...")
    relative_poses, relative_covs = run_pose_graph_optimization_pipeline(db)

    # --- Exercise 7: Loop Closure Detection ---
    print("\n[Stage 7/7] Running Loop Closure Detection and Pose Graph Refinement...")
    pg_results = run_loop_closure_pipeline(db, relative_poses, relative_covs)

    # --- Performance Analysis Stage ---
    print("\n[Stage 8] Running Performance Analysis...")
    run_performance_analysis_pipeline(db, bundle_stats["global_keyframe_poses"], pg_results)

    print("\n==================================================")
    print("  PIPELINE COMPLETE!                         ")
    print("==================================================")


if __name__ == "__main__":
    main()