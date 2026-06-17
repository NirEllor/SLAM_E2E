# ex4.py
import van_utils as lib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from tracking_database_custom import TrackingDB

# The total number of images to process across the sequence loop
NUM_FRAMES = lib.get_num_frames()


def q4_1(num_frames=NUM_FRAMES):
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

    prev_data = lib.run_single_pair(
        idx=0,
        display=False,
        plot_3d=False
    )

    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    for idx in range(1, num_frames):
        print(f"Processing Frame Sequence Node: {idx}/{num_frames - 1}")

        curr_data = lib.run_single_pair(
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

        try:
            correspondences = lib.build_pnp_correspondences(
                prev_data,
                curr_data,
                temporal_matches
            )

            if len(correspondences) < 4:
                raise RuntimeError("Not enough correspondences for PnP-RANSAC.")

            best_inliers = []
            best_outliers = []
            best_R = None
            best_t = None

            k_matrix, _, _ = lib.read_cameras()

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

                inliers, outliers = lib.evaluate_supporters(
                    correspondences,
                    prev_data,
                    curr_data,
                    R_candidate,
                    tvec,
                    threshold=2
                )

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
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
            # X_curr = R_rel X_prev + t_rel
            R_global, t_global = lib.compose_transform(
                R_global,
                t_global,
                best_R,
                best_t
            )

        except RuntimeError as e:
            print(f"PnP-RANSAC failure at frame link {idx - 1}->{idx}: {e}")
            inlier_percentages.append(0.0)

            # If pose estimation fails, keep previous pose
            R_global = R_global.copy()
            t_global = t_global.copy()

        camera_poses.append(
            (R_global.copy(), t_global.copy())
        )

        prev_stereo = lib.build_stereo_dict(prev_data)
        curr_stereo = lib.build_stereo_dict(curr_data)

        db.update_tracks(
            idx,
            temporal_matches,
            prev_stereo,
            curr_stereo,
            prev_data,
            curr_data
        )

        prev_data = curr_data

    db.inlier_percentages = inlier_percentages
    db.camera_poses = camera_poses

    return db

def q4_2(db):
    """
    4.2: Computes and prints descriptive tracking statistics.
    """
    stats = lib.compute_tracking_statistics(db)
    lib.print_tracking_statistics(stats)
    return db


def q4_3(db):
    """
    4.3: Visualizes localized crop windows of a specific multi-frame feature track.
    """
    track_id = lib.select_track_by_min_length(db, min_length=6)

    print(f"\n--- Task 4.3: Track Patch Visualization ---")
    print(f"Selected Track ID: {track_id}")
    print(f"Track Continuity Lifespan: {len(db.frames(track_id))} frames")
    print(f"Spanned Frame Nodes: {db.frames(track_id)}")

    lib.plot_track_observations(db, track_id, crop_size=20)


def q4_4(db):
    """
    4.4: Computes and charts track connectivity linkages between successive frames.
    """
    connectivity = lib.compute_connectivity(db)
    print(f"Mean Track Continuity Connectivity Coefficient: {np.mean(connectivity):.2f}")
    lib.plot_connectivity(connectivity)


def q4_5(db):
    """
    4.5: Charts the proportion of verification inliers across the timeline.
    """
    print(f"\n--- Task 4.5: RANSAC Inlier Proportion Analysis ---")
    mean_inliers = np.mean(db.inlier_percentages)
    print(f"Mean RANSAC Inlier Ratio: {mean_inliers:.2f}%")
    lib.plot_inlier_percentage(db.inlier_percentages)


def q4_6(db):
    """
    4.6: Generates a track lifespan frequency distribution histogram.
    """
    lib.plot_track_length_histogram(db, min_length=2)

def q4_7(db):
    """
    4.7: Analyzes spatial error baseline drift metrics over long duration paths.
    Fixes the geometric reference system by rotating the stereo baseline vector
    into the camera's local coordinate frame for both triangulation and projection.
    """
    print("\n--- Task 4.7: Reprojection Error Analysis ---")
    
    # 1. Isolate non-trivial tracking lines that run across at least 10 keyframes
    long_tracks = [t_id for t_id in db.track_to_frames if len(db.frames(t_id)) >= 10]
    if not long_tracks:
        raise RuntimeError("No structural track found with an observation lifetime >= 10 frames.")
    
    track_id = random.choice(long_tracks)
    frames = db.frames(track_id)
    print(f"Selected Validation Track ID: {track_id} | Lifespan: {len(frames)} | Nodes: {frames}")
    
    # 2. Extract camera metrics and absolute ground truth positional poses
    K, m_left0, m_right0 = lib.read_cameras()
    # Stereo baseline translation vector in left camera coordinates
    t_stereo = np.linalg.inv(K) @ m_right0[:, 3]
    gt_poses = lib.read_ground_truth_poses()
    
    # 3. Triangulate spatial reference location from the FIRST frame node (frames[0])
    first_frame_id = frames[0]
    obs_first = db.observation(first_frame_id, track_id)
    
    R_first_gt, t_first_gt = gt_poses[first_frame_id]
    
    # Transform Ground Truth matrices from Camera-to-World to World-to-Camera (w2c)
    R_first_w2c = R_first_gt.T
    t_first_w2c = -R_first_gt.T @ t_first_gt.reshape(3, 1)
    
    # Construct corrected projection matrices for triangulation
    P_L_first = K @ np.hstack([R_first_w2c, t_first_w2c])
    # Rotate the stereo baseline vector into the world coordinate frame properly
    P_R_first = K @ np.hstack([R_first_w2c, t_first_w2c + R_first_w2c @ t_stereo.reshape(3, 1)])
    
    p_left_first = np.array([obs_first.x_left, obs_first.y])
    p_right_first = np.array([obs_first.x_right, obs_first.y])
    
    # Compute the fixed 3D reference world coordinate
    X_world = lib.triangulate_point_linear(p_left_first, p_right_first, P_L_first, P_R_first)
    
    left_errors = []
    right_errors = []
    distances_from_reference = []
    
    # 4. Calculate projection residuals relative to moving baselines across the track
    for idx, frame_id in enumerate(frames):
        obs = db.observation(frame_id, track_id)
        R_curr_gt, t_curr_gt = gt_poses[frame_id]
        
        # Transform current camera pose from Camera-to-World to World-to-Camera (w2c)
        R_curr_w2c = R_curr_gt.T
        t_curr_w2c = -R_curr_gt.T @ t_curr_gt.reshape(3, 1)
        
        # Build corrected local projection matrices for the current frame
        P_left = K @ np.hstack([R_curr_w2c, t_curr_w2c])
        P_right = K @ np.hstack([R_curr_w2c, t_curr_w2c + R_curr_w2c @ t_stereo.reshape(3, 1)])
        
        # Project world point onto current frame cameras using the corrected matrices
        proj_l = lib.project_point(P_left, X_world)
        proj_r = lib.project_point(P_right, X_world)
        
        # Actual pixel observations from database
        obs_l = np.array([obs.x_left, obs.y])
        obs_r = np.array([obs.x_right, obs.y])
        
        # Calculate L2 errors (Euclidean distance)
        err_l = np.linalg.norm(proj_l - obs_l)
        err_r = np.linalg.norm(proj_r - obs_r)
        
        left_errors.append(err_l)
        right_errors.append(err_r)
        distances_from_reference.append(idx)
        
    # 5. Generate the finalized diagnostic metric plot
    plt.figure(figsize=(10, 6))
    plt.plot(distances_from_reference, left_errors, label='Left Channel Residuals', color='#2F4F4F', linewidth=2)
    plt.plot(distances_from_reference, right_errors, label='Right Channel Residuals', color='#FFA500', linewidth=2)
    
    plt.title("PnP - projection error vs track length")
    plt.xlabel("distance from reference (frames)")
    plt.ylabel("projection error (pixels)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()

def q4():
    print("========== Running Exercise 4 Execution Script ==========")
    
    # Trigger 4.1 sequence ingestion loop
    db = q4_1(num_frames=NUM_FRAMES)
    
    # Trigger analytics pipeline tasks sequentially
    q4_2(db)
    q4_3(db)
    q4_4(db)
    q4_5(db)
    q4_6(db)
    q4_7(db)
    
    plt.show()


def main():
    q4()


if __name__ == '__main__':
    main()