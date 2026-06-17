# ex5.py
import gtsam
from gtsam import symbol
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import van_utils as lib
from gtsam.utils import plot as gtsam_plot
from ex4 import q4_1

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)  # Creates the folder if it doesn't exist

NUM_FRAMES = lib.get_num_frames()


def q5_1(db):
    print("\n--- Task 5.1: Single Track Error Analysis with GTSAM ---")
    
    long_tracks = [t_id for t_id in db.track_to_frames if len(db.frames(t_id)) >= 10]
    if not long_tracks: raise RuntimeError("No track found with length >= 10 frames.")
        
    track_id = random.choice(long_tracks)
    frames = db.frames(track_id)
    
    K_gtsam = lib.init_gtsam_stereo_calibration()
    poses_dict = {
        f_id: lib.pnp_pose_to_gtsam_pose(
            db.camera_poses[f_id][0],
            db.camera_poses[f_id][1]
        )
        for f_id in frames
    }
    
    # Triangulate initial point configuration from last frame
    last_frame_id = frames[-1]
    obs_last = db.observation(last_frame_id, track_id)
    X_world_gtsam = gtsam.StereoCamera(poses_dict[last_frame_id], K_gtsam).backproject(
        gtsam.StereoPoint2(obs_last.x_left, obs_last.x_right, obs_last.y)
    )
    
    reprojection_errors, factor_errors = [], []
    for f_id in frames:
        obs = db.observation(f_id, track_id)
        
        # Call generalized evaluation functions directly
        reprojection_errors.append(lib.compute_stereo_reprojection_error(poses_dict[f_id], K_gtsam, X_world_gtsam, obs))
        factor_errors.append(lib.compute_single_factor_error(poses_dict[f_id], K_gtsam, X_world_gtsam, obs))

    # Output Plotting Visualization pass
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(frames, reprojection_errors, marker='o', color='b')
    ax1.set_title("Reprojection Error ($L_2$ Norm)"); ax1.grid(True)
    ax2.plot(frames, factor_errors, marker='s', color='r')
    ax2.set_title("GTSAM Factor Error"); ax2.grid(True)
    output_path = os.path.join(output_dir, "task_5_1_output.png")
    plt.savefig(output_path, dpi=300); plt.close()

def q5_3(db):
    print("\n================================================================================")
    print("SECTION 5.3: FIRST BUNDLE ADJUSTMENT WINDOW")
    print("================================================================================")
    
    K_gtsam = lib.init_gtsam_stereo_calibration()
    K_mat, _, m_right0 = lib.read_cameras()
    t_stereo = np.linalg.inv(K_mat) @ m_right0[:, 3]
    camera_poses = db.camera_poses
    # Keyframe selection logic bound to 2.5 meters
    kf_indices, accumulated_dist = [0], 0.0
    for idx in range(1, db.frame_num()):
        R_prev, t_prev = camera_poses[idx - 1]
        R_curr, t_curr = camera_poses[idx]

        p_prev = lib.camera_center(R_prev, t_prev)
        p_curr = lib.camera_center(R_curr, t_curr)
        accumulated_dist += np.linalg.norm(p_curr - p_prev)
        if accumulated_dist >= 2.5:
            kf_indices.append(idx); break
            
    start_frame, end_frame = kf_indices[0], kf_indices[1]
    window_frames = list(range(start_frame, end_frame + 1))
    
    print(f"Bundle Window 1 spans from Frame {start_frame} to Keyframe {end_frame} (Total {len(window_frames)} frames)")
    
    landmarks_in_window = set()
    for f_id in window_frames: 
        landmarks_in_window.update(db.tracks(f_id))
    print(f"Number of landmarks participating in this bundle: {len(landmarks_in_window)}")
        
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    base_noise = gtsam.noiseModel.Isotropic.Sigma(
        3,
        1.0
    )

    measurement_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(2.0),
        base_noise
    )
    # Add Camera Poses Initial values
    for f_id in window_frames:
        pose_key = symbol('c', f_id)
        R_init, t_init = camera_poses[f_id]
        pose_initial = lib.pnp_pose_to_gtsam_pose(R_init, t_init)
        initial_estimate.insert(pose_key, pose_initial)
        if f_id == start_frame:
            graph.add(gtsam.PriorFactorPose3(pose_key, pose_initial, gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*1e-6)))
            
    # Triangulate Landmarks & Populate graph structures
    for t_id in landmarks_in_window:
        point_key = symbol('q', t_id)
        track_frames = [f for f in db.frames(t_id) if f in window_frames]
        if not track_frames: continue
            
        init_f_id = track_frames[0]
        obs_init = db.observation(init_f_id, t_id)
        R_w2c, t_w2c = camera_poses[init_f_id]

        P_L = K_mat @ np.hstack([
            R_w2c,
            t_w2c.reshape(3, 1)
        ])

        P_R = K_mat @ np.hstack([
            R_w2c,
            (t_w2c.reshape(3, 1) + R_w2c @ t_stereo.reshape(3, 1))
        ])
        
        X_init = lib.triangulate_point_linear(np.array([obs_init.x_left, obs_init.y]), np.array([obs_init.x_right, obs_init.y]), P_L, P_R)
        initial_estimate.insert(point_key, gtsam.Point3(X_init[0], X_init[1], X_init[2]))
        
        for f_id in track_frames:
            obs = db.observation(f_id, t_id)
            graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(obs.x_left, obs.x_right, obs.y), 
                                                 measurement_noise, symbol('c', f_id), point_key, K_gtsam))
            
    # Evaluation Pass BEFORE Optimization execution
    num_factors = graph.size()
    initial_total_error = graph.error(initial_estimate)
    print(f"Total Factors in Graph: {num_factors}")
    print(f"Total Factor Graph Error BEFORE Optimization: {initial_total_error:.4f}")
    print(f"Average Factor Error BEFORE Optimization: {initial_total_error / num_factors:.4f}")
    
    # Locate Worst Performing Structural Factor
    max_err, worst_factor = -1, None
    for idx in range(num_factors):
        f = graph.at(idx)
        if isinstance(f, gtsam.GenericStereoFactor3D) and f.error(initial_estimate) > max_err:
            max_err = f.error(initial_estimate)
            worst_factor = f
                
    worst_pose_key, worst_point_key = worst_factor.keys()[0], worst_factor.keys()[1]
    
    print(f"\nWorst Initial Factor Info:")
    print(f"  Frame Key: {worst_pose_key} | Landmark Key: {worst_point_key}")
    print(f"  Initial Factor Error: {max_err:.4f}")
    
    proj_init = gtsam.StereoCamera(initial_estimate.atPose3(worst_pose_key), K_gtsam).project(initial_estimate.atPoint3(worst_point_key))
    meas_worst = worst_factor.measured()
    
    print(f"  Distance from measurements (Pixels) BEFORE:")
    print(f"    Left Camera Dist: {np.sqrt((proj_init.uL() - meas_worst.uL())**2 + (proj_init.v() - meas_worst.v())**2):.2f}")
    print(f"    Right Camera Dist: {np.sqrt((proj_init.uR() - meas_worst.uR())**2 + (proj_init.v() - meas_worst.v())**2):.2f}")
    
    # Optimization Sequence Execution
    print("\nRunning Levenberg-Marquardt Optimization...")
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate).optimize()
    print("Optimization Complete!")
    
    final_total_error = graph.error(result)
    print(f"\nTotal Factor Graph Error AFTER Optimization: {final_total_error:.4f}")
    print(f"Average Factor Error AFTER Optimization: {final_total_error / num_factors:.4f}")
    
    # Collect post-optimization projection validation anchors
    proj_final = gtsam.StereoCamera(result.atPose3(worst_pose_key), K_gtsam).project(result.atPoint3(worst_point_key))
    worst_frame_id = gtsam.symbolIndex(worst_pose_key)
    worst_obs = db.observation(worst_frame_id, gtsam.symbolIndex(worst_point_key))
    
    print(f"\nWorst Factor Error AFTER Optimization: {worst_factor.error(result):.4f}")
    print(f"  Distance from measurements (Pixels) AFTER:")
    print(f"    Left Camera Dist: {np.sqrt((proj_final.uL() - meas_worst.uL())**2 + (proj_final.v() - meas_worst.v())**2):.2f}")
    print(f"    Right Camera Dist: {np.sqrt((proj_final.uR() - meas_worst.uR())**2 + (proj_final.v() - meas_worst.v())**2):.2f}")
    
    # Trigger modular image visualizations
    try: 
        lib.draw_projection_validation_frames(worst_frame_id, worst_obs, proj_init, proj_final)
    except Exception as e: 
        print(f"[Warning] Imaging validation dropped: {e}")

    cam_positions, lm_filtered = lib.extract_optimized_geometry(result, window_frames, landmarks_in_window)

    # 3D Path rendering setup
    fig3d = plt.figure(figsize=(10, 7)); axes3d = fig3d.add_subplot(111, projection='3d')
    axis_length = max(np.ptp(lm_filtered, axis=0).max() * 0.05, 0.5) if len(lm_filtered) > 0 else 0.5
    for f_id in window_frames: 
        gtsam_plot.plot_pose3_on_axes(axes3d, result.atPose3(symbol('c', f_id)), axis_length=axis_length)
    if len(lm_filtered) > 0: 
        axes3d.scatter(lm_filtered[:, 0], lm_filtered[:, 1], lm_filtered[:, 2], s=1, c='gray', alpha=0.3)
    output_path = os.path.join(output_dir, "task_5_3_bundle1_3D.png")
    plt.savefig(output_path, dpi=150); plt.close()

    # 2D Top-down plot rendering setup
    fig2d, ax2d = plt.subplots(figsize=(10, 7))
    if len(lm_filtered) > 0: 
        ax2d.scatter(lm_filtered[:, 0], lm_filtered[:, 2], s=1, c='gray', alpha=0.4, label='Landmarks')
    ax2d.plot(cam_positions[:, 0], cam_positions[:, 2], 'b-o', markersize=4, label='Camera Trajectory')
    for f_idx, pos in [(start_frame, cam_positions[0]), (end_frame, cam_positions[-1])]: 
        ax2d.scatter(pos[0], pos[2], s=80, c='red', zorder=5)
    ax2d.set_aspect('equal'); ax2d.grid(True, alpha=0.3); ax2d.legend()
    output_path = os.path.join(output_dir, "task_5_3_bundle1_2D.png")
    plt.savefig(output_path, dpi=150); plt.close()
    print("[Success] Fully optimized visual validation assets saved.")


def plot_q5_4_results(keyframes, global_keyframe_poses, all_points_global):
    gt_poses = lib.read_ground_truth_poses()

    estimated_positions = []
    gt_positions = []

    valid_keyframes = [
        kf for kf in keyframes
        if kf in global_keyframe_poses
    ]

    for kf in valid_keyframes:
        est_pose = global_keyframe_poses[kf]
        estimated_positions.append(
            lib.pose_translation_np(est_pose)
        )

        R_gt, t_gt = gt_poses[kf]
        gt_positions.append(
            lib.camera_center(R_gt, t_gt)
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
            alpha=0.2,
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

    output_path = os.path.join(
        output_dir,
        "task_5_4_keyframes_vs_gt.png"
    )

    plt.savefig(output_path, dpi=200)


def plot_keyframe_localization_error(keyframes, global_keyframe_poses):
    gt_poses = lib.read_ground_truth_poses()

    errors = []
    valid_keyframes = []

    print("\n[Debug localization comparison]")

    for kf in keyframes[:10]:
        if kf not in global_keyframe_poses:
            continue

        est_pos = lib.pose_translation_np(global_keyframe_poses[kf])

        R_gt, t_gt = gt_poses[kf]
        gt_pos_direct = t_gt.flatten()
        gt_pos_center = lib.camera_center(R_gt, t_gt)

        # print(f"KF {kf}")
        # print(f"  est_pos:       {est_pos}")
        # print(f"  gt_pos_direct: {gt_pos_direct}")
        # print(f"  gt_pos_center: {gt_pos_center}")
        # print(f"  err_direct:    {np.linalg.norm(est_pos - gt_pos_direct):.3f}")
        # print(f"  err_center:    {np.linalg.norm(est_pos - gt_pos_center):.3f}")

    for kf in keyframes:

        if kf not in global_keyframe_poses:
            continue

        est_pos = lib.pose_translation_np(
            global_keyframe_poses[kf]
        )

        R_gt, t_gt = gt_poses[kf]
        gt_pos = lib.camera_center(R_gt, t_gt)
        err = np.linalg.norm(est_pos - gt_pos)

        errors.append(err)
        valid_keyframes.append(kf)

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

def q5_4(db):
    print("\n================================================================================")
    print("SECTION 5.4: FULL SLIDING BUNDLE ADJUSTMENT")
    print("================================================================================")

    keyframes = lib.choose_keyframes(
        db,
        distance_threshold=2.5,
        min_gap=5,
        max_gap=20
    )

    bundle_windows = [
        (keyframes[i], keyframes[i + 1])
        for i in range(len(keyframes) - 1)
    ]

    print(f"Number of keyframes: {len(keyframes)}")
    print(f"Number of bundle windows: {len(bundle_windows)}")

    # Global optimized keyframe poses in frame0 coordinates
    global_keyframe_poses = {
        keyframes[0]: gtsam.Pose3()
    }

    all_points_global = []
    last_bundle_result = None
    failed_bundles = 0
    for start_frame, end_frame in bundle_windows:
        if start_frame not in global_keyframe_poses:
            continue
        try:
            bundle_result = lib.solve_bundle_window(
                db,
                start_frame,
                end_frame
            )
        except Exception as e:
            print(type(e))
            print(repr(e))
            print(f"[Warning] Bundle {start_frame}->{end_frame} failed: {e}")
            failed_bundles += 1

            if start_frame in global_keyframe_poses:
                global_keyframe_poses[end_frame] = global_keyframe_poses[start_frame]
            continue

        last_bundle_result = bundle_result

        start_global_pose = global_keyframe_poses[start_frame]
        relative_pose = bundle_result["relative_pose"]

        rel_t = lib.pose_translation_np(relative_pose)
        start_t = lib.pose_translation_np(start_global_pose)
        end_global_pose = start_global_pose.compose(relative_pose)
        end_t = lib.pose_translation_np(end_global_pose)



        global_keyframe_poses[end_frame] = end_global_pose

        # Transform local landmarks to global frame0 coordinates
        points_local = bundle_result["optimized_points_local"]

        if len(points_local) > 0:
            for p_local in points_local:
                p_global = start_global_pose.transformFrom(
                    gtsam.Point3(*p_local)
                )
                all_points_global.append(
                    np.array(p_global).reshape(3)
                )

    print(f"Failed bundles: {failed_bundles}/{len(bundle_windows)}")
    if last_bundle_result is None:
        raise RuntimeError("No bundle window was successfully optimized.")

    # -------------------------
    # Last bundle diagnostics
    # -------------------------


    # -------------------------
    # Plotting
    # -------------------------
    plot_q5_4_results(
        keyframes,
        global_keyframe_poses,
        all_points_global
    )

    plot_keyframe_localization_error(
        keyframes,
        global_keyframe_poses
    )

    return global_keyframe_poses, all_points_global

if __name__ == '__main__':
    import pickle

    with open("tracking_db.pkl", "rb") as f:
        db = pickle.load(f)

    q5_4(db)