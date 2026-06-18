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
    print("SECTION 5.3: FIRST BUNDLE ADJUSTMENT WINDOW (8 FRAMES)")
    print("================================================================================")
    
    K_gtsam = lib.init_gtsam_stereo_calibration()
    K_mat, _, m_right0 = lib.read_cameras()
    t_stereo = np.linalg.inv(K_mat) @ m_right0[:, 3]
    camera_poses = db.camera_poses

    # Target exactly 8 frames for the local window optimization
    TARGET_FRAMES_COUNT = 8
    start_frame = 0
    end_frame = start_frame + TARGET_FRAMES_COUNT - 1
    window_frames = list(range(start_frame, end_frame + 1))
    
    print(f"Bundle Window spans from Frame {start_frame} to Frame {end_frame} (Total {len(window_frames)} frames)")
    
    landmarks_in_window = set()
    for f_id in window_frames: 
        landmarks_in_window.update(db.tracks(f_id))
    print(f"Number of landmarks participating in this bundle: {len(landmarks_in_window)}")
        
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()
    base_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)

    measurement_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(2.0),
        base_noise
    )
    
    # Track the start pose matrix to convert everything into local coordinates
    R_start, t_start = camera_poses[start_frame]

    # Add Camera Poses Initial values (Transformed to local window frame)
    for f_id in window_frames:
        pose_key = symbol('c', f_id)
        R_f, t_f = camera_poses[f_id]
        
        # Map extrinsics into local GTSAM coordinates
        pose_local = lib.w2c_to_local_gtsam_pose(R_start, t_start, R_f, t_f)
        initial_estimate.insert(pose_key, pose_local)
        
        if f_id == start_frame:
            graph.add(gtsam.PriorFactorPose3(pose_key, pose_local, gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*1e-6)))
            
    # Triangulate Landmarks & Populate graph structures
    for t_id in landmarks_in_window:
        point_key = symbol('q', t_id)
        track_frames = [f for f in db.frames(t_id) if f in window_frames]
        if not track_frames: continue
            
        # Require at least 3 frames of support for this landmark inside the
        # window. A landmark with only 1-2 stereo observations is poorly
        # constrained and very sensitive to a single bad triangulation,
        # which can destabilize the whole bundle (see PDF: "take care to
        # avoid creating an ill-formed problem").
        if len(track_frames) < 3: continue

        init_f_id = track_frames[0]
        obs_init = db.observation(init_f_id, t_id)
        R_w2c, t_w2c = camera_poses[init_f_id]

        # Skip points with degenerate/near-zero disparity: these blow up to
        # huge or NaN/garbage depths under linear triangulation and corrupt
        # the initial landmark estimate (and therefore the whole graph).
        if not lib.valid_stereo_obs(obs_init, min_disp=1.0): continue

        P_L = K_mat @ np.hstack([R_w2c, t_w2c.reshape(3, 1)])
        # t_stereo is already expressed in the LEFT camera's coordinate frame
        # (it's the right camera's extrinsic translation relative to the left
        # camera, i.e. K^-1 @ m_right0[:, 3]). The world-to-right-camera
        # extrinsic translation is therefore a simple vector sum t_w2c + t_stereo
        # (same convention used in evaluate_supporters / project_stereo_point in
        # exercises 3/4). Do NOT rotate t_stereo by R_w2c here - that incorrectly
        # treats t_stereo as if it were expressed in world coordinates, and
        # introduces a frame-dependent (because R_w2c differs per frame) error
        # in the stereo baseline used for triangulation, corrupting the 3D
        # landmark positions and poisoning the whole bundle initialization.
        P_R = K_mat @ np.hstack([R_w2c, (t_w2c.reshape(3, 1) + t_stereo.reshape(3, 1))])
        
        X_global = lib.triangulate_point_linear(np.array([obs_init.x_left, obs_init.y]), np.array([obs_init.x_right, obs_init.y]), P_L, P_R)
        
        if not np.all(np.isfinite(X_global)): continue
 
        # Reject points triangulated behind the camera or absurdly far away
        # (linear triangulation occasionally produces these from noisy
        # matches even with valid disparity). Same bound used in 5.4's
        # add_valid_tracks_to_graph for consistency.
        if X_global[2] < 5 or X_global[2] > 60: continue
            
        pose_init_frame = initial_estimate.atPose3(symbol('c', init_f_id))
        X_init_local = pose_init_frame.transformFrom(gtsam.Point3(float(X_global[0]), float(X_global[1]), float(X_global[2])))
        
        initial_estimate.insert(point_key, X_init_local)
        
        for f_id in track_frames:
            obs = db.observation(f_id, t_id)
            graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(obs.x_left, obs.x_right, obs.y), 
                                                 measurement_noise, symbol('c', f_id), point_key, K_gtsam))
            
    # Evaluation Pass BEFORE Optimization execution
    num_factors = graph.size()
    initial_total_error = graph.error(initial_estimate)
    print(f"Total Factors in Graph: {num_factors}")
    print(f"Total Factor Graph Error BEFORE Optimization: {initial_total_error:.4f}")
    
    # Optimization Sequence Execution
    print("\nRunning Levenberg-Marquardt Optimization...")
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate).optimize()
    print("Optimization Complete!")
    
    # Extract clean local geometry configurations
    cam_positions, lm_filtered = lib.extract_optimized_geometry(result, window_frames, landmarks_in_window)
    initial_cam_positions = np.array([initial_estimate.atPose3(symbol('c', f_id)).translation() for f_id in window_frames])

    # -------------------------------------------------------------
    # PLOTTING PASS
    # -------------------------------------------------------------
    
    # 1. Plot 3D Optimized Trajectory (8 Poses Connected with Standard Axes)
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    xs_plot = cam_positions[:, 2]  
    ys_plot = cam_positions[:, 0]  
    zs_plot = -cam_positions[:, 1] 

    # Draw connection baseline sequence for all 8 frames
    ax3d.plot(xs_plot, ys_plot, zs_plot, 'k--', linewidth=1.5, zorder=1)
    
    axis_length = 0.25  
    for i in range(len(window_frames)):
        pose = result.atPose3(symbol('c', window_frames[i]))
        R = pose.rotation().matrix()
        cx, cy, cz = xs_plot[i], ys_plot[i], zs_plot[i]
        
        ax3d.scatter(cx, cy, cz, color='black', s=15, zorder=2)
        
        ax_right = R[:, 0]
        ax_down = R[:, 1]
        ax_forward = R[:, 2]
        
        # Red line: Right (X)
        ax3d.plot([cx, cx + axis_length * ax_right[2]],
                  [cy, cy + axis_length * ax_right[0]],
                  [cz, cz - axis_length * ax_right[1]], color='r', linewidth=1.5)
        
        # Green line: Up (-Y)
        ax3d.plot([cx, cx - axis_length * ax_down[2]],
                  [cy, cy - axis_length * ax_down[0]],
                  [cz, cz + axis_length * ax_down[1]], color='g', linewidth=1.5)
                  
        # Blue line: Forward (Z)
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
    plt.savefig(os.path.join(output_dir, "task_5_3_bundle1_3D.png"), dpi=200, bbox_inches='tight'); plt.close()

    # 2. GTSAM Factor Graph State with Marginal Covariances (8-Frame State)
    fig_cov = plt.figure(figsize=(8, 6))
    ax_cov = fig_cov.add_subplot(111, projection='3d')
    ax_cov.set_title("Plot Trajectory\nGTSAM Factor Graph State with Marginal Covariances\n", fontsize=11, fontweight='bold')
    
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
    plt.savefig(os.path.join(output_dir, "task_5_3_marginal_covariances.png"), dpi=200, bbox_inches='tight'); plt.close()

    # 3. 2D Bird's-Eye View 
    fig2d_full, ax2d_full = plt.subplots(figsize=(7, 7))
    if len(lm_filtered) > 0:
        ax2d_full.scatter(lm_filtered[:, 0], lm_filtered[:, 2], s=1, c='gray', alpha=0.4, label='Landmarks')
    ax2d_full.plot(initial_cam_positions[:, 0], initial_cam_positions[:, 2], 'b-+', alpha=0.6, label='Initial (PnP)')
    ax2d_full.plot(cam_positions[:, 0], cam_positions[:, 2], 'r-o', markersize=4, label='Optimized (BA)')
    ax2d_full.set_title("Local Window Bundle Adjustment: 2D Bird's-Eye View Trajectory")
    ax2d_full.set_xlabel("X Coordinate (East) [m]")
    ax2d_full.set_ylabel("Z Coordinate (North) [m]")
    ax2d_full.grid(True, alpha=0.3)
    ax2d_full.legend()
    plt.savefig(os.path.join(output_dir, "task_5_3_bundle1_2D_full.png"), dpi=150); plt.close()

    # 4. 2D Bird's-Eye View Zoomed Trajectory Focus 
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
    plt.savefig(os.path.join(output_dir, "task_5_3_bundle1_2D_zoomed.png"), dpi=150); plt.close()

    print("[Success] Visual verification assets for exactly 8 frames successfully updated.")

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

    with open("code/tracking_db.pkl", "rb") as f:
        db = pickle.load(f)

    q5_3(db)