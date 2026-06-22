# ex5.py
from gtsam import symbol
import os
import van_utils as lib
from gtsam.utils import plot as gtsam_plot

output_dir = "./outputs"
DB_PKL_PATH = "tracking_db_ex5.pkl"
os.makedirs(output_dir, exist_ok=True)  # Creates the folder if it doesn't exist
import os
import random
import matplotlib.pyplot as plt
import gtsam
import numpy as np
import pickle



def q5_1(db):
    print("\n--- Task 5.1: Single Track Error Analysis with GTSAM ---")
    
    exact_tracks = [t_id for t_id in db.track_to_frames if len(db.frames(t_id)) == 15]
    if not exact_tracks: 
        raise RuntimeError("No track found with length == 15 frames.")
        
    track_id = random.choice(exact_tracks)
    frames = db.frames(track_id)
    
    K_gtsam = lib.init_gtsam_stereo_calibration()
    gt_poses = lib.read_ground_truth_poses()
    
    poses_dict = {f_id: lib.get_gtsam_camera_pose(gt_poses, frame_id=f_id) for f_id in frames}
    
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
        
        factor_errors.append(lib.compute_single_factor_error(poses_dict[f_id], K_gtsam, X_world_gtsam, obs))

    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, left_reprojection_errors, label="Left camera reprojection error", color="#1f77b4")
    plt.plot(frame_indices, right_reprojection_errors, label="Right camera reprojection error", color="#ff7f0e")
    plt.title("Right camera reprojection error") # כותרת הגרף כפי שמופיעה בדוגמה
    plt.xlabel("Frame index")
    plt.ylabel("Reprojection error (L2 norm) - pixels")
    plt.grid(True)
    plt.legend(loc="upper right")
    reproj_path = os.path.join(output_dir, "reprojection_error_graph.png")
    plt.savefig(reproj_path, dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, factor_errors, color="red", marker='s')
    plt.title("Factor error graph")
    plt.xlabel("Frame index")
    plt.ylabel("Factor Error")
    plt.grid(True)
    factor_path = os.path.join(output_dir, "factor_error_graph.png")
    plt.savefig(factor_path, dpi=300)
    plt.close()


def q5_3(db):
    print("\n================================================================================")
    print("SECTION 5.3: FIRST BUNDLE ADJUSTMENT WINDOW (10 FRAMES)")
    print("================================================================================")
    
    K_gtsam = lib.init_gtsam_stereo_calibration()
    K_mat, _, m_right0 = lib.read_cameras()
    t_stereo = np.linalg.inv(K_mat) @ m_right0[:, 3]
    camera_poses = db.camera_poses

    TARGET_FRAMES_COUNT = 10
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
    
    R_start, t_start = camera_poses[start_frame]

    # Add Camera Poses Initial values (Transformed to local window frame)
    for f_id in window_frames:
        pose_key = symbol('c', f_id)
        R_f, t_f = camera_poses[f_id]
        
        pose_local = lib.w2c_to_local_gtsam_pose(R_start, t_start, R_f, t_f)
        initial_estimate.insert(pose_key, pose_local)
        
        if f_id == start_frame:
            graph.add(gtsam.PriorFactorPose3(pose_key, pose_local, gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*1e-6)))
            
    # Triangulate Landmarks & Populate graph structures
    landmarks_added = set()
    for t_id in landmarks_in_window:
        point_key = symbol('q', t_id)
        track_frames = [f for f in db.frames(t_id) if f in window_frames]
        if len(track_frames) < 3: continue

        init_f_id = track_frames[0]
        obs_init = db.observation(init_f_id, t_id)
        if not lib.valid_stereo_obs(obs_init, min_disp=1.0): continue

        pose_init_local = initial_estimate.atPose3(symbol('c', init_f_id))
        
        try:
            camera_stereo_local = gtsam.StereoCamera(pose_init_local, K_gtsam)
            stereo_point = gtsam.StereoPoint2(obs_init.x_left, obs_init.x_right, obs_init.y)
            X_local = camera_stereo_local.backproject(stereo_point)
        except RuntimeError:
            continue

        X_cam_coords = pose_init_local.inverse().transformTo(X_local)
        if X_cam_coords[2] < 2.0 or X_cam_coords[2] > 80.0: 
            continue

        temporary_factors = []
        for f_id in track_frames:
            obs = db.observation(f_id, t_id)
            if lib.valid_stereo_obs(obs, min_disp=1.0):
                factor = gtsam.GenericStereoFactor3D(
                    gtsam.StereoPoint2(obs.x_left, obs.x_right, obs.y), 
                    measurement_noise, symbol('c', f_id), point_key, K_gtsam
                )
                temporary_factors.append(factor)

        if len(temporary_factors) >= 2:
            initial_estimate.insert(point_key, X_local)
            landmarks_added.add(t_id)  
            for factor in temporary_factors:
                graph.add(factor)

    # -------------------------------------------------------------------------
    # EVALUATION PASS BEFORE OPTIMIZATION
    # -------------------------------------------------------------------------
    num_factors = graph.size()
    num_projection_factors = num_factors - 1  # subtract the 1 prior factor on frame 0

    initial_total_error = graph.error(initial_estimate)
    initial_avg_error = initial_total_error / num_factors

    print("\n--------------------------------------------------------------------------------")
    print("FACTOR GRAPH STATISTICS")
    print("--------------------------------------------------------------------------------")
    print(f"Total Factors in Graph (including prior):        {num_factors}")
    print(f"  - Prior factors (anchoring start pose):        1")
    print(f"  - Stereo projection factors:                   {num_projection_factors}")
    print(f"  - Landmarks inserted into graph:               {len(landmarks_added)}")
    print(f"\nTotal Factor Graph Error   BEFORE Optimization:  {initial_total_error:.4f}")
    print(f"Average Factor Error       BEFORE Optimization:  {initial_avg_error:.4f}")

    # -------------------------------------------------------------------------
    # FIND THE FACTOR WITH THE LARGEST INITIAL ERROR
    # -------------------------------------------------------------------------
    max_init_error = -1.0
    worst_factor = None
    worst_factor_idx = -1
    worst_c_key = None
    worst_q_key = None

    for i in range(num_factors):
        factor = graph.at(i)
        keys = factor.keys()
        if len(keys) != 2:
            continue
        key_a, key_b = keys[0], keys[1]
        sym_a = gtsam.Symbol(key_a)
        sym_b = gtsam.Symbol(key_b)
        
        if sym_a.chr() == ord('c') and sym_b.chr() == ord('q'):
            c_key_cand, q_key_cand = key_a, key_b
        elif sym_a.chr() == ord('q') and sym_b.chr() == ord('c'):
            c_key_cand, q_key_cand = key_b, key_a
        else:
            continue
            
        err = factor.error(initial_estimate)
        if err > max_init_error:
            max_init_error = err
            worst_factor = factor
            worst_factor_idx = i
            worst_c_key = c_key_cand
            worst_q_key = q_key_cand

    print("\n--------------------------------------------------------------------------------")
    print("LARGEST INITIAL ERROR FACTOR — INITIAL VALUES")
    print("--------------------------------------------------------------------------------")
    if worst_factor is not None:
        c_sym   = gtsam.Symbol(worst_c_key)
        q_sym   = gtsam.Symbol(worst_q_key)
        c_index = c_sym.index()
        q_index = q_sym.index()

        print(f"Factor Index in Graph:           {worst_factor_idx}")
        print(f"Camera Frame Key  (c):           symbol('c', {c_index})  [GTSAM key = {worst_c_key}]")
        print(f"Landmark Point Key (q):          symbol('q', {q_index})  [GTSAM key = {worst_q_key}]")
        print(f"Initial Factor Error:            {max_init_error:.6f}")

        measured        = worst_factor.measured()
        stereo_cam_init = gtsam.StereoCamera(initial_estimate.atPose3(worst_c_key), K_gtsam)
        projected_init  = stereo_cam_init.project(initial_estimate.atPoint3(worst_q_key))

        print(f"\nMeasurement  — Left X: {measured.uL():8.2f}  |  Right X: {measured.uR():8.2f}  |  Y: {measured.v():8.2f}")
        print(f"Projection   — Left X: {projected_init.uL():8.2f}  |  Right X: {projected_init.uR():8.2f}  |  Y: {projected_init.v():8.2f}")

        dist_left_init  = np.sqrt((projected_init.uL() - measured.uL())**2 + (projected_init.v() - measured.v())**2)
        dist_right_init = np.sqrt((projected_init.uR() - measured.uR())**2 + (projected_init.v() - measured.v())**2)
        print(f"\nL2 Pixel Distance from Measurement (Left  camera): {dist_left_init:.4f} px")
        print(f"L2 Pixel Distance from Measurement (Right camera): {dist_right_init:.4f} px")
    else:
        print("[Warning] No stereo projection factor found.")

    # -------------------------------------------------------------------------
    # OPTIMIZATION
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("RUNNING LEVENBERG-MARQUARDT OPTIMIZATION...")
    print("--------------------------------------------------------------------------------")
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate).optimize()
    print("Optimization Complete!")

    # -------------------------------------------------------------------------
    # EVALUATION PASS AFTER OPTIMIZATION
    # -------------------------------------------------------------------------
    final_total_error = graph.error(result)
    final_avg_error = final_total_error / num_factors

    print(f"\nTotal Factor Graph Error   AFTER  Optimization:  {final_total_error:.4f}")
    print(f"Average Factor Error       AFTER  Optimization:  {final_avg_error:.4f}")
    print(f"\nError Reduction:  {initial_total_error:.4f}  -->  {final_total_error:.4f}  "
          f"(reduced by {100.0 * (1.0 - final_total_error / initial_total_error):.1f}%)")

    # -------------------------------------------------------------------------
    # SAME WORST FACTOR — OPTIMIZED VALUES
    # -------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("LARGEST INITIAL ERROR FACTOR — OPTIMIZED VALUES")
    print("--------------------------------------------------------------------------------")
    if worst_factor is not None:
        final_factor_error = worst_factor.error(result)
        stereo_cam_opt = gtsam.StereoCamera(result.atPose3(worst_c_key), K_gtsam)
        projected_opt  = stereo_cam_opt.project(result.atPoint3(worst_q_key))

        dist_left_opt  = np.sqrt((projected_opt.uL() - measured.uL())**2 + (projected_opt.v() - measured.v())**2)
        dist_right_opt = np.sqrt((projected_opt.uR() - measured.uR())**2 + (projected_opt.v() - measured.v())**2)

        print(f"Same Factor:  symbol('c', {c_index})  x  symbol('q', {q_index})")
        print(f"Optimized Factor Error:          {final_factor_error:.6f}  (was {max_init_error:.6f})")
        print(f"\nMeasurement  — Left X: {measured.uL():8.2f}  |  Right X: {measured.uR():8.2f}  |  Y: {measured.v():8.2f}")
        print(f"Projection   — Left X: {projected_opt.uL():8.2f}  |  Right X: {projected_opt.uR():8.2f}  |  Y: {projected_opt.v():8.2f}")
        print(f"\nL2 Pixel Distance from Measurement (Left  camera): {dist_left_opt:.4f} px  (was {dist_left_init:.4f} px)")
        print(f"L2 Pixel Distance from Measurement (Right camera): {dist_right_opt:.4f} px  (was {dist_right_init:.4f} px)")
        obs_object = db.observation(c_index, q_index) 
        lib.draw_projection_validation_frames(c_index, obs_object, projected_init, projected_opt)

    # -------------------------------------------------------------------------
    # EXTRACT GEOMETRY FOR PLOTTING
    # -------------------------------------------------------------------------
    cam_positions = np.array([result.atPose3(symbol('c', f_id)).translation() for f_id in window_frames])
    initial_cam_positions = np.array([initial_estimate.atPose3(symbol('c', f_id)).translation() for f_id in window_frames])

    lm_pts = []
    for t_id in landmarks_added:
        pt = result.atPoint3(symbol('q', t_id))
        lm_pts.append([pt[0], pt[1], pt[2]])
    lm_filtered = np.array(lm_pts) if lm_pts else np.empty((0, 3))

    # -------------------------------------------------------------------------
    # PLOT 1: 3D Optimized Trajectory with Camera Axes
    # -------------------------------------------------------------------------
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')
    
    xs_plot = cam_positions[:, 2]  
    ys_plot = cam_positions[:, 0]  
    zs_plot = -cam_positions[:, 1] 

    ax3d.plot(xs_plot, ys_plot, zs_plot, 'k--', linewidth=1.5, zorder=1)
    
    axis_length = 0.25  
    for i in range(len(window_frames)):
        pose = result.atPose3(symbol('c', window_frames[i]))
        R = pose.rotation().matrix()
        cx, cy, cz = xs_plot[i], ys_plot[i], zs_plot[i]
        
        ax3d.scatter(cx, cy, cz, color='black', s=15, zorder=2)
        
        ax_right   = R[:, 0]
        ax_down    = R[:, 1]
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

    # -------------------------------------------------------------------------
    # PLOT 2: GTSAM Factor Graph State with Marginal Covariances
    # -------------------------------------------------------------------------
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
    plt.savefig(os.path.join(output_dir, "task_5_3_marginal_covariances.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # PLOT 3: 2D Bird's-Eye View — All Cameras + All Landmarks
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # PLOT 4: 2D Bird's-Eye View — Zoomed Trajectory Comparison
    # -------------------------------------------------------------------------
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

    print("\n[Success] All plots and statistics for Section 5.3 successfully generated.")



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

    global_keyframe_poses = {
        keyframes[0]: gtsam.Pose3()
    }

    global_landmarks_dict = {}
    last_bundle_result = None
    failed_bundles = 0

    for start_frame, end_frame in bundle_windows:
        print(f"Solving bundle {start_frame}->{end_frame}", flush=True)

        try:
            bundle_result = lib.solve_bundle_window(
                db,
                start_frame,
                end_frame
            )
        except Exception as e:
            failed_bundles += 1
            print(type(e))
            print(repr(e))
            print(f"[Warning] Bundle {start_frame}->{end_frame} failed: {e}")

            if start_frame in global_keyframe_poses:
                global_keyframe_poses[end_frame] = global_keyframe_poses[start_frame]
            continue

        last_bundle_result = bundle_result

        start_global_pose = global_keyframe_poses[start_frame]
        relative_pose = bundle_result["relative_pose"]

        end_global_pose = start_global_pose.compose(relative_pose)
        global_keyframe_poses[end_frame] = end_global_pose

        optimized_landmark_ids = bundle_result["optimized_landmark_ids"]
        points_local = bundle_result["optimized_points_local"]

        for track_id, p_local in zip(optimized_landmark_ids, points_local):
            p_global = start_global_pose.transformFrom(
                gtsam.Point3(*p_local)
            )
            global_landmarks_dict[track_id] = np.array(p_global).reshape(3)

    print(f"Failed bundles: {failed_bundles}/{len(bundle_windows)}")

    if last_bundle_result is None:
        raise RuntimeError("No bundle window was successfully optimized.")

    all_points_global = list(global_landmarks_dict.values())

    last_result = last_bundle_result["result"]
    last_start = last_bundle_result["start_frame"]
    last_start_pose = last_result.atPose3(symbol("c", last_start))

    print("\n--- Last Bundle Diagnostics ---")
    print(f"Last bundle start frame: {last_start}")
    print(
        "Position of first frame after optimization:",
        lib.pose_translation_np(last_start_pose)
    )

    anchor_error = last_bundle_result["anchor_factor"].error(last_result)
    print(f"Anchoring factor final error: {anchor_error:.12f}")
    print(
        "The anchoring error is approximately zero because the first camera "
        "in each bundle is fixed to the local origin using a strong prior."
    )

    lib.debug_coordinate_system_alignment(
        keyframes,
        global_keyframe_poses
    )

    lib.debug_scale_drift(
        keyframes,
        global_keyframe_poses
    )

    lib.plot_q5_4_results(
        keyframes,
        global_keyframe_poses,
        all_points_global,
        output_dir=output_dir
    )

    lib.plot_keyframe_localization_error(
        keyframes,
        global_keyframe_poses,
        output_dir=output_dir
    )

    return global_keyframe_poses, all_points_global

    
if __name__ == "__main__":
    db = lib.load_or_build_db(
        force_rebuild=False,
        num_frames=lib.get_num_frames(),
    )

    q5_4(db)