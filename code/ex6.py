# ex6.py
import os
import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam import symbol
from gtsam.utils import plot as gtsam_plot
import van_utils as lib

def symbol(char, index):
    # הגרסה שלך מצפה לתו עצמו כסטרינג, ולא ל-ord שלו
    return gtsam.symbol(char, index)

def _solve_bundle_with_prior_sigma(db, start_frame, end_frame, prior_sigma):
    """
    Re-solves a bundle window identically to lib.solve_bundle_window but
    allows overriding the prior-factor noise sigma on the anchor pose.
    Returns (graph, result, window_frames).
    """
    import random
    K_gtsam   = lib.init_gtsam_stereo_calibration()
    K_mat, P_left0, P_right0 = lib.read_cameras()

    graph            = gtsam.NonlinearFactorGraph()
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

    # Populate landmarks (same logic as lib.solve_bundle_window)
    candidate_tracks = set()
    for f_id in window_frames:
        candidate_tracks.update(db.tracks(f_id))

    guaranteed_tracks = set()
    for f_id in window_frames:
        ts = list(db.tracks(f_id))
        guaranteed_tracks.update(random.sample(ts, min(15, len(ts))))

    remaining  = 150 - len(guaranteed_tracks)
    extras     = list(set(candidate_tracks) - guaranteed_tracks)
    final_tracks = list(guaranteed_tracks) + (
        random.sample(extras, min(remaining, len(extras))) if remaining > 0 else []
    )

    for track_id in final_tracks:
        track_frames = [f for f in db.frames(track_id) if f in window_frames]
        if len(track_frames) < 2:
            continue
        init_frame = track_frames[0]
        obs_init   = db.observation(init_frame, track_id)
        if not lib.valid_stereo_obs(obs_init, min_disp=1.0):
            continue
        X_cam = lib.triangulate_point_linear(
            np.array([obs_init.x_left, obs_init.y]),
            np.array([obs_init.x_right, obs_init.y]),
            P_left0, P_right0
        )
        if not np.all(np.isfinite(X_cam)) or X_cam[2] <= 2.0 or X_cam[2] > 120.0:
            continue

        init_pose = initial_estimate.atPose3(symbol("c", init_frame))
        X_local   = init_pose.transformFrom(
            gtsam.Point3(float(X_cam[0]), float(X_cam[1]), float(X_cam[2]))
        )
        point_key   = symbol("q", track_id)
        temp_factors = []
        for f_id in track_frames:
            obs = db.observation(f_id, track_id)
            if lib.valid_stereo_obs(obs, min_disp=1.0):
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


# =============================================================================
# 6.1: Extract relative pose constraint from Bundle Adjustment
# =============================================================================

def q6_1(db, output_dir="."):
    print("\n================================================================================")
    print("SECTION 6.1: RELATIVE POSE CONSTRAINTS FROM BUNDLE ADJUSTMENT")
    print("================================================================================")

    # ── Keyframes (same selection as ex5) ────────────────────────────────────
    keyframes = lib.choose_keyframes(
        db, distance_threshold=2.5, min_gap=20, max_gap=20
    )
    c0_idx = keyframes[0]
    ck_idx = keyframes[1]
    print(f"Total keyframes : {len(keyframes)}")
    print(f"First two       : c0={c0_idx}, c_k={ck_idx}")

    # ── PART A: Plot with three different prior-noise scales ─────────────────
    prior_configs = [
        ("Unit matrix (σ=1.0)",   1.0,   "task_6_1_cov_prior_1p0.png"),
        ("I × 0.05  (σ=0.05)",    0.05,  "task_6_1_cov_prior_0p05.png"),
        ("I × 1e-6  (σ=1e-6)",    1e-6,  "task_6_1_cov_prior_1e-6.png"),
    ]

    print(f"\n--- Plotting covariance sensitivity for bundle {c0_idx}→{ck_idx} ---")

    for label, sigma, fname in prior_configs:
        print(f"  Prior noise σ = {sigma}  ({label}) ...", end="", flush=True)

        graph_s, result_s, wf_s = _solve_bundle_with_prior_sigma(
            db, c0_idx, ck_idx, prior_sigma=sigma
        )

        marginals_s = gtsam.Marginals(graph_s, result_s)

        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_title(
            f"6.1 – Bundle {c0_idx}→{ck_idx}  |  prior noise: {label}\n"
            "Frame locations with marginal covariances",
            fontsize=10, fontweight='bold'
        )

        for f_id in wf_s:
            key  = symbol('c', f_id)
            pose = result_s.atPose3(key)
            try:
                cov = marginals_s.marginalCovariance(key)
                gtsam_plot.plot_pose3_on_axes(ax, pose, axis_length=0.3, P=cov)
            except Exception:
                gtsam_plot.plot_pose3_on_axes(ax, pose, axis_length=0.3)

        ax.set_xlabel("X axis"); ax.set_ylabel("Y axis"); ax.set_zlabel("Z axis")
        ax.view_init(elev=20, azim=-60)
        plt.tight_layout()
        path = os.path.join(output_dir, fname)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f" saved → {path}")

    # ── PART B: Relative pose + conditional covariance for first window ──────
    print(f"\n--- Relative pose & covariance for c0={c0_idx} → c_k={ck_idx} ---")

    bundle_result = lib.solve_bundle_window(db, c0_idx, ck_idx)
    graph   = bundle_result["graph"]
    result  = bundle_result["result"]

    marginals = gtsam.Marginals(graph, result)

    key_c0 = symbol('c', c0_idx)
    key_ck = symbol('c', ck_idx)

    # שליפה בטוחה של הבלוקים הייעודיים מתוך ה-Joint Marginal למניעת שגיאות מיון פנימי
    joint_keys = gtsam.KeyVector([key_c0, key_ck])
    joint_marginal = marginals.jointMarginalCovariance(joint_keys)
    
    Sigma_00 = joint_marginal.at(key_c0, key_c0)
    Sigma_0k = joint_marginal.at(key_c0, key_ck)
    Sigma_kk = joint_marginal.at(key_ck, key_ck)

    # חישוב הטרנספורמציה המותנית באמצעות Schur Complement
    Sigma_rel    = Sigma_kk - Sigma_0k.T @ np.linalg.inv(Sigma_00) @ Sigma_0k
    relative_pose = bundle_result["relative_pose"]

    print(f"\nRelative Pose between keyframes c{c0_idx} and c{ck_idx}:")
    print(relative_pose)
    print(f"\nConditional covariance P(c{ck_idx} | c{c0_idx}):")
    print(np.array2string(Sigma_rel, precision=8, suppress_small=True))

    # ── PART C: All consecutive keyframe pairs ───────────────────────────────
    print("\n--- Processing all consecutive keyframe pairs ---")

    bundle_windows = [
        (keyframes[i], keyframes[i + 1])
        for i in range(len(keyframes) - 1)
    ]

    relative_poses = {}
    relative_covs  = {}

    for sf, ef in bundle_windows:
        print(f"  Bundle {sf} -> {ef} ...", end="", flush=True)
        try:
            br = lib.solve_bundle_window(db, sf, ef)
            g, r = br["graph"], br["result"]

            margs   = gtsam.Marginals(g, r)
            k_sf   = symbol('c', sf)
            k_ef   = symbol('c', ef)
            
            jk     = gtsam.KeyVector([k_sf, k_ef])
            jc_marginal = margs.jointMarginalCovariance(jk)

            # שליפה בטוחה לפי מפתח גם בלולאה הכללית
            S00 = jc_marginal.at(k_sf, k_sf)
            S0k = jc_marginal.at(k_sf, k_ef)
            Skk = jc_marginal.at(k_ef, k_ef)

            rel_cov  = Skk - S0k.T @ np.linalg.inv(S00) @ S0k
            rel_pose = br["relative_pose"]

            relative_poses[(sf, ef)] = rel_pose
            relative_covs [(sf, ef)] = rel_cov
            print(" OK")

        except Exception as e:
            print(f" FAILED ({e})")

    print(f"\nSuccessfully computed {len(relative_poses)}/{len(bundle_windows)} "
          "relative pose constraints.")

    # Summary table
    print(f"\n{'Pair':<15} {'|t| [m]':>10}  {'det(Σ_rel)':>14}")
    print("-" * 43)
    for (sf, ef), rp in relative_poses.items():
        t_norm = float(np.linalg.norm(rp.translation()))
        det    = float(np.linalg.det(relative_covs[(sf, ef)]))
        print(f"({sf:4d},{ef:4d})   {t_norm:10.4f}   {det:14.6e}")

    return relative_poses, relative_covs

# =============================================================================
# ENTRY POINT
# =============================================================================


def q6_2(relative_poses, relative_covs, output_dir="."):
    print("\n================================================================================")
    print("SECTION 6.2: POSE GRAPH OPTIMIZATION")
    print("================================================================================")


    print("\n[Covariance Debug]")
    for edge, cov in list(relative_covs.items())[:5]:
        print("Edge:", edge)
        print("Diagonal:", np.diag(cov))
        print()

    relative_poses, relative_covs = lib.keep_connected_pose_graph_component(
        relative_poses,
        relative_covs
    )

    print(
        "First 20 edges:",
        sorted(relative_poses.keys())[:20]
    )

    print(
        "Last 20 edges:",
        sorted(relative_poses.keys())[-20:]
    )

    edges = sorted(relative_poses.keys())

    for (a, b), (c, d) in zip(edges[:-1], edges[1:]):
        if b != c:
            print(f"Gap after edge ({a}, {b}); next edge is ({c}, {d})")


    graph = lib.build_pose_graph(relative_poses, relative_covs)
    initial = lib.build_pose_graph_initial_estimate(relative_poses)

    missing = []
    for i in range(graph.size()):
        factor = graph.at(i)
        for key in factor.keys():
            if not initial.exists(key):
                missing.append(gtsam.Symbol(key).index())

    print("Missing keys:", sorted(set(missing)))

    print(f"Pose graph factors: {graph.size()}")
    print(f"Initial poses: {initial.size()}")

    initial_error = graph.error(initial)
    print(f"Pose graph error BEFORE optimization: {initial_error:.6f}")

    lib.plot_pose_graph_trajectory(
        initial,
        title="6.2: Initial Pose Graph Trajectory",
        output_path=os.path.join(output_dir, "task_6_2_initial_pose_graph.png")
    )

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    final_error = graph.error(result)
    print(f"Pose graph error AFTER optimization: {final_error:.6f}")

    lib.plot_pose_graph_trajectory(
        result,
        title="6.2: Optimized Pose Graph Trajectory",
        output_path=os.path.join(output_dir, "task_6_2_optimized_pose_graph.png")
    )

    marginals = gtsam.Marginals(graph, result)

    lib.plot_pose_graph_with_covariances(
        result,
        marginals,
        title="6.2: Optimized Pose Graph With Marginal Covariances",
        output_path=os.path.join(output_dir, "task_6_2_pose_graph_covariances.png")
    )

    return result


if __name__ == "__main__":
    db = lib.load_or_build_db(
        force_rebuild=False,
        num_frames=lib.get_num_frames()
    )

    relative_poses, relative_covs = q6_1(db)
    q6_2(relative_poses, relative_covs, output_dir=".")