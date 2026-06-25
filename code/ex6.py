# ex6.py
import os
import numpy as np
import matplotlib.pyplot as plt
import gtsam
from gtsam import symbol
from gtsam.utils import plot as gtsam_plot
import van_utils as lib




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

    # lib.debug_target_frame_geometry(
    #     db,
    #     start_frame=2840,
    #     end_frame=2860,
    #     target_frame=2860,
    #     max_tracks_per_window=150,
    #     output_dir=output_dir
    # )

    # ── PART A: Plot with three different prior-noise scales ─────────────────
    prior_configs = [
        ("Unit matrix (σ=1.0)",   1.0,   "task_6_1_cov_prior_1p0.png"),
        ("I × 0.05  (σ=0.05)",    0.05,  "task_6_1_cov_prior_0p05.png"),
        ("I × 1e-6  (σ=1e-6)",    1e-6,  "task_6_1_cov_prior_1e-6.png"),
    ]

    print(f"\n--- Plotting covariance sensitivity for bundle {c0_idx}→{ck_idx} ---")

    for label, sigma, fname in prior_configs:
        print(f"  Prior noise σ = {sigma}  ({label}) ...", end="", flush=True)

        graph_s, result_s, wf_s = lib.solve_bundle_with_prior_sigma(
            db, c0_idx, ck_idx, prior_sigma=sigma
        )

        marginal_s = gtsam.Marginals(graph_s, result_s)

        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection='3d')
        ax.set_title(
            f"6.1 – Bundle {c0_idx}→{ck_idx}  |  prior noise: {label}\n"
            "Frame locations with marginal covariances",
            fontsize=10, fontweight='bold'
        )

        for f_id in wf_s:
            key  = gtsam.symbol('c', f_id)
            pose = result_s.atPose3(key)
            try:
                cov = marginal_s.marginalCovariance(key)
                gtsam_plot.plot_pose3_on_axes(ax, pose, axis_length=0.3, P=cov)
            except Exception:
                gtsam_plot.plot_pose3_on_axes(ax, pose, axis_length=0.3)
        if sigma == 1:
            ax.set_xlabel("X axis"); ax.set_ylabel("Y axis"); ax.set_zlabel("Z axis")
            ax.view_init(elev=20, azim=-60)
            plt.tight_layout()
            path = os.path.join(output_dir, fname)
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f" saved → {path}")
        else:
            ax.set_xlabel("X axis"); ax.set_ylabel("Y axis"); ax.set_zlabel("Z axis")
            ax.set_xlim(-10.0, 10.0)
            ax.set_ylim(-10.0, 10.0)  
            ax.view_init(elev=20, azim=-60)
            plt.tight_layout()
            path = os.path.join(output_dir, fname)
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()

    # ── PART B: Relative pose + conditional covariance for first window ──────
    print(f"\n--- Relative pose & covariance for c0={c0_idx} → c_k={ck_idx} ---")

    bundle_result = lib.solve_bundle_window(db, c0_idx, ck_idx)
    graph   = bundle_result["graph"]
    result  = bundle_result["result"]

    marginals = gtsam.Marginals(graph, result)

    key_c0 = gtsam.symbol('c', c0_idx)
    key_ck = gtsam.symbol('c', ck_idx)

    # שליפת המטריצה המשותפת המלאה כ-NumPy Array בגודל 12x12
    joint_keys = gtsam.KeyVector([key_c0, key_ck])
    joint_cov_matrix = marginals.jointMarginalCovariance(joint_keys).fullMatrix()
    
    # חילוץ תתי-הבלוקים (כל פוזה היא בגודל 6x6)
    Sigma_00 = joint_cov_matrix[0:6, 0:6]
    Sigma_0k = joint_cov_matrix[0:6, 6:12]
    Sigma_k0 = joint_cov_matrix[6:12, 0:6]
    Sigma_kk = joint_cov_matrix[6:12, 6:12]

    # חישוב ה-Schur Complement במערכת הקואורדינטות הגלובלית
    Sigma_conditional_global = Sigma_kk - Sigma_k0 @ np.linalg.inv(Sigma_00) @ Sigma_0k
    
    # מעבר למערכת הצירים המקומית (Body Frame) בעזרת ה-Adjoint Map של הפוזה היחסית
    relative_pose = bundle_result["relative_pose"]
    Ad_k = relative_pose.AdjointMap()
    Sigma_rel = Ad_k @ Sigma_conditional_global @ Ad_k.T

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

        try:
            br = lib.solve_bundle_window(db, sf, ef)
            g, r = br["graph"], br["result"]

            margs   = gtsam.Marginals(g, r)
            k_sf   = gtsam.symbol('c', sf)
            k_ef   = gtsam.symbol('c', ef)
            
            # שליפת המטריצה המשותפת המלאה עבור זוג הפריים הנוכחי
            jk     = gtsam.KeyVector([k_sf, k_ef])
            jc_matrix = margs.jointMarginalCovariance(jk).fullMatrix()

            # חילוץ הבלוקים בצורה בטוחה מהמטריצה
            S00 = jc_matrix[0:6, 0:6]
            S0k = jc_matrix[0:6, 6:12]
            Sk0 = jc_matrix[6:12, 0:6]
            Skk = jc_matrix[6:12, 6:12]

            # חישוב Schur Complement
            S_cond_global = Skk - Sk0 @ np.linalg.inv(S00) @ S0k
            
            # הטלה למערכת הצירים המקומית של המצלמה הנוכחית
            rel_pose = br["relative_pose"]
            Ad_curr = rel_pose.AdjointMap()
            rel_cov = Ad_curr @ S_cond_global @ Ad_curr.T

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


def q6_2(db, relative_poses, relative_covs, output_dir="."):
    print("\n================================================================================")
    print("SECTION 6.2: POSE GRAPH OPTIMIZATION")
    print("================================================================================")

    # 1. מיון וניקוי קצוות - נוודא שאין קפיצות קיצוניות או קובריאנס סינגולרי
    edges = sorted(relative_poses.keys())
    cleaned_poses = {}
    cleaned_covs = {}
    
    for edge in edges:
        cov = relative_covs[edge]
        # בדיקה שהקובריאנס תקין ולא מכיל ערכים שליליים או אפסים על האלכסון
        if np.any(np.diag(cov) <= 0) or np.any(np.isnan(cov)):
            print(f"Skipping bad covariance edge: {edge}")
            continue
        cleaned_poses[edge] = relative_poses[edge]
        cleaned_covs[edge] = relative_covs[edge]

    # 2. בניית הגרף בעזרת הפונקציה המקורית מהספריה עם הקצוות הנקיים
    graph = lib.build_pose_graph(cleaned_poses, cleaned_covs)
    
    # 3. בנייה חסינה של ה-Initial Estimate כדי למנוע נפילה לראשית (0,0)
    initial = gtsam.Values()
    sorted_cleaned_edges = sorted(cleaned_poses.keys())
    
    if len(sorted_cleaned_edges) == 0:
        raise RuntimeError("No valid relative poses left after cleaning.")
        
    # קביעת הקיפריים הראשון כראשית
    first_kf = sorted_cleaned_edges[0][0]
    current_global_pose = gtsam.Pose3()
    initial.insert(gtsam.symbol("c", first_kf), current_global_pose)
    
    # שרשור רציף וחסין לפערים
    for start_kf, end_kf in sorted_cleaned_edges:
        start_key = gtsam.symbol("c", start_kf)
        end_key = gtsam.symbol("c", end_kf)
        
        # אם פריים ההתחלה חסר בגלל Gap, נשתמש בפוזה הגלובלית האחרונה שחישבנו
        if not initial.exists(start_key):
            initial.insert(start_key, current_global_pose)
        else:
            current_global_pose = initial.atPose3(start_key)
            
        rel_pose = cleaned_poses[(start_kf, end_kf)]
        end_pose = current_global_pose.compose(rel_pose)
        
        if not initial.exists(end_key):
            initial.insert(end_key, end_pose)
            current_global_pose = end_pose

    # בדיקת מפתחות חסרים למניעת קפיצות בגרף
    missing = []
    for i in range(graph.size()):
        factor = graph.at(i)
        for key in factor.keys():
            if not initial.exists(key):
                missing.append(gtsam.Symbol(key).index())
    if missing:
        print("Warning - Fixed missing keys that would have collapsed to 0:", sorted(set(missing)))
        for m_key in missing:
            initial.insert(gtsam.symbol("c", m_key), gtsam.Pose3())

    print(f"Pose graph factors: {graph.size()}")
    print(f"Initial poses: {initial.size()}")

    initial_error = graph.error(initial)
    print(f"Pose graph error BEFORE optimization: {initial_error:.6f}")

    # שמירת גרף הטרקטוריה הראשונית המתוקנת
    lib.plot_pose_graph_trajectory(
        initial,
        title="6.2: Initial Pose Graph Trajectory (Fixed Gaps)",
        output_path=os.path.join(output_dir, "task_6_2_initial_pose_graph.png")
    )

    # 4. הרצת האופטימיזציה
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    final_error = graph.error(result)
    print(f"Pose graph error AFTER optimization: {final_error:.6f}")

    # שמירת הגרפים הסופיים
    lib.plot_pose_graph_trajectory(
        result,
        title="6.2: Optimized Pose Graph Trajectory",
        output_path=os.path.join(output_dir, "task_6_2_optimized_pose_graph.png")
    )

    marginals = gtsam.Marginals(graph, result)
    lib.plot_pose_graph_with_covariances(
        result,
        marginals,
        title="6.2: Optimized Pose Graph With Final Marginal Covariances",
        output_path=os.path.join(output_dir, "task_6_2_pose_graph_covariances.png"),
        covariance_step=5,
    )

    return result


if __name__ == "__main__":
    db = lib.load_or_build_db(
        force_rebuild=False,
        num_frames=lib.get_num_frames()
    )

    relative_poses, relative_covs = q6_1(db)
    q6_2(db, relative_poses, relative_covs, output_dir=".")