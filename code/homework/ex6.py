import os
import van_utils as lib

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)


def q6_1(db, output_dir="./outputs"):
    """6.1: Extracts relative pose constraints from bundle adjustment keyframe windows."""
    print("\n" + "=" * 80)
    print("SECTION 6.1: RELATIVE POSE CONSTRAINTS FROM BUNDLE ADJUSTMENT")
    print("=" * 80)

    # 1. Choose keyframes and run sensitivity analysis plots
    keyframes = lib.choose_keyframes(db, distance_threshold=2.5, min_gap=20, max_gap=20)
    c0_idx, ck_idx = keyframes[0], keyframes[1]
    print(f"Total keyframes : {len(keyframes)}")
    print(f"First two       : c0={c0_idx}, c_k={ck_idx}")

    print(f"\n--- Plotting covariance sensitivity for bundle {c0_idx}→{ck_idx} ---")
    lib.run_and_plot_prior_sensitivity(db, c0_idx, ck_idx, output_dir)

    # 2. Extract first relative pose and conditional covariance
    print(f"\n--- Relative pose & covariance for c0={c0_idx} → c_k={ck_idx} ---")
    relative_pose, Sigma_rel = lib.compute_relative_pose_and_covariance(db, c0_idx, ck_idx)
    print(f"\nRelative Pose between keyframes c{c0_idx} and c{ck_idx}:")
    print(relative_pose)
    print(f"\nConditional covariance P(c{ck_idx} | c{c0_idx}):")
    import numpy as np
    print(np.array2string(Sigma_rel, precision=8, suppress_small=True))

    # 3. Process all consecutive pairs
    print("\n--- Processing all consecutive keyframe pairs ---")
    relative_poses, relative_covs = lib.compute_all_relative_constraints(db, keyframes)

    return relative_poses, relative_covs


def q6_2(db, relative_poses, relative_covs, output_dir="./outputs"):
    """6.2: Optimizes pose graph using relative pose constraints and covariances."""
    print("\n" + "=" * 80)
    print("SECTION 6.2: POSE GRAPH OPTIMIZATION")
    print("=" * 80)

    # 1. Clean bad edges
    cleaned_poses, cleaned_covs = lib.clean_pose_graph_edges(relative_poses, relative_covs)

    # 2. Build graph and robustly initialize estimates
    graph, initial = lib.build_and_initialize_pose_graph(cleaned_poses, cleaned_covs)
    print(f"Pose graph factors: {graph.size()}")
    print(f"Initial poses: {initial.size()}")
    print(f"Pose graph error BEFORE optimization: {graph.error(initial):.8e}")

    # 3. Plot initial state
    lib.plot_pose_graph_trajectory(
        initial,
        title="6.2: Initial Pose Graph Trajectory (Fixed Gaps)",
        output_path=os.path.join(output_dir, "task_6_2_initial_pose_graph.png")
    )

    # 4. Optimize
    result, marginals = lib.optimize_pose_graph(graph, initial)
    print(f"Pose graph error AFTER optimization: {graph.error(result):.8e}")

    # 5. Save final visual artifacts
    lib.plot_pose_graph_trajectory(
        result,
        title="6.2: Optimized Pose Graph Trajectory",
        output_path=os.path.join(output_dir, "task_6_2_optimized_pose_graph.png")
    )
    lib.plot_pose_graph_with_covariances(
        result,
        marginals,
        title="6.2: Optimized Pose Graph With Final Marginal Covariances",
        output_path=os.path.join(output_dir, "task_6_2_pose_graph_covariances.png"),
        covariance_step=5,
    )
    lib.plot_pose_graph_2d_ellipses(
        result,
        marginals,
        output_path=os.path.join(output_dir, "task_6_2_pose_graph_2d_covariances.png")
    )

    return result


if __name__ == "__main__":
    db = lib.load_or_build_db(force_rebuild=False, num_frames=lib.get_num_frames())
    relative_poses, relative_covs = q6_1(db)
    q6_2(db, relative_poses, relative_covs)