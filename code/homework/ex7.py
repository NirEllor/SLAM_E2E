import os
import van_utils as lib

output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)


def q7_1_detect_candidates(db, relative_poses, relative_covs, keyframes, mahalanobis_threshold=15.0, output_dir="./outputs"):
    """7.1: Detects loop closure candidates using Mahalanobis distance filtering."""
    print("\n" + "=" * 80)
    print("RUNNING: Q7.1 - DETECT LOOP CLOSURE CANDIDATES (SHORTEST-PATH COVARIANCE VERSION)")
    print("=" * 80)

    # Detect loop candidates using graph search and Mahalanobis filtering
    candidates, total_found = lib.detect_loop_closure_candidates(
        relative_poses, relative_covs, keyframes, mahalanobis_threshold
    )

    print("\n" + "=" * 80)
    print(f"Total Loop Closure Candidates Found: {total_found}")
    print("=" * 80)

    return candidates


def q7_2_consensus_matching(db, loop_candidates, inlier_ratio_threshold=0.6, output_dir="./outputs"):
    """7.2: Verifies loop closures using RANSAC fundamental matrix consensus."""
    print("\n" + "=" * 80)
    print("RUNNING: Q7.2 - CONSENSUS MATCHING (VISUAL VERIFICATION)")
    print("=" * 80)

    # Run visual verification via RANSAC fundamental matrix analysis
    verified_loops, total_verified = lib.verify_loop_closures_consensus(
        db, loop_candidates, inlier_ratio_threshold, output_dir
    )

    print("\n" + "=" * 80)
    print(f"SUMMARY Q7.2: Total Visually Verified Loops: {total_verified}")
    print("=" * 80)

    return verified_loops


def q7_3_relative_pose_estimation(db, verified_loops, output_dir="./outputs"):
    """7.3: Estimates relative poses for verified loop closures via bundle adjustment."""
    print("\n" + "=" * 80)
    print("RUNNING: Q7.3 - RELATIVE POSE ESTIMATION FROM SMALL BUNDLE")
    print("=" * 80)
    loop_measurements = lib.estimate_verified_loop_relative_poses(
        db, verified_loops, output_dir=output_dir
    )
    print(f"SUMMARY Q7.3: Successful relative loop measurements: {len(loop_measurements)}")
    return loop_measurements


def q7_4_update_pose_graph(cleaned_poses, cleaned_covs, loop_measurements, output_dir="./outputs"):
    """7.4: Adds loop closure constraints to pose graph and re-optimizes."""
    print("\n" + "=" * 80)
    print("RUNNING: Q7.4 - UPDATE POSE GRAPH WITH LOOP CLOSURES")
    print("=" * 80)

    pg_results = lib.add_loop_closures_and_optimize(
        cleaned_poses,
        cleaned_covs,
        loop_measurements,
        output_dir=output_dir
    )

    print(
        f"SUMMARY Q7.4: Added {pg_results['num_added_loop_closures']} loop closure factors"
    )

    return pg_results


def q7_5_results(pg_results, loop_measurements, output_dir="./outputs"):
    """7.5: Generates final reports and visualizations for loop closure results."""
    print("\n" + "=" * 80)
    print("RUNNING: Q7.5 - FINAL PLOTS AND REPORT NUMBERS")
    print("=" * 80)
    print(f"Visually verified / estimated loop measurements: {len(loop_measurements)}")
    print(f"Loop closure factors added to pose graph: {pg_results['num_added_loop_closures']}")

    lib.plot_pose_graph_vs_ground_truth(
        pg_results["no_loop_result"],
        pg_results["loop_result"],
        output_dir=output_dir,
    )
    lib.plot_absolute_location_error(
        pg_results["no_loop_result"],
        pg_results["loop_result"],
        output_dir=output_dir,
    )
    lib.plot_location_uncertainty_size(
        pg_results["no_loop_result"],
        pg_results["no_loop_marginals"],
        pg_results["loop_result"],
        pg_results["loop_marginals"],
        output_dir=output_dir,
    )

    print("\nSnapshot choice for report:")
    print("1. before adding loop closures")
    print("2-4. after evenly spaced loop-closure additions, including the final graph")
    print("Uncertainty size measure: sqrt(det(Sigma_xz)), i.e. 1-sigma location ellipse area divided by pi.")


if __name__ == "__main__":
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    db = lib.load_or_build_db(force_rebuild=False, num_frames=lib.get_num_frames())
    keyframes = lib.choose_keyframes(db, distance_threshold=2.5, min_gap=20, max_gap=20)

    print("\n[Initialization] Extracting relative edge constraints from exercise 6...")
    relative_poses, relative_covs = lib.compute_all_relative_constraints(db, keyframes)

    print("\n[Initialization] Cleaning and optimizing no-loop baseline pose graph...")
    cleaned_poses, cleaned_covs = lib.clean_pose_graph_edges(relative_poses, relative_covs)
    graph, initial = lib.build_and_initialize_pose_graph(cleaned_poses, cleaned_covs)
    optimized_values, marginals = lib.optimize_pose_graph(graph, initial)

    # 3. Execute Step 7.1
    mahalanobis_threshold = 1000.0
    candidates = q7_1_detect_candidates(
        db, relative_poses, relative_covs, keyframes,
        mahalanobis_threshold=mahalanobis_threshold
    )

    # Plot loop shortcuts over the newly decoupled trajectory values
    lib.plot_loop_candidates(keyframes, candidates, optimized_values, output_dir="./outputs")

    # 4. Execute Step 7.2
    inlier_ratio_threshold = 0.75
    verified_loops = q7_2_consensus_matching(
        db, candidates,
        inlier_ratio_threshold,
        output_dir="./outputs"
    )

    loop_measurements = q7_3_relative_pose_estimation(
        db, verified_loops,
        output_dir=output_dir,
    )

    pg_results = q7_4_update_pose_graph(
        cleaned_poses, cleaned_covs, loop_measurements,
        output_dir=output_dir,
    )

    q7_5_results(pg_results, loop_measurements, output_dir=output_dir)
