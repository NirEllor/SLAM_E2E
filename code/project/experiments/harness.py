"""
Shared experiment harness for pipeline comparison studies.
Provides DB building and pipeline-running infrastructure for all 5 comparison topics.
"""

import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'code' / 'project'))
sys.path.insert(0, str(PROJECT_ROOT / 'code'))

from utils.geometry import get_num_frames
from utils.tracking import load_or_build_db, build_data
from utils.bundle_adjustment import choose_keyframes, solve_bundle_window
from utils.pose_graph import (
    compute_relative_pose_and_covariance, compute_all_relative_constraints,
    clean_pose_graph_edges, build_and_initialize_pose_graph, optimize_pose_graph,
    compute_pose_graph_absolute_error, compute_bundle_relative_error_vs_gt,
    extract_trajectory_and_ids
)
from utils.loop_closure import (
    detect_loop_closure_candidates, verify_loop_closures_consensus,
    estimate_verified_loop_relative_poses, add_loop_closures_and_optimize
)


def build_variant_db(detector_type='akaze', pnp_threshold=2, pnp_iterations=50,
                     pnp_max_no_improvement=12, num_frames=None, cache_tag=None, orb_n_features=None):
    """
    Builds or loads a variant TrackingDB with specified feature detector and PnP-RANSAC params.

    Saves to a distinct pickle path per variant (cache_tag) to avoid clobbering baseline.

    Args:
        detector_type: 'akaze' (default), 'orb', or 'sift'
        pnp_threshold: reprojection error threshold (pixels) for RANSAC inlier cutoff
        pnp_iterations: max RANSAC iterations
        pnp_max_no_improvement: early stopping after N iterations with no improvement
        num_frames: number of frames to process (default: all available)
        cache_tag: identifier for variant pickle (e.g., 'akaze_baseline', 'orb_3000', 'sift')
        orb_n_features: number of features for ORB detector (default 700 if None)

    Returns:
        TrackingDB object
    """
    if num_frames is None:
        num_frames = get_num_frames()

    if cache_tag is None:
        cache_tag = f"{detector_type}_t{pnp_threshold}_i{pnp_iterations}"

    print(f"\n{'='*80}")
    print(f"Variant DB: {cache_tag}")
    print(f"  Detector: {detector_type}, PnP threshold: {pnp_threshold}px, "
          f"iterations: {pnp_iterations}, max_no_improve: {pnp_max_no_improvement}")
    print(f"{'='*80}")

    # Special case: baseline uses the shared code/tracking_db.pkl (never modified by experiments)
    if cache_tag == 'baseline' and detector_type == 'akaze' and pnp_threshold == 2 and pnp_iterations == 50:
        print(f"Loading BASELINE DB from shared code/tracking_db.pkl (NEVER modified by experiments)")
        db = load_or_build_db(force_rebuild=False, num_frames=num_frames)
        print(f"✓ Loaded {db.frame_num()} frames from shared baseline")
        return db

    # For all variant DBs, use a separate pickle file to avoid modifying the shared baseline
    variant_pkl = PROJECT_ROOT / 'code' / f'tracking_db_{cache_tag}.pkl'
    print(f"  Cache path: {variant_pkl}")

    # Check if variant pickle exists and is usable
    if variant_pkl.exists():
        print(f"Loading variant DB from cache")
        try:
            with open(variant_pkl, 'rb') as f:
                db = pickle.load(f)
            print(f"✓ Loaded {db.frame_num()} frames from variant cache")
            return db
        except Exception as e:
            print(f"  [Note] Pickle incompatible ({e}), rebuilding...")

    # Build new DB with variant parameters
    # NOTE: We call build_data directly (not load_or_build_db) to ensure we save to variant_pkl,
    # not the shared code/tracking_db.pkl
    print(f"Building variant DB from scratch (calling build_data directly)...")
    db = build_data(
        num_frames=num_frames,
        detector_type=detector_type,
        pnp_threshold=pnp_threshold,
        pnp_iterations=pnp_iterations,
        pnp_max_no_improvement=pnp_max_no_improvement,
        orb_n_features=orb_n_features
    )

    # Save to variant pickle (NOT the shared baseline DB_PKL_PATH)
    print(f"Saving variant DB to {variant_pkl}...")
    with open(variant_pkl, 'wb') as f:
        pickle.dump(db, f)
    print(f"✓ Variant DB saved")
    print(f"  PROTECTION: Baseline code/tracking_db.pkl remains untouched")

    return db


def run_pipeline_variant(db, keyframe_kwargs=None, bundle_kwargs=None, loop_kwargs=None,
                        run_loop_closure=True, output_dir=None):
    """
    Runs the pipeline stages (keyframe selection, bundle, pose-graph, loop closure) with specified params.

    Returns a dict with standardized metrics reusable across all topics.

    Args:
        db: TrackingDB object
        keyframe_kwargs: dict of params for choose_keyframes (distance_threshold, min_gap, max_gap)
        bundle_kwargs: dict of params for bundle adjustment (max_tracks_per_window, prior_sigma, stereo_sigma, huber_k)
        loop_kwargs: dict of params for loop closure (mahalanobis_threshold, inlier_ratio_threshold)
        run_loop_closure: whether to run loop-closure detection/verification
        output_dir: directory for optional plots (default: code/project/outputs)

    Returns:
        dict with keys: 'db', 'keyframes', 'relative_poses', 'relative_covs', 'pg_no_lc', 'pg_with_lc',
        'absolute_errors_no_lc', 'absolute_errors_lc', 'kitti_errors', 'loop_count', 'runtime_sec'
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'code' / 'project' / 'experiments' / 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    if keyframe_kwargs is None:
        keyframe_kwargs = {}
    if bundle_kwargs is None:
        bundle_kwargs = {}
    if loop_kwargs is None:
        loop_kwargs = {}

    start_time = time.time()
    results = {'db': db}

    # Stage 1: Keyframe selection
    kf_params = {'distance_threshold': 2.5, 'min_gap': 5, 'max_gap': 20}
    kf_params.update(keyframe_kwargs)
    keyframes = choose_keyframes(db, **kf_params)
    results['keyframes'] = keyframes
    print(f"Keyframes: {len(keyframes)}")

    # Stage 2: Bundle adjustment + relative pose extraction
    bundle_params = {'max_tracks_per_window': 150, 'prior_sigma': 1e-6}
    bundle_params.update(bundle_kwargs)

    relative_poses, relative_covs = compute_all_relative_constraints(db, keyframes, bundle_kwargs=bundle_params)
    results['relative_poses'] = relative_poses
    results['relative_covs'] = relative_covs
    print(f"Relative constraints: {len(relative_poses)} edges")

    # Stage 3: Pose graph (without loop closure)
    cleaned_poses, cleaned_covs = clean_pose_graph_edges(relative_poses, relative_covs)
    graph_no_lc, initial_no_lc = build_and_initialize_pose_graph(cleaned_poses, cleaned_covs)
    pg_result_no_lc, marginals_no_lc = optimize_pose_graph(graph_no_lc, initial_no_lc)
    # Note: Don't store graph/marginals (GTSAM objects, not picklable); only store result for trajectory extraction

    # Absolute errors (pose graph without LC)
    abs_errors_no_lc = compute_pose_graph_absolute_error(pg_result_no_lc)
    results['absolute_errors_no_lc'] = abs_errors_no_lc
    print(f"PG (no LC) mean location error: {np.mean(abs_errors_no_lc['err_norm']):.4f} m")

    # Stage 4: Loop closure (optional)
    pg_result_with_lc = pg_result_no_lc
    loop_count = 0

    if run_loop_closure:
        lc_params = {'mahalanobis_threshold': 1000.0, 'inlier_ratio_threshold': 0.6}
        lc_params.update(loop_kwargs)

        try:
            candidates, _ = detect_loop_closure_candidates(
                relative_poses, relative_covs, keyframes, lc_params['mahalanobis_threshold']
            )
            verified, loop_count = verify_loop_closures_consensus(db, candidates, lc_params['inlier_ratio_threshold'])

            if verified:
                loop_measurements = estimate_verified_loop_relative_poses(db, verified)
                pg_results = add_loop_closures_and_optimize(
                    cleaned_poses, cleaned_covs, loop_measurements, output_dir=output_dir
                )
                pg_result_with_lc = pg_results['loop_result']
                print(f"Loop closures: {loop_count} verified")
        except Exception as e:
            print(f"Loop closure error: {e}")

    results['loop_count'] = loop_count

    # Absolute errors (pose graph with LC)
    abs_errors_lc = compute_pose_graph_absolute_error(pg_result_with_lc)
    results['absolute_errors_lc'] = abs_errors_lc
    print(f"PG (with LC) mean location error: {np.mean(abs_errors_lc['err_norm']):.4f} m")

    # Extract trajectory (keyframe poses) as numpy arrays for pickling
    # GTSAM objects (Values, Marginals) can't be pickled, so extract positions now
    try:
        import gtsam
        positions = []
        for key in pg_result_with_lc.keys():
            sym = gtsam.Symbol(key)
            if sym.chr() == ord('c'):  # 'c' = camera pose
                pose = pg_result_with_lc.atPose3(key)
                from utils.bundle_adjustment import pose_translation_np
                pos = pose_translation_np(pose)
                positions.append(pos)
        results['trajectory_positions'] = np.array(positions) if positions else np.array([])
    except Exception as e:
        print(f"Warning: Could not extract trajectory: {e}")
        results['trajectory_positions'] = np.array([])

    # KITTI segment errors (optional, skip for now)
    results['kitti_errors'] = {}

    results['runtime_sec'] = time.time() - start_time
    print(f"Runtime: {results['runtime_sec']:.1f} sec")

    return results


def save_variant_result(results, topic, variant_name, output_dir=None):
    """Saves results dict to a pickle file for later plotting."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'code' / 'project' / 'experiment_results'

    topic_dir = Path(output_dir) / topic
    topic_dir.mkdir(parents=True, exist_ok=True)

    result_path = topic_dir / f'{variant_name}.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved results to {result_path}")
    return result_path


def load_variant_result(topic, variant_name, output_dir=None):
    """Loads results dict from a pickle file."""
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'code' / 'project' / 'experiment_results'

    result_path = Path(output_dir) / topic / f'{variant_name}.pkl'
    with open(result_path, 'rb') as f:
        return pickle.load(f)


