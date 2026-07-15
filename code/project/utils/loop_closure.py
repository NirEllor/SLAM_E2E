"""
utils/loop_closure.py
Loop closure detection, verification, and pose estimation utilities using GTSAM.
"""

import numpy as np
import cv2
import gtsam
from gtsam import symbol
import networkx as nx

from .geometry import read_images, get_akaze_features, run_single_pair
from .bundle_adjustment import init_gtsam_stereo_calibration
from .pose_graph import optimize_pose_graph, build_and_initialize_pose_graph


def _shortest_path_pose_graph(relative_poses, relative_covs):
    """Constructs a NetworkX graph weighted by the scalar trace of relative edge covariances."""
    G = nx.Graph()
    for (start_kf, end_kf), cov in relative_covs.items():
        G.add_edge(start_kf, end_kf, weight=float(np.trace(cov)))
    return G


def _get_edge_pose_and_cov(u, v, relative_poses, relative_covs):
    """Retrieves edge constraints between keyframe nodes, automatically handling directional inversions."""
    if (u, v) in relative_poses:
        return relative_poses[(u, v)], relative_covs[(u, v)]

    if (v, u) in relative_poses:
        pose_uv = relative_poses[(v, u)].inverse()
        Ad_inv = pose_uv.AdjointMap()
        return pose_uv, Ad_inv @ relative_covs[(v, u)] @ Ad_inv.T

    raise KeyError(f"No pose-graph edge between keyframes {u} and {v}")


def detect_loop_closure_candidates(relative_poses, relative_covs, keyframes, mahalanobis_threshold):
    """Identifies candidate loop closures below a Mahalanobis distance threshold using Dijkstra shortest paths."""
    G = _shortest_path_pose_graph(relative_poses, relative_covs)
    loop_candidates = {}
    total_candidates_counter = 0

    for n_idx, c_n in enumerate(keyframes):
        if n_idx == 0 or c_n not in G:
            continue

        loop_candidates[c_n] = []
        for c_i in keyframes[:n_idx]:
            if c_i not in G:
                continue

            try:
                path = nx.shortest_path(G, source=c_n, target=c_i, weight="weight")
                suffix_pose = gtsam.Pose3()
                Sigma_rel = np.zeros((6, 6))

                for u, v in reversed(list(zip(path[:-1], path[1:]))):
                    edge_pose, edge_cov = _get_edge_pose_and_cov(u, v, relative_poses, relative_covs)
                    Ad_suffix_inv = suffix_pose.inverse().AdjointMap()
                    Sigma_rel += Ad_suffix_inv @ edge_cov @ Ad_suffix_inv.T
                    suffix_pose = edge_pose.compose(suffix_pose)

                xi_ni = gtsam.Pose3.Logmap(suffix_pose)
                Sigma_inv = np.linalg.inv(Sigma_rel + np.eye(6) * 1e-6)
                mahalanobis_dist = xi_ni.T @ Sigma_inv @ xi_ni

                if mahalanobis_dist < mahalanobis_threshold:
                    loop_candidates[c_n].append({
                        "candidate_kf": c_i, "mahalanobis_distance": mahalanobis_dist,
                        "relative_pose_estimate": suffix_pose, "relative_covariance": Sigma_rel, "path": path,
                    })
                    total_candidates_counter += 1
            except Exception:
                continue

    return loop_candidates, total_candidates_counter


def verify_loop_closures_consensus(db, loop_candidates, inlier_ratio_threshold, output_dir=None):
    """Verifies candidate loops by performing AKAZE matching and Fundamental matrix RANSAC filtering."""
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    verified_loops = {}
    total_verified_loops = 0
    MIN_ABSOLUTE_INLIERS = 20

    for c_n, cands in loop_candidates.items():
        if not cands:
            continue

        verified_loops[c_n] = []
        img_n, _ = read_images(c_n)
        kp_n, des_n = get_akaze_features(img_n)

        for cand in cands:
            c_i = cand["candidate_kf"]
            img_i, _ = read_images(c_i)
            kp_i, des_i = get_akaze_features(img_i)

            if des_n is None or des_i is None:
                continue

            knn_matches = bf_matcher.knnMatch(des_i, des_n, k=2)
            good_matches = [m for m, n in knn_matches if m.distance < 0.7 * n.distance]
            if len(good_matches) < 4:
                continue

            pts_i = np.array([kp_i[m.queryIdx].pt for m in good_matches], dtype=np.float32)
            pts_n = np.array([kp_n[m.trainIdx].pt for m in good_matches], dtype=np.float32)

            _, mask = cv2.findFundamentalMat(pts_i, pts_n, cv2.FM_RANSAC, 3.0, 0.99)
            if mask is None:
                continue

            inliers_count = int(np.sum(mask))
            inlier_ratio = inliers_count / len(good_matches)

            if inlier_ratio >= inlier_ratio_threshold and inliers_count >= MIN_ABSOLUTE_INLIERS:
                inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]
                verified_loops[c_n].append({"candidate_kf": c_i, "inliers_count": inliers_count, "inlier_matches": inlier_matches})
                total_verified_loops += 1

    return verified_loops, total_verified_loops


def estimate_loop_relative_pose_bundle(db, c_i, c_n, inlier_matches=None, max_landmarks=120, min_landmarks=8, output_dir="."):
    """Estimates relative pose and covariance across confirmed loop closure frames using 2-frame BA."""
    from .geometry import read_cameras

    K_gtsam = init_gtsam_stereo_calibration()
    K_mat, _, _ = read_cameras()

    data_i = run_single_pair(c_i, display=False, plot_3d=False)
    data_n = run_single_pair(c_n, display=False, plot_3d=False)

    if inlier_matches is None:
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = bf_matcher.knnMatch(data_i["des_left"], data_n["des_left"], k=2)
        good = [m for m, nn in knn if m.distance < 0.7 * nn.distance]
        pts_i = np.array([data_i["kp_left"][m.queryIdx].pt for m in good], dtype=np.float32)
        pts_n = np.array([data_n["kp_left"][m.trainIdx].pt for m in good], dtype=np.float32)
        _, mask = cv2.findFundamentalMat(pts_i, pts_n, cv2.FM_RANSAC, 3.0, 0.99)
        inlier_matches = [good[k] for k in range(len(good)) if mask[k][0] == 1]

    i_q_to_pt = {m.queryIdx: idx for idx, m in enumerate(data_i["stereo_inliers"])}
    i_q_to_stereo = {m.queryIdx: m for m in data_i["stereo_inliers"]}
    n_q_to_stereo = {m.queryIdx: m for m in data_n["stereo_inliers"]}

    usable = []
    for m in inlier_matches:
        if m.queryIdx in i_q_to_pt and m.queryIdx in i_q_to_stereo and m.trainIdx in n_q_to_stereo:
            X_i = np.array(data_i["points_3d"][i_q_to_pt[m.queryIdx]], dtype=np.float64).reshape(3)
            if np.all(np.isfinite(X_i)) and 2.0 < X_i[2] < 120.0:
                usable.append((m, X_i))

    if len(usable) < min_landmarks:
        raise RuntimeError(f"Loop {c_i}->{c_n}: only {len(usable)} usable stereo landmarks")

    usable = sorted(usable, key=lambda item: item[0].distance)[:max_landmarks]

    obj_pts = np.array([X for _, X in usable], dtype=np.float32)
    img_pts = np.array([data_n["kp_left"][m.trainIdx].pt for m, _ in usable], dtype=np.float32)
    success, rvec, tvec, pnp_inliers = cv2.solvePnPRansac(obj_pts, img_pts, K_mat, None, iterationsCount=200, reprojectionError=3.0, confidence=0.99)

    if not success or pnp_inliers is None or len(pnp_inliers) < min_landmarks:
        raise RuntimeError(f"Loop {c_i}->{c_n}: PnP failed")

    R_ni, _ = cv2.Rodrigues(rvec)
    t_ni = tvec.reshape(3)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    key_i, key_n = symbol("c", int(c_i)), symbol("c", int(c_n))

    initial.insert(key_i, gtsam.Pose3())
    initial.insert(key_n, gtsam.Pose3(gtsam.Rot3(R_ni.T), gtsam.Point3(*(-R_ni.T @ t_ni).reshape(3))))
    graph.add(gtsam.PriorFactorPose3(key_i, gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)))

    meas_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Huber.Create(2.0), gtsam.noiseModel.Isotropic.Sigma(3, 1.0))
    kept_indices = set(int(x[0]) for x in pnp_inliers.reshape(-1, 1))

    for local_idx, (m, X_i) in enumerate(usable):
        if local_idx not in kept_indices:
            continue
        point_key = symbol("q", int(10_000_000 + c_i * 10_000 + c_n * 10 + local_idx))
        initial.insert(point_key, gtsam.Point3(float(X_i[0]), float(X_i[1]), float(X_i[2])))

        kp_li, kp_ri = data_i["kp_left"][m.queryIdx], data_i["kp_right"][i_q_to_stereo[m.queryIdx].trainIdx]
        kp_ln, kp_rn = data_n["kp_left"][m.trainIdx], data_n["kp_right"][n_q_to_stereo[m.trainIdx].trainIdx]

        graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(kp_li.pt[0], kp_ri.pt[0], kp_li.pt[1]), meas_noise, key_i, point_key, K_gtsam))
        graph.add(gtsam.GenericStereoFactor3D(gtsam.StereoPoint2(kp_ln.pt[0], kp_rn.pt[0], kp_ln.pt[1]), meas_noise, key_n, point_key, K_gtsam))

    result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
    marginals = gtsam.Marginals(graph, result)

    keys = gtsam.KeyVector()
    keys.append(key_i)
    keys.append(key_n)

    rel_cov = np.linalg.inv(marginals.jointMarginalInformation(keys).fullMatrix()[-6:, -6:])

    return {
        "start_kf": int(c_i), "end_kf": int(c_n),
        "relative_pose": result.atPose3(key_i).between(result.atPose3(key_n)),
        "relative_covariance": rel_cov, "graph": graph, "result": result
    }


def estimate_verified_loop_relative_poses(db, verified_loops, output_dir="."):
    """Executes relative pose estimation for all verified loop candidate pairs."""
    loop_measurements = []
    for c_n, loops in verified_loops.items():
        for loop in loops:
            try:
                m = estimate_loop_relative_pose_bundle(db, loop["candidate_kf"], c_n, inlier_matches=loop.get("inlier_matches"), output_dir=output_dir)
                loop_measurements.append(m)
            except Exception as e:
                print(f"[Q7.3] Loop {loop['candidate_kf']}->{c_n}: FAILED ({e})")
    return loop_measurements


def add_loop_closures_and_optimize(cleaned_poses, cleaned_covs, loop_measurements, output_dir=".", snapshot_count=4):
    """Adds loop closure constraints to full pose graph and resolves global SLAM optimization."""
    from .visualization import plot_pose_graph_2d_ellipses

    graph0, initial0 = build_and_initialize_pose_graph(cleaned_poses, cleaned_covs)
    no_loop_result, no_loop_marginals = optimize_pose_graph(graph0, initial0)

    final_graph, _ = build_and_initialize_pose_graph(cleaned_poses, cleaned_covs)
    relative_poses_lc, relative_covs_lc = dict(cleaned_poses), dict(cleaned_covs)
    current_initial, final_result, final_marginals = no_loop_result, no_loop_result, no_loop_marginals

    for meas in loop_measurements:
        edge = (meas["end_kf"], meas["start_kf"])
        meas_rel = meas["relative_pose"].inverse()
        loop_cov = 0.5 * (meas["relative_covariance"] + meas["relative_covariance"].T) + np.eye(6) * 1e-9

        final_graph.add(gtsam.BetweenFactorPose3(
            symbol("c", meas["end_kf"]), symbol("c", meas["start_kf"]),
            meas_rel, gtsam.noiseModel.Gaussian.Covariance(loop_cov)
        ))

        relative_poses_lc[edge] = meas_rel
        relative_covs_lc[edge] = loop_cov

        final_result, final_marginals = optimize_pose_graph(final_graph, current_initial)
        current_initial = final_result

    import os
    plot_pose_graph_2d_ellipses(final_result, final_marginals, os.path.join(output_dir, "task_7_5_final_loop_closed.png"), step=5)

    return {
        "no_loop_graph": graph0, "no_loop_result": no_loop_result, "no_loop_marginals": no_loop_marginals,
        "loop_graph": final_graph, "loop_result": final_result, "loop_marginals": final_marginals,
        "relative_poses_with_loops": relative_poses_lc, "relative_covs_with_loops": relative_covs_lc,
        "num_added_loop_closures": len(loop_measurements),
    }
