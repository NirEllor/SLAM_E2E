"""
utils/visualization.py
Visualization and plotting tools for stereo pairs, trajectories, factor graphs, and covariances.
"""

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import gtsam
from gtsam import symbol
from gtsam.utils import plot as gtsam_plot

from .geometry import read_images, camera_center, crop_around_point, read_ground_truth_poses


def plot_stereo_side_by_side(img_l, img_r, title="Stereo Pair"):
    """Displays left and right stereo images side-by-side."""
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img_l, cmap='gray')
    plt.title(f"{title} - Left")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_r, cmap='gray')
    plt.title(f"{title} - Right")
    plt.axis('off')
    plt.tight_layout()


def draw_matches_custom(img1, kp1, img2, kp2, matches, title, num=20):
    """Draws a random subset of feature matches between two images."""
    if len(matches) > num:
        matches = random.sample(matches, num)

    img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.imshow(img_rgb)
    plt.axis('off')


def plot_deviation_histogram(deviations):
    """Plots a histogram of y-deviation values for stereo matches."""
    plt.figure(figsize=(10, 6))
    plt.hist(deviations, bins=50)
    plt.xlabel("Deviation from rectified stereo pattern (pixels)")
    plt.ylabel("Number of matches")
    plt.title("Histogram of deviations from rectified stereo pattern")
    plt.tight_layout()


def print_large_deviation_percentage(deviations, threshold=2):
    """Prints percentage of matches exceeding the epipolar error threshold."""
    num_matches = len(deviations)
    num_bad_matches = np.sum(deviations > threshold)
    percentage = 100 * num_bad_matches / num_matches
    print(f"Matches with deviation > {threshold} pixels: {num_bad_matches}/{num_matches}")
    print(f"Percentage of matches with deviation > {threshold} pixels: {percentage:.2f}%")


def draw_inliers_outliers(img1, kp1, img2, kp2, inliers, outliers):
    """Overlays inlier (orange) and outlier (cyan) keypoint markers on stereo image pairs."""
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    for match in outliers:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        cv2.circle(img1_color, (int(pt1[0]), int(pt1[1])), 3, (0, 255, 255), -1)
        cv2.circle(img2_color, (int(pt2[0]), int(pt2[1])), 3, (0, 255, 255), -1)

    for match in inliers:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        cv2.circle(img1_color, (int(pt1[0]), int(pt1[1])), 3, (255, 165, 0), -1)
        cv2.circle(img2_color, (int(pt2[0]), int(pt2[1])), 3, (255, 165, 0), -1)

    plot_stereo_side_by_side(img1_color, img2_color, title="Inliers (orange) vs Outliers (cyan)")


def plot_3d_points(points_3d, title="3D Point Cloud"):
    """Plots a 3D scatter plot of a landmark point cloud."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=2)
    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()


def plot_four_cameras(R, t, baseline=0.54):
    """Plots bird's-eye view 2D positions (X, Z) of four relative stereo cameras (2 time steps)."""
    c_left_0 = np.array([0.0, 0.0])
    c_right_0 = np.array([baseline, 0.0])

    c_world_3d = camera_center(R, t)
    c_left_1 = np.array([c_world_3d[0], c_world_3d[2]])

    right_shift_world = R.T[:, 0] * baseline
    c_right_1 = c_left_1 + np.array([right_shift_world[0], right_shift_world[2]])

    plt.figure(figsize=(8, 6))
    plt.scatter(c_left_0[0], c_left_0[1], color='blue', marker='^', s=150, label='Left Camera (t0)')
    plt.scatter(c_right_0[0], c_right_0[1], color='cyan', marker='^', s=150, label='Right Camera (t0)')
    plt.scatter(c_left_1[0], c_left_1[1], color='red', marker='s', s=120, label='Left Camera (t1)')
    plt.scatter(c_right_1[0], c_right_1[1], color='orange', marker='s', s=120, label='Right Camera (t1)')

    plt.plot([c_left_0[0], c_right_0[0]], [c_left_0[1], c_right_0[1]], 'b--', alpha=0.5)
    plt.plot([c_left_1[0], c_right_1[0]], [c_left_1[1], c_right_1[1]], 'r--', alpha=0.5)

    plt.title("Relative Positions of Four Cameras (Bird's-Eye View)")
    plt.xlabel("X (Width / Lateral movement [m])")
    plt.ylabel("Z (Depth / Forward movement [m])")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.axis('equal')


def draw_match_canvas(img0_gray, kp0, img1_gray, kp1, matches_green, matches_red, title):
    """Generic helper to visualize matches across two images (green=inliers, red=outliers)."""
    img0 = cv2.cvtColor(img0_gray, cv2.COLOR_GRAY2RGB)
    img1 = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2RGB)
    w0 = img0.shape[1]
    canvas = np.hstack([img0, img1])

    def draw_line(match, color):
        q_idx = match.queryIdx if hasattr(match, 'queryIdx') else match['temporal_match'].queryIdx
        t_idx = match.trainIdx if hasattr(match, 'trainIdx') else match['temporal_match'].trainIdx
        pt0 = tuple(map(int, kp0[q_idx].pt))
        pt1_raw = kp1[t_idx].pt
        pt1 = (int(pt1_raw[0] + w0), int(pt1_raw[1]))
        cv2.circle(canvas, pt0, 3, color, -1)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.line(canvas, pt0, pt1, color, 1)

    for match in matches_red:
        draw_line(match, (255, 0, 0))
    for match in matches_green:
        draw_line(match, (0, 255, 0))

    plt.figure(figsize=(16, 8))
    plt.imshow(canvas)
    plt.title(title)
    plt.axis("off")


def draw_temporal_supporters(img_left0, kp_left0, img_left1, kp_left1, supporters, non_supporters, title="Supporters vs Non-supporters"):
    """Plots temporal tracking matches with green lines for supporters and red for outliers."""
    draw_match_canvas(img_left0, kp_left0, img_left1, kp_left1, supporters, non_supporters, title + " | green=supporters, red=outliers")


def draw_ransac_results(frame0_data, frame1_data, inliers, outliers):
    """Visualizes PnP-RANSAC temporally tracked inliers and outliers between consecutive frames."""
    draw_match_canvas(
        frame0_data['img_left'], frame0_data['kp_left'],
        frame1_data['img_left'], frame1_data['kp_left'],
        inliers, outliers, "3.5: RANSAC Inliers (green) vs Outliers (red)"
    )


def plot_transformed_clouds(frame0_data, frame1_data, R, t):
    """Visualizes point cloud alignment from Frame 0 transformed into Frame 1 coordinate system."""
    cloud0 = frame0_data['points_3d']
    cloud1 = frame1_data['points_3d']

    cloud0_transformed = (R @ cloud0.T + t.reshape(3, 1)).T

    mask0 = (cloud0_transformed[:, 2] > 0) & (cloud0_transformed[:, 2] < 80)
    mask1 = (cloud1[:, 2] > 0) & (cloud1[:, 2] < 80)

    plt.figure(figsize=(10, 8))
    plt.scatter(cloud0_transformed[mask0, 0], cloud0_transformed[mask0, 2], s=2, c='red', label='Pair0 after T')
    plt.scatter(cloud1[mask1, 0], cloud1[mask1, 2], s=2, c='blue', label='Pair1')

    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title("3.5: Point Clouds Alignment")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)


def plot_trajectory(est_positions, gt_positions):
    """Plots estimated vehicle trajectory against ground truth (X-Z Top View)."""
    est_positions, gt_positions = np.array(est_positions), np.array(gt_positions)
    plt.figure(figsize=(10, 8))
    plt.plot(est_positions[:, 0], est_positions[:, 2], label='Estimated trajectory', linewidth=2)
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], label='Ground truth', linewidth=2)
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.title("3.6: Camera Trajectory (Top View)")
    plt.legend()
    plt.axis('equal')
    plt.grid(True)


def plot_track_observations(db, track_id, crop_size=20):
    """Displays tracked feature patch crops across all frames where the track is present."""
    frames = db.frames(track_id)
    num_rows = len(frames)
    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 2.2 * num_rows))
    if num_rows == 1:
        axes = np.array([axes])

    fig.suptitle(f"Track #{track_id}, length={len(frames)}", fontsize=14)

    for row, frame_id in enumerate(frames):
        obs = db.observation(frame_id, track_id)
        img_left, _ = read_images(frame_id)
        crop, lx, ly = crop_around_point(img_left, obs.x_left, obs.y, crop_size=crop_size)

        axes[row, 0].imshow(img_left, cmap="gray")
        axes[row, 0].scatter([obs.x_left], [obs.y], c="red", marker="x", s=40)
        axes[row, 0].set_title(f"Frame {frame_id}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(crop, cmap="gray")
        axes[row, 1].scatter([lx], [ly], c="red", marker="x", s=40)
        axes[row, 1].set_title(f"{crop_size}x{crop_size} crop")
        axes[row, 1].axis("off")

    plt.tight_layout()


def plot_connectivity(connectivity):
    """Plots consecutive frame feature connectivity sequence over time."""
    plt.figure(figsize=(12, 5))
    plt.plot(connectivity, linewidth=1)
    plt.axhline(np.mean(connectivity), color="green", linestyle="--", label=f"Mean={np.mean(connectivity):.1f}")
    plt.title("Connectivity")
    plt.xlabel("Frame")
    plt.ylabel("Outgoing Tracks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_inlier_percentage(inlier_percentages):
    """Plots the percentage of PnP RANSAC inliers per frame transition."""
    plt.figure(figsize=(12, 5))
    plt.plot(inlier_percentages, linewidth=1)
    mean_val = np.mean(inlier_percentages)
    plt.axhline(mean_val, color="green", linestyle="--", label=f"Mean={mean_val:.2f}%")
    plt.title("Inlier Percentage Per Frame")
    plt.xlabel("Frame")
    plt.ylabel("Inlier Percentage (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_track_length_histogram(db, min_length=2):
    """Plots a log-scale histogram of feature track lengths."""
    track_lengths = [len(db.frames(t_id)) for t_id in db.track_to_frames if len(db.frames(t_id)) >= min_length]
    plt.figure(figsize=(12, 5))
    bins = range(min(track_lengths), max(track_lengths) + 2)
    plt.hist(track_lengths, bins=bins, edgecolor="black")
    plt.yscale("log")
    plt.title("Track Length Histogram")
    plt.xlabel("Track Length")
    plt.ylabel("Track Count")
    plt.grid(True, axis="y")
    plt.tight_layout()


def plot_track_reprojection_error_analysis(frames, left_errors, right_errors):
    """Plots left and right reprojection errors across track frames."""
    distances_from_reference = list(range(len(frames)))

    plt.figure(figsize=(10, 6))
    plt.plot(distances_from_reference, left_errors, label='Left Channel Residuals', color='#2F4F4F', linewidth=2)
    plt.plot(distances_from_reference, right_errors, label='Right Channel Residuals', color='#FFA500', linewidth=2)

    plt.title("4.7: PnP - projection error vs track length")
    plt.xlabel("distance from reference (frames)")
    plt.ylabel("projection error (pixels)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()


def draw_projection_validation_frames(frame_id, obs, proj_init, proj_final):
    """Draws measurement (Blue), pre-optimization (Red), and post-optimization (Green) pixel overlays."""
    img_left_gray, img_right_gray = read_images(frame_id)
    img_left = cv2.cvtColor(img_left_gray, cv2.COLOR_GRAY2BGR)
    img_right = cv2.cvtColor(img_right_gray, cv2.COLOR_GRAY2BGR)

    points_map = [
        ((int(obs.x_left), int(obs.y)), (int(proj_init.uL()), int(proj_init.v())), (int(proj_final.uL()), int(proj_final.v()))),
        ((int(obs.x_right), int(obs.y)), (int(proj_init.uR()), int(proj_init.v())), (int(proj_final.uR()), int(proj_final.v())))
    ]

    for img, (pt_meas, pt_before, pt_after) in zip([img_left, img_right], points_map):
        cv2.circle(img, pt_meas, radius=6, color=(255, 0, 0), thickness=-1)
        cv2.circle(img, pt_before, radius=6, color=(0, 0, 255), thickness=-1)
        cv2.circle(img, pt_after, radius=6, color=(0, 255, 0), thickness=-1)

    cv2.putText(img_left, "Blue: Meas | Red: Before | Green: After", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"task_5_3_worst_frame_{frame_id}_left.png"), img_left)
    cv2.imwrite(os.path.join(output_dir, f"task_5_3_worst_frame_{frame_id}_right.png"), img_right)


def reprojection_error_graph(frame_indices, left_reprojection_errors, right_reprojection_errors, output_dir):
    """Plots stereo reprojection errors across frame indices."""
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, left_reprojection_errors, label="Left camera reprojection error", color="#1f77b4")
    plt.plot(frame_indices, right_reprojection_errors, label="Right camera reprojection error", color="#ff7f0e")
    plt.title("Right camera reprojection error")
    plt.xlabel("Frame index")
    plt.ylabel("Reprojection error (L2 norm) - pixels")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, "reprojection_error_graph.png"), dpi=300)
    plt.close()


def factor_error_graph(frame_indices, factor_errors, output_dir):
    """Plots GTSAM factor cost graph evolution across frame indices."""
    plt.figure(figsize=(10, 5))
    plt.plot(frame_indices, factor_errors, color="red", marker='s')
    plt.title("Factor error graph")
    plt.xlabel("Frame index")
    plt.ylabel("Factor Error")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "factor_error_graph.png"), dpi=300)
    plt.close()


def bundle1_3D(window_frames, cam_positions, result, axis_length, output_dir):
    """Renders 3D local bundle trajectory visualization with oriented pose axes."""
    fig3d = plt.figure(figsize=(8, 6))
    ax3d = fig3d.add_subplot(111, projection='3d')

    xs_plot, ys_plot, zs_plot = cam_positions[:, 2], cam_positions[:, 0], -cam_positions[:, 1]
    ax3d.plot(xs_plot, ys_plot, zs_plot, 'k--', linewidth=1.5, zorder=1)

    for i in range(len(window_frames)):
        pose = result.atPose3(symbol('c', window_frames[i]))
        R = pose.rotation().matrix()
        cx, cy, cz = xs_plot[i], ys_plot[i], zs_plot[i]

        ax3d.scatter(cx, cy, cz, color='black', s=15, zorder=2)
        ax3d.plot([cx, cx + axis_length * R[2, 0]], [cy, cy + axis_length * R[0, 0]], [cz, cz - axis_length * R[1, 0]], color='r', linewidth=1.5)
        ax3d.plot([cx, cx - axis_length * R[2, 1]], [cy, cy - axis_length * R[0, 1]], [cz, cz + axis_length * R[1, 1]], color='g', linewidth=1.5)
        ax3d.plot([cx, cx + axis_length * R[2, 2]], [cy, cy + axis_length * R[0, 2]], [cz, cz - axis_length * R[1, 2]], color='b', linewidth=1.5)

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


def marginal_covariances(window_frames, graph, result, output_dir):
    """Plots GTSAM 3D trajectory with marginal covariance ellipsoids on camera poses."""
    fig_cov = plt.figure(figsize=(8, 6))
    ax_cov = fig_cov.add_subplot(111, projection='3d')
    ax_cov.set_title("Plot Trajectory\nGTSAM Factor Graph State with Marginal Covariances\n", fontsize=11, fontweight='bold')

    try:
        marginals = gtsam.Marginals(graph, result)
        for f_id in window_frames:
            pose_key = symbol('c', f_id)
            gtsam_plot.plot_pose3_on_axes(ax_cov, result.atPose3(pose_key), axis_length=0.4, P=marginals.marginalCovariance(pose_key))
    except Exception as e:
        print(f"[Warning] Covariance layout fallback: {e}")
        for f_id in window_frames:
            gtsam_plot.plot_pose3_on_axes(ax_cov, result.atPose3(symbol('c', f_id)), axis_length=0.4)

    ax_cov.set_xlabel("X axis")
    ax_cov.set_ylabel("Y axis")
    ax_cov.set_zlabel("Z axis")
    ax_cov.view_init(elev=20, azim=-35)
    plt.savefig(os.path.join(output_dir, "task_5_3_marginal_covariances.png"), dpi=200, bbox_inches='tight')
    plt.close()


def bundle1_2D_full(window_frames, initial_cam_positions, cam_positions, lm_filtered, output_dir):
    """Plots top-down (X-Z) overview of camera trajectories and triangulated landmarks."""
    fig2d_full, ax2d_full = plt.subplots(figsize=(8, 8))

    if len(lm_filtered) > 0:
        ax2d_full.scatter(lm_filtered[:, 0], lm_filtered[:, 2], s=2, c='orange', alpha=0.6, label=f'Landmarks ({len(lm_filtered)})')

    ax2d_full.plot(initial_cam_positions[:, 0], initial_cam_positions[:, 2], 'b-+', alpha=0.6, label='Initial Cameras (PnP)')
    ax2d_full.plot(cam_positions[:, 0], cam_positions[:, 2], 'r-^', markersize=6, linewidth=1.5, label='Optimized Cameras (BA)')

    for i, f_id in enumerate(window_frames):
        ax2d_full.annotate(str(f_id), (cam_positions[i, 0], cam_positions[i, 2]), textcoords="offset points", xytext=(4, 4), fontsize=7, color='darkred')

    ax2d_full.set_title("Top-Down View (X-Z) of Bundle Window\nAll Cameras & Landmarks", fontsize=11, fontweight='bold')
    ax2d_full.set_xlabel("X [m]")
    ax2d_full.set_ylabel("Z (Forward) [m]")
    ax2d_full.set_xlim([-50, 50])
    ax2d_full.set_ylim([-10, 100])
    ax2d_full.grid(True, alpha=0.5)
    ax2d_full.legend(loc="upper right")
    plt.savefig(os.path.join(output_dir, "task_5_3_bundle1_2D_full.png"), dpi=150, bbox_inches='tight')
    plt.close()


def bundle1_2D_zoomed(initial_cam_positions, cam_positions, output_dir):
    """Plots zoomed 2D top-down trajectory comparison of local bundle optimization."""
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
    plt.savefig(os.path.join(output_dir, "task_5_3_bundle1_2D_zoomed.png"), dpi=150)
    plt.close()


def plot_q5_4_results(keyframes, global_keyframe_poses, all_points_global, output_dir="./outputs"):
    """Plots optimized bundle keyframe positions against ground truth trajectories."""
    gt_poses = read_ground_truth_poses()
    valid_keyframes = [kf for kf in keyframes if kf in global_keyframe_poses]

    estimated_positions = np.array([np.array(global_keyframe_poses[kf].translation()).reshape(3) for kf in valid_keyframes])
    gt_positions = np.array([camera_center(*gt_poses[kf]) for kf in valid_keyframes])
    points = np.array(all_points_global)

    plt.figure(figsize=(10, 8))
    if len(points) > 0:
        plt.scatter(points[:, 0], points[:, 2], s=1, alpha=0.25, label="Optimized 3D points")

    plt.plot(estimated_positions[:, 0], estimated_positions[:, 2], "bo-", markersize=3, label="Optimized keyframes")
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], "r--", linewidth=2, label="Ground truth keyframes")

    plt.title("5.4: Optimized Keyframe Trajectory vs Ground Truth")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "task_5_4_keyframes_vs_gt.png"), dpi=200)


def plot_keyframe_localization_error(keyframes, global_keyframe_poses, output_dir="./outputs"):
    """Plots Euclidean keyframe localization errors [m] relative to ground truth."""
    gt_poses = read_ground_truth_poses()
    errors, valid_keyframes = [], []

    for kf in keyframes:
        if kf not in global_keyframe_poses:
            continue
        est_pos = np.array(global_keyframe_poses[kf].translation()).reshape(3)
        gt_pos = camera_center(*gt_poses[kf])
        errors.append(np.linalg.norm(est_pos - gt_pos))
        valid_keyframes.append(kf)

    if not errors:
        print("No valid localization errors to plot.")
        return

    plt.figure(figsize=(12, 5))
    plt.plot(valid_keyframes, errors, marker="o", linewidth=1)
    plt.title("5.4: Keyframe Localization Error")
    plt.xlabel("Frame")
    plt.ylabel("Localization Error [m]")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "task_5_4_keyframe_error.png"), dpi=200)
    print(f"Mean keyframe localization error: {np.mean(errors):.3f} m")
    print(f"Max keyframe localization error: {np.max(errors):.3f} m")


def plot_pose_graph_trajectory(values, title, output_path):
    """Plots top-down 2D trajectory of pose graph state."""
    items = []
    for key in values.keys():
        sym = gtsam.Symbol(key)
        if sym.chr() == ord("c"):
            items.append((sym.index(), np.array(values.atPose3(key).translation()).reshape(3)))
    items.sort(key=lambda x: x[0])
    positions = np.array([x[1] for x in items])

    plt.figure(figsize=(10, 8))
    plt.plot(positions[:, 0], positions[:, 2], "bo-", markersize=3, linewidth=1.5)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


def plot_pose_graph_with_covariances(values, marginals, title, output_path, covariance_step=10, covariance_scale=1.0):
    """Visualizes 3D pose graph trajectory with marginal translation covariance ellipsoids."""
    items = []
    for key in values.keys():
        sym = gtsam.Symbol(key)
        if sym.chr() == ord("c"):
            items.append((sym.index(), np.array(values.atPose3(key).translation()).reshape(3)))
    items.sort(key=lambda x: x[0])
    frame_ids = [x[0] for x in items]
    positions = np.array([x[1] for x in items])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], linestyle="--", marker=".", markersize=3, linewidth=1.0, label="Optimized keyframes")

    for frame_id in frame_ids[::covariance_step]:
        key = symbol("c", frame_id)
        if not values.exists(key):
            continue
        p = np.array(values.atPose3(key).translation()).reshape(3)

        try:
            cov6 = marginals.marginalCovariance(key)
            cov_xyz = 0.5 * (cov6[3:6, 3:6] + cov6[3:6, 3:6].T)

            eigvals, eigvecs = np.linalg.eigh(cov_xyz)
            radii = np.minimum(covariance_scale * np.sqrt(np.maximum(eigvals, 0.0)), 20.0)

            u, v = np.linspace(0, 2 * np.pi, 18), np.linspace(0, np.pi, 9)
            xs = radii[0] * np.outer(np.cos(u), np.sin(v))
            ys = radii[1] * np.outer(np.sin(u), np.sin(v))
            zs = radii[2] * np.outer(np.ones_like(u), np.cos(v))

            ellipsoid = np.stack([xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)], axis=0)
            rotated = eigvecs @ ellipsoid

            ax.plot_wireframe(
                rotated[0, :].reshape(xs.shape) + p[0],
                rotated[1, :].reshape(ys.shape) + p[1],
                rotated[2, :].reshape(zs.shape) + p[2],
                linewidth=0.4, alpha=0.45
            )
        except Exception as e:
            print(f"[Warning] Could not plot covariance for frame {frame_id}: {e}")

    ax.set_title(title)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.grid(True)
    ax.legend()
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pose_graph_2d_ellipses(result, marginals, output_path, step=1, sigma_scale=20):
    """Renders top-down 2D keyframe trajectory overlaid with 2D marginal confidence ellipses."""
    items = []
    for key in result.keys():
        sym = gtsam.Symbol(key)
        if sym.chr() == ord("c"):
            items.append((sym.index(), np.array(result.atPose3(key).translation()).reshape(3)))
    items.sort(key=lambda x: x[0])
    frame_ids = [x[0] for x in items]
    positions = np.array([x[1] for x in items])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(positions[:, 0], positions[:, 2], label="optimised trajectory", color="#1f77b4", linewidth=1.2)
    ax.scatter(positions[0, 0], positions[0, 2], color="black", marker="s", s=80, label="c_0", zorder=5)

    for frame_id in frame_ids[::step]:
        key = symbol("c", frame_id)
        if not result.exists(key):
            continue
        p = np.array(result.atPose3(key).translation()).reshape(3)

        try:
            cov6 = marginals.marginalCovariance(key)
            cov_xz = cov6[3:6, 3:6][np.ix_([0, 2], [0, 2])]

            eigvals, eigvecs = np.linalg.eigh(cov_xz)
            eigvals = np.maximum(eigvals, 0.0)

            width = 2 * sigma_scale * np.sqrt(eigvals[0])
            height = 2 * sigma_scale * np.sqrt(eigvals[1])
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

            ax.add_patch(Ellipse(
                xy=(p[0], p[2]), width=width, height=height, angle=angle,
                edgecolor="red", facecolor="none", alpha=0.4, linewidth=0.8
            ))
        except Exception:
            pass

    ax.set_title("Q6.2: keyframes with marginal covariances", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z — forward (m)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")
    ax.axis("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_loop_candidates(keyframes, candidates, optimized_values, output_dir="."):
    """Plots trajectory with candidate loop closure shortcut connections overlaid."""
    kf_positions = {
        kf: (optimized_values.atPose3(symbol('c', kf)).translation()[0], optimized_values.atPose3(symbol('c', kf)).translation()[2])
        for kf in keyframes if optimized_values.exists(symbol('c', kf))
    }

    positions_array = np.array([kf_positions[kf] for kf in keyframes if kf in kf_positions])

    plt.figure(figsize=(12, 9))
    plt.plot(positions_array[:, 0], positions_array[:, 1], color='gray', linestyle='-', alpha=0.5, label='Estimated Trajectory')
    plt.scatter(positions_array[:, 0], positions_array[:, 1], color='black', s=5, alpha=0.3)

    for c_n, cands in candidates.items():
        if not cands or c_n not in kf_positions:
            continue
        x_n, z_n = kf_positions[c_n]
        plt.scatter(x_n, z_n, color='green', s=30, zorder=3)

        for cand in cands:
            c_i = cand["candidate_kf"]
            if c_i in kf_positions:
                x_i, z_i = kf_positions[c_i]
                plt.scatter(x_i, z_i, color='magenta', s=15, zorder=3)
                plt.plot([x_n, x_i], [z_n, z_i], color='red', linestyle='--', alpha=0.6, linewidth=1.0)

    plt.title("Q7.1: Detected Loop Closure Candidates on ESTIMATED Trajectory", fontsize=12, fontweight='bold')
    plt.xlabel("X (Width) [m]")
    plt.ylabel("Z (Depth / Forward) [m]")
    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "task_7_1_loop_candidates_trajectory.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pose_graph_vs_ground_truth(no_loop_values, loop_values, output_dir="."):
    """Plots optimized trajectories with and without loop closures against ground truth."""
    items_no, items_lc = [], []
    for key in no_loop_values.keys():
        sym = gtsam.Symbol(key)
        if sym.chr() == ord("c"):
            items_no.append((sym.index(), np.array(no_loop_values.atPose3(key).translation()).reshape(3)))
    for key in loop_values.keys():
        sym = gtsam.Symbol(key)
        if sym.chr() == ord("c"):
            items_lc.append((sym.index(), np.array(loop_values.atPose3(key).translation()).reshape(3)))

    items_no.sort(key=lambda x: x[0])
    items_lc.sort(key=lambda x: x[0])

    ids_no = [x[0] for x in items_no]
    pos_no = np.array([x[1] for x in items_no])
    pos_lc = np.array([x[1] for x in items_lc])

    gt = read_ground_truth_poses()
    gt_pts = np.array([camera_center(*gt[f]) for f in ids_no if f < len(gt)])

    plt.figure(figsize=(10, 8))
    plt.plot(pos_no[:, 0], pos_no[:, 2], "o-", markersize=3, linewidth=1, label="without loop closures")
    plt.plot(pos_lc[:, 0], pos_lc[:, 2], "o-", markersize=3, linewidth=1, label="with loop closures")
    if len(gt_pts) == len(pos_no):
        plt.plot(gt_pts[:, 0], gt_pts[:, 2], "--", linewidth=2, label="ground truth")

    plt.title("Q7.5: Pose Graph vs Ground Truth")
    plt.xlabel("X [m]")
    plt.ylabel("Z [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_7_5_pose_graph_vs_ground_truth.png"), dpi=200)
    plt.close()


def plot_absolute_location_error(no_loop_values, loop_values, output_dir="."):
    """Plots keyframe position errors relative to ground truth before and after loop closures."""
    items_no, items_lc = [], []
    for key in no_loop_values.keys():
        sym = gtsam.Symbol(key)
        if sym.chr() == ord("c"):
            items_no.append((sym.index(), np.array(no_loop_values.atPose3(key).translation()).reshape(3)))
    for key in loop_values.keys():
        sym = gtsam.Symbol(key)
        if sym.chr() == ord("c"):
            items_lc.append((sym.index(), np.array(loop_values.atPose3(key).translation()).reshape(3)))

    items_no.sort(key=lambda x: x[0])
    items_lc.sort(key=lambda x: x[0])

    ids_no = [x[0] for x in items_no]
    ids_lc = [x[0] for x in items_lc]
    pos_no = np.array([x[1] for x in items_no])
    pos_lc = np.array([x[1] for x in items_lc])

    gt = read_ground_truth_poses()
    gt_no = np.array([camera_center(*gt[f]) for f in ids_no if f < len(gt)])
    gt_lc = np.array([camera_center(*gt[f]) for f in ids_lc if f < len(gt)])

    plt.figure(figsize=(12, 5))
    plt.plot(ids_no[:len(gt_no)], np.linalg.norm(pos_no[:len(gt_no)] - gt_no, axis=1), marker="o", linewidth=1, label="without loop closures")
    plt.plot(ids_lc[:len(gt_lc)], np.linalg.norm(pos_lc[:len(gt_lc)] - gt_lc, axis=1), marker="o", linewidth=1, label="with loop closures")
    plt.title("Q7.5: Absolute Location Error")
    plt.xlabel("Keyframe")
    plt.ylabel("Position error [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_7_5_absolute_location_error.png"), dpi=200)
    plt.close()


def plot_location_uncertainty_size(no_loop_values, no_loop_marginals, loop_values, loop_marginals, output_dir="."):
    """Plots marginal position uncertainty area over keyframes before and after loop closures."""
    def _uncertainty_trace(values, marginals):
        items = []
        for key in values.keys():
            sym = gtsam.Symbol(key)
            if sym.chr() == ord("c"):
                items.append((sym.index(), np.array(values.atPose3(key).translation()).reshape(3)))
        items.sort(key=lambda x: x[0])
        ids = [x[0] for x in items]

        sizes, good_ids = [], []
        for f in ids:
            try:
                cov6 = marginals.marginalCovariance(symbol("c", int(f)))
                cov_xz = cov6[np.ix_([3, 5], [3, 5])]
                sizes.append(float(np.sqrt(max(np.linalg.det(cov_xz), 0.0))))
                good_ids.append(f)
            except Exception:
                pass
        return good_ids, np.array(sizes)

    ids_no, unc_no = _uncertainty_trace(no_loop_values, no_loop_marginals)
    ids_lc, unc_lc = _uncertainty_trace(loop_values, loop_marginals)

    plt.figure(figsize=(12, 5))
    plt.plot(ids_no, unc_no, marker="o", linewidth=1, label="without loop closures")
    plt.plot(ids_lc, unc_lc, marker="o", linewidth=1, label="with loop closures")
    plt.title("Q7.5: Location Uncertainty Size")
    plt.xlabel("Keyframe")
    plt.ylabel(r"$\sqrt{\det(\Sigma_{xz})}$ [m²]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_7_5_location_uncertainty_size.png"), dpi=200)
    plt.close()


def plot_matches_per_frame(matches_per_frame, output_dir="."):
    """Displays the number of matches per frame with mean and variance bands."""
    frames = np.arange(len(matches_per_frame))
    mean_matches = np.mean(matches_per_frame)
    plt.figure(figsize=(14, 5))
    plt.plot(frames, matches_per_frame, linewidth=1.5, label="matches per frame")
    plt.axhline(mean_matches, color="r", linestyle="--", linewidth=1, label=f"mean = {mean_matches:.1f}")
    plt.xlabel("Frame")
    plt.ylabel("Number of Matches")
    plt.grid(True)
    plt.legend()
    plt.title("Task 4.8: Matches per Frame")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_4_8_matches_per_frame.png"), dpi=200)
    plt.close()


def plot_absolute_location_error_components(frame_ids, err_x, err_y, err_z, err_norm, title, output_path):
    """Plots absolute location error components (X, Y, Z, norm) on one figure."""
    plt.figure(figsize=(12, 5))
    plt.plot(frame_ids, err_x, linewidth=1, label="X error (m)", alpha=0.8)
    plt.plot(frame_ids, err_y, linewidth=1, label="Y error (m)", alpha=0.8)
    plt.plot(frame_ids, err_z, linewidth=1, label="Z error (m)", alpha=0.8)
    plt.plot(frame_ids, err_norm, linewidth=1.5, label="Total error norm (m)", alpha=0.9)
    plt.xlabel("Frame")
    plt.ylabel("Error (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_absolute_angle_error(frame_ids, err_angle, title, output_path):
    """Plots absolute angle error in degrees."""
    plt.figure(figsize=(12, 5))
    plt.plot(frame_ids, err_angle, linewidth=1, label="Angle error (deg)")
    plt.xlabel("Frame")
    plt.ylabel("Angle Error (degrees)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_bundle_optimization_error(window_start_ids, mean_initial_errors, mean_final_errors, output_dir="./outputs"):
    """Plots mean factor error before and after bundle adjustment per window."""
    plt.figure(figsize=(12, 5))
    plt.plot(window_start_ids, mean_initial_errors, marker="o", linewidth=1.5, label="Initial Error", color="orange")
    plt.plot(window_start_ids, mean_final_errors, marker="o", linewidth=1.5, label="Optimized Error", color="blue")
    plt.xlabel("Bundle Starting at Frame idx")
    plt.ylabel("Mean Factor Error")
    plt.title("Task 5.4: Bundle Optimization Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_5_4_mean_factor_error.png"), dpi=200)
    plt.close()


def plot_bundle_projection_error(window_start_ids, median_initial_errors, median_final_errors, output_dir="./outputs"):
    """Plots median projection error before and after bundle adjustment per window."""
    plt.figure(figsize=(12, 5))
    plt.plot(window_start_ids, median_initial_errors, marker="o", linewidth=1.5, label="Initial Error", color="orange")
    plt.plot(window_start_ids, median_final_errors, marker="o", linewidth=1.5, label="Optimized Error", color="blue")
    plt.xlabel("Bundle Starting at Frame idx")
    plt.ylabel("Median Projection Error (pixels)")
    plt.title("Task 5.4: Bundle Projection Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_5_4_median_projection_error.png"), dpi=200)
    plt.close()


def plot_projection_error_vs_distance(distances, median_errors, title, output_path):
    """Plots median projection error vs distance from reference frame."""
    plt.figure(figsize=(12, 5))
    plt.plot(distances, median_errors, marker="o", linewidth=1.5)
    plt.xlabel("Distance from Reference Frame")
    plt.ylabel("Median Projection Error (pixels)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_relative_error_comparison(edge_ids, bundle_errors, pnp_errors, ylabel, title, output_path):
    """Plots bundle vs PnP relative pose errors on the same figure."""
    plt.figure(figsize=(12, 5))
    plt.plot(edge_ids, bundle_errors, marker="o", linewidth=1, label="Bundle", alpha=0.8)
    plt.plot(edge_ids, pnp_errors, marker="s", linewidth=1, label="PnP", alpha=0.8)
    plt.xlabel("Edge Index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_full_trajectory_comparison(pnp_positions, bundle_ids, bundle_positions, pg_ids, pg_positions, gt_positions, output_dir="."):
    """Plots overlaid trajectories (PnP, Bundle, Pose-Graph+LC, GT) bird's-eye X-Z view."""
    plt.figure(figsize=(12, 8))
    plt.plot(gt_positions[:, 0], gt_positions[:, 2], "g--", linewidth=2, label="Ground Truth", alpha=0.7)
    plt.plot(pnp_positions[:, 0], pnp_positions[:, 2], linewidth=1.5, label="PnP", alpha=0.7)
    if len(bundle_positions) > 0:
        plt.scatter(bundle_positions[:, 0], bundle_positions[:, 2], marker="o", s=30, label="Bundle Keyframes", alpha=0.6)
    if len(pg_positions) > 0:
        plt.scatter(pg_positions[:, 0], pg_positions[:, 2], marker="^", s=30, label="Pose-Graph+LC Keyframes", alpha=0.6)
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title("Task Summary: Full Trajectory Comparison (Bird's Eye)")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_summary_full_trajectory_comparison.png"), dpi=200)
    plt.close()


def plot_kitti_sequence_error(segment_results, ylabel, title, output_path):
    """Plots KITTI sequence error for segments of different lengths (100, 400, 800 frames)."""
    # Early abort if no data at all
    total_data_points = sum(len(v) if isinstance(v, list) else 0 for v in segment_results.values())
    if total_data_points == 0:
        return

    plt.figure(figsize=(12, 5))
    colors = {100: "blue", 400: "orange", 800: "red"}
    has_data = False
    for seg_len in sorted(segment_results.keys()):
        errs = segment_results[seg_len]
        if errs and len(errs) > 0:
            # Filter out NaN values for plotting
            valid_indices = [i for i, err in enumerate(errs) if np.isfinite(err)]
            if valid_indices:
                valid_errs = [errs[i] for i in valid_indices]
                plt.plot(valid_indices, valid_errs, marker="o", linewidth=1, label=f"Segment Length {seg_len}", color=colors.get(seg_len, "gray"), alpha=0.8)
                has_data = True
    if has_data:
        plt.xlabel("Segment Index")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
    plt.close()


def plot_loop_closure_match_stats(loop_measurements, output_dir="."):
    """Plots loop closure match count and inlier ratio per successful loop."""
    labels = [f"{m['start_kf']}-{m['end_kf']}" for m in loop_measurements]
    match_counts = [m["good_matches_count"] for m in loop_measurements]
    inlier_ratios = [m["inlier_ratio"] * 100 for m in loop_measurements]
    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.bar(x, match_counts, width, label="Good Matches")
    ax1.set_ylabel("Match Count")
    ax1.set_title("Task 7.5: Loop Closure Match Statistics")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.bar(x, inlier_ratios, width, label="Inlier Ratio")
    ax2.set_ylabel("Inlier Ratio (%)")
    ax2.set_xlabel("Loop Closure Pair")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_7_5_loop_closure_match_stats.png"), dpi=200)
    plt.close()


def plot_angle_uncertainty_size(no_loop_values, no_loop_marginals, loop_values, loop_marginals, output_dir="."):
    """Plots angle uncertainty (rotation block) for pose graph with and without loop closures."""
    def _angle_uncertainty_trace(values, marginals):
        ids, sizes = [], []
        for key in values.keys():
            sym = gtsam.Symbol(key)
            if sym.chr() == ord("c"):
                try:
                    cov6 = marginals.marginalCovariance(key)
                    cov_rot = cov6[np.ix_([0, 1, 2], [0, 1, 2])]
                    det_val = max(np.linalg.det(cov_rot), 0.0)
                    size_rad = np.power(det_val, 1.0 / 3.0)
                    size_deg = float(np.degrees(size_rad))
                    sizes.append(size_deg)
                    ids.append(sym.index())
                except Exception:
                    pass
        sort_idx = np.argsort(ids)
        return [ids[i] for i in sort_idx], np.array([sizes[i] for i in sort_idx])

    ids_no, unc_no = _angle_uncertainty_trace(no_loop_values, no_loop_marginals)
    ids_lc, unc_lc = _angle_uncertainty_trace(loop_values, loop_marginals)

    plt.figure(figsize=(12, 5))
    plt.plot(ids_no, unc_no, marker="o", linewidth=1, label="without loop closures")
    plt.plot(ids_lc, unc_lc, marker="o", linewidth=1, label="with loop closures")
    plt.title("Task 7.5: Angle Uncertainty Size")
    plt.xlabel("Keyframe")
    plt.ylabel(r"$(|\Sigma_{rot}|)^{1/3}$ [degrees]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "task_7_5_angle_uncertainty_size.png"), dpi=200)
    plt.close()