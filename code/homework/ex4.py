# ex4.py
import van_utils as lib
import matplotlib.pyplot as plt
import numpy as np

# The total number of images to process across the sequence loop
NUM_FRAMES = lib.get_num_frames()

def q4_1(nun_frames=NUM_FRAMES):
    """4.1: Builds and caches tracking database from frame sequence."""
    return lib.build_data(nun_frames)

def q4_2(db):
    """
    4.2: Computes and prints descriptive tracking statistics.
    """
    stats = lib.compute_tracking_statistics(db)
    lib.print_tracking_statistics(stats)
    return db


def q4_3(db):
    """
    4.3: Visualizes localized crop windows of a specific multi-frame feature track.
    """
    track_id = lib.select_track_by_min_length(db, min_length=6)

    print(f"\n--- Task 4.3: Track Patch Visualization ---")
    print(f"Selected Track ID: {track_id}")
    print(f"Track Continuity Lifespan: {len(db.frames(track_id))} frames")
    print(f"Spanned Frame Nodes: {db.frames(track_id)}")

    lib.plot_track_observations(db, track_id, crop_size=20)


def q4_4(db):
    """
    4.4: Computes and charts track connectivity linkages between successive frames.
    """
    connectivity = lib.compute_connectivity(db)
    print(f"Mean Track Continuity Connectivity Coefficient: {np.mean(connectivity):.2f}")
    lib.plot_connectivity(connectivity)


def q4_5(db):
    """
    4.5: Charts the proportion of verification inliers across the timeline.
    """
    print(f"\n--- Task 4.5: RANSAC Inlier Proportion Analysis ---")
    mean_inliers = np.mean(db.inlier_percentages)
    print(f"Mean RANSAC Inlier Ratio: {mean_inliers:.2f}%")
    lib.plot_inlier_percentage(db.inlier_percentages)


def q4_6(db):
    """
    4.6: Generates a track lifespan frequency distribution histogram.
    """
    lib.plot_track_length_histogram(db, min_length=2)

def q4_7(db):
    """
    4.7: Analyzes spatial error baseline drift metrics over long duration paths.
    Fixes the geometric reference system by rotating the stereo baseline vector
    into the camera's local coordinate frame for both triangulation and projection.
    """
    print("\n--- Task 4.7: Reprojection Error Analysis ---")

    track_id = lib.select_long_track(db, min_length=10)
    frames = db.frames(track_id)
    print(f"Selected Validation Track ID: {track_id} | Lifespan: {len(frames)} | Nodes: {frames}")

    K, m_left0, m_right0 = lib.read_cameras()

    first_frame_id = frames[0]
    X_world = lib.triangulate_track_reference_point(db, track_id, first_frame_id, K, m_left0, m_right0)

    left_errors, right_errors = lib.compute_track_reprojection_errors(db, track_id, frames, X_world, K, m_left0, m_right0)

    lib.plot_track_reprojection_error_analysis(frames, left_errors, right_errors)

def q4():
    """Exercise 4: Orchestrates tracking database construction and analysis pipeline."""
    print("========== Running Exercise 4 Execution Script ==========")
    
    # Trigger 4.1 sequence ingestion loop
    db = q4_1()
    
    # Trigger analytics pipeline tasks sequentially
    q4_2(db)
    q4_3(db)
    q4_4(db)
    q4_5(db)
    q4_6(db)
    q4_7(db)
    
    plt.show()


def main():
    """Entry point for Exercise 4."""
    q4()


if __name__ == '__main__':
    main()