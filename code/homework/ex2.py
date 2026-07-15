# ex2.py
import van_utils as lib
import matplotlib.pyplot as plt
from ex1 import *
def q2_1(kp1, kp2, matches):
    """2.1: Analyze deviations from the rectified stereo pattern."""
    print("\n--- Task 2.1: Rectified Stereo Pattern ---")

    deviations = lib.compute_rectified_stereo_deviations(kp1, kp2, matches)

    lib.plot_deviation_histogram(deviations)
    lib.print_large_deviation_percentage(deviations, threshold=2)

    return deviations


def q2_2(img1, img2, kp1, kp2, matches, threshold=2):
    """2.2: Reject matches using the rectified stereo pattern."""
    print("\n--- Task 2.2: Reject Matches by Rectified Stereo Constraint ---")

    inliers, outliers = lib.split_matches_by_rectified_pattern(
        kp1, kp2, matches, threshold=threshold
    )

    print(f"Inliers: {len(inliers)}")
    print(f"Outliers: {len(outliers)}")
    print(f"Discarded matches: {len(outliers)}")

    lib.draw_inliers_outliers(img1, kp1, img2, kp2, inliers, outliers)

    return inliers, outliers

def q2_3(kp1, kp2, matches):
    """2.3: Linear least-squares triangulation and OpenCV comparison."""
    print("\n--- Task 2.3: Triangulation ---")

    k, m1, m2 = lib.read_cameras()

    points1, points2 = lib.get_matched_points(kp1, kp2, matches)

    points_3d_linear = lib.triangulate_points_linear(points1, points2, m1, m2)

    lib.plot_3d_points(points_3d_linear, title="2.3: Linear Least-Squares Triangulation")

    points_3d_cv = lib.triangulate_points_opencv(points1, points2, m1, m2)

    lib.plot_3d_points(points_3d_cv, title="2.3: OpenCV triangulatePoints")

    median_dist = lib.median_3d_distance(points_3d_linear, points_3d_cv)

    print(f"Median distance between linear and OpenCV 3D points: {median_dist}")

    return points_3d_linear, points_3d_cv


def q2():
    """Exercise 2: Orchestrates rectified stereo filtering and triangulation pipeline."""
    print("========== Running Exercise 2 Pipeline ==========")
    
    # Load foundational data components from frame 0
    img1, img2 = lib.read_images(0)
    kp1, des1 = lib.get_orb_features(img1)
    kp2, des2 = lib.get_orb_features(img2)
    matches = q1_3(img1, img2, kp1, kp2, des1, des2)
    
    # 2.1: Analyze & chart deviations from rectified epipolar lines
    # (Enforced BEFORE spatial pipeline extensions)
    deviations = q2_1(kp1, kp2, matches)
    
    # 2.2: Isolate inliers and strip geometric outliers
    inliers, outliers = q2_2(img1, img2, kp1, kp2, matches, threshold=2)
    
    # 2.3: Triangulate 3D coordinates and contrast against OpenCV
    points_3d_linear, points_3d_cv = q2_3(kp1, kp2, inliers)
    
    plt.show()

def main():
    """Entry point for Exercise 2."""
    q2()

if __name__ == '__main__':
    main()