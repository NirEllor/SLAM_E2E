import os

# Ensure output directory exists
OUTPUT_DIR = "project_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


"""
ex1.py
Exercise 1: Feature Detection, Description, and Matching Pipeline.
"""

import random
import cv2
import matplotlib.pyplot as plt

# Import from utils package
from utils.geometry import read_images, get_orb_features
from utils.visualization import plot_stereo_side_by_side, draw_matches_custom


def detect_and_visualize_orb_features(frame_index=0, plot=True):
    """Detects ORB keypoints on a stereo image pair and displays them side-by-side."""
    img_left, img_right = read_images(frame_index)
    kp_left, des_left = get_orb_features(img_left)
    kp_right, des_right = get_orb_features(img_right)

    assert len(kp_left) >= 500 and len(kp_right) >= 500, "Insufficient keypoints detected!"

    if plot:
        img_left_kp = cv2.drawKeypoints(img_left, kp_left, None, color=(0, 255, 0))
        img_right_kp = cv2.drawKeypoints(img_right, kp_right, None, color=(0, 255, 0))
        plot_stereo_side_by_side(img_left_kp, img_right_kp, "ORB Keypoints")

    return img_left, img_right, kp_left, kp_right, des_left, des_right


def inspect_feature_descriptors(des_left):
    """Prints raw feature descriptor values to console for inspection."""
    print("\n--- Task 1.2: Descriptor Inspection ---")
    if des_left is not None and len(des_left) >= 2:
        print("Left Image - Descriptor 0:", des_left[0])
        print("Left Image - Descriptor 1:", des_left[1])


def compute_raw_bruteforce_matches(img_left, img_right, kp_left, kp_right, des_left, des_right):
    """Computes raw Hamming brute-force matches between left and right descriptors."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des_left, des_right)

    draw_matches_custom(img_left, kp_left, img_right, kp_right, matches, "1.3: Random Raw Matches")
    return matches


def apply_lowe_ratio_test(img_left, img_right, kp_left, kp_right, des_left, des_right, ratio_threshold=0.7):
    """Filters ambiguous matches using Lowe's ratio test on k-Nearest Neighbors matches."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des_left, des_right, k=2)

    good_matches, rejected_matches = [], []
    for m, n in knn_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
        else:
            rejected_matches.append(m)

    # Visualize filtered matches
    draw_matches_custom(img_left, kp_left, img_right, kp_right, good_matches, f"1.4: Good Matches (Ratio={ratio_threshold})")

    # Print summary metrics
    print("\n--- Task 1.4: Lowe's Ratio Test Summary ---")
    print(f"Ratio Threshold: {ratio_threshold}")
    print(f"Total Valid Matches: {len(good_matches)}")
    print(f"Total Discarded Matches: {len(rejected_matches)}")

    # Show a rejected match sample if available
    if rejected_matches:
        fail = random.choice(rejected_matches)
        draw_matches_custom(
            img_left, kp_left, img_right, kp_right, [fail],
            "1.4: Failed Ratio Test Match Example", num=1
        )


def run_feature_detection_and_matching_pipeline(frame_index=0):
    """Executes the complete Exercise 1 pipeline sequentially using descriptive function calls."""
    print("==================================================")
    print("  EXERCISE 1: FEATURE DETECTION & MATCHING       ")
    print("==================================================")

    # 1. Detect keypoints
    img_left, img_right, kp_left, kp_right, des_left, des_right = detect_and_visualize_orb_features(frame_index)

    # 2. Inspect descriptor structure
    inspect_feature_descriptors(des_left)

    # 3. Compute raw matches
    compute_raw_bruteforce_matches(img_left, img_right, kp_left, kp_right, des_left, des_right)

    # 4. Filter matches with Lowe's Ratio Test
    apply_lowe_ratio_test(img_left, img_right, kp_left, kp_right, des_left, des_right)


def main():
    print("==================================================")
    print("  VISUAL AERIAL NAVIGATION (VAN) - FULL PIPELINE  ")
    print("==================================================\n")

    # --- Exercise 1: Feature Extraction & Matching ---
    print("\n[Stage 1/7] Running Feature Detection and Matching...")
    run_feature_detection_and_matching_pipeline(frame_index=0)