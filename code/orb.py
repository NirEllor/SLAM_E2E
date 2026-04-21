import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

# 1. Get the directory where orb.py is located (.../VAN_ex/code)
script_dir = Path(__file__).parent

# 2. Go one level up to the project root (.../VAN_ex) and then into the dataset
DATA_PATH = script_dir.parent / 'dataset' / 'dataset_2026' / 'sequences' / '00'


def read_images(idx):
    # Format the image name with leading zeros
    img_name = f'{idx:06d}.png'

    # Pathlib handles slashes automatically for any OS
    path1 = DATA_PATH / 'image_0' / img_name
    path2 = DATA_PATH / 'image_1' / img_name

    # cv2.imread usually accepts Path objects directly,
    # but str() ensures compatibility with older versions
    img1 = cv2.imread(str(path1), 0)
    img2 = cv2.imread(str(path2), 0)

    return img1, img2


def detect_orb(img, n_features=500):
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def visualize_side_by_side(img_left, img_right, title="Comparison", cmap=None):
    """Helper to plot two images side by side."""
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f"{title} - Left")
    plt.imshow(img_left, cmap=cmap)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"{title} - Right")
    plt.imshow(img_right, cmap=cmap)
    plt.axis('off')

    plt.tight_layout()


def print_descriptor_info(des1, des2):
    """Handles the printing requirements for task 1.2."""
    if des1 is not None and len(des1) >= 2 and des2 is not None and len(des2) >= 2:
        print("\n--- Image 1: First Two Descriptors ---")
        print(des1[0], des1[1], sep="\n")
        print("\n--- Image 2: First Two Descriptors ---")
        print(des2[0], des2[1], sep="\n")
    else:
        print("Warning: Could not extract enough descriptors.")

def match_descriptors(des1, des2):
    """
    Matches each descriptor in the left image to its closest descriptor in the right image using BFMatcher.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # crossCheck=False => as written in the instruction: for each descriptor in the left image, the closest feature in the right image.
    matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda m: m.distance)
    return matches

def draw_random_matches(img1, kp1, img2, kp2, matches, num_matches=20):
    """
    Draws random matches between the left and right images.
    """
    if len(matches) < num_matches:
        num_matches = len(matches)

    sampled_matches = random.sample(matches, num_matches)

    matched_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        sampled_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 8))
    plt.title(f"{num_matches} Random Matches")
    plt.imshow(matched_img_rgb)
    plt.axis('off')
    plt.tight_layout()


def ratio_test_match(des1, des2, ratio=0.75):
    """
    Performs KNN matching and applies Lowe's ratio test.
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    knn_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    rejected = []

    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
        else:
            rejected.append(m)

    return good_matches, rejected

def draw_matches(img1, kp1, img2, kp2, matches, title="Matches", num=20):
    if len(matches) > num:
        matches = random.sample(matches, num)

    img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 8))
    plt.title(title)
    plt.imshow(img_rgb)
    plt.axis('off')

def show_multiple_rejected(img1, kp1, img2, kp2, rejected, num=3):
    selected = random.sample(rejected, min(num, len(rejected)))

    img1_copy = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2_copy = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    for match in selected:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt

        cv2.circle(img1_copy, (int(pt1[0]), int(pt1[1])), 6, (255, 0, 0), -1)
        cv2.circle(img2_copy, (int(pt2[0]), int(pt2[1])), 6, (255, 0, 0), -1)

    visualize_side_by_side(img1_copy, img2_copy, title="Rejected Matches (Circled)")


def main():
    # 1. Setup and Data Loading
    idx = 0
    img1, img2 = read_images(idx)

    # 2. Processing (Detection & Extraction)
    kp1, des1 = detect_orb(img1)
    kp2, des2 = detect_orb(img2)

    # 3. Validation & Reporting (Task 1.2)
    print(f"Keypoints found: Image1: {len(kp1)}, Image2: {len(kp2)}")
    print_descriptor_info(des1, des2)
    assert len(kp1) >= 500 and len(kp2) >= 500, "Insufficient keypoints detected!"

    # 4. Visualization (Task 1.1)
    # Plot Original Images
    visualize_side_by_side(img1, img2, title="Original Images", cmap='gray')

    # Plot Images with Keypoints
    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Convert for Matplotlib
    img1_kp_rgb = cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB)
    img2_kp_rgb = cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB)

    visualize_side_by_side(img1_kp_rgb, img2_kp_rgb, title="ORB Keypoints")
    matches = match_descriptors(des1, des2)
    print(f"Total matches found: {len(matches)}")
    draw_random_matches(img1, kp1, img2, kp2, matches, num_matches=20)
    good_matches, rejected = ratio_test_match(des1, des2, ratio=0.6)
    print(f"Good matches: {len(good_matches)}")
    print(f"Rejected matches: {len(rejected)}")
    draw_matches(img1, kp1, img2, kp2, good_matches, title="Good Matches (After Ratio Test)")
    show_multiple_rejected(img1, kp1, img2, kp2, rejected)
    # draw_matches(img1, kp1, img2, kp2, rejected, title="Rejected Matches (After Ratio Test)")
    plt.show()


if __name__ == '__main__':
    main()