import cv2
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path(r'C:\Users\Nir\PycharmProjects\VAN_ex\dataset\dataset_2026\sequences\00')


def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    path1 = str(DATA_PATH / 'image_0' / img_name)
    path2 = str(DATA_PATH / 'image_1' / img_name)

    img1 = cv2.imread(path1, 0)
    img2 = cv2.imread(path2, 0)
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

    plt.show()


if __name__ == '__main__':
    main()