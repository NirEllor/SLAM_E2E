import matplotlib.pyplot as plt
from pathlib import Path
import camera_lib as lib  # Import your library

# Update this to your local path
DATA_PATH = Path(r'C:\Hebrew University\computer scince\second year second sem\slam\VAN_ex\dataset\dataset_2026\sequences\00')

def q1_1_to_1_2():
    """Detect, extract, and print descriptor info[cite: 27, 39]."""
    img1, img2 = lib.read_images(DATA_PATH, 0)
    kp1, des1 = lib.detect_and_extract(img1)
    kp2, des2 = lib.detect_and_extract(img2)

    print(f"Task 1.2: Image 1 First Descriptor: {des1[0]}")
    # Add your visualization code here
    return img1, kp1, des1, img2, kp2, des2


def q1_3():
    """Perform initial matching[cite: 40, 43]."""
    img1, kp1, des1, img2, kp2, des2 = q1_1_to_1_2()
    matches = lib.match_features(des1, des2)
    print(f"Task 1.3: Total matches: {len(matches)}")
    # Add code to draw 20 random matches here


def q1_4():
    """Apply ratio test and report results[cite: 46, 48]."""
    img1, kp1, des1, img2, kp2, des2 = q1_1_to_1_2()
    ratio = 0.6
    good, rejected = lib.apply_ratio_test(des1, des2, ratio=ratio)
    print(f"Task 1.4: Ratio used: {ratio}")
    print(f"Matches discarded: {len(rejected)}")
    # Add code to show a 'correct' match that failed


def main():
    """Runs all needed functions."""
    q1_1_to_1_2()
    q1_3()
    q1_4()
    plt.show()


if __name__ == "__main__":
    main()