# ex1.py
import van_utils as lib
import matplotlib.pyplot as plt
import cv2
import random

#--------------------------------------ex1---------------------------------------------------------
def q1_1(idx=0, plot = True):
    """1.1: Detect & Present Keypoints."""
    img1, img2 = lib.read_images(idx)
    kp1, des1 = lib.get_orb_features(img1)
    kp2, des2 = lib.get_orb_features(img2)

    assert len(kp1) >= 500 and len(kp2) >= 500, "Insufficient keypoints!"

    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))
    lib.plot_stereo_side_by_side(img1_kp, img2_kp, "ORB Keypoints")

    return img1, img2, kp1, kp2, des1, des2


def q1_2(des1):
    """1.2: Calculate & Print Descriptors."""
    print("\n--- Task 1.2: Descriptor Info ---")
    if des1 is not None and len(des1) >= 2:
        print("Image 1 - Descriptor 0:", des1[0])
        print("Image 1 - Descriptor 1:", des1[1])


def q1_3(img1, img2, kp1, kp2, des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(des1, des2)

    lib.draw_matches_custom(img1, kp1, img2, kp2, matches, "1.3: 20 Random Raw Matches")
    return matches


def q1_4(img1, img2, kp1, kp2, des1, des2):
    """1.4: Significance (Ratio) Test."""
    ratio_value = 0.7
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    rejected_matches = []

    for m, n in knn_matches:
        if m.distance < ratio_value * n.distance:
            good_matches.append(m)
        else:
            rejected_matches.append(m)

    # 1. Output 20 resulting matches
    lib.draw_matches_custom(img1, kp1, img2, kp2, good_matches, f"1.4: Good Matches (Ratio={ratio_value})")

    # 2. Statistics
    print("\n--- Task 1.4: Significance Test ---")
    print(f"Ratio Value Used: {ratio_value}")
    print(f"Matches Discarded: {len(rejected_matches)}")

    # 3. Present a failed match (rejected but visually correct)
    if rejected_matches:
        # Pick one rejected match to display
        fail = random.choice(rejected_matches)
        img1_fail = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        img2_fail = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

        pt1 = tuple(map(int, kp1[fail.queryIdx].pt))
        pt2 = tuple(map(int, kp2[fail.trainIdx].pt))

        cv2.circle(img1_fail, pt1, 10, (255, 0, 0), -1)
        cv2.circle(img2_fail, pt2, 10, (255, 0, 0), -1)
        lib.plot_stereo_side_by_side(img1_fail, img2_fail, "1.4: Failed Significance Test Match (Rejected Match Example"
                                                           ")")

def q1():
    print("========== Running Exercise 1 Pipeline ==========")
    
    # 1.1: Detect and show keypoints
    img1, img2, kp1, kp2, des1, des2 = q1_1(idx=0)
    
    # 1.2: Print sample descriptor information
    q1_2(des1)
    
    # 1.3: Run standard crosscheck matching
    matches = q1_3(img1, img2, kp1, kp2, des1, des2)
    
    # 1.4: Run Lowe Significance Ratio Test validation
    q1_4(img1, img2, kp1, kp2, des1, des2)
    
    plt.show()

def main():
    q1()

if __name__ == '__main__':
    main()