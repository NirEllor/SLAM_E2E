import cv2

def read_images(data_path, idx):
    """Reads stereo pair for a given index[cite: 33]."""
    img_name = '{:06d}.png'.format(idx)
    path1 = str(data_path / 'image_0' / img_name)
    path2 = str(data_path / 'image_1' / img_name)
    return cv2.imread(path1, 0), cv2.imread(path2, 0)

def detect_and_extract(img, n_features=500):
    """Detects keypoints and extracts descriptors[cite: 27, 38]."""
    orb = cv2.ORB_create(nfeatures=n_features)
    return orb.detectAndCompute(img, None)

def match_features(des1, des2):
    """Brute-force matching between two sets of descriptors[cite: 40]."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return bf.match(des1, des2)

def apply_ratio_test(des1, des2, ratio=0.75):
    """Applies significance test (Lowe's ratio test)[cite: 46, 52]."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good, rejected = [], []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good.append(m)
        else:
            rejected.append(m)
    return good, rejected