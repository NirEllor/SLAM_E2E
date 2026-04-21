import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'dataset' / 'dataset_2026' / 'sequences' / '00'

def read_images(idx):
    img_name = f'{idx:06d}.png'
    p1, p2 = DATA_PATH / 'image_0' / img_name, DATA_PATH / 'image_1' / img_name
    img1, img2 = cv2.imread(str(p1), 0), cv2.imread(str(p2), 0)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Could not find images at {DATA_PATH}")
    return img1, img2

def get_orb_features(img, n_features=1000):
    orb = cv2.ORB_create(nfeatures=n_features)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des

def plot_stereo_side_by_side(img_l, img_r, title="Stereo Pair"):
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1); plt.imshow(img_l, cmap='gray'); plt.title(f"{title} - Left"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(img_r, cmap='gray'); plt.title(f"{title} - Right"); plt.axis('off')
    plt.tight_layout()

def draw_matches_custom(img1, kp1, img2, kp2, matches, title, num=20):
    if len(matches) > num:
        matches = random.sample(matches, num)
    out_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 7))
    plt.title(title)
    plt.imshow(out_img)
    plt.axis('off')