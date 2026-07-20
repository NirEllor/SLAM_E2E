#!/usr/bin/env python3
"""
Generate missing Exercise-4 graphs (connectivity, inlier %, track length histogram).
These are built by visualization.py functions in main.py but never persisted because
the functions don't call plt.savefig() internally.
"""

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO_ROOT = Path("/mnt/c/Users/Nir/PycharmProjects/VAN_ex") if sys.platform != "win32" else Path("C:/Users/Nir/PycharmProjects/VAN_ex")
CANONICAL_TREE = REPO_ROOT / "code/project"
OUTPUT_DIR = CANONICAL_TREE / "outputs"
WORK_DIR = REPO_ROOT / ".van_report_work"

# Add project to path
sys.path.insert(0, str(CANONICAL_TREE))
sys.path.insert(0, str(REPO_ROOT / "code"))

print("=" * 80)
print("GENERATING MISSING EXERCISE-4 GRAPHS")
print("=" * 80)

# Load DB
print("\nLoading TrackingDB...")
try:
    db_path = REPO_ROOT / "code" / "tracking_db.pkl"
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    print(f"✓ Loaded TrackingDB")
except Exception as e:
    print(f"✗ Failed to load DB: {e}")
    sys.exit(1)

# Import functions
try:
    from utils.visualization import (
        plot_connectivity,
        plot_track_length_histogram,
        plot_inlier_percentage,
    )
    from utils.tracking import compute_connectivity
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Connectivity
# ============================================================================
print("\nGenerating: Connectivity Graph...")
try:
    connectivity = compute_connectivity(db)
    plt.figure(figsize=(12, 5))
    plot_connectivity(connectivity)
    plt.savefig(OUTPUT_DIR / "task_4_4_connectivity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ task_4_4_connectivity.png")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# 2. Inlier Percentage
# ============================================================================
print("\nGenerating: Inlier Percentage...")
try:
    plt.figure(figsize=(12, 5))
    plot_inlier_percentage(db.inlier_percentages)
    plt.savefig(OUTPUT_DIR / "task_4_5_inlier_percentage.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ task_4_5_inlier_percentage.png")
except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# 3. Track Length Histogram
# ============================================================================
print("\nGenerating: Track Length Histogram...")
try:
    plt.figure(figsize=(12, 5))
    plot_track_length_histogram(db, min_length=2)
    plt.savefig(OUTPUT_DIR / "task_4_6_track_length_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ task_4_6_track_length_histogram.png")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)
