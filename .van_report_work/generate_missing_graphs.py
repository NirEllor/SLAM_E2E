#!/usr/bin/env python3
"""
External savefig wrapper — generates missing Exercise-4 graphs without modifying source.
These graphs are built by visualization.py functions but never persisted internally.
"""

import sys
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path("/mnt/c/Users/Nir/PycharmProjects/VAN_ex") if sys.platform != "win32" else Path("C:/Users/Nir/PycharmProjects/VAN_ex")
CANONICAL_TREE = REPO_ROOT / "code/project"
OUTPUT_DIR = CANONICAL_TREE / "outputs"
WORK_DIR = REPO_ROOT / ".van_report_work"

# Add project to path
sys.path.insert(0, str(CANONICAL_TREE))
sys.path.insert(0, str(REPO_ROOT / "code"))

print("=" * 80)
print("EXTERNAL SAVEFIG WRAPPER — Missing Graph Generation")
print("=" * 80)

# Load the real DB
print("\nLoading TrackingDB...")
try:
    db_path = REPO_ROOT / "code" / "tracking_db.pkl"
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    print(f"✓ Loaded TrackingDB")
except Exception as e:
    print(f"✗ Failed to load DB: {e}")
    sys.exit(1)

try:
    from utils.visualization import (
        plot_connectivity,
        plot_track_length_histogram,
        plot_matches_per_frame,
        plot_inlier_percentage,
        plot_full_trajectory_comparison,
        plot_projection_error_vs_distance,
        plot_bundle_optimization_error,
        plot_bundle_projection_error,
    )
except ImportError as e:
    print(f"✗ Failed to import visualization functions: {e}")
    sys.exit(1)

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Graph 1: Connectivity
print("\nGenerating Graph 1: Connectivity...")
try:
    plt.figure()
    plot_connectivity(db)
    plt.savefig(OUTPUT_DIR / "task_4_4_connectivity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated task_4_4_connectivity.png")
except Exception as e:
    print(f"✗ Error generating connectivity: {e}")

# Graph 2: Track Length Histogram
print("\nGenerating Graph 2: Track Length Histogram...")
try:
    plt.figure()
    plot_track_length_histogram(db)
    plt.savefig(OUTPUT_DIR / "task_4_5_track_length_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated task_4_5_track_length_histogram.png")
except Exception as e:
    print(f"✗ Error generating track length histogram: {e}")

# Graph 3: Matches per Frame
print("\nGenerating Graph 3: Matches per Frame...")
try:
    if hasattr(db, 'matches_per_frame'):
        plt.figure()
        plot_matches_per_frame(db)
        plt.savefig(OUTPUT_DIR / "task_4_6_matches_per_frame.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Generated task_4_6_matches_per_frame.png")
    else:
        print("⚠ matches_per_frame attribute missing from DB")
except Exception as e:
    print(f"✗ Error generating matches per frame: {e}")

# Graph 4: Inlier Percentage
print("\nGenerating Graph 4: Inlier Percentage...")
try:
    if hasattr(db, 'inlier_percentages'):
        plt.figure()
        plot_inlier_percentage(db)
        plt.savefig(OUTPUT_DIR / "task_4_7_inlier_percentage.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Generated task_4_7_inlier_percentage.png")
    else:
        print("⚠ inlier_percentages attribute missing from DB")
except Exception as e:
    print(f"✗ Error generating inlier percentage: {e}")

# Graph 3: Trajectory bird's-eye view
print("\nGenerating Graph: Trajectory Comparison...")
try:
    plt.figure()
    plot_trajectory_comparison(db)
    plt.savefig(OUTPUT_DIR / "task_summary_full_trajectory_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated task_summary_full_trajectory_comparison.png")
except Exception as e:
    print(f"⚠ Error generating trajectory comparison: {e}")

# Graph: Bundle Optimization Error
print("\nGenerating Graph: Bundle Optimization Error...")
try:
    plt.figure()
    plot_bundle_optimization_error(db)
    plt.savefig(OUTPUT_DIR / "task_5_4_mean_factor_error.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated task_5_4_mean_factor_error.png")
except Exception as e:
    print(f"⚠ Error generating bundle optimization error: {e}")

# Graph: Track Link Error vs Distance
print("\nGenerating Graphs: Track Link Error vs Distance...")
try:
    plt.figure()
    plot_bundle_projection_error_distance(db)
    plt.savefig(OUTPUT_DIR / "task_5_4_projection_error_vs_distance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Generated task_5_4_projection_error_vs_distance.png")
except Exception as e:
    print(f"⚠ Error generating projection error vs distance: {e}")

print("\n" + "=" * 80)
print("GRAPH GENERATION COMPLETE")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Generated files available for report inclusion.")
