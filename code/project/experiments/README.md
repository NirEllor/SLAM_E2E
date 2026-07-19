# Pipeline Comparison Experiments

This package contains scripts and infrastructure for running systematic comparison experiments across the visual SLAM pipeline. Each experiment varies a single parameter and compares the resulting trajectories, absolute errors, and other metrics.

## Architecture

- **`harness.py`**: Core experiment infrastructure
  - `build_variant_db()` - Builds or loads a TrackingDB with specified feature detector and PnP-RANSAC parameters
  - `run_pipeline_variant()` - Runs the full pipeline (bundle adjustment, pose-graph optimization, optional loop closure) with specified parameters
  - `save_variant_result()` / `load_variant_result()` - Pickle/unpickle experiment results for later plotting

- **`plotting.py`**: Multi-variant comparison visualization functions
  - `plot_multi_variant_trajectory()` - Bird's-eye-view overlay of trajectories from different variants + ground truth
  - `plot_multi_variant_absolute_error()` - Overlay absolute location and angle errors vs frame
  - `plot_multi_variant_scalar_comparison()` - Bar charts of keyframe count, loop-closure count, runtime
  - `plot_multi_variant_kitti_bar()` - KITTI-style segment error bars

- **`smoke_test.py`**: Quick validation script
  - Tests the harness on a small frame count (100 frames) to verify everything works before full-sequence runs

## Experiments (5 topics)

### 1. Loop-Closure Gating Strictness (`compare_loop_closure_gating.py`)
- **Parameters**: `mahalanobis_threshold`, `inlier_ratio_threshold`
- **Variants**: strict (300, 0.8), baseline (1000, 0.6), loose (3000, 0.4)
- **Cost**: Cheap (no DB rebuild; reuses baseline DB)

### 2. Keyframe Density (`compare_keyframe_density.py`)
- **Parameters**: `distance_threshold` (others fixed at defaults)
- **Variants**: dense (1.25 m), baseline (2.5 m), sparse (5.0 m)
- **Cost**: Cheap (no DB rebuild)

### 3. Bundle Adjustment Noise Model (`compare_bundle_noise_model.py`)
- **Parameters**: `prior_sigma`, `stereo_sigma`, `huber_k`
- **Variants**: Sweeps of prior_sigma and stereo_sigma
- **Cost**: Cheap (no DB rebuild)

### 4. PnP-RANSAC Parameters (`compare_pnp_ransac_params.py`)
- **Parameters**: `pnp_threshold` (pixels), `pnp_iterations`
- **Variants**: strict (1px, 100 iter), baseline (2px, 50), loose (4px, 25)
- **Cost**: Expensive (requires DB rebuild per variant)

### 5. Feature Detectors (`compare_feature_detectors.py`)
- **Parameters**: `detector_type` (akaze/orb/sift)
- **Variants**: AKAZE (baseline), ORB (700 features), SIFT
- **Cost**: Expensive (requires full DB rebuild per variant)

## Running Experiments

### Smoke Test (validation only)
```bash
cd code/project
wsl bash -lc "source gtsam_venv/bin/activate && python experiments/smoke_test.py"
```

### Single Comparison Script
```bash
wsl bash -lc "source gtsam_venv/bin/activate && cd code/project && python experiments/compare_loop_closure_gating.py"
```

### Results and Plots
- Variant results pickled to: `code/project/experiment_results/<topic>/<variant>.pkl`
- Comparison plots saved to: `code/project/outputs/cmp_<topic>_*.png`

## Modified Core Files

The following core pipeline files were updated with backward-compatible signature changes (all new parameters have defaults matching the original behavior):

- **`utils/geometry.py`**
  - Added `get_sift_features(img)` function
  - Added `detector_type='akaze'` parameter to `run_single_pair()`

- **`utils/tracking.py`**
  - Added `detector_type`, `pnp_threshold`, `pnp_iterations`, `pnp_max_no_improvement` parameters to `build_data()`
  - Added same parameters to `load_or_build_db()`

- **`utils/bundle_adjustment.py`**
  - Added `stereo_sigma=1.0`, `huber_k=2.0` parameters to `build_and_solve_bundle_core()`
  - Added same parameters to `solve_bundle_window()`

All changes are backward-compatible: existing code (e.g., `main.py`, `exN.py`) continues to work without modification.

## Notes

- Full-sequence experiments (~2760 frames) take ~2-5 minutes per variant depending on loop closure detection complexity
- Cheap experiments (topics 1-3) can run sequentially; expensive experiments (topics 4-5) can be parallelized across machines if needed
- KITTI segment error computation is optional (currently skipped in harness) but can be added once utility functions are properly wired
