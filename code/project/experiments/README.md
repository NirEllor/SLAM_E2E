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
- **Loop Closure**: ✅ **ENABLED** (this experiment specifically tests loop-closure gating)
- **Cost**: Expensive (~30-60 min for full sequence due to RANSAC consensus verification on loop candidates)
- **DB**: Full sequence, dedicated DB (`tracking_db_lc_test_full.pkl`)

### 2. Keyframe Density (`compare_keyframe_density.py`)
- **Parameters**: `distance_threshold` (others fixed at defaults)
- **Variants**: dense (1.25 m), baseline (2.5 m), sparse (5.0 m)
- **Loop Closure**: ❌ **DISABLED** (loop closure is orthogonal to keyframe selection; enabling it would confound results by varying loop counts per variant)
- **Cost**: Cheap (~2-3 min total)
- **DB**: Baseline reuse

### 3. Bundle Adjustment Noise Model (`compare_bundle_noise_model.py`)
- **Parameters**: `prior_sigma`, `stereo_sigma`, `huber_k`
- **Variants**: Sweeps of prior_sigma ∈ {1.0, 0.05, 1e-6} and stereo_sigma ∈ {0.5, 1.0, 2.0}
- **Loop Closure**: ❌ **DISABLED** (loop closure is independent of bundle adjustment noise modeling)
- **Cost**: Cheap (~3-5 min total)
- **DB**: Baseline reuse

### 4. PnP-RANSAC Parameters (`compare_pnp_ransac_params.py`)
- **Parameters**: `pnp_threshold` (pixels), `pnp_iterations`
- **Variants**: strict (1px, 100 iter), baseline (2px, 50), loose (4px, 25)
- **Loop Closure**: ❌ **DISABLED** (loop closure is independent of PnP feature tracking quality; disabling keeps comparison fair across variants)
- **Cost**: Expensive (~10-15 min total, requires DB rebuild per variant)
- **DB**: Separate DBs per variant (`tracking_db_strict_pnp.pkl`, etc.)

### 5. Feature Detectors (`compare_feature_detectors.py`)
- **Parameters**: `detector_type` (akaze/orb/sift)
- **Variants**: AKAZE (baseline), ORB (700 features), SIFT
- **Loop Closure**: ❌ **DISABLED** (loop closure is independent of feature detection; disabling ensures fair comparison)
- **Cost**: Expensive (~15-25 min total, 2 full DB rebuilds)
- **DB**: Baseline for AKAZE, separate DBs for ORB and SIFT

## Why Loop Closure is Disabled for Most Experiments

**Loop closure is a POST-PROCESSING stage** that runs after pose graph optimization. It is:
- ✅ Enabled for the **loop-closure-gating experiment** (topic 1), which specifically tests loop-closure detection and gating parameters
- ❌ Disabled for experiments 2-5 because:
  1. **Fair comparison**: Each experiment tests ONE component (keyframes, bundle noise, RANSAC, detector). Loop closure would confound results by introducing variable numbers of detected loops per variant
  2. **Orthogonal stages**: Loop closure doesn't depend on keyframe selection, bundle noise modeling, RANSAC parameters, or feature detectors
  3. **Isolate effects**: Disabling loop closure lets us measure the pure effect of the parameter being tested

**Example**: If keyframe density affected loop closure detection, it would be impossible to tell if improved trajectories came from better keyframes or more loop closures. Disabling LC ensures we measure only keyframe effects.

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
- Comparison plots saved to: `code/project/experiments/outputs/cmp_<topic>_*.png`

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

## Runtime Estimates (Full Sequence)

| Experiment | Loop Closure | Time per Variant | Total (all variants) |
|-----------|--------------|------------------|----------------------|
| **1. Loop-Closure Gating** | ✅ Yes | 30-60 min | 90-180 min |
| **2. Keyframe Density** | ❌ No | 3-5 min | 10-15 min |
| **3. Bundle Noise Model** | ❌ No | 2-3 min | 10-15 min |
| **4. PnP-RANSAC** | ❌ No | 3-5 min | 10-15 min |
| **5. Feature Detectors** | ❌ No | 5-10 min | 15-30 min |
| **Total (all 5 topics)** | Mixed | - | **2-4 hours** |

**Notes:**
- Cheap experiments (2-3) complete in ~30 min total
- Expensive experiments (4-5) require DB rebuild per variant but skip loop closure (15-30 min total)
- Loop-closure-gating (1) is slowest due to RANSAC consensus verification on all candidate loop pairs
- You can run cheap experiments first while loop-closure-gating runs in the background
- Variant DBs are cached (e.g., `tracking_db_strict_pnp.pkl`), so re-running experiments is instant if results already exist

## Database Protection

**Shared baseline DB is NEVER modified** (`code/tracking_db.pkl` used by both project and homework):
- Cheap experiments load baseline without modification
- Expensive experiments build and cache separate variant pickles
- See `DB_PROTECTION.md` for details

## Notes

- KITTI segment error computation is optional (currently skipped in harness) but can be added once utility functions are properly wired
- All new parameters in core pipeline files are backward-compatible with defaults matching original behavior
- Trajectory and error data are extracted as numpy arrays before pickling to ensure results can be re-plotted later
