# Quick Start Guide: Pipeline Comparison Experiments

## Status: ✅ Complete and Tested

All 5 comparison experiment scripts have been created and tested (smoke test passed).

## What Was Built

### 5 Comparison Experiments

| # | Topic | Script | DB Rebuild | Variants |
|---|-------|--------|-----------|----------|
| 1 | Loop-Closure Gating | `compare_loop_closure_gating.py` | ❌ No | strict / baseline / loose |
| 2 | Keyframe Density | `compare_keyframe_density.py` | ❌ No | dense / baseline / sparse |
| 3 | Bundle Noise Model | `compare_bundle_noise_model.py` | ❌ No | prior_sigma + stereo_sigma sweeps |
| 4 | PnP-RANSAC Params | `compare_pnp_ransac_params.py` | ✅ Yes | strict / baseline / loose |
| 5 | Feature Detectors | `compare_feature_detectors.py` | ✅ Yes (2x) | AKAZE / ORB / SIFT |

### Infrastructure

- **`harness.py`** - Core experiment runner (DB building, pipeline execution, result persistence)
- **`plotting.py`** - Multi-variant comparison visualizations
- **`smoke_test.py`** - Quick validation (100 frames, already tested ✓)
- **`run_all_comparisons.py`** - Master runner for all 5 experiments sequentially

## How to Run

### Smoke Test (validation only, ~30 seconds)
```bash
cd code/project
wsl bash -lc "source gtsam_venv/bin/activate && python experiments/smoke_test.py"
```

### Single Comparison (pick one)

**Loop-Closure Gating** (~2-3 min per variant, 3 total):
```bash
wsl bash -lc "source gtsam_venv/bin/activate && cd code/project && python experiments/compare_loop_closure_gating.py"
```

**Keyframe Density** (~2-3 min per variant, 3 total):
```bash
wsl bash -lc "source gtsam_venv/bin/activate && cd code/project && python experiments/compare_keyframe_density.py"
```

**Bundle Noise Model** (~2-3 min per variant, 5 total):
```bash
wsl bash -lc "source gtsam_venv/bin/activate && cd code/project && python experiments/compare_bundle_noise_model.py"
```

**PnP-RANSAC Parameters** (~5-7 min per variant, 3 total — includes DB rebuild):
```bash
wsl bash -lc "source gtsam_venv/bin/activate && cd code/project && python experiments/compare_pnp_ransac_params.py"
```

**Feature Detectors** (~5-7 min per variant, 3 total — includes 2 full DB rebuilds):
```bash
wsl bash -lc "source gtsam_venv/bin/activate && cd code/project && python experiments/compare_feature_detectors.py"
```

### All Experiments Sequentially
```bash
wsl bash -lc "source gtsam_venv/bin/activate && cd code/project && python experiments/run_all_comparisons.py"
```
**Est. total runtime: 30-40 minutes** (cheap experiments: ~15 min, expensive: ~15-25 min)

## Output Structure

After running experiments:

```
code/project/
  outputs/
    cmp_lc_gating_trajectory.png              # Trajectory overlay
    cmp_lc_gating_absolute_error.png          # Error curves
    cmp_lc_gating_scalars.png                 # Bar charts (keyframes, LC count, runtime)
    cmp_kf_density_*.png                      # 3 plots for keyframe density
    cmp_bundle_noise_*.png                    # 3 plots for bundle noise
    cmp_pnp_ransac_*.png                      # 3 plots for PnP-RANSAC
    cmp_feature_detectors_*.png               # 3 plots for feature detectors

  experiment_results/
    loop_closure_gating/
      strict_lc.pkl
      baseline_lc.pkl
      loose_lc.pkl
    keyframe_density/
      dense_kf.pkl
      baseline_kf.pkl
      sparse_kf.pkl
    bundle_noise_model/
      prior_1p0_stereo_1p0.pkl
      prior_0p05_stereo_1p0.pkl
      ... (5 total)
    pnp_ransac_params/
      strict_pnp.pkl
      baseline_pnp.pkl
      loose_pnp.pkl
    feature_detectors/
      akaze_baseline.pkl
      orb_700features.pkl
      sift_default.pkl
```

## Variant Parameters at a Glance

### 1. Loop-Closure Gating
| Variant | Mahalanobis | Inlier Ratio |
|---------|-------------|--------------|
| strict  | 300 | 0.8 |
| baseline | 1000 | 0.6 |
| loose | 3000 | 0.4 |

### 2. Keyframe Density
| Variant | Distance Threshold |
|---------|-----------------|
| dense | 1.25 m |
| baseline | 2.5 m |
| sparse | 5.0 m |

### 3. Bundle Noise Model
| Variant | Prior Sigma | Stereo Sigma | Huber K |
|---------|------------|--------------|---------|
| prior_1p0_stereo_1p0 | 1.0 | 1.0 | 2.0 |
| prior_0p05_stereo_1p0 | 0.05 | 1.0 | 2.0 |
| prior_1e-6_stereo_1p0 | 1e-6 | 1.0 | 2.0 |
| prior_1e-6_stereo_0p5 | 1e-6 | 0.5 | 2.0 |
| prior_1e-6_stereo_2p0 | 1e-6 | 2.0 | 2.0 |

### 4. PnP-RANSAC
| Variant | Threshold | Iterations |
|---------|-----------|-----------|
| strict | 1 px | 100 |
| baseline | 2 px | 50 |
| loose | 4 px | 25 |

### 5. Feature Detectors
| Variant | Detector | Parameters |
|---------|----------|-----------|
| akaze_baseline | AKAZE | threshold=0.0001 |
| orb_700features | ORB | nfeatures=700 |
| sift_default | SIFT | defaults |

## Core Files Modified (Backward Compatible)

All changes maintain backward compatibility — existing code works unchanged:

- **`utils/geometry.py`**
  - Added: `get_sift_features(img)`
  - Modified: `run_single_pair(..., detector_type='akaze')`

- **`utils/tracking.py`**
  - Modified: `build_data(..., detector_type='akaze', pnp_threshold=2, pnp_iterations=50, pnp_max_no_improvement=12)`
  - Modified: `load_or_build_db(..., same_params)`

- **`utils/bundle_adjustment.py`**
  - Modified: `build_and_solve_bundle_core(..., stereo_sigma=1.0, huber_k=2.0)`
  - Modified: `solve_bundle_window(..., stereo_sigma=1.0, huber_k=2.0)`

## Next Steps

1. **Run smoke test** to validate setup (~30 sec)
2. **Run cheap experiments** (topics 1-3) to get initial results (~15 min)
3. **Run expensive experiments** (topics 4-5) to complete the study (~20-25 min)
4. **Analyze plots** in `code/project/outputs/cmp_*.png`
5. **Use results** for Section 3 (Performance Analysis) in your project report

## Notes

- Variant DBs are cached separately (e.g., `tracking_db_strict_pnp.pkl`) so experiments can be rerun without rebuild
- Plots are automatically generated and saved after each experiment
- All experiments reuse the same baseline feature points/tracks (except feature detector topic which rebuilds)
- Full-sequence experiments use `get_num_frames()` (all ~2760 frames in sequence 00)
