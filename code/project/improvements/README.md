# Robust Pose-Graph Optimization Improvements

This folder contains implementations of two robust back-end techniques for pose-graph SLAM, evaluated against the VAN_ex stereo visual-SLAM pipeline baseline.

## Overview

The baseline VAN_ex pipeline uses **hard-threshold pre-optimization filtering** (Mahalanobis distance gating + RANSAC consensus) to reject false loop-closure candidates *before* adding them to the pose graph. While effective, this binary accept/reject strategy cannot recover from false positives that slip through the gates.

This folder implements two alternative, **continuous in-the-loop down-weighting** strategies that allow the back-end optimizer to automatically suppress (but not completely ignore) outlier constraints during optimization:

1. **Switchable Constraints** (Sünderhauf & Protzel, IROS 2012)
2. **Dynamic Covariance Scaling** (Agarwal et al., ICRA 2013)

## Non-Invasive Design

All code in this folder:
- **Imports only** from `utils.geometry`, `utils.tracking`, `utils.pose_graph`, `utils.loop_closure`, `utils.bundle_adjustment` (read-only)
- Never modifies any files under `utils/`, `main.py`, or `tracking_database.py`
- Reuses the same `tracking_database.pkl` and produces the same `(keyframes, relative_poses, relative_covs, loop_measurements)` pipeline inputs as the baseline
- Builds independent `gtsam.NonlinearFactorGraph` objects from those inputs, applying the new techniques only at the pose-graph optimization stage

This ensures reproducibility and prevents contamination of the original pipeline.

## Mathematical Formulations

### Switchable Constraints

**Switch variable** per loop-closure edge `(i, j)`: scalar `s_ij ∈ [0, 1]`.

**Switched constraint error**: 
```
e_ij_switched = s_ij * e_ij_original(pose_i, pose_j)
```

**Switch prior factor**: pulls each switch toward 1.0 with Gaussian noise:
```
PriorFactor(s_ij, 1.0, sigma=0.3)
```

**Optimization**: jointly optimizes poses **and** switch variables. Switches near 1.0 indicate trusted loop closures; switches near 0.0 indicate suppressed outliers. Critically, this happens *during* optimization, not before it — the back-end can learn to down-weight bad loops on its own.

**Reference**: Sünderhauf, N., & Protzel, P. (2012). Switchable Constraints for Robust Pose Graph SLAM. In *Proceedings of IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*.

### Dynamic Covariance Scaling (DCS)

**Scale factor** per loop-closure constraint, recomputed each iteration:
```
w_ij = scale(χ²_ij) = min(1.0, Φ / χ²_ij)
```
where `χ²_ij` is the squared Mahalanobis distance of the constraint's current residual, and `Φ ≈ 1.0` is the threshold.

**Information matrix reweighting**:
```
Λ_ij^{new} = w_ij * Λ_ij^{original}
```

The more a constraint violates its covariance model (high `χ²`), the more it gets down-weighted. This is implemented in GTSAM as a native robust kernel (`mEstimator.DCS`) — no separate switch optimization needed. The weighting is *closed-form and continuous*, providing smooth downgrading of outliers.

**Reference**: Agarwal, P., Tipaldi, G. D., Spinello, L., Stachniss, C., & Burgard, W. (2013). Robust Map Optimization Using Dynamic Covariance Scaling. In *Proceedings of IEEE International Conference on Robotics and Automation (ICRA)*, pp. 62–69.

## Contents

- `switchable_constraints.py` — CustomFactor-based implementation of switch variables + runner
- `dynamic_covariance_scaling.py` — Native mEstimator.DCS wrapping + runner
- `stress_test.py` — Deliberately-loosened loop-closure gates to inject a false positive; compares baseline vs both techniques
- `outputs/` — Generated PNG comparison plots (trajectory, error, scalars, diagnostics)
- `results/` — Pickled result dictionaries (metrics, runtime, trajectory positions)

## Running the Improvements

Each script is standalone and can be run independently:

```bash
cd /mnt/c/Users/Nir/PycharmProjects/VAN_ex/code/project/improvements
source ../../gtsam_venv/bin/activate

python switchable_constraints.py   # ~5-10 min: builds graph, optimizes, evaluates
python dynamic_covariance_scaling.py  # ~3-5 min
python stress_test.py              # ~10-15 min: runs all three variants (baseline, SC, DCS) on loosened gates
```

Each script prints:
- Loop closures detected/verified
- Optimization convergence stats
- Per-variant metrics (location error m, angle error deg, KITTI sub-segment errors, runtime)
- PNG output paths

Results are pickled to `results/` and plots saved to `outputs/`.

## Integration into the Report

The `/van-report` skill's Section 4.2 "Future Work" has been updated to:
1. Load results from `improvements/results/*.pkl`
2. Cite exact function implementations (switchable_constraints.py:line, dynamic_covariance_scaling.py:line)
3. Embed real metrics and plots from `improvements/outputs/`
4. Replace generic speculation with grounded evidence

See `.van_report_work/render_report.py:424-431+` for the code that generates this section.
