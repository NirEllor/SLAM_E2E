# Database Protection Mechanism

## Guarantee: Shared Baseline DB Never Modified

The shared `code/tracking_db.pkl` (used by both project/ and homework/ folders) is **guaranteed to never be modified** by any experiment.

## How It Works

### Baseline Loading (Cheap Experiments)
When experiments load the baseline DB with:
```python
db = build_variant_db(detector_type='akaze', cache_tag='baseline')
```

The harness:
1. **Detects baseline request** via special check:
   - `cache_tag == 'baseline'`
   - `detector_type == 'akaze'`
   - `pnp_threshold == 2`
   - `pnp_iterations == 50`

2. **Loads from shared DB** using `load_or_build_db(..., force_rebuild=False)`
   - Reuses existing `code/tracking_db.pkl` (never modified)
   - No caching to variant file
   - Returns immediately

### Variant DB Saving (All Experiments)
When experiments build variant DBs (with different parameters or cache_tags):

1. **Saves to variant-specific pickle** at `code/tracking_db_{cache_tag}.pkl`
   - Example: `tracking_db_strict_pnp.pkl`, `tracking_db_sift_default.pkl`
   - Never touches the shared `code/tracking_db.pkl`

2. **Calls `build_data` directly** (not `load_or_build_db`)
   - This is the critical safety step
   - `build_data` returns a DB object without saving
   - Harness saves to variant_pkl (not DB_PKL_PATH)
   - Ensures no accidental writes to shared baseline

## Experiment Types

### Cheap Experiments (No DB Rebuild)
- `compare_loop_closure_gating.py`
- `compare_keyframe_density.py`
- `compare_bundle_noise_model.py`

**Action**: Calls `build_variant_db(..., cache_tag='baseline')`
**Result**: Loads shared `code/tracking_db.pkl` (never modified)

### Expensive Experiments (DB Rebuild)
- `compare_pnp_ransac_params.py`
- `compare_feature_detectors.py`

**Action**: Calls `build_variant_db(detector_type=..., cache_tag=...)`
**Result**: Saves to `code/tracking_db_{cache_tag}.pkl` (variant-specific)

## Safety Verification

### Before Running Experiments
```bash
# Note the baseline DB timestamp
ls -la code/tracking_db.pkl
```

### After Running All Experiments
```bash
# Verify baseline DB unchanged
ls -la code/tracking_db.pkl  # Same timestamp

# Verify variant DBs created separately
ls -la code/tracking_db_*.pkl  # Should see variant files:
# code/tracking_db_strict_pnp.pkl
# code/tracking_db_baseline_pnp.pkl
# code/tracking_db_loose_pnp.pkl
# code/tracking_db_orb_700.pkl
# code/tracking_db_sift_default.pkl
```

## Code Reference

**Protection mechanism**: `code/project/experiments/harness.py`, function `build_variant_db`

```python
# Special case: baseline uses the shared code/tracking_db.pkl (never modified)
if cache_tag == 'baseline' and detector_type == 'akaze' and pnp_threshold == 2 and pnp_iterations == 50:
    print(f"Loading BASELINE DB from shared code/tracking_db.pkl (NEVER modified by experiments)")
    db = load_or_build_db(force_rebuild=False, num_frames=num_frames)
    return db

# For all variant DBs, use a separate pickle file
variant_pkl = PROJECT_ROOT / 'code' / f'tracking_db_{cache_tag}.pkl'

# Build new DB and save to VARIANT pickle (not shared DB_PKL_PATH)
db = build_data(...)  # Calls build_data directly, NOT load_or_build_db
with open(variant_pkl, 'wb') as f:
    pickle.dump(db, f)
```

## Why This Matters

- **Shared environment**: Both `code/project/` and `code/homework/` use `code/tracking_db.pkl`
- **Experiment isolation**: Each variant experiment has its own cached DB
- **No conflicts**: Multiple experiments can run independently without interference
- **Replayability**: Variant DBs persist so experiments can be rerun instantly without rebuilding

## Summary

✅ **Baseline** (`code/tracking_db.pkl`) — **PROTECTED** — never modified
✅ **Variants** (`code/tracking_db_{tag}.pkl`) — **ISOLATED** — each in own file
✅ **Cheap experiments** — reuse baseline
✅ **Expensive experiments** — build and cache separately
