# Cleanup Summary

**Date:** 2025-01-XX

## Files and Directories Archived

The following unnecessary files have been moved to `archive/old_files/`:

### Log Files
- `benchmark_20clients.log`
- `benchmark_final.log`
- `benchmark_optimized.log`
- `benchmark_output.log`
- `feature_analysis_all.log`
- `feature_analysis_all_subjects.log`
- `test_output.log`

### Old Documentation Files
- `ADAPTIVE_CLUSTERING.md`
- `FEDCLUSTERPROXXAI_EXPLANATION.md`
- `OPTIMIZATIONS.md`
- `PROJECT_STRUCTURE.md`
- `SUMMARY.md`

### Old Source Code
- `src/` (superseded by `src_v1/` and `src_v2/`)

### Old Configuration
- `configs/` (superseded by `configs_v1/` and `configs_v2/`)

### Old Scripts
- `scripts_v1_old/` containing:
  - `check_feature_analysis_status.py`
  - `create_feature_analysis_md.py`
  - `run_feature_analysis_all.py`
  - `show_feature_analysis_summary.py`
  - `test_feature_explanations.py`

### Empty Directories
- `experiments/` (empty, superseded by `experiments_v1/` and `experiments_v2/`)
- `logs/` (empty)

## Active Structure

The project now uses a clean v1/v2 structure:

- **`src_v1/`** and **`src_v2/`** - Source code
- **`scripts_v1/`** and **`scripts_v2/`** - Runner scripts
- **`configs_v1/`** and **`configs_v2/`** - Configuration files
- **`experiments_v1/`** and **`experiments_v2/`** - Experimental results
- **`data/`** - Data directories (raw and processed)
- **`experiments_v2/cgm_fl_benchmark/`** - Publication framework

## Root-Level Files Kept

- `README.md` - Main project README
- `README_V1_V2.md` - Version structure explanation
- `VERSION_STRUCTURE.md` - Detailed version documentation
- `EXPERIMENT_PLAN_V2.md` - Experiment plan
- `requirements.txt` - Python dependencies

## Note

All archived files are preserved in `archive/old_files/` for reference but are not part of the active codebase.

