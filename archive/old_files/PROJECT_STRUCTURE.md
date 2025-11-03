# Project Structure - V1 & V2 Organization

**Date:** 2025-11-02  
**Status:** Complete

---

## Overview

The project is organized into **V1** (original/baseline) and **V2** (feature-optimized) versions. All code, scripts, configs, and experiments are separated to allow parallel development and easy comparison.

---

## Directory Organization

### Source Code

**V1 (Original):**
- Location: `src_v1/`
- Features: All 42 original features
- Preprocessing: Standard (mean/std)
- Subjects: All available
- Status: Baseline reference

**V2 (Feature-Optimized):**
- Location: `src_v2/`
- Features: 24 core features (≥70% consistency)
- Preprocessing: Robust (median/IQR)
- Subjects: 47 validated only
- Status: Feature-optimized implementation

### Runner Scripts

**V1 Scripts:**
- Location: `scripts_v1/`
- Main runner: `run_experiment.py`
- Uses: `src_v1/`, `configs_v1/`, `experiments_v1/`

**V2 Scripts:**
- Location: `scripts_v2/`
- Main runner: `run_experiment_v2.py`
- Uses: `src_v2/`, `configs_v2/`, `experiments_v2/`

### Configurations

**V1 Configs:**
- Location: `configs_v1/`
- Files: `experiment_config.json`, `data_config.json`
- Settings: Original/baseline

**V2 Configs:**
- Location: `configs_v2/`
- Files: `experiment_config.json`, `data_config.json`
- Settings: Feature-optimized

### Experiments

**V1 Experiments:**
- Location: `experiments_v1/cgm_fl_benchmark/`
- Results: Baseline experiments
- Features: All 42 features

**V2 Experiments:**
- Location: `experiments_v2/cgm_fl_benchmark/`
- Results: Feature-optimized experiments
- Features: 24 core features + comparisons

---

## Quick Reference

### Running V1 (Baseline)

```bash
cd scripts_v1
python run_experiment.py
```

**Uses:**
- Source: `src_v1/`
- Config: `configs_v1/`
- Results: `experiments_v1/`

### Running V2 (Feature-Optimized)

```bash
cd scripts_v2
python run_experiment_v2.py
```

**Uses:**
- Source: `src_v2/`
- Config: `configs_v2/`
- Results: `experiments_v2/`

---

## Shared Resources

- **Data:** `data/` (used by both V1 and V2)
  - `data/raw/` - Raw subject files
  - `data/processed/` - Processed subject files
  
- **Documentation:** Root level markdown files
  - `EXPERIMENT_PLAN_V2.md` - V2 experimental plan
  - `VERSION_STRUCTURE.md` - Version organization details
  - `PROJECT_STRUCTURE.md` - This file

---

## Version Comparison

| Component | V1 | V2 |
|-----------|----|----|
| **Source** | `src_v1/` | `src_v2/` |
| **Scripts** | `scripts_v1/` | `scripts_v2/` |
| **Configs** | `configs_v1/` | `configs_v2/` |
| **Experiments** | `experiments_v1/` | `experiments_v2/` |
| **Features** | 42 (all) | 24 (core) |
| **Scaling** | Mean/Std | Median/IQR |
| **Subjects** | All | 47 validated |
| **Feature Selection** | None | Automatic |
| **Status** | Baseline | Optimized |

---

## Migration Path

### Development
- Keep V1 as stable baseline
- Develop V2 independently
- Compare results between versions
- Use V2 for new development

### Testing
1. Test V1 scripts (verify baseline works)
2. Test V2 scripts (verify new implementation)
3. Compare results side-by-side
4. Document differences

---

**See `VERSION_STRUCTURE.md` for detailed information.**

