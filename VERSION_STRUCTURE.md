# Version Structure - V1 & V2 Separation

**Date:** 2025-11-02  
**Status:** Complete

---

## Overview

The codebase has been organized into **V1** (original) and **V2** (feature-optimized) versions. This separation allows for:
- Clear comparison between baseline and feature-optimized approaches
- Independent development and testing
- Easy rollback if needed
- Parallel experimentation

---

## Directory Structure

```
Experiments2/
├── README.md                          # Main project README
├── VERSION_STRUCTURE.md               # This file
│
├── src_v1/                            # V1 Source Code (Original)
│   ├── README.md                      # V1 documentation
│   ├── data_processing.py            # Original data processing (42 features)
│   ├── fl_benchmark.py               # Original FL benchmark
│   ├── strategies.py                 # Original FL strategies
│   ├── generate_explanations.py      # Original XAI generation
│   └── utils.py                      # Original utilities
│
├── src_v2/                            # V2 Source Code (Feature-Optimized)
│   ├── README.md                      # V2 documentation
│   ├── data_processing.py            # Enhanced with feature selection (24 features)
│   ├── fl_benchmark.py               # Enhanced FL benchmark
│   ├── strategies.py                 # Enhanced FL strategies
│   ├── generate_explanations.py      # Enhanced XAI with feature validation
│   └── utils.py                      # Enhanced utilities (includes feature analysis)
│
├── scripts_v1/                        # V1 Runner Scripts
│   ├── run_experiment.py             # Main V1 experiment runner
│   ├── preprocess_data.py            # V1 data preprocessing
│   ├── run_feature_analysis_all.py   # Feature analysis (for reference)
│   └── ...                           # Other V1 scripts
│
├── scripts_v2/                        # V2 Runner Scripts
│   └── run_experiment_v2.py          # Main V2 experiment runner
│
├── configs_v1/                        # V1 Configuration Files
│   ├── experiment_config.json        # Original experiment config
│   └── data_config.json              # Original data config
│
├── configs_v2/                        # V2 Configuration Files
│   ├── experiment_config.json        # Feature-optimized experiment config
│   └── data_config.json              # Enhanced data config
│
├── experiments_v1/                    # V1 Experiments
│   └── cgm_fl_benchmark/             # Original experiment results
│       ├── fedavg/
│       ├── fedprox/
│       ├── fedclusterproxxai/
│       └── ...
│
├── experiments_v2/                     # V2 Experiments
│   └── cgm_fl_benchmark/             # Feature-optimized experiments
│       ├── data_preparation/
│       ├── models/
│       ├── experiments/
│       └── results/
│
└── data/                              # Shared Data (used by both versions)
    ├── raw/                           # Raw subject data
    └── processed/                     # Processed subject data
```

---

## Version Differences

### V1 (Original)

**Source Code:**
- All 42 features (original feature set)
- Standard preprocessing (mean/std scaling)
- All available subjects (no filtering)
- Original model architectures (input: 20/42 features)

**Configurations:**
- `experiment_config.json` - Original settings
- `data_config.json` - Original data settings

**Experiments:**
- Baseline federated learning experiments
- Original feature set
- All subjects included

**Scripts:**
- `run_experiment.py` - Original experiment runner
- Uses `src_v1/` imports

### V2 (Feature-Optimized)

**Source Code:**
- 24 core features (consistently important ≥70%)
- Robust preprocessing (median/IQR scaling)
- 47 validated subjects only (10 excluded)
- Enhanced model architectures (input: 24 features)
- Feature selection integrated
- Redundancy removal
- Feature interaction engineering

**Configurations:**
- `experiment_config.json` - Feature-optimized settings
- `data_config.json` - Enhanced with feature selection

**Experiments:**
- Feature-optimized federated learning experiments
- Multiple feature set comparisons
- Subject validation applied
- Feature importance validation

**Scripts:**
- `run_experiment_v2.py` - Feature-optimized experiment runner
- Uses `src_v2/` imports

---

## Usage

### Running V1 Experiments

```bash
# Navigate to scripts
cd scripts_v1

# Run V1 experiment
python run_experiment.py

# Or use direct path
python scripts_v1/run_experiment.py
```

### Running V2 Experiments

```bash
# Navigate to scripts
cd scripts_v2

# Run V2 experiment
python run_experiment_v2.py

# Or use direct path
python scripts_v2/run_experiment_v2.py
```

---

## Key Differences Summary

| Aspect | V1 (Original) | V2 (Feature-Optimized) |
|--------|---------------|------------------------|
| **Features** | 42 (all features) | 24 (core features) |
| **Scaling** | Mean/Std | Median/IQR (robust) |
| **Subjects** | All available | 47 validated only |
| **Feature Selection** | None | Automatic (core_24) |
| **Redundancy Removal** | None | Automatic |
| **Interaction Features** | None | Optional |
| **Model Input** | 20/42 features | 24 features |
| **Experiments** | Baseline | Feature-optimized + comparisons |

---

## Migration Notes

### From V1 to V2

1. **Update imports:**
   ```python
   # V1
   from src.data_processing import ...
   
   # V2
   from src_v2.data_processing import ...
   ```

2. **Update config paths:**
   ```python
   # V1
   config_path = 'configs/experiment_config.json'
   
   # V2
   config_path = 'configs_v2/experiment_config.json'
   ```

3. **Update experiment directories:**
   ```python
   # V1
   experiment_dir = 'experiments/cgm_fl_benchmark'
   
   # V2
   experiment_dir = 'experiments_v2/cgm_fl_benchmark'
   ```

### Maintaining Both Versions

- Keep V1 as baseline reference
- Develop V2 with feature optimizations
- Compare results between versions
- Use V2 as primary going forward

---

## File Locations

### V1 Files

- **Source:** `src_v1/`
- **Scripts:** `scripts_v1/`
- **Configs:** `configs_v1/`
- **Experiments:** `experiments_v1/`

### V2 Files

- **Source:** `src_v2/`
- **Scripts:** `scripts_v2/`
- **Configs:** `configs_v2/`
- **Experiments:** `experiments_v2/`

### Shared Files

- **Data:** `data/` (used by both versions)
- **Documentation:** Root level markdown files
- **Requirements:** `requirements.txt` (same for both)

---

## Next Steps

1. ✅ Directory structure created
2. ✅ Files separated into V1 and V2
3. ✅ Scripts updated with correct paths
4. ⏳ Test V1 scripts (verify they still work)
5. ⏳ Test V2 scripts (verify imports work)
6. ⏳ Run comparison experiments

---

**Status:** Complete  
**Last Updated:** 2025-11-02

