# CGM Federated Learning Benchmark V2 - Feature Optimized
**Version:** 2.0  
**Status:** Setup Complete  
**Based on:** Feature Analysis Results (47 subjects, 24 core features)

---

## Overview

This is the Version 2 experiment structure implementing the feature-optimized experimental plan based on comprehensive feature analysis. This version focuses on using the 24 consistently important features identified across 47 subjects.

## Directory Structure

```
cgm_fl_benchmark_v2/
├── README.md (this file)
├── data_preparation/          # Feature selection and preprocessing
├── models/                    # Model architectures by feature set
│   ├── baseline/             # 42 features (original)
│   ├── core_24/              # 24 core features (primary)
│   ├── top_10/               # Top 10 features (minimal)
│   └── temporal_focused/     # 10 temporal-focused features
├── experiments/               # Experiment definitions
│   ├── feature_comparison/   # Compare feature sets
│   ├── redundancy_impact/    # Evaluate redundancy removal
│   ├── architecture_comparison/ # Compare architectures
│   └── fl_strategy_comparison/  # Compare FL strategies
├── results/                   # Experiment results
│   ├── global/               # Global results
│   ├── by_experiment/        # Results by experiment type
│   └── comparisons/          # Comparison analyses
├── logs/                      # Training and experiment logs
├── feature_visualizations/    # Feature analysis plots
└── checkpoints/              # Model checkpoints
```

## Key Features

### Core 24 Features (Primary Feature Set)

**Temporal (4):** `hour`, `hour_cos`, `hourly_baseline`, `hour_sin`  
**Recent Values (3):** `roc_60min`, `current_glucose`, `roc_30min`  
**Rolling Statistics (14):** Various time windows (1h-24h)  
**Medical Indicators (2):** `in_tight_range`, `is_hyperglycemic`  
**Pattern Analysis (1):** `deviation_from_recent_trend`

### Experiment Phases

1. **Phase 1: Data Preparation**
   - Feature selection implementation
   - Redundancy removal
   - Robust scaling setup

2. **Phase 2: Model Updates**
   - Architecture optimization for 24 features
   - Feature interaction engineering
   - Model variants

3. **Phase 3: Experiments**
   - Feature set comparison
   - Redundancy impact
   - Architecture comparison
   - FL strategy comparison

4. **Phase 4: Evaluation**
   - Cross-validation
   - Feature importance validation
   - Performance metrics

## Quick Start

### 1. Data Preparation
```bash
cd data_preparation/
python prepare_features.py --feature_set core_24
```

### 2. Train Models
```bash
cd models/core_24/
python train_model.py --config ../../configs/experiment_config_v2.json
```

### 3. Run Experiments
```bash
cd experiments/feature_comparison/
python run_comparison.py
```

## Configuration

Main configuration file: `configs/experiment_config_v2.json`

Key settings:
- Feature set: Core 24 features
- Model input: 24 features (vs. 42 original)
- Subjects: 47 valid subjects
- Outlier handling: Robust scaling (keep outliers)

## Results

Results will be saved in:
- `results/global/` - Overall results
- `results/by_experiment/` - Experiment-specific results
- `results/comparisons/` - Comparison analyses

## References

- **Feature Analysis:** `../cgm_fl_benchmark/feature_visualizations/COMPREHENSIVE_DATA_ANALYSIS_REPORT.md`
- **Experiment Plan:** `../../EXPERIMENT_PLAN_V2.md`
- **Configuration:** `../../configs/experiment_config_v2.json`

---

*Created: 2025-11-02*  
*Version: 2.0*

