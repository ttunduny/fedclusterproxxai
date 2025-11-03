# V2 Experiment Setup Summary
**Feature-Optimized CGM Prediction Experiments**

---

## Setup Status: ✅ COMPLETE

All directory structures and initial files have been created for Version 2 experiments.

---

## Directory Structure Created

```
experiments/cgm_fl_benchmark_v2/
├── README.md                          # Overview and quick start
├── SETUP_SUMMARY.md                   # This file
├── data_preparation/                  # Feature selection and preprocessing
│   ├── feature_selector.py           # Feature selection module
│   └── prepare_data.py               # Data preparation script
├── models/                            # Model architectures by feature set
│   ├── baseline/                     # 42 features (original)
│   ├── core_24/                      # 24 core features (primary)
│   ├── top_10/                       # Top 10 features (minimal)
│   ├── temporal_focused/             # 10 temporal features
│   └── MODEL_ARCHITECTURES.md        # Architecture documentation
├── experiments/                       # Experiment definitions
│   ├── feature_comparison/           # Compare feature sets
│   ├── redundancy_impact/            # Evaluate redundancy removal
│   ├── architecture_comparison/      # Compare architectures
│   ├── fl_strategy_comparison/       # Compare FL strategies
│   └── EXPERIMENT_RUNNER.md          # Experiment runner guide
├── results/                           # Experiment results
│   ├── global/                       # Global results
│   ├── by_experiment/                # Results by experiment type
│   └── comparisons/                  # Comparison analyses
├── logs/                              # Training and experiment logs
├── feature_visualizations/            # Feature analysis plots
└── checkpoints/                      # Model checkpoints
```

---

## Key Files Created

### 1. Feature Selection Module
**File:** `data_preparation/feature_selector.py`

- **Core 24 Features:** 24 consistently important features
- **Top 10 Features:** Highest composite scores
- **Temporal-Focused:** 10 temporal features
- **Redundancy Handling:** Automatic removal of redundant features
- **Interaction Features:** Creates feature interactions

### 2. Data Preparation Script
**File:** `data_preparation/prepare_data.py`

- Loads subject data
- Applies feature selection
- Performs robust scaling
- Creates interaction features (optional)
- Handles 47 valid subjects

### 3. Documentation
- **README.md:** Overview and quick start guide
- **MODEL_ARCHITECTURES.md:** Architecture documentation
- **EXPERIMENT_RUNNER.md:** Experiment execution guide

---

## Configuration

**Main Config:** `configs/experiment_config_v2.json`

**Key Settings:**
- Feature set: Core 24 features
- Model input: 24 features
- Subjects: 47 valid subjects
- Outlier handling: Robust scaling (keep outliers)
- FL strategies: FedAvg, FedProx, FedClusterProxXAI

---

## Quick Start

### 1. Verify Setup
```bash
python run_experiment_v2.py
```

This will:
- Validate configuration
- Create directory structure
- Display next steps

### 2. Prepare Data
```bash
cd experiments/cgm_fl_benchmark_v2/data_preparation
python prepare_data.py --feature_set core_24 --output_dir ../data_prepared
```

### 3. Run Experiments
```bash
# See EXPERIMENT_PLAN_V2.md for detailed experiment designs
# Or navigate to specific experiment directory
cd experiments/feature_comparison/
# Run comparison experiment
```

---

## Feature Sets Available

### Core 24 (RECOMMENDED ✅)
- 24 features with ≥70% consistency
- Balanced across categories
- High correlation with target
- **Use this for primary experiments**

### Top 10 (Minimal)
- 10 highest composite score features
- Fastest training
- Good for baseline comparison
- **Use for computational efficiency tests**

### Temporal-Focused (10 features)
- Focused on temporal patterns
- Emphasizes circadian rhythms
- **Use for temporal pattern analysis**

### All Features (42 features)
- Original feature set
- Baseline for comparison
- **Use only for comparison experiments**

---

## Next Steps

### Immediate (Week 1)
1. ✅ Directory structure created
2. ✅ Feature selection module implemented
3. ✅ Data preparation script ready
4. ⏳ Prepare data for all feature sets
5. ⏳ Update model architectures for 24 features

### Short-term (Weeks 2-3)
1. ⏳ Run feature comparison experiments
2. ⏳ Evaluate redundancy impact
3. ⏳ Test model architectures
4. ⏳ Compare FL strategies

### Medium-term (Weeks 4-5)
1. ⏳ Comprehensive evaluation
2. ⏳ Feature importance validation (SHAP/LIME)
3. ⏳ Performance optimization
4. ⏳ Results documentation

---

## References

- **Feature Analysis:** `../cgm_fl_benchmark/feature_visualizations/COMPREHENSIVE_DATA_ANALYSIS_REPORT.md`
- **Experiment Plan:** `../../EXPERIMENT_PLAN_V2.md`
- **Configuration:** `../../configs/experiment_config_v2.json`

---

**Setup Date:** 2025-11-02  
**Status:** Ready for Implementation  
**Version:** 2.0

