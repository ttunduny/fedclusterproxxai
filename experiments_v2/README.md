# Experiments V2 - Feature-Optimized Results

This directory contains results from the feature-optimized (V2) experiments.

## Structure

```
cgm_fl_benchmark/
├── data_preparation/          # Feature selection and preprocessing
├── models/                    # Models by feature set
│   ├── baseline/             # 42 features (comparison)
│   ├── core_24/             # 24 core features (primary)
│   ├── top_10/               # Top 10 features (minimal)
│   └── temporal_focused/     # 10 temporal features
├── experiments/               # Experiment definitions
│   ├── feature_comparison/
│   ├── redundancy_impact/
│   ├── architecture_comparison/
│   └── fl_strategy_comparison/
├── results/                   # Experiment results
│   ├── global/
│   ├── by_experiment/
│   └── comparisons/
└── feature_visualizations/   # Feature analysis plots
```

## Results

- Feature-optimized performance with 24 core features
- Robust preprocessing (median/IQR scaling)
- 47 validated subjects only
- Feature-optimized model architectures
- Multiple experiment comparisons

## Usage

These results demonstrate the impact of feature optimization and can be compared with V1 baseline.

---

*Version: 2.0*  
*Status: Feature-Optimized*

