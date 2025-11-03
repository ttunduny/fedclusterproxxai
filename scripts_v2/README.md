# Scripts V2 - Feature-Optimized Implementation

This directory contains runner scripts for the feature-optimized (V2) implementation.

## Scripts

- **`run_experiment_v2.py`** - Main V2 experiment runner (24 core features)

## Usage

```bash
# Run main V2 experiment
python run_experiment_v2.py
```

## Configuration

Uses configurations from `configs_v2/`:
- `experiment_config.json` (feature-optimized)
- `data_config.json` (with feature selection settings)

## Source Code

Uses source code from `src_v2/`:
- 24 core features (consistently important)
- Robust scaling (median/IQR)
- Feature selection integrated
- 47 validated subjects only

## Features

- **Feature Selection:** Automatic selection of 24 core features
- **Robust Scaling:** Median/IQR-based normalization
- **Redundancy Removal:** Automatic removal of redundant features
- **Subject Validation:** Only uses 47 validated subjects
- **Feature Interactions:** Optional interaction features

---

*Version: 2.0*  
*Status: Feature-Optimized*  
*Based on: Feature analysis results*

