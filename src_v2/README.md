# Source Code V2 - Feature-Optimized Implementation

This directory contains the feature-optimized source code based on comprehensive feature analysis.

## Files

- **`data_processing.py`** - Enhanced data processing with feature selection (24 core features)
- **`fl_benchmark.py`** - Updated benchmark runner for feature-optimized experiments
- **`strategies.py`** - Enhanced FL strategies with feature-optimized models
- **`generate_explanations.py`** - Enhanced XAI with feature importance validation
- **`utils.py`** - Enhanced utilities with feature analysis functions

## Features

- **Core 24 Features** - Consistently important features (≥70% consistency)
- **Feature Selection** - Automatic feature selection and redundancy removal
- **Robust Scaling** - Median/IQR-based scaling (outlier-aware)
- **Feature Interactions** - Interaction feature engineering
- **47 Valid Subjects** - Uses validated subjects only

## Key Differences from V1

1. **Feature Selection**: Uses 24 core features instead of all 42
2. **Robust Scaling**: Median/IQR instead of mean/std
3. **Redundancy Removal**: Automatic removal of redundant features
4. **Interaction Features**: Creates feature interactions based on analysis
5. **Subject Validation**: Only uses 47 validated subjects

## Usage

See `scripts_v2/run_experiment_v2.py` for running V2 experiments.

---

*Version: 2.0*  
*Status: Feature-optimized implementation*  
*Based on: Feature analysis (47 subjects, 24 core features)*

