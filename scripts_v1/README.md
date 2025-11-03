# Scripts V1 - Original Implementation

This directory contains runner scripts for the original (V1) implementation.

## Scripts

- **`run_experiment.py`** - Main experiment runner (all features, baseline)
- **`preprocess_data.py`** - Data preprocessing (original method)
- **`run_feature_analysis_all.py`** - Feature analysis script
- **`create_feature_analysis_md.py`** - Markdown report generation
- **`check_feature_analysis_status.py`** - Analysis status checker
- **`show_feature_analysis_summary.py`** - Analysis summary display
- **`test_feature_explanations.py`** - XAI explanation testing

## Usage

```bash
# Run main experiment
python run_experiment.py

# Preprocess data
python preprocess_data.py

# Run feature analysis
python run_feature_analysis_all.py
```

## Configuration

Uses configurations from `configs_v1/`:
- `experiment_config.json`
- `data_config.json`

## Source Code

Uses source code from `src_v1/`:
- All 42 features
- Standard preprocessing
- Original implementations

---

*Version: 1.0*  
*Status: Baseline*

