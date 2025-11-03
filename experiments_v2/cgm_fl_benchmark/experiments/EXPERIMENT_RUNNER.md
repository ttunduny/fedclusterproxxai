# Experiment Runner Guide - V2
**Feature-Optimized CGM Prediction Experiments**

---

## Experiment Overview

This directory contains experiment runners for the four main experiment types defined in the experimental plan.

### Experiment Types

1. **Feature Comparison** (`feature_comparison/`)
   - Compare different feature sets
   - Conditions: all_42, core_24, top_10, temporal_focused

2. **Redundancy Impact** (`redundancy_impact/`)
   - Evaluate impact of removing redundant features
   - Conditions: with_redundant, without_redundant

3. **Architecture Comparison** (`architecture_comparison/`)
   - Compare model architectures
   - Conditions: lstm_optimized, transformer_enhanced, lstm_baseline

4. **FL Strategy Comparison** (`fl_strategy_comparison/`)
   - Compare federated learning strategies
   - Conditions: fedavg_core, fedprox_core, fedclusterproxxai_core

---

## Running Experiments

### Prerequisites

1. Data prepared: `python data_preparation/prepare_data.py --feature_set core_24`
2. Models configured: Update model configurations for each feature set
3. Environment: Python 3.10+, TensorFlow 2.15+, Flower (FLWR)

### Example Commands

```bash
# Feature comparison experiment
cd experiments/feature_comparison/
python run_comparison.py --config ../../../configs/experiment_config_v2.json

# Redundancy impact
cd experiments/redundancy_impact/
python run_experiment.py --with_redundant --without_redundant

# Architecture comparison
cd experiments/architecture_comparison/
python compare_architectures.py

# FL strategy comparison
cd experiments/fl_strategy_comparison/
python compare_strategies.py
```

---

## Expected Results

Results will be saved in `../results/by_experiment/` with subdirectories for each experiment type.

---

*See EXPERIMENT_PLAN_V2.md for detailed experiment designs*

