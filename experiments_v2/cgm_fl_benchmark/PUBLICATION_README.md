# Publication-Quality Experiments - Nature Standards

**Purpose:** Rigorous experimental framework for Nature journal submission

---

## Overview

This directory contains publication-quality experiments following Nature journal standards with:
- Comprehensive baselines
- Rigorous statistical testing
- Ablation studies
- Reproducibility protocols

---

## Experimental Framework

See `NATURE_PUBLICATION_EXPERIMENTAL_FRAMEWORK.md` for complete experimental design.

---

## Running Publication Experiments

### Full Experimental Suite
```bash
cd experiments_v2/cgm_fl_benchmark
python run_publication_experiments.py
# Select option 4: Full Experimental Suite
```

### Individual Phases
```bash
# Option 1: Baseline experiments only
# Option 2: Novel method experiments only
# Option 3: Ablation studies only
```

---

## Results Structure

```
publication/
├── baselines/           # Baseline method results (FedAvg, FedProx, etc.)
├── novel_method/        # FedClusterProxXAI results
├── ablation/            # Ablation study results
├── results/             # Aggregated results and statistics
└── checkpoints/         # Model checkpoints
```

---

## Expected Deliverables

1. **Primary Results Table** - All methods on test set
2. **Statistical Comparison Table** - Significance tests
3. **Ablation Study Table** - Component contributions
4. **Visualization Figures** - Convergence, error grids, etc.
5. **Statistical Analysis** - p-values, effect sizes, CIs

---

*See NATURE_PUBLICATION_EXPERIMENTAL_FRAMEWORK.md for complete details*

