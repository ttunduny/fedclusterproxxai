# How to Proceed - Step-by-Step Guide
**From Framework to Publication-Ready Results**

---

## Prerequisites Check

Before starting, ensure you have:
- ✅ Processed subject data in `data/processed/subjects/` (54 subjects available)
- ✅ Python 3.10+ with required packages (TensorFlow, Flower, pandas, numpy, etc.)
- ✅ Publication framework files created
- ✅ Access to compute resources (experiments may take hours/days)

---

## Step 1: Data Quality Assurance (REQUIRED FIRST STEP)

**Purpose:** Validate all processed data meets publication quality standards

**Action:**
```bash
cd /Users/gtlabs/Desktop/Experiments2/experiments_v2/cgm_fl_benchmark
python data_quality_assurance.py
```

**Expected Output:**
- `data/processed/quality_assurance_report.json` - Detailed validation results
- `data/processed/quality_assurance_report.csv` - Summary table
- `data/processed/high_quality_subjects.json` - List of subjects passing all checks
- `data/processed/quality_assurance.log` - Detailed log file

**What to Check:**
- Number of subjects that passed (should be ≥50 for publication)
- Review any failed subjects' issues in the report
- Note which subjects to use for experiments

**Time:** ~5-10 minutes

---

## Step 2: Review Publication Framework

**Purpose:** Understand the experimental design before running experiments

**Action:**
```bash
# Read the framework document
open experiments_v2/cgm_fl_benchmark/NATURE_PUBLICATION_EXPERIMENTAL_FRAMEWORK.md

# Or review the summary
open experiments_v2/cgm_fl_benchmark/PUBLICATION_SUMMARY.md
```

**Key Points to Understand:**
- Baseline methods: FedAvg, FedProx, FedSGD, FedCluster, SOTA methods
- Novel method: FedClusterProxXAI (should outperform all)
- Ablation studies: Component-wise analysis
- Statistical tests: Friedman, Wilcoxon, effect sizes
- Expected results: RMSE ≤15 mg/dL, TIR ≥87%

**Time:** 15-30 minutes to review

---

## Step 3: Run Data Quality Assurance Results Review

**Purpose:** Ensure you have high-quality data before proceeding

**Action:**
```bash
# Check how many subjects passed
python -c "
import json
with open('data/processed/high_quality_subjects.json', 'r') as f:
    data = json.load(f)
    print(f'High-quality subjects: {len(data[\"high_quality_subjects\"])}/{data[\"total_subjects\"]}')
    print(f'\nFirst 10 subjects:')
    for subj in data['high_quality_subjects'][:10]:
        print(f'  - {subj}')
"
```

**Decision Point:**
- ✅ If ≥50 subjects passed: Proceed to Step 4
- ⚠️ If 30-49 subjects passed: Proceed but note in results
- ❌ If <30 subjects passed: Fix data quality issues first

---

## Step 4: Initial Configuration Check

**Purpose:** Verify experiment configurations are set correctly

**Action:**
```bash
# Check if experiment runner exists and is executable
cd experiments_v2/cgm_fl_benchmark
python -c "
import sys
sys.path.append('../../src_v2')
from run_publication_experiments import PublicationExperimentRunner
print('✓ Publication experiment runner is ready')
print('✓ All imports successful')
"
```

**If errors occur:**
- Check that `src_v2` directory exists
- Verify all required packages are installed
- Review import paths in `run_publication_experiments.py`

---

## Step 5: Start with Baseline Experiments (RECOMMENDED FIRST)

**Purpose:** Establish performance baselines before testing novel method

**Action:**
```bash
cd experiments_v2/cgm_fl_benchmark
python run_publication_experiments.py
```

**Select Option 1:** Baseline Experiments

**What This Does:**
- Runs FedAvg, FedProx, FedSGD, FedCluster (4 methods)
- Each method runs 10 times (different random seeds)
- Saves results to `publication/baselines/`
- Estimated time: 2-6 hours (depending on compute)

**Monitor Progress:**
```bash
# In another terminal, check logs
tail -f publication/logs/baseline_experiments.log
```

**What to Expect:**
- Baseline RMSE: ~16-18 mg/dL (FedAvg)
- Baseline TIR: ~82-86%
- These become your comparison targets

**Time:** 2-6 hours for all baselines

---

## Step 6: Verify Baseline Results

**Purpose:** Ensure baselines are reasonable before proceeding

**Action:**
```bash
# Check baseline results summary
python -c "
import json
import os

baseline_results = {}
for method in ['fedavg', 'fedprox', 'fedsgd', 'fedcluster']:
    results_file = f'publication/baselines/{method}/results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            data = json.load(f)
            baseline_results[method] = {
                'rmse': data.get('test_metrics', {}).get('rmse', 'N/A'),
                'tir': data.get('test_metrics', {}).get('time_in_range_accuracy', 'N/A')
            }

print('Baseline Results Summary:')
for method, metrics in baseline_results.items():
    print(f'  {method.upper()}:')
    print(f'    RMSE: {metrics[\"rmse\"]}')
    print(f'    TIR: {metrics[\"tir\"]}')
"
```

**Decision Point:**
- ✅ If RMSE ~16-18 mg/dL: Baselines are reasonable, proceed
- ⚠️ If RMSE >20 mg/dL: Check data preprocessing
- ⚠️ If RMSE <10 mg/dL: Check for data leakage

---

## Step 7: Run Novel Method Experiments (FEDCLUSTERPROXXAI)

**Purpose:** Test your novel method against established baselines

**Action:**
```bash
python run_publication_experiments.py
```

**Select Option 2:** Novel Method Experiments

**What This Does:**
- Runs FedClusterProxXAI (novel method)
- Runs 10 times with different random seeds
- Saves results to `publication/novel_method/`
- Estimated time: 3-8 hours (depends on complexity)

**Monitor Progress:**
```bash
tail -f publication/logs/novel_method_experiments.log
```

**What to Expect:**
- Novel RMSE: ≤15.0 mg/dL (target)
- Novel TIR: ≥87% (target)
- Improvement: ≥10% over FedAvg

**Time:** 3-8 hours for novel method

---

## Step 8: Verify Novel Method Outperforms Baselines

**Purpose:** Confirm your method achieves target improvements

**Action:**
```bash
# Compare novel method vs baseline
python -c "
import json
import os

# Load baseline (FedAvg)
with open('publication/baselines/fedavg/results.json', 'r') as f:
    baseline = json.load(f)

# Load novel method
with open('publication/novel_method/results.json', 'r') as f:
    novel = json.load(f)

baseline_rmse = baseline.get('test_metrics', {}).get('rmse', 0)
novel_rmse = novel.get('test_metrics', {}).get('rmse', 0)

improvement = ((baseline_rmse - novel_rmse) / baseline_rmse) * 100

print(f'Baseline (FedAvg) RMSE: {baseline_rmse:.2f} mg/dL')
print(f'Novel Method RMSE: {novel_rmse:.2f} mg/dL')
print(f'Improvement: {improvement:.1f}%')
print()
if improvement >= 10:
    print('✅ Target achieved! (≥10% improvement)')
else:
    print('⚠️ Below target. Consider hyperparameter tuning.')
"
```

**Decision Point:**
- ✅ If improvement ≥10%: Excellent! Proceed to ablation
- ⚠️ If improvement 5-9%: Acceptable, but consider tuning
- ❌ If improvement <5%: Review method implementation

---

## Step 9: Run Ablation Studies

**Purpose:** Quantify contribution of each component (clustering, proximal, XAI)

**Action:**
```bash
python run_publication_experiments.py
```

**Select Option 3:** Ablation Studies

**What This Does:**
- Runs 4 ablation variants:
  1. FedClusterProx (no XAI)
  2. FedClusterXAI (no Proximal)
  3. FedClusterProxXAI (fixed μ, no adaptive)
  4. FedAvg+Adaptiveμ (no clustering)
- Each runs 10 times
- Saves results to `publication/ablation/`

**Time:** 8-16 hours for all ablation variants

---

## Step 10: Run Statistical Analysis

**Purpose:** Generate publication-ready statistical comparisons

**Action:**
```bash
python run_publication_experiments.py
```

**Select Option 5:** Statistical Analysis Only

**What This Does:**
- Aggregates all results (baselines + novel + ablation)
- Performs statistical tests:
  - Friedman test (non-parametric)
  - Wilcoxon signed-rank test (pairwise)
  - Effect sizes (Cohen's d)
  - Bootstrap confidence intervals
- Generates comparison tables (CSV, Excel)
- Creates visualization-ready summaries

**Output:**
- `publication/results/statistical_comparison.csv`
- `publication/results/statistical_comparison.xlsx`
- `publication/results/statistical_summary.json`

**Time:** 5-15 minutes

---

## Step 11: Review Results for Publication

**Purpose:** Prepare results for manuscript submission

**Action:**
```bash
# Review statistical comparison
open publication/results/statistical_comparison.xlsx

# Check all required metrics
python -c "
import json
import pandas as pd

# Load statistical summary
with open('publication/results/statistical_summary.json', 'r') as f:
    stats = json.load(f)

print('Publication Readiness Checklist:')
print('=' * 50)
print(f'✓ Statistical significance: {stats.get(\"significance\", \"Not checked\")}')
print(f'✓ Effect size (Cohen\'s d): {stats.get(\"effect_size\", \"Not calculated\")}')
print(f'✓ Primary metrics: {stats.get(\"primary_metrics\", \"Not available\")}')
print()
print('Review publication/results/ for complete analysis.')
"
```

**What to Check:**
- ✅ Statistical significance (p < 0.05)
- ✅ Effect size (Cohen's d > 0.8 for large effect)
- ✅ All required metrics present
- ✅ Results tables formatted correctly

---

## Step 12: Generate Final Report

**Purpose:** Create comprehensive results document

**Action:**
```bash
# Generate publication-ready summary
python -c "
import json
import os
from datetime import datetime

# Collect all results
report = {
    'generation_date': datetime.now().isoformat(),
    'experiments': {},
    'statistical_comparison': {}
}

# Load baselines
for method in ['fedavg', 'fedprox', 'fedsgd', 'fedcluster']:
    results_file = f'publication/baselines/{method}/results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            report['experiments'][method] = json.load(f)

# Load novel method
novel_file = 'publication/novel_method/results.json'
if os.path.exists(novel_file):
    with open(novel_file, 'r') as f:
        report['experiments']['fedclusterproxxai'] = json.load(f)

# Load statistical summary
stats_file = 'publication/results/statistical_summary.json'
if os.path.exists(stats_file):
    with open(stats_file, 'r') as f:
        report['statistical_comparison'] = json.load(f)

# Save comprehensive report
with open('publication/FINAL_RESULTS_REPORT.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print('✓ Final results report generated: publication/FINAL_RESULTS_REPORT.json')
"
```

---

## Timeline Summary

| Step | Task | Time | Priority |
|------|------|------|----------|
| 1 | Data Quality Assurance | 5-10 min | **Required** |
| 2 | Review Framework | 15-30 min | **Required** |
| 3 | Review Quality Results | 5 min | **Required** |
| 4 | Configuration Check | 5 min | **Required** |
| 5 | Baseline Experiments | 2-6 hours | **Required** |
| 6 | Verify Baselines | 10 min | **Required** |
| 7 | Novel Method Experiments | 3-8 hours | **Required** |
| 8 | Verify Novel Method | 10 min | **Required** |
| 9 | Ablation Studies | 8-16 hours | Recommended |
| 10 | Statistical Analysis | 5-15 min | **Required** |
| 11 | Review Results | 30-60 min | **Required** |
| 12 | Generate Final Report | 10 min | Recommended |

**Total Time (Minimum):** ~6-15 hours  
**Total Time (Complete):** ~14-25 hours

---

## Quick Start (Minimal Path)

If you want to quickly verify the setup works:

```bash
# Step 1: Quality assurance (required)
cd experiments_v2/cgm_fl_benchmark
python data_quality_assurance.py

# Step 2: Run a single baseline test (optional, for verification)
python run_publication_experiments.py
# Select Option 1, then select only FedAvg, single run

# Step 3: Check results
ls -lh publication/baselines/fedavg/
```

---

## Troubleshooting

### Issue: Data quality assurance fails for many subjects
**Solution:**
- Check data preprocessing pipeline
- Review `quality_assurance_report.json` for specific issues
- Consider relaxing some checks (but document why)

### Issue: Baseline experiments fail or produce errors
**Solution:**
- Check TensorFlow/Flower installation
- Verify subject data files are accessible
- Review logs in `publication/logs/`

### Issue: Novel method doesn't outperform baselines
**Solution:**
- Review hyperparameters (cluster count, μ schedule)
- Check feature set (should be 24 core features)
- Consider additional tuning

### Issue: Statistical tests fail
**Solution:**
- Ensure you have ≥10 runs per method
- Check that results files exist
- Verify statistical analysis script

---

## Next Steps After Experiments

1. **Manuscript Preparation:**
   - Write Methods section (use framework document)
   - Create Results section (use statistical comparisons)
   - Generate figures from results

2. **Supplementary Materials:**
   - Prepare complete hyperparameter tables
   - Document preprocessing pipeline
   - Include code repository links

3. **Code & Data Availability:**
   - Prepare GitHub repository
   - Create documentation (README, API docs)
   - Optionally create Docker environment

---

**Ready to start? Begin with Step 1!**

```bash
cd experiments_v2/cgm_fl_benchmark
python data_quality_assurance.py
```

