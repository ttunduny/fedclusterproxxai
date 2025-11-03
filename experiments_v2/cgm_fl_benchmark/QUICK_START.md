# Quick Start - Publication Experiments
**Get started in 3 steps**

---

## 🚀 Start Here

### Step 1: Validate Your Data (5 minutes)
```bash
cd /Users/gtlabs/Desktop/Experiments2/experiments_v2/cgm_fl_benchmark
python data_quality_assurance.py
```

**Expected:** ≥50 subjects pass validation  
**Output:** `data/processed/high_quality_subjects.json`

---

### Step 2: Run Baseline Experiments (2-6 hours)
```bash
python run_publication_experiments.py
# Select: Option 1 (Baseline Experiments)
```

**What happens:**
- Runs FedAvg, FedProx, FedSGD, FedCluster
- 10 runs per method (for statistical significance)
- Results saved to `publication/baselines/`

**Expected results:**
- FedAvg RMSE: ~16-18 mg/dL
- FedAvg TIR: ~82-86%

---

### Step 3: Run Novel Method (3-8 hours)
```bash
python run_publication_experiments.py
# Select: Option 2 (Novel Method Experiments)
```

**What happens:**
- Runs FedClusterProxXAI (your novel method)
- 10 runs (for statistical significance)
- Results saved to `publication/novel_method/`

**Target results:**
- RMSE: ≤15.0 mg/dL (≥10% improvement over FedAvg)
- TIR: ≥87%

---

### Step 4: Statistical Analysis (5 minutes)
```bash
python run_publication_experiments.py
# Select: Option 5 (Statistical Analysis Only)
```

**What happens:**
- Aggregates all results
- Performs statistical tests (Friedman, Wilcoxon)
- Generates comparison tables

**Output:** `publication/results/statistical_comparison.xlsx`

---

## 📊 Check Your Results

After running experiments, check:
```bash
# View statistical comparison
open publication/results/statistical_comparison.xlsx

# Check if targets met
python -c "
import json
with open('publication/results/statistical_summary.json', 'r') as f:
    stats = json.load(f)
    print('Novel Method vs FedAvg:')
    print(f'  Improvement: {stats.get(\"improvement_pct\", \"N/A\")}%')
    print(f'  Statistical significance: {stats.get(\"significance\", \"N/A\")}')
"
```

---

## 📚 Documentation

- **`HOW_TO_PROCEED.md`** - Detailed step-by-step guide
- **`NATURE_PUBLICATION_EXPERIMENTAL_FRAMEWORK.md`** - Complete experimental design
- **`PUBLICATION_SUMMARY.md`** - Quick reference

---

## ⚡ Estimated Timeline

| Task | Time |
|------|------|
| Data Quality Assurance | 5-10 min |
| Baseline Experiments | 2-6 hours |
| Novel Method | 3-8 hours |
| Statistical Analysis | 5 min |
| **Total (Minimum)** | **6-15 hours** |

---

## ✅ Success Criteria

Your experiments are successful if:
- ✅ Novel method RMSE ≤15.0 mg/dL
- ✅ Novel method improvement ≥10% over FedAvg
- ✅ Statistical significance: p < 0.05
- ✅ Effect size: Cohen's d > 0.8

---

**Ready? Start with Step 1!**

