# Server Setup & Running Instructions

## Quick Start on Server

### 1. Clone/Pull Repository
```bash
git clone <repository-url>
# OR if already cloned:
cd Experiments2
git pull
```

### 2. Install Dependencies
```bash
pip install tensorflow flwr pandas numpy scipy scikit-learn openpyxl matplotlib seaborn
# OR if using requirements.txt:
pip install -r requirements.txt
```

### 3. Verify Data Quality (Optional but Recommended)
```bash
cd experiments_v2/cgm_fl_benchmark
python data_quality_assurance.py
```

### 4. Run Experiments

**Option A: Using the script (Recommended)**
```bash
# From project root
./run_on_server.sh baseline    # Baseline experiments (2-6 hours)
./run_on_server.sh novel       # Novel method (3-8 hours)
./run_on_server.sh ablation    # Ablation studies (8-16 hours)
./run_on_server.sh all         # Full suite (14-25 hours)
```

**Option B: Direct Python execution**
```bash
cd experiments_v2/cgm_fl_benchmark
python run_publication_experiments.py
# Then select option 1, 2, 3, or 4
```

### 5. Run in Background (Recommended for Long Experiments)
```bash
nohup ./run_on_server.sh baseline > baseline_run.log 2>&1 &
# Check progress:
tail -f baseline_run.log
# Or check process:
ps aux | grep python
```

### 6. Monitor Progress
```bash
# Check logs
tail -f experiments_v2/cgm_fl_benchmark/publication/logs/publication_experiments.log

# Check results as they're generated
ls -lh experiments_v2/cgm_fl_benchmark/publication/baselines/*/run_*/results.json

# Check experiment status
python experiments_v2/cgm_fl_benchmark/quick_test.py
```

## Expected Results Location

```
experiments_v2/cgm_fl_benchmark/publication/
├── baselines/
│   ├── fedavg/run_0/results.json
│   ├── fedavg/run_1/results.json
│   └── ... (10 runs per method)
│   ├── fedprox/
│   ├── fedsgd/
│   └── fedcluster/
├── novel_method/
│   └── run_0/... (10 runs)
├── ablation/
│   └── ... (ablation variants)
└── results/
    ├── all_baselines.json
    ├── novel_method_aggregated.json
    └── statistical_comparison.csv
```

## Troubleshooting

### Issue: Module not found
```bash
# Ensure you're using the correct Python path
python3 --version
# Add project to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src_v2"
```

### Issue: Out of memory
```bash
# Reduce batch size in configs_v2/experiment_config.json
# Or reduce number of clients per round
```

### Issue: Experiments taking too long
- Consider reducing `num_runs` from 10 to 5 for initial testing
- Reduce `num_rounds` in config for faster iteration
- Use fewer clients per round

## Resource Requirements

- **Memory:** 8GB+ RAM recommended
- **Storage:** ~5GB for results and logs
- **CPU:** Multi-core recommended (experiments can use parallel processing)
- **GPU:** Optional (TensorFlow can use GPU if available)

## Estimated Runtime

| Experiment Type | Time | Runs per Method |
|----------------|------|-----------------|
| Baseline | 2-6 hours | 10 |
| Novel Method | 3-8 hours | 10 |
| Ablation | 8-16 hours | 10 per variant |
| Full Suite | 14-25 hours | All |

## Status Check

```bash
# Count completed runs
find experiments_v2/cgm_fl_benchmark/publication/baselines -name "results.json" | wc -l

# Expected: 40 results.json files (4 methods × 10 runs)
```

