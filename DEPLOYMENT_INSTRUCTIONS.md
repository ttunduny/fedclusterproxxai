# Deployment Instructions - Server Setup

## Quick Deployment Steps

### 1. On Local Machine - Push to Repository
```bash
cd /Users/gtlabs/Desktop/Experiments2
git push origin main
```

### 2. On Server - Clone/Pull Repository
```bash
# If first time:
git clone <repository-url> Experiments2
cd Experiments2

# If already cloned:
git pull origin main
```

### 3. Install Dependencies
```bash
# Install Python packages
pip install tensorflow flwr pandas numpy scipy scikit-learn openpyxl matplotlib seaborn

# OR use requirements.txt if available:
pip install -r requirements.txt
```

### 4. Verify Setup
```bash
cd experiments_v2/cgm_fl_benchmark
python quick_test.py
```

Expected output: All tests should pass ✓

### 5. Run Experiments

#### Option A: Using Script (Recommended)
```bash
cd /path/to/Experiments2

# Baseline experiments (2-6 hours)
./run_on_server.sh baseline

# Novel method (3-8 hours)
./run_on_server.sh novel

# Ablation studies (8-16 hours)
./run_on_server.sh ablation

# Full suite (14-25 hours)
./run_on_server.sh all
```

#### Option B: Background Execution (For Long Runs)
```bash
# Run in background with nohup
nohup ./run_on_server.sh baseline > baseline_run_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Monitor progress
tail -f baseline_run_*.log

# Check if process is running
ps aux | grep "run_publication_experiments"
```

#### Option C: Using Screen/Tmux (Recommended for Remote Servers)
```bash
# Start a screen session
screen -S experiments

# Run experiments
cd experiments_v2/cgm_fl_benchmark
python run_publication_experiments.py
# Select option 1, 2, 3, or 4

# Detach: Ctrl+A, then D
# Reattach: screen -r experiments
```

### 6. Monitor Progress

```bash
# Watch log files
tail -f experiments_v2/cgm_fl_benchmark/publication/logs/publication_experiments.log

# Check completed runs
find experiments_v2/cgm_fl_benchmark/publication/baselines -name "results.json" | wc -l

# Expected for baselines: 40 files (4 methods × 10 runs)

# Check latest results
ls -lth experiments_v2/cgm_fl_benchmark/publication/baselines/fedavg/run_*/results.json | head -5
```

### 7. Check Results

```bash
# View aggregated results
cat experiments_v2/cgm_fl_benchmark/publication/results/all_baselines.json

# View statistical comparison
cat experiments_v2/cgm_fl_benchmark/publication/results/statistical_comparison.csv
```

## Directory Structure on Server

```
Experiments2/
├── run_on_server.sh              # Main server script
├── SERVER_SETUP.md               # Detailed server instructions
├── experiments_v2/
│   └── cgm_fl_benchmark/
│       ├── run_publication_experiments.py  # Main experiment runner
│       ├── quick_test.py                  # Quick test script
│       ├── data_quality_assurance.py       # Data validation
│       └── publication/                    # Results directory
│           ├── baselines/                  # Baseline results
│           ├── novel_method/              # Novel method results
│           ├── ablation/                  # Ablation results
│           ├── results/                   # Aggregated results
│           └── logs/                      # Experiment logs
└── src_v2/                       # Source code (v2)
```

## Troubleshooting

### Issue: Module not found
```bash
# Check Python path
python3 --version
which python3

# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src_v2"
```

### Issue: Out of Memory
```bash
# Edit config to reduce batch size or clients
nano configs_v2/experiment_config.json
# Reduce: batch_size, clients_per_round
```

### Issue: Experiments Interrupted
```bash
# Check if process is still running
ps aux | grep python

# Resume from last completed run
# (Script will skip already completed runs)
python run_publication_experiments.py
```

### Issue: Data Path Errors
```bash
# Ensure data exists
ls -la data/processed/subjects/*.xlsx | wc -l
# Should show 57 files

# Run data quality check
cd experiments_v2/cgm_fl_benchmark
python data_quality_assurance.py
```

## Expected Timeline

| Experiment | Time | Location |
|------------|------|----------|
| Baseline | 2-6 hours | `publication/baselines/` |
| Novel Method | 3-8 hours | `publication/novel_method/` |
| Ablation | 8-16 hours | `publication/ablation/` |
| Full Suite | 14-25 hours | `publication/` |

## Resource Requirements

- **CPU:** Multi-core recommended
- **RAM:** 8GB+ recommended
- **Storage:** ~5GB for results
- **GPU:** Optional (TensorFlow will use if available)

## Success Criteria

Experiments are successful when:
- ✓ All runs complete without errors
- ✓ Results files exist: `results.json` in each run directory
- ✓ Aggregated results generated: `all_baselines.json`
- ✓ Statistical comparison created: `statistical_comparison.csv`

## Next Steps After Experiments

1. **Download Results:**
   ```bash
   # Compress results for download
   tar -czf publication_results_$(date +%Y%m%d).tar.gz experiments_v2/cgm_fl_benchmark/publication/
   ```

2. **Run Statistical Analysis:**
   ```bash
   cd experiments_v2/cgm_fl_benchmark
   python run_publication_experiments.py
   # Select option 5: Statistical Analysis Only
   ```

3. **Generate Final Report:**
   ```bash
   # Review results
   cat publication/results/statistical_comparison.csv
   ```

---

**Status:** Ready for deployment ✓

