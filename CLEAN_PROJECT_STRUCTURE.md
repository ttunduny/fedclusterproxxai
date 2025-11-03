# Clean Project Structure

**Status:** Cleaned and organized  
**Date:** 2025-01-XX

---

## Active Project Structure

```
Experiments2/
├── README.md                    # Main project README
├── README_V1_V2.md             # Version structure explanation
├── VERSION_STRUCTURE.md        # Detailed version documentation
├── EXPERIMENT_PLAN_V2.md       # Experiment plan
├── requirements.txt            # Python dependencies
│
├── src_v1/                     # V1 Source Code (Original)
│   ├── README.md
│   ├── data_processing.py
│   ├── fl_benchmark.py
│   ├── strategies.py
│   ├── generate_explanations.py
│   └── utils.py
│
├── src_v2/                     # V2 Source Code (Feature-Optimized)
│   ├── README.md
│   ├── data_processing.py
│   ├── fl_benchmark.py
│   ├── strategies.py
│   ├── generate_explanations.py
│   └── utils.py
│
├── scripts_v1/                  # V1 Runner Scripts
│   ├── README.md
│   ├── run_experiment.py
│   └── preprocess_data.py
│
├── scripts_v2/                  # V2 Runner Scripts
│   ├── README.md
│   └── run_experiment_v2.py
│
├── configs_v1/                 # V1 Configuration Files
│   ├── README.md
│   ├── data_config.json
│   └── experiment_config.json
│
├── configs_v2/                  # V2 Configuration Files
│   ├── README.md
│   ├── data_config.json
│   └── experiment_config.json
│
├── experiments_v1/               # V1 Experimental Results
│   ├── README.md
│   └── cgm_fl_benchmark/
│       ├── feature_visualizations/
│       ├── fedavg/
│       ├── fedprox/
│       ├── fedsgd/
│       └── fedclusterproxxai/
│
├── experiments_v2/               # V2 Experimental Results & Publication Framework
│   ├── README.md
│   └── cgm_fl_benchmark/
│       ├── README.md
│       ├── QUICK_START.md                      # Quick start guide
│       ├── HOW_TO_PROCEED.md                   # Detailed step-by-step guide
│       ├── NATURE_PUBLICATION_EXPERIMENTAL_FRAMEWORK.md  # Complete framework
│       ├── PUBLICATION_README.md               # Publication guide
│       ├── PUBLICATION_SUMMARY.md              # Quick reference
│       ├── run_publication_experiments.py      # Main experiment runner
│       ├── data_quality_assurance.py           # Data validation
│       ├── data_preparation/                   # Data preparation scripts
│       ├── experiments/                        # Experiment definitions
│       ├── models/                             # Model architecture docs
│       ├── results/                            # Experimental results
│       └── logs/                               # Experiment logs
│
├── data/                        # Data Directory
│   ├── raw/                     # Raw subject data
│   └── processed/               # Processed subject data
│       ├── subjects/           # 54 processed subject files
│       └── metadata/           # Processing metadata
│
└── archive/                     # Archived Files (NOT part of active codebase)
    ├── README.md
    ├── CLEANUP_SUMMARY.md
    └── old_files/               # All archived files
```

---

## Key Directories

### **Source Code**
- **`src_v1/`** - Original implementation (42 features)
- **`src_v2/`** - Feature-optimized implementation (24 core features)

### **Scripts**
- **`scripts_v1/`** - V1 experiment runners
- **`scripts_v2/`** - V2 experiment runners

### **Configurations**
- **`configs_v1/`** - V1 experiment configurations
- **`configs_v2/`** - V2 experiment configurations

### **Experiments**
- **`experiments_v1/`** - V1 experimental results and outputs
- **`experiments_v2/`** - V2 experimental results + **Publication Framework**

### **Publication Framework**
Location: `experiments_v2/cgm_fl_benchmark/`

**Key Files:**
- `QUICK_START.md` - Fastest way to begin
- `HOW_TO_PROCEED.md` - Complete step-by-step guide
- `NATURE_PUBLICATION_EXPERIMENTAL_FRAMEWORK.md` - Full experimental design
- `run_publication_experiments.py` - Main experiment runner
- `data_quality_assurance.py` - Data validation tool

---

## Next Steps

To start running publication-quality experiments:

```bash
cd experiments_v2/cgm_fl_benchmark
python data_quality_assurance.py
```

See `QUICK_START.md` for immediate next steps.

---

## Archived Files

All unnecessary files have been moved to `archive/old_files/`:
- Old log files
- Redundant documentation
- Superseded source code (`src/`, `configs/`, etc.)
- Old analysis scripts
- Empty directories

These files are preserved for reference but are **NOT** part of the active codebase.

See `archive/CLEANUP_SUMMARY.md` for a complete list.

