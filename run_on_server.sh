#!/bin/bash
# Script to run publication experiments on server
# Usage: ./run_on_server.sh [baseline|novel|ablation|all]

set -e

EXPERIMENT_TYPE=${1:-baseline}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$SCRIPT_DIR/experiments_v2/cgm_fl_benchmark"

cd "$EXPERIMENT_DIR"

echo "================================================================================="
echo "PUBLICATION EXPERIMENTS - Server Run"
echo "================================================================================="
echo "Experiment Type: $EXPERIMENT_TYPE"
echo "Script Directory: $EXPERIMENT_DIR"
echo "Timestamp: $(date)"
echo "================================================================================="
echo ""

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import tensorflow, flwr, pandas, numpy" 2>/dev/null || {
    echo "Error: Required packages not installed"
    echo "Please install: pip install tensorflow flwr pandas numpy"
    exit 1
}

# Run experiments based on type
case "$EXPERIMENT_TYPE" in
    baseline)
        echo "Running baseline experiments (FedAvg, FedProx, FedSGD, FedCluster)..."
        echo "This will take 2-6 hours..."
        python3 -c "
from run_publication_experiments import PublicationExperimentRunner
runner = PublicationExperimentRunner(experiment_dir=None, random_seed=42, num_runs=10)
runner.run_baseline_experiments()
" 2>&1 | tee publication/logs/baseline_experiments_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    novel)
        echo "Running novel method experiments (FedClusterProxXAI)..."
        echo "This will take 3-8 hours..."
        python3 -c "
from run_publication_experiments import PublicationExperimentRunner
runner = PublicationExperimentRunner(experiment_dir=None, random_seed=42, num_runs=10)
runner.run_novel_method_experiments()
" 2>&1 | tee publication/logs/novel_method_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    ablation)
        echo "Running ablation studies..."
        echo "This will take 8-16 hours..."
        python3 -c "
from run_publication_experiments import PublicationExperimentRunner
runner = PublicationExperimentRunner(experiment_dir=None, random_seed=42, num_runs=10)
runner.run_ablation_studies()
" 2>&1 | tee publication/logs/ablation_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    all)
        echo "Running full experimental suite..."
        echo "This will take 14-25 hours..."
        python3 run_publication_experiments.py <<EOF
4
EOF
        2>&1 | tee publication/logs/full_suite_$(date +%Y%m%d_%H%M%S).log
        ;;
    
    *)
        echo "Usage: $0 [baseline|novel|ablation|all]"
        exit 1
        ;;
esac

echo ""
echo "================================================================================="
echo "EXPERIMENTS COMPLETE"
echo "================================================================================="
echo "Results saved to: $EXPERIMENT_DIR/publication/"
echo "Timestamp: $(date)"
echo "================================================================================="

