#!/usr/bin/env python3
"""
Publication-Quality Experiment Runner
Runs comprehensive experiments for Nature publication submission
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Add paths - use absolute paths for reliability
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up 2 levels: cgm_fl_benchmark -> experiments_v2 -> Experiments2
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
src_v2_path = os.path.join(project_root, 'src_v2')
scripts_v2_path = os.path.join(project_root, 'scripts_v2')

# Debug: print paths (comment out in production)
# print(f"Script dir: {script_dir}")
# print(f"Project root: {project_root}")
# print(f"src_v2 path: {src_v2_path}")

if src_v2_path not in sys.path:
    sys.path.insert(0, src_v2_path)
if scripts_v2_path not in sys.path:
    sys.path.insert(0, scripts_v2_path)

from data_processing import CGMDataProcessor, DataConfig
from fl_benchmark import BenchmarkRunner, ExperimentConfig
try:
    from utils import setup_logging
except ImportError:
    # Fallback if setup_logging not available
    def setup_logging():
        pass

class PublicationExperimentRunner:
    """Publication-quality experiment runner with rigorous protocols"""
    
    def __init__(self, 
                 experiment_dir: str = None,
                 random_seed: int = 42,
                 num_runs: int = 10):
        """
        Initialize publication experiment runner
        
        Args:
            experiment_dir: Directory for publication experiments (default: publication/ in script directory)
            random_seed: Random seed for reproducibility
            num_runs: Number of independent runs for statistical significance
        """
        if experiment_dir is None:
            # Default to publication/ directory in same location as script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            experiment_dir = os.path.join(script_dir, 'publication')
        self.experiment_dir = Path(experiment_dir)
        self.random_seed = random_seed
        self.num_runs = num_runs
        
        # Create directory structure FIRST (before logging)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / 'baselines').mkdir(exist_ok=True)
        (self.experiment_dir / 'novel_method').mkdir(exist_ok=True)
        (self.experiment_dir / 'ablation').mkdir(exist_ok=True)
        (self.experiment_dir / 'sota').mkdir(exist_ok=True)
        (self.experiment_dir / 'results').mkdir(exist_ok=True)
        (self.experiment_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.experiment_dir / 'logs').mkdir(exist_ok=True)
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        log_file = self.experiment_dir / 'logs' / 'publication_experiments.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_baseline_experiments(self) -> Dict:
        """
        Run all baseline methods with identical conditions
        
        Returns:
            Dictionary of results for all baselines
        """
        self.logger.info("=" * 80)
        self.logger.info("BASELINE EXPERIMENTS - Publication Quality")
        self.logger.info("=" * 80)
        
        baselines = {
            'fedavg': {'name': 'FedAvg', 'params': {}},
            'fedprox': {'name': 'FedProx', 'params': {'mu': 0.1}},
            'fedsgd': {'name': 'FedSGD', 'params': {}},
            'fedcluster': {'name': 'FedCluster', 'params': {'num_clusters': 3}},
        }
        
        all_results = {}
        
        for baseline_id, baseline_config in baselines.items():
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"Running: {baseline_config['name']}")
            self.logger.info(f"{'=' * 80}")
            
            # Run multiple times for statistical significance
            baseline_results = []
            
            for run_id in range(self.num_runs):
                self.logger.info(f"\nRun {run_id + 1}/{self.num_runs}")
                
                # Set random seed for this run
                np.random.seed(self.random_seed + run_id)
                import tensorflow as tf
                tf.random.set_seed(self.random_seed + run_id)
                
                try:
                    # Load configuration
                    config = self._load_experiment_config(baseline_id)
                    
                    # Get absolute path to data directory
                    script_dir = Path(__file__).parent
                    project_root = script_dir.parent.parent
                    data_path = str(project_root / 'data' / 'processed')
                    
                    # Initialize and run experiment
                    runner = BenchmarkRunner(config, data_path)
                    results = runner.run_benchmark()
                    
                    # Save results
                    run_dir = self.experiment_dir / 'baselines' / baseline_id / f'run_{run_id}'
                    run_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(run_dir / 'results.json', 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    
                    baseline_results.append(results)
                    self.logger.info(f"✓ Run {run_id + 1} completed")
                    
                except Exception as e:
                    self.logger.error(f"✗ Run {run_id + 1} failed: {e}")
                    continue
            
            # Aggregate results
            if baseline_results:
                aggregated = self._aggregate_results(baseline_results, baseline_id)
                all_results[baseline_id] = aggregated
                
                # Save aggregated results
                with open(self.experiment_dir / 'baselines' / f'{baseline_id}_aggregated.json', 'w') as f:
                    json.dump(aggregated, f, indent=2, default=str)
        
        # Save all baseline results
        with open(self.experiment_dir / 'results' / 'all_baselines.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        self.logger.info(f"\n✓ Baseline experiments complete!")
        return all_results
    
    def run_novel_method_experiments(self) -> Dict:
        """
        Run FedClusterProxXAI with optimal hyperparameters
        
        Returns:
            Dictionary of results for novel method
        """
        self.logger.info("=" * 80)
        self.logger.info("NOVEL METHOD EXPERIMENTS - FedClusterProxXAI")
        self.logger.info("=" * 80)
        
        novel_results = []
        
        for run_id in range(self.num_runs):
            self.logger.info(f"\nRun {run_id + 1}/{self.num_runs}")
            
            # Set random seed
            np.random.seed(self.random_seed + run_id)
            import tensorflow as tf
            tf.random.set_seed(self.random_seed + run_id)
            
            try:
                # Load FedClusterProxXAI configuration
                config = self._load_experiment_config('fedclusterproxxai')
                
                # Run experiment
                runner = BenchmarkRunner(config, 'data/processed')
                results = runner.run_benchmark()
                
                # Save results
                run_dir = self.experiment_dir / 'novel_method' / f'run_{run_id}'
                run_dir.mkdir(parents=True, exist_ok=True)
                
                with open(run_dir / 'results.json', 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                novel_results.append(results)
                self.logger.info(f"✓ Run {run_id + 1} completed")
                
            except Exception as e:
                self.logger.error(f"✗ Run {run_id + 1} failed: {e}")
                continue
        
        # Aggregate results
        if novel_results:
            aggregated = self._aggregate_results(novel_results, 'fedclusterproxxai')
            
            # Save aggregated results
            with open(self.experiment_dir / 'results' / 'novel_method_aggregated.json', 'w') as f:
                json.dump(aggregated, f, indent=2, default=str)
            
            return aggregated
        
        return {}
    
    def run_ablation_studies(self) -> Dict:
        """
        Run ablation studies to quantify component contributions
        
        Returns:
            Dictionary of ablation results
        """
        self.logger.info("=" * 80)
        self.logger.info("ABLATION STUDIES")
        self.logger.info("=" * 80)
        
        ablation_variants = {
            'fedclusterprox': {'description': 'Clustering + Proximal (no XAI)', 'params': {'xai_enabled': False}},
            'fedclusterxai': {'description': 'Clustering + XAI (no Proximal)', 'params': {'proximal_enabled': False}},
            'fedclusterproxxai_fixed_mu': {'description': 'Fixed μ (no adaptive)', 'params': {'adaptive_mu': False}},
            'fedavg_adaptive_mu': {'description': 'FedAvg + Adaptive μ (no clustering)', 'params': {'num_clusters': 1}},
        }
        
        ablation_results = {}
        
        for variant_id, variant_config in ablation_variants.items():
            self.logger.info(f"\n{'=' * 80}")
            self.logger.info(f"Ablation: {variant_config['description']}")
            self.logger.info(f"{'=' * 80}")
            
            variant_results = []
            
            for run_id in range(self.num_runs):
                self.logger.info(f"Run {run_id + 1}/{self.num_runs}")
                
                # Set random seed
                np.random.seed(self.random_seed + run_id)
                import tensorflow as tf
                tf.random.set_seed(self.random_seed + run_id)
                
                try:
                    # Load configuration with variant parameters
                    config = self._load_experiment_config('fedclusterproxxai', variant_config['params'])
                    
                    # Run experiment
                    runner = BenchmarkRunner(config, 'data/processed')
                    results = runner.run_benchmark()
                    
                    # Save results
                    run_dir = self.experiment_dir / 'ablation' / variant_id / f'run_{run_id}'
                    run_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(run_dir / 'results.json', 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    
                    variant_results.append(results)
                    
                except Exception as e:
                    self.logger.error(f"✗ Run {run_id + 1} failed: {e}")
                    continue
            
            # Aggregate ablation results
            if variant_results:
                aggregated = self._aggregate_results(variant_results, variant_id)
                ablation_results[variant_id] = aggregated
        
        # Save ablation results
        with open(self.experiment_dir / 'results' / 'ablation_studies.json', 'w') as f:
            json.dump(ablation_results, f, indent=2, default=str)
        
        self.logger.info(f"\n✓ Ablation studies complete!")
        return ablation_results
    
    def _load_experiment_config(self, strategy_id: str, variant_params: Dict = None) -> ExperimentConfig:
        """
        Load experiment configuration for a strategy
        
        Args:
            strategy_id: Strategy identifier
            variant_params: Variant-specific parameters for ablation
            
        Returns:
            ExperimentConfig instance
        """
        # Load base V2 configuration
        # Go up 2 levels: cgm_fl_benchmark -> experiments_v2 -> Experiments2
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        config_path = project_root / 'configs_v2' / 'experiment_config.json'
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create experiment config
        config = ExperimentConfig()
        config.__dict__.update(config_dict)
        
        # Update strategy-specific settings
        if variant_params:
            for key, value in variant_params.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Update experiment directory
        config.experiment_dir = str(self.experiment_dir)
        
        return config
    
    def _aggregate_results(self, results_list: List[Dict], method_id: str) -> Dict:
        """
        Aggregate results across multiple runs
        
        Args:
            results_list: List of results dictionaries
            method_id: Method identifier
            
        Returns:
            Aggregated results with statistics
        """
        if not results_list:
            return {}
        
        # Extract metrics
        metrics = ['rmse', 'mae', 'mape', 'r2']
        aggregated = {
            'method': method_id,
            'num_runs': len(results_list),
            'metrics': {}
        }
        
        for metric in metrics:
            values = []
            for result in results_list:
                if 'fedclusterproxxai' in result:
                    strategy_results = result.get('fedclusterproxxai', {})
                else:
                    strategy_results = result.get(method_id, {})
                
                if 'metrics' in strategy_results:
                    metric_value = strategy_results['metrics'].get(metric)
                    if metric_value:
                        values.append(float(metric_value))
            
            if values:
                aggregated['metrics'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'cv': float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 0
                }
        
        return aggregated
    
    def generate_statistical_comparison(self, baseline_results: Dict, novel_results: Dict) -> pd.DataFrame:
        """
        Generate statistical comparison table
        
        Args:
            baseline_results: Results from baseline methods
            novel_results: Results from novel method
            
        Returns:
            DataFrame with statistical comparisons
        """
        from scipy.stats import wilcoxon, friedmanchisquare
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STATISTICAL COMPARISON")
        self.logger.info("=" * 80)
        
        # Prepare data for Friedman test
        all_methods = list(baseline_results.keys()) + ['fedclusterproxxai']
        
        # Extract RMSE values for each method
        rmse_data = {}
        for method_id in all_methods:
            if method_id == 'fedclusterproxxai':
                data = novel_results.get('metrics', {}).get('rmse', {})
            else:
                data = baseline_results.get(method_id, {}).get('metrics', {}).get('rmse', {})
            
            if data:
                rmse_data[method_id] = data
        
        # Friedman test
        if len(rmse_data) >= 3:
            friedman_values = [list(rmse_data[m].values()) for m in rmse_data.keys()]
            # Note: This is simplified - need actual per-run data
            
        # Pairwise Wilcoxon tests
        comparisons = []
        
        if novel_results and 'metrics' in novel_results:
            novel_rmse = novel_results['metrics'].get('rmse', {})
            novel_mean = novel_rmse.get('mean', 0)
            
            for method_id, method_results in baseline_results.items():
                baseline_rmse = method_results.get('metrics', {}).get('rmse', {})
                baseline_mean = baseline_rmse.get('mean', 0)
                
                improvement = ((baseline_mean - novel_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
                
                comparisons.append({
                    'Comparison': f'FedClusterProxXAI vs {method_id.upper()}',
                    'Baseline RMSE': f"{baseline_mean:.2f} ± {baseline_rmse.get('std', 0):.2f}",
                    'Novel RMSE': f"{novel_mean:.2f} ± {novel_rmse.get('std', 0):.2f}",
                    'Improvement %': f"{improvement:.1f}%",
                    'Effect Size (d)': 'TBD',  # Calculate Cohen's d
                    'p-value': 'TBD'  # Calculate Wilcoxon test
                })
        
        comparison_df = pd.DataFrame(comparisons)
        
        # Save comparison table
        comparison_df.to_csv(self.experiment_dir / 'results' / 'statistical_comparison.csv', index=False)
        comparison_df.to_excel(self.experiment_dir / 'results' / 'statistical_comparison.xlsx', index=False)
        
        self.logger.info("\n" + comparison_df.to_string())
        
        return comparison_df

def main():
    """Main execution function"""
    print("=" * 80)
    print("PUBLICATION-QUALITY EXPERIMENT RUNNER")
    print("Nature Journal Standards")
    print("=" * 80)
    
    runner = PublicationExperimentRunner(
        experiment_dir=None,  # Uses default: publication/ in script directory
        random_seed=42,
        num_runs=10  # 10 independent runs for statistical significance
    )
    
    print("\nSelect experiment phase:")
    print("1. Baseline Experiments (FedAvg, FedProx, FedSGD, FedCluster)")
    print("2. Novel Method Experiments (FedClusterProxXAI)")
    print("3. Ablation Studies")
    print("4. Full Experimental Suite (All)")
    print("5. Statistical Analysis Only")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        runner.run_baseline_experiments()
    elif choice == '2':
        runner.run_novel_method_experiments()
    elif choice == '3':
        runner.run_ablation_studies()
    elif choice == '4':
        print("\nRunning full experimental suite...")
        baseline_results = runner.run_baseline_experiments()
        novel_results = runner.run_novel_method_experiments()
        ablation_results = runner.run_ablation_studies()
        
        # Statistical comparison
        comparison_df = runner.generate_statistical_comparison(baseline_results, novel_results)
        
        print("\n" + "=" * 80)
        print("✓ ALL EXPERIMENTS COMPLETE!")
        print("=" * 80)
    elif choice == '5':
        # Load existing results and generate statistics
        baseline_results = json.load(open(runner.experiment_dir / 'results' / 'all_baselines.json'))
        novel_results = json.load(open(runner.experiment_dir / 'results' / 'novel_method_aggregated.json'))
        comparison_df = runner.generate_statistical_comparison(baseline_results, novel_results)
    else:
        print("Invalid choice. Exiting.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

