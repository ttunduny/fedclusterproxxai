#!/usr/bin/env python3
"""
Main execution script for CGM Federated Learning Benchmark V2
Feature-Optimized Experiments
"""
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiments', 'cgm_fl_benchmark_v2', 'data_preparation'))

from utils import setup_logging

def main():
    """Main execution function"""
    print("=" * 80)
    print("CGM Federated Learning Benchmark V2 - Feature Optimized")
    print("=" * 80)
    print("\nBased on Comprehensive Feature Analysis")
    print("  - 24 Core Features (≥70% consistency across 47 subjects)")
    print("  - Outlier Rate: 3.80% (normal, acceptable)")
    print("  - 47 Valid Subjects (10 excluded due to data issues)")
    print("\n" + "=" * 80)
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Load V2 configuration
        config_path = 'configs/experiment_config_v2.json'
        if not os.path.exists(config_path):
            print(f"\n✗ Configuration file not found: {config_path}")
            print("Please ensure configs/experiment_config_v2.json exists")
            return 1
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"\n✓ Loaded configuration: {config['experiment_name']}")
        print(f"  Version: {config['version']}")
        print(f"  Feature Set: {config['feature_selection']['strategy']}")
        print(f"  Core Features: {len(config['feature_selection']['core_features'])}")
        print(f"  Valid Subjects: {config['data_config']['valid_subjects_count']}")
        
        # Create experiment directory structure
        experiment_dir = Path(config['experiment_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        subdirs = [
            'data_preparation',
            'models/baseline', 'models/core_24', 'models/top_10', 'models/temporal_focused',
            'experiments/feature_comparison', 'experiments/redundancy_impact',
            'experiments/architecture_comparison', 'experiments/fl_strategy_comparison',
            'results/global', 'results/by_experiment', 'results/comparisons',
            'logs', 'feature_visualizations', 'checkpoints'
        ]
        
        for subdir in subdirs:
            (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        print(f"\n✓ Experiment directory structure ready: {experiment_dir}")
        
        # Display next steps
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Prepare Data:")
        print("   cd experiments/cgm_fl_benchmark_v2/data_preparation")
        print("   python prepare_data.py --feature_set core_24")
        
        print("\n2. Run Experiments:")
        print("   See EXPERIMENT_PLAN_V2.md for detailed experiment designs")
        print("   Or run individual experiments:")
        print("   - Feature comparison: experiments/feature_comparison/")
        print("   - Redundancy impact: experiments/redundancy_impact/")
        print("   - Architecture comparison: experiments/architecture_comparison/")
        print("   - FL strategy comparison: experiments/fl_strategy_comparison/")
        
        print("\n3. View Results:")
        print(f"   Results will be saved in: {experiment_dir}/results/")
        
        print("\n" + "=" * 80)
        print("✓ V2 Experiment Setup Complete!")
        print("=" * 80)
        
        # Save setup metadata
        setup_metadata = {
            'setup_date': str(datetime.now()),
            'config_path': config_path,
            'experiment_dir': str(experiment_dir),
            'feature_set': config['feature_selection']['strategy'],
            'core_features_count': len(config['feature_selection']['core_features']),
            'valid_subjects_count': config['data_config']['valid_subjects_count'],
            'excluded_subjects': config['data_config']['excluded_subjects']
        }
        
        metadata_path = experiment_dir / 'setup_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(setup_metadata, f, indent=2)
        
        print(f"\n✓ Setup metadata saved to: {metadata_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

