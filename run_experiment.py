#!/usr/bin/env python3
"""
Main execution script for CGM Federated Learning Benchmark
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import CGMDataProcessor, DataConfig
from fl_benchmark import BenchmarkRunner, ExperimentConfig
from utils import setup_logging

def main():
    """Main execution function"""
    print("=" * 60)
    print("CGM Federated Learning Benchmark")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Load configurations
        with open('configs/data_config.json', 'r') as f:
            data_config_dict = json.load(f)
        
        with open('configs/experiment_config.json', 'r') as f:
            experiment_config_dict = json.load(f)
        
        # Step 1: Data Processing
        print("\n" + "="*50)
        print("STEP 1: DATA PROCESSING")
        print("="*50)
        
        data_config = DataConfig()
        data_config.__dict__.update(data_config_dict)
        
        processor = CGMDataProcessor(data_config)
        
        # Process ONLY real raw data (no synthetic SUB### subjects)
        processed_data, metadata = processor.process_raw_data(data_loader_function=None)
        
        if not processed_data:
            logger.error("Data processing failed. Exiting.")
            return
        
        # Step 2: Federated Learning Benchmark
        print("\n" + "="*50)
        print("STEP 2: FEDERATED LEARNING BENCHMARK")
        print("="*50)
        
        experiment_config = ExperimentConfig()
        experiment_config.__dict__.update(experiment_config_dict)
        
        runner = BenchmarkRunner(experiment_config, data_config.processed_data_dir)
        results = runner.run_benchmark()
        
        print("\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"📊 Results saved in: experiments/{experiment_config.experiment_name}/")
        print(f"📁 Processed data: {data_config.processed_data_dir}/")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise

if __name__ == "__main__":
    main()
