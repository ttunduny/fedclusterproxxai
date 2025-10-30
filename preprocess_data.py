#!/usr/bin/env python3
"""
Standalone script for CGM data preprocessing
Run this before the main FL benchmark to generate processed subject files
"""

import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import CGMDataProcessor, DataConfig, create_sample_data_loader
from utils import setup_logging

def main():
    """Run data preprocessing only"""
    print("=" * 60)
    print("CGM Data Preprocessing")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Load data configuration
        with open('configs/data_config.json', 'r') as f:
            data_config_dict = json.load(f)
        
        # Create data config
        data_config = DataConfig()
        data_config.__dict__.update(data_config_dict)
        
        print(f"\nConfiguration:")
        print(f"  - Prediction horizon: {data_config.prediction_horizon} steps (60 minutes)")
        print(f"  - History window: {data_config.history_days} days")
        print(f"  - Min samples per subject: {data_config.min_samples_per_subject}")
        
        # Initialize processor
        processor = CGMDataProcessor(data_config)
        
        # Create sample data loader
        # Generates 2 years of data, but processor will use last 15 days
        print(f"\n{'='*60}")
        print("Generating sample CGM data...")
        print(f"{'='*60}\n")
        
        sample_loader = create_sample_data_loader(
            num_subjects=5,      # Number of subjects
            days_per_subject=730  # 2 years of data (will use last 15 days)
        )
        
        # Process data
        processed_data, metadata = processor.process_raw_data(
            data_loader_function=sample_loader
        )
        
        if not processed_data:
            logger.error("❌ Data processing failed. No subjects processed.")
            return 1
        
        # Print summary
        print(f"\n{'='*60}")
        print("✅ PREPROCESSING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"\nResults:")
        print(f"  📊 Subjects processed: {len(processed_data)}")
        print(f"  📁 Output directory: {data_config.processed_data_dir}/subjects/")
        print(f"  📋 Metadata: {data_config.processed_data_dir}/metadata/")
        
        print(f"\nSubject Details:")
        for subject_id, data in processed_data.items():
            num_features = len([col for col in data.columns if col not in ['target', 'subject_id']])
            print(f"  - {subject_id}: {len(data):,} samples, {num_features} features")
        
        print(f"\n{'='*60}")
        print("Next step: Run FL benchmark with:")
        print("  python run_experiment.py")
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


