#!/usr/bin/env python3
"""
Data Quality Assurance Module
Ensures publication-quality data preprocessing and validation
"""
import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

class DataQualityAssurance:
    """Ensures data quality for publication-quality experiments"""
    
    def __init__(self, processed_data_dir: str = "data/processed/subjects"):
        """
        Initialize data quality assurance
        
        Args:
            processed_data_dir: Directory containing processed subject data
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging"""
        log_file = self.processed_data_dir.parent / 'quality_assurance.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_all_subjects(self) -> Dict:
        """
        Validate all processed subjects for publication quality
        
        Returns:
            Dictionary with validation results for each subject
        """
        self.logger.info("=" * 80)
        self.logger.info("DATA QUALITY ASSURANCE - Publication Quality")
        self.logger.info("=" * 80)
        
        # Find all subject files
        subject_files = list(self.processed_data_dir.glob('Subject_*.xlsx'))
        
        self.logger.info(f"\nFound {len(subject_files)} subject files")
        
        validation_results = {}
        
        for subject_file in subject_files:
            subject_id = subject_file.stem.replace('Subject_', '')
            self.logger.info(f"\nValidating: {subject_id}")
            
            try:
                # Load subject data
                df = pd.read_excel(subject_file, sheet_name='processed_data', index_col=0)
                
                # Run validation checks
                validation = self._validate_subject_data(subject_id, df)
                validation_results[subject_id] = validation
                
                if validation['passed']:
                    self.logger.info(f"✓ {subject_id}: PASSED")
                else:
                    self.logger.warning(f"✗ {subject_id}: FAILED")
                    self.logger.warning(f"  Issues: {validation['issues']}")
                    
            except Exception as e:
                self.logger.error(f"✗ {subject_id}: ERROR - {e}")
                validation_results[subject_id] = {
                    'passed': False,
                    'error': str(e),
                    'issues': [f'Failed to load: {e}']
                }
        
        # Summary
        passed = sum(1 for v in validation_results.values() if v.get('passed', False))
        failed = len(validation_results) - passed
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("VALIDATION SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total subjects: {len(validation_results)}")
        self.logger.info(f"Passed: {passed}")
        self.logger.info(f"Failed: {failed}")
        
        # Save validation report
        report_path = self.processed_data_dir.parent / 'quality_assurance_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'validation_date': datetime.now().isoformat(),
                'total_subjects': len(validation_results),
                'passed': passed,
                'failed': failed,
                'results': validation_results
            }, f, indent=2, default=str)
        
        self.logger.info(f"\n✓ Validation report saved: {report_path}")
        
        return validation_results
    
    def _validate_subject_data(self, subject_id: str, df: pd.DataFrame) -> Dict:
        """
        Validate a single subject's data
        
        Args:
            subject_id: Subject identifier
            df: Subject data DataFrame
            
        Returns:
            Validation results dictionary
        """
        validation = {
            'subject_id': subject_id,
            'passed': True,
            'issues': [],
            'checks': {}
        }
        
        # Check 1: Non-empty DataFrame
        if df.empty:
            validation['passed'] = False
            validation['issues'].append('Empty DataFrame')
            return validation
        
        validation['checks']['non_empty'] = True
        
        # Check 2: Minimum samples (≥500)
        min_samples = 500
        if len(df) < min_samples:
            validation['passed'] = False
            validation['issues'].append(f'Insufficient samples: {len(df)} < {min_samples}')
        else:
            validation['checks']['min_samples'] = True
        
        # Check 3: Required columns
        required_columns = ['target', 'current_glucose']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation['passed'] = False
            validation['issues'].append(f'Missing required columns: {missing_columns}')
        else:
            validation['checks']['required_columns'] = True
        
        # Check 4: No NaN in target
        target_nan_count = df['target'].isna().sum()
        if target_nan_count > 0:
            validation['passed'] = False
            validation['issues'].append(f'NaN in target: {target_nan_count}')
        else:
            validation['checks']['target_no_nan'] = True
        
        # Check 5: Target values in valid range [0, 1] (normalized)
        target_min = df['target'].min()
        target_max = df['target'].max()
        if target_min < 0 or target_max > 1:
            validation['passed'] = False
            validation['issues'].append(f'Target out of range: [{target_min:.3f}, {target_max:.3f}]')
        else:
            validation['checks']['target_range'] = True
        
        # Check 6: Feature completeness (<5% missing)
        feature_columns = [col for col in df.columns if col not in ['target', 'subject_id']]
        max_missing_pct = 5.0
        
        for col in feature_columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            if missing_pct > max_missing_pct:
                validation['passed'] = False
                validation['issues'].append(f'High missing data in {col}: {missing_pct:.1f}%')
        
        if validation['checks'].get('feature_completeness', True):
            validation['checks']['feature_completeness'] = True
        
        # Check 7: Temporal consistency (datetime index)
        if not isinstance(df.index, pd.DatetimeIndex):
            validation['passed'] = False
            validation['issues'].append('Index is not DatetimeIndex')
        else:
            validation['checks']['temporal_index'] = True
            
            # Check for large gaps (>1 hour)
            time_diffs = df.index.to_series().diff()
            large_gaps = (time_diffs > pd.Timedelta(hours=1)).sum()
            if large_gaps > len(df) * 0.05:  # >5% large gaps
                validation['passed'] = False
                validation['issues'].append(f'Too many temporal gaps: {large_gaps}')
        
        # Check 8: Glucose values in physiological range (40-400 mg/dL if not normalized)
        if 'current_glucose' in df.columns:
            glucose_min = df['current_glucose'].min()
            glucose_max = df['current_glucose'].max()
            
            # Check if normalized (0-1) or raw (40-400)
            if glucose_max <= 1.0:
                # Normalized - check range
                if glucose_min < 0 or glucose_max > 1:
                    validation['passed'] = False
                    validation['issues'].append(f'Normalized glucose out of range: [{glucose_min:.3f}, {glucose_max:.3f}]')
                else:
                    validation['checks']['glucose_range'] = True
            else:
                # Raw - check physiological range
                if glucose_min < 40 or glucose_max > 400:
                    validation['passed'] = False
                    validation['issues'].append(f'Glucose out of physiological range: [{glucose_min:.1f}, {glucose_max:.1f}]')
                else:
                    validation['checks']['glucose_range'] = True
        
        # Check 9: Feature count (should have ~24 features)
        expected_feature_count = 24
        actual_feature_count = len(feature_columns)
        if abs(actual_feature_count - expected_feature_count) > 5:
            validation['issues'].append(f'Unexpected feature count: {actual_feature_count} (expected ~{expected_feature_count})')
        else:
            validation['checks']['feature_count'] = True
        
        # Check 10: Data statistics
        validation['statistics'] = {
            'num_samples': len(df),
            'num_features': len(feature_columns),
            'date_range': {
                'start': str(df.index.min()),
                'end': str(df.index.max()),
                'days': (df.index.max() - df.index.min()).days
            },
            'target_stats': {
                'mean': float(df['target'].mean()),
                'std': float(df['target'].std()),
                'min': float(df['target'].min()),
                'max': float(df['target'].max())
            }
        }
        
        return validation
    
    def generate_quality_report(self, validation_results: Dict) -> pd.DataFrame:
        """
        Generate quality assurance report
        
        Args:
            validation_results: Validation results dictionary
            
        Returns:
            DataFrame with quality metrics
        """
        report_rows = []
        
        for subject_id, validation in validation_results.items():
            stats = validation.get('statistics', {})
            target_stats = stats.get('target_stats', {})
            
            report_rows.append({
                'subject_id': subject_id,
                'passed': validation.get('passed', False),
                'num_samples': stats.get('num_samples', 0),
                'num_features': stats.get('num_features', 0),
                'date_range_days': stats.get('date_range', {}).get('days', 0),
                'target_mean': target_stats.get('mean', 0),
                'target_std': target_stats.get('std', 0),
                'num_issues': len(validation.get('issues', [])),
                'issues': '; '.join(validation.get('issues', []))
            })
        
        report_df = pd.DataFrame(report_rows)
        
        # Save report
        report_path = self.processed_data_dir.parent / 'quality_assurance_report.csv'
        report_df.to_csv(report_path, index=False)
        
        self.logger.info(f"\n✓ Quality report saved: {report_path}")
        
        return report_df
    
    def get_high_quality_subjects(self, validation_results: Dict) -> List[str]:
        """
        Get list of high-quality subjects (passed all checks)
        
        Args:
            validation_results: Validation results dictionary
            
        Returns:
            List of subject IDs that passed all checks
        """
        high_quality = [
            subject_id 
            for subject_id, validation in validation_results.items()
            if validation.get('passed', False)
        ]
        
        self.logger.info(f"\nHigh-quality subjects: {len(high_quality)}/{len(validation_results)}")
        
        return high_quality

def main():
    """Main execution function"""
    print("=" * 80)
    print("DATA QUALITY ASSURANCE - Publication Quality")
    print("=" * 80)
    
    qa = DataQualityAssurance(processed_data_dir="data/processed/subjects")
    
    # Validate all subjects
    validation_results = qa.validate_all_subjects()
    
    # Generate quality report
    report_df = qa.generate_quality_report(validation_results)
    
    # Get high-quality subjects
    high_quality_subjects = qa.get_high_quality_subjects(validation_results)
    
    print("\n" + "=" * 80)
    print("QUALITY ASSURANCE COMPLETE")
    print("=" * 80)
    print(f"\nHigh-quality subjects: {len(high_quality_subjects)}")
    print(f"\nFirst 10 high-quality subjects:")
    for subject_id in high_quality_subjects[:10]:
        print(f"  - {subject_id}")
    
    # Save high-quality subject list
    high_quality_path = qa.processed_data_dir.parent / 'high_quality_subjects.json'
    with open(high_quality_path, 'w') as f:
        json.dump({
            'high_quality_subjects': high_quality_subjects,
            'total_subjects': len(validation_results),
            'validation_date': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✓ High-quality subject list saved: {high_quality_path}")

if __name__ == "__main__":
    main()

