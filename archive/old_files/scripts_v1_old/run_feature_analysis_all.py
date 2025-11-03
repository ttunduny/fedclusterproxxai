#!/usr/bin/env python3
"""
Run comprehensive feature analysis for all subjects in data/processed/subjects
"""
import os
import sys
from src.utils import multi_subject_feature_analysis

def main():
    """Run feature analysis for all subjects"""
    print("=" * 80)
    print("RUNNING FEATURE ANALYSIS FOR ALL SUBJECTS")
    print("=" * 80)
    
    # Set up directories
    subjects_dir = "data/processed/subjects"
    output_dir = "experiments/cgm_fl_benchmark/feature_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis for all subjects
    try:
        result = multi_subject_feature_analysis(
            subject_ids=None,  # Analyze all available subjects
            subjects_dir=subjects_dir,
            save_aggregated_report=os.path.join(output_dir, 'aggregated_feature_analysis_all_subjects.json'),
            output_dir=output_dir
        )
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nSuccessfully analyzed {result.get('total_subjects_analyzed', 0)} subjects")
        print(f"Report saved to: {os.path.join(output_dir, 'aggregated_feature_analysis_all_subjects.json')}")
        
        # Print summary
        if 'consistently_important_features' in result:
            print(f"\nConsistently Important Features: {len(result['consistently_important_features'])}")
            print("\nTop 20 consistently important features:")
            for i, feat in enumerate(result['consistently_important_features'][:20], 1):
                print(f"  {i:2d}. {feat}")
        
        return 0
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

