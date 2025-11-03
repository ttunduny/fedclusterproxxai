#!/usr/bin/env python3
"""
Check the status of the feature analysis
"""
import os
import json
import glob

def check_status():
    """Check analysis status"""
    output_dir = "experiments/cgm_fl_benchmark/feature_visualizations"
    
    # Count completed subjects
    report_files = glob.glob(os.path.join(output_dir, "feature_analysis_report_*.json"))
    completed = len(report_files)
    
    # Count total subjects
    subject_files = glob.glob("data/processed/subjects/Subject_*.xlsx")
    total = len(subject_files)
    
    print(f"=" * 80)
    print(f"FEATURE ANALYSIS STATUS")
    print(f"=" * 80)
    print(f"\nProgress: {completed}/{total} subjects completed ({100*completed/total:.1f}%)")
    
    # Check if aggregated report exists
    aggregated_report = os.path.join(output_dir, "aggregated_feature_analysis_all_subjects.json")
    if os.path.exists(aggregated_report):
        print(f"\n✓ Analysis COMPLETE!")
        print(f"  Report: {aggregated_report}")
        
        with open(aggregated_report, 'r') as f:
            data = json.load(f)
        
        print(f"\n  Subjects analyzed: {data.get('total_subjects_analyzed', 0)}")
        print(f"  Failed subjects: {len(data.get('failed_subjects', []))}")
        print(f"  Consistently important features: {len(data.get('consistently_important_features', []))}")
        
        if 'consistently_important_features' in data and data['consistently_important_features']:
            print(f"\n  Top 20 consistently important features:")
            for i, feat in enumerate(data['consistently_important_features'][:20], 1):
                stats = data.get('feature_statistics', {}).get(feat, {})
                consistency = stats.get('consistency', 0) * 100
                print(f"    {i:2d}. {feat:<30} ({consistency:.1f}% consistency)")
    else:
        print(f"\n⏳ Analysis IN PROGRESS...")
        print(f"  Completed reports: {completed}/{total}")
        
        if completed > 0:
            print(f"\n  Recently completed subjects:")
            recent = sorted(report_files, key=os.path.getmtime, reverse=True)[:5]
            for f in recent:
                subject = os.path.basename(f).replace('feature_analysis_report_', '').replace('.json', '')
                print(f"    - {subject}")
    
    print(f"\n" + "=" * 80)

if __name__ == "__main__":
    check_status()

