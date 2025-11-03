#!/usr/bin/env python3
"""
Display summary of aggregated feature analysis results
"""
import json
import os

def show_summary():
    """Display analysis summary"""
    report_path = "experiments/cgm_fl_benchmark/feature_visualizations/aggregated_feature_analysis_all_subjects.json"
    
    if not os.path.exists(report_path):
        print(f"Error: Report not found at {report_path}")
        return
    
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("COMPREHENSIVE FEATURE ANALYSIS - ALL SUBJECTS SUMMARY")
    print("=" * 80)
    
    print(f"\nAnalysis Statistics:")
    print(f"  - Subjects analyzed: {data['total_subjects_analyzed']}")
    print(f"  - Failed subjects: {len(data.get('failed_subjects', []))}")
    if data.get('failed_subjects'):
        print(f"  - Failed subject IDs: {', '.join(data['failed_subjects'])}")
    
    print(f"\n{'=' * 80}")
    print("CONSISTENTLY IMPORTANT FEATURES (Important in ≥70% of subjects)")
    print("=" * 80)
    
    consistently_important = data.get('consistently_important_features', [])
    print(f"\nTotal: {len(consistently_important)} features")
    print(f"\n{'Rank':<6} {'Feature':<35} {'Consistency':<15} {'Avg Correlation':<15} {'Avg Composite Score':<20}")
    print("-" * 80)
    
    for i, feat in enumerate(consistently_important, 1):
        stats = data.get('feature_statistics', {}).get(feat, {})
        consistency = stats.get('consistency', 0) * 100
        avg_corr = stats.get('avg_correlation', 0)
        avg_score = stats.get('avg_composite_score', 0)
        
        print(f"{i:<6} {feat:<35} {consistency:>13.1f}%  {avg_corr:>14.4f}  {avg_score:>19.4f}")
    
    # Show top features by average correlation
    print(f"\n{'=' * 80}")
    print("TOP 20 FEATURES BY AVERAGE CORRELATION WITH TARGET")
    print("=" * 80)
    
    # Get all features with statistics
    all_features = []
    for feat, stats in data.get('feature_statistics', {}).items():
        all_features.append((feat, stats))
    
    # Sort by average correlation
    sorted_by_corr = sorted(all_features, 
                           key=lambda x: x[1].get('avg_correlation', 0), 
                           reverse=True)
    
    print(f"\n{'Rank':<6} {'Feature':<35} {'Avg Correlation':<15} {'Frequency':<15} {'Consistency':<15}")
    print("-" * 80)
    
    for i, (feat, stats) in enumerate(sorted_by_corr[:20], 1):
        freq = stats.get('frequency', 0)
        consistency = stats.get('consistency', 0) * 100
        avg_corr = stats.get('avg_correlation', 0)
        
        print(f"{i:<6} {feat:<35} {avg_corr:>14.4f}  {freq:>14}/{data['total_subjects_analyzed']}  {consistency:>13.1f}%")
    
    # Common redundant features
    if data.get('common_redundant_features'):
        print(f"\n{'=' * 80}")
        print("COMMON REDUNDANT FEATURES (found across multiple subjects)")
        print("=" * 80)
        print(f"\n{len(data['common_redundant_features'])} features identified as redundant:")
        for feat in sorted(data['common_redundant_features']):
            print(f"  - {feat}")
    
    print(f"\n{'=' * 80}")
    print(f"Report saved to: {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    show_summary()

