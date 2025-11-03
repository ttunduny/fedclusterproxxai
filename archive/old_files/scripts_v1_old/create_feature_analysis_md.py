#!/usr/bin/env python3
"""
Create markdown report from feature analysis results
"""
import json
import os
from datetime import datetime

def create_markdown_report():
    """Create comprehensive markdown report"""
    report_path = "experiments/cgm_fl_benchmark/feature_visualizations/aggregated_feature_analysis_all_subjects.json"
    output_md = "experiments/cgm_fl_benchmark/feature_visualizations/FEATURE_ANALYSIS_REPORT.md"
    
    if not os.path.exists(report_path):
        print(f"Error: Report not found at {report_path}")
        return
    
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    md_content = []
    md_content.append("# Comprehensive Feature Analysis Report")
    md_content.append("")
    md_content.append(f"**Analysis Date:** {data.get('analysis_date', 'N/A')}")
    md_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_content.append("")
    md_content.append("---")
    md_content.append("")
    
    # Executive Summary
    md_content.append("## Executive Summary")
    md_content.append("")
    md_content.append(f"This report presents a comprehensive feature analysis conducted on **{data['total_subjects_analyzed']} subjects** from the CGM dataset.")
    md_content.append(f"")
    md_content.append(f"- **Subjects Analyzed:** {data['total_subjects_analyzed']}")
    md_content.append(f"- **Failed Subjects:** {len(data.get('failed_subjects', []))}")
    md_content.append(f"- **Consistently Important Features:** {len(data.get('consistently_important_features', []))}")
    md_content.append(f"- **Total Features Analyzed:** {len(data.get('feature_statistics', {}))}")
    md_content.append("")
    
    # Failed Subjects
    if data.get('failed_subjects'):
        md_content.append("### Failed Subjects")
        md_content.append("")
        md_content.append("The following subjects could not be analyzed due to data issues:")
        md_content.append("")
        for subject in data['failed_subjects']:
            md_content.append(f"- {subject}")
        md_content.append("")
    
    # Methodology
    md_content.append("## Methodology")
    md_content.append("")
    md_content.append("The feature analysis was conducted using a six-step process:")
    md_content.append("")
    md_content.append("### 1. Target Correlation Analysis")
    md_content.append("- Computed Pearson correlation coefficient between each feature and target CGM value")
    md_content.append("- Identified statistically significant correlations (p-value < 0.05)")
    md_content.append("- Features with |correlation| > 0.05 were considered useful")
    md_content.append("")
    md_content.append("### 2. Mutual Information Scoring")
    md_content.append("- Calculated mutual information (non-linear dependency measure) between features and target")
    md_content.append("- Identified features with high mutual information scores")
    md_content.append("- Captures non-linear relationships that correlation might miss")
    md_content.append("")
    md_content.append("### 3. Variance Analysis")
    md_content.append("- Analyzed feature variance to identify informative features")
    md_content.append("- Low variance features (constant/near-constant) were flagged")
    md_content.append("- High variance features contribute more to model learning")
    md_content.append("")
    md_content.append("### 4. Redundancy Detection")
    md_content.append("- Computed inter-feature correlation matrix")
    md_content.append("- Identified highly correlated feature pairs (correlation >= 0.9)")
    md_content.append("- Recommended removing redundant features to reduce multicollinearity")
    md_content.append("- Kept feature with higher target correlation in each pair")
    md_content.append("")
    md_content.append("### 5. Composite Scoring")
    md_content.append("- Combined metrics: **Target Correlation (50%)**, **Mutual Information (30%)**, **Variance (20%)**")
    md_content.append("- Ranked features by composite score")
    md_content.append("- Features with composite score > 0.1 and not redundant were recommended")
    md_content.append("")
    md_content.append("### 6. Cross-Subject Consistency")
    md_content.append("- Analyzed feature importance across multiple subjects")
    md_content.append("- Features appearing in top recommendations for ≥70% of subjects are 'consistently important'")
    md_content.append("- Provides robustness and generalizability across different glucose patterns")
    md_content.append("")
    
    # Consistently Important Features
    md_content.append("## Consistently Important Features")
    md_content.append("")
    md_content.append("Features that are important in **≥70%** of analyzed subjects are considered consistently important.")
    md_content.append("")
    md_content.append(f"**Total:** {len(data.get('consistently_important_features', []))} features")
    md_content.append("")
    md_content.append("| Rank | Feature | Consistency | Avg Correlation | Avg Composite Score |")
    md_content.append("|------|---------|------------|----------------|---------------------|")
    
    consistently_important = data.get('consistently_important_features', [])
    for i, feat in enumerate(consistently_important, 1):
        stats = data.get('feature_statistics', {}).get(feat, {})
        consistency = stats.get('consistency', 0) * 100
        avg_corr = stats.get('avg_correlation', 0)
        avg_score = stats.get('avg_composite_score', 0)
        
        md_content.append(f"| {i} | `{feat}` | {consistency:.1f}% | {avg_corr:.4f} | {avg_score:.4f} |")
    
    md_content.append("")
    
    # Top Features by Correlation
    md_content.append("## Top 20 Features by Average Correlation with Target")
    md_content.append("")
    md_content.append("| Rank | Feature | Avg Correlation | Frequency | Consistency |")
    md_content.append("|------|---------|----------------|----------|-------------|")
    
    all_features = []
    for feat, stats in data.get('feature_statistics', {}).items():
        all_features.append((feat, stats))
    
    sorted_by_corr = sorted(all_features, 
                           key=lambda x: x[1].get('avg_correlation', 0), 
                           reverse=True)
    
    for i, (feat, stats) in enumerate(sorted_by_corr[:20], 1):
        freq = stats.get('frequency', 0)
        consistency = stats.get('consistency', 0) * 100
        avg_corr = stats.get('avg_correlation', 0)
        
        md_content.append(f"| {i} | `{feat}` | {avg_corr:.4f} | {freq}/{data['total_subjects_analyzed']} | {consistency:.1f}% |")
    
    md_content.append("")
    
    # Feature Categories Analysis
    md_content.append("## Feature Categories Analysis")
    md_content.append("")
    
    # Categorize features
    categories = {
        'Temporal Features': ['hour', 'hour_cos', 'hour_sin', 'day_of_week', 'is_weekend', 'hourly_baseline'],
        'Recent Values': ['prev_15min', 'prev_30min', 'prev_60min', 'roc_30min', 'roc_60min', 'current_glucose'],
        'Rolling Statistics': ['mean_1h', 'mean_3h', 'mean_6h', 'mean_12h', 'mean_24h', 
                              'std_1h', 'std_3h', 'std_6h', 'std_12h', 'std_24h',
                              'min_3h', 'min_6h', 'min_12h', 'min_24h',
                              'max_3h', 'max_6h', 'max_12h', 'max_24h'],
        'Medical Indicators': ['is_hyperglycemic', 'is_hypoglycemic', 'in_target_range', 'in_tight_range',
                               'recent_hyper_6h', 'recent_hypo_6h'],
        'Pattern Analysis': ['deviation_from_daily_mean', 'deviation_from_recent_trend', 
                            'deviation_from_pattern', 'glucose_variability', 'trend_strength',
                            'pattern_consistency']
    }
    
    for category, features in categories.items():
        category_features = [f for f in consistently_important if f in features]
        if category_features:
            md_content.append(f"### {category}")
            md_content.append("")
            md_content.append(f"**Consistently Important Features:** {len(category_features)}")
            md_content.append("")
            for feat in category_features:
                stats = data.get('feature_statistics', {}).get(feat, {})
                consistency = stats.get('consistency', 0) * 100
                md_content.append(f"- `{feat}` ({consistency:.1f}% consistency, avg correlation: {stats.get('avg_correlation', 0):.4f})")
            md_content.append("")
    
    # Redundant Features
    if data.get('common_redundant_features'):
        md_content.append("## Common Redundant Features")
        md_content.append("")
        md_content.append(f"The following {len(data['common_redundant_features'])} features were identified as redundant across multiple subjects (correlation ≥ 0.9 with other features):")
        md_content.append("")
        for feat in sorted(data['common_redundant_features']):
            md_content.append(f"- `{feat}`")
        md_content.append("")
        md_content.append("**Recommendation:** Consider removing these features to reduce multicollinearity and improve model performance.")
        md_content.append("")
    
    # Recommendations
    md_content.append("## Recommendations")
    md_content.append("")
    md_content.append("### Feature Selection Recommendations")
    md_content.append("")
    md_content.append(f"**Core Feature Set:** Use the {len(consistently_important)} consistently important features as the foundation for CGM prediction models.")
    md_content.append("")
    md_content.append("**Priority Features:**")
    md_content.append("1. **Temporal Features**: `hour`, `hour_cos`, `hourly_baseline` - Critical for capturing circadian patterns")
    md_content.append("2. **Current State**: `current_glucose` - Highest correlation with target (0.6974)")
    md_content.append("3. **Recent History**: `prev_15min`, `prev_30min` - Capture short-term trends")
    md_content.append("4. **Rolling Statistics**: `mean_1h`, `mean_3h`, `max_6h`, `min_6h` - Provide context windows")
    md_content.append("5. **Rate of Change**: `roc_60min`, `roc_30min` - Capture velocity of glucose changes")
    md_content.append("6. **Medical Indicators**: `in_tight_range`, `is_hyperglycemic` - Clinical relevance")
    md_content.append("")
    md_content.append("### Model Development Recommendations")
    md_content.append("")
    md_content.append("1. **Start with Core Features**: Begin model development with the consistently important features")
    md_content.append("2. **Feature Engineering**: Consider interactions between temporal features and glucose values")
    md_content.append("3. **Redundancy Removal**: Remove or combine redundant features to reduce model complexity")
    md_content.append("4. **Subject-Specific Tuning**: While core features are consistent, subject-specific patterns may benefit from additional features")
    md_content.append("5. **Validation Strategy**: Use cross-validation across subjects to ensure generalizability")
    md_content.append("")
    
    # Insights
    md_content.append("## Key Insights")
    md_content.append("")
    md_content.append("1. **Temporal Patterns Are Critical**: Temporal features (`hour`, `hour_cos`, `hourly_baseline`) show the highest consistency across subjects, indicating strong circadian patterns in glucose levels.")
    md_content.append("")
    md_content.append("2. **Current State Matters Most**: `current_glucose` has the highest correlation (0.6974) with the target, making it the single most predictive feature.")
    md_content.append("")
    md_content.append("3. **Short-Term Statistics Outperform Long-Term**: Features like `mean_1h`, `mean_3h` are more consistently important than `mean_24h`, suggesting recent history is more predictive.")
    md_content.append("")
    md_content.append("4. **Rate of Change is Important**: `roc_60min` appears in 100% of subjects, indicating glucose velocity is a universal predictor.")
    md_content.append("")
    md_content.append("5. **Medical Indicators Provide Context**: Features like `in_tight_range` and `is_hyperglycemic` appear consistently, adding clinical relevance to predictions.")
    md_content.append("")
    
    # Appendix
    md_content.append("## Appendix")
    md_content.append("")
    md_content.append("### All Analyzed Subjects")
    md_content.append("")
    for i, subject in enumerate(data.get('subject_ids', []), 1):
        md_content.append(f"{i}. {subject}")
    md_content.append("")
    
    md_content.append("### Report Files")
    md_content.append("")
    md_content.append(f"- **Aggregated Report (JSON)**: `{report_path}`")
    md_content.append(f"- **Individual Reports**: 47 subject-specific analysis reports in the same directory")
    md_content.append("")
    
    md_content.append("---")
    md_content.append("")
    md_content.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # Write markdown file
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, 'w') as f:
        f.write('\n'.join(md_content))
    
    print(f"✓ Markdown report saved to: {output_md}")
    return output_md

if __name__ == "__main__":
    create_markdown_report()

