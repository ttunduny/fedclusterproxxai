import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import json

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pass

def setup_logging():
    """Setup logging configuration"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics for predictions"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        'mae': mae,
        'mse': mse, 
        'rmse': rmse
    }

def plot_training_history(history: Dict, save_path: str = None):
    """Plot training history metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Training Loss')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
    
    # Plot metrics
    if 'mae' in history:
        axes[1].plot(history['mae'], label='MAE')
        axes[1].set_title('Training Metrics')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def save_results(results: Dict, filepath: str):
    """Save results to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_results(filepath: str) -> Dict:
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def create_sample_glucose_data(num_points: int = 1000) -> pd.DataFrame:
    """Create sample glucose data for testing"""
    dates = pd.date_range('2024-01-01', periods=num_points, freq='5T')
    glucose = np.random.normal(120, 30, num_points)
    glucose = np.clip(glucose, 40, 400)
    
    return pd.DataFrame({'glucose': glucose}, index=dates)

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """Validate DataFrame structure and content"""
    if df.empty:
        return False
    
    if required_columns:
        for col in required_columns:
            if col not in df.columns:
                return False
    
    return True

def compute_z_index(data: Any, axis: int = 0, ddof: int = 0) -> Any:
    """
    Compute z-scores (z-index) for data normalization.
    
    Z-score formula: z = (x - μ) / σ
    where μ is the mean and σ is the standard deviation.
    
    Args:
        data: Input data as numpy array or pandas DataFrame/Series
        axis: Axis along which to compute z-scores (0 for columns, 1 for rows)
              Default is 0 (standardize each column independently)
        ddof: Delta degrees of freedom for std calculation (default 0 for population std)
    
    Returns:
        Normalized data with the same shape as input
        - For numpy arrays: returns numpy array
        - For pandas DataFrames: returns DataFrame with same index/columns
        - For pandas Series: returns Series with same index
    
    Example:
        >>> data = np.array([[1, 2], [3, 4], [5, 6]])
        >>> z_scores = compute_z_index(data)
        >>> # Returns normalized array where each column has mean=0, std=1
    """
    if isinstance(data, pd.DataFrame):
        # Compute z-scores for DataFrame columns
        result = data.apply(lambda x: (x - x.mean()) / x.std(ddof=ddof), axis=axis)
        # Handle case where std is 0 (constant column)
        result = result.fillna(0)
        return result
    elif isinstance(data, pd.Series):
        # Compute z-scores for Series
        mean = data.mean()
        std = data.std(ddof=ddof)
        if std == 0 or np.isnan(std):
            return pd.Series(0, index=data.index)
        return (data - mean) / std
    elif isinstance(data, np.ndarray):
        # Compute z-scores for numpy array
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, ddof=ddof, keepdims=True)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        z_scores = (data - mean) / std
        return z_scores
    else:
        # Try to convert to numpy array
        data_array = np.array(data)
        mean = np.mean(data_array, axis=axis, keepdims=True)
        std = np.std(data_array, axis=axis, ddof=ddof, keepdims=True)
        std = np.where(std == 0, 1, std)
        z_scores = (data_array - mean) / std
        return z_scores

def analyze_subject_z_index(subject_id: str = "SUB001", subjects_dir: str = "data/processed/subjects") -> Dict[str, Any]:
    """
    Compute and analyze z-index (z-scores) for a subject's data.
    
    Args:
        subject_id: Subject ID to analyze (default: "SUB001" for subject 1)
        subjects_dir: Directory containing processed subject files
    
    Returns:
        Dictionary containing z-index statistics and analysis
    """
    import os
    
    # Load subject data
    filepath = os.path.join(subjects_dir, f"Subject_{subject_id}.xlsx")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Subject file not found: {filepath}")
    
    # Load data
    df = pd.read_excel(filepath, sheet_name='processed_data', index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Remove non-numeric columns for z-index computation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'subject_id' in numeric_cols:
        numeric_cols.remove('subject_id')
    
    df_numeric = df[numeric_cols].copy()
    
    # Compute z-index for all numeric columns
    z_scores_df = compute_z_index(df_numeric)
    
    # Compute statistics for each column
    z_stats = {}
    for col in z_scores_df.columns:
        z_col = z_scores_df[col].dropna()
        if len(z_col) > 0:
            z_stats[col] = {
                'mean': float(z_col.mean()),
                'std': float(z_col.std()),
                'min': float(z_col.min()),
                'max': float(z_col.max()),
                'pct_within_2std': float(((z_col >= -2) & (z_col <= 2)).sum() / len(z_col) * 100),
                'pct_outliers_high': float((z_col > 2).sum() / len(z_col) * 100),
                'pct_outliers_low': float((z_col < -2).sum() / len(z_col) * 100),
            }
    
    # Focus on glucose-related columns for detailed analysis
    glucose_cols = [col for col in z_scores_df.columns if 'glucose' in col.lower() or col == 'target' or 'current_glucose' in col]
    glucose_analysis = {}
    
    if glucose_cols:
        for col in glucose_cols:
            if col in z_stats:
                glucose_analysis[col] = z_stats[col]
                
                # Assessment
                mean_z = abs(z_stats[col]['mean'])
                std_z = z_stats[col]['std']
                pct_in_range = z_stats[col]['pct_within_2std']
                outliers = z_stats[col]['pct_outliers_high'] + z_stats[col]['pct_outliers_low']
                
                # Determine quality
                if mean_z < 0.1 and 0.9 < std_z < 1.1 and pct_in_range > 95:
                    assessment = "Excellent - Well normalized, minimal outliers"
                elif mean_z < 0.2 and 0.8 < std_z < 1.2 and pct_in_range > 90:
                    assessment = "Good - Well normalized, acceptable outliers"
                elif mean_z < 0.5 and pct_in_range > 85:
                    assessment = "Fair - Some deviation from ideal normalization"
                elif outliers > 10:
                    assessment = "Poor - High number of outliers (>10%)"
                else:
                    assessment = "Needs Review - Significant deviations detected"
                
                glucose_analysis[col]['assessment'] = assessment
    
    # Overall summary
    all_mean_zs = [abs(stats['mean']) for stats in z_stats.values()]
    all_std_zs = [stats['std'] for stats in z_stats.values()]
    all_outlier_pcts = [stats['pct_outliers_high'] + stats['pct_outliers_low'] for stats in z_stats.values()]
    
    overall_assessment = "Good"
    if np.mean(all_mean_zs) > 0.3 or np.mean(all_std_zs) < 0.8 or np.mean(all_std_zs) > 1.2:
        overall_assessment = "Fair - Some columns show deviation"
    if np.mean(all_outlier_pcts) > 8:
        overall_assessment = "Needs Attention - High outlier rate"
    if np.mean(all_mean_zs) > 0.5 or np.mean(all_std_zs) < 0.7 or np.mean(all_outlier_pcts) > 12:
        overall_assessment = "Poor - Significant normalization issues"
    
    summary = {
        'subject_id': subject_id,
        'total_samples': len(df),
        'numeric_features': len(numeric_cols),
        'overall_assessment': overall_assessment,
        'avg_mean_z_index': float(np.mean(all_mean_zs)),
        'avg_std_z_index': float(np.mean(all_std_zs)),
        'avg_outlier_percentage': float(np.mean(all_outlier_pcts)),
        'glucose_features_analysis': glucose_analysis,
        'all_features_stats': z_stats
    }
    
    return summary

def print_z_index_analysis(analysis: Dict[str, Any]):
    """Print formatted z-index analysis results"""
    print("=" * 80)
    print(f"Z-INDEX ANALYSIS FOR SUBJECT: {analysis['subject_id']}")
    print("=" * 80)
    print(f"\nDataset Info:")
    print(f"  - Total Samples: {analysis['total_samples']:,}")
    print(f"  - Numeric Features: {analysis['numeric_features']}")
    
    print(f"\nOverall Statistics:")
    print(f"  - Average Mean Z-Index: {analysis['avg_mean_z_index']:.4f} (ideal: ~0.0)")
    print(f"  - Average Std Z-Index: {analysis['avg_std_z_index']:.4f} (ideal: ~1.0)")
    print(f"  - Average Outlier %: {analysis['avg_outlier_percentage']:.2f}% (ideal: <5%)")
    
    print(f"\nOverall Assessment: {analysis['overall_assessment']}")
    
    if analysis['glucose_features_analysis']:
        print(f"\n{'=' * 80}")
        print("GLUCOSE FEATURES DETAILED ANALYSIS:")
        print("=" * 80)
        for col, stats in analysis['glucose_features_analysis'].items():
            print(f"\n  Column: {col}")
            print(f"    Mean Z-Score: {stats['mean']:.4f}")
            print(f"    Std Z-Score: {stats['std']:.4f}")
            print(f"    Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
            print(f"    % within ±2 std: {stats['pct_within_2std']:.2f}%")
            print(f"    % outliers (>2 std): {stats['pct_outliers_high']:.2f}%")
            print(f"    % outliers (<-2 std): {stats['pct_outliers_low']:.2f}%")
            print(f"    Assessment: {stats['assessment']}")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print("""
    Z-Index (Z-Score) Interpretation:
    - Mean Z-Index ≈ 0: Data is well-centered (good)
    - Std Z-Index ≈ 1: Proper normalization achieved (good)
    - % within ±2 std > 95%: Most data points are normal (good for healthy distribution)
    - Outliers > 10%: Indicates high variability or extreme values
    
    For CGM Data:
    - Good: Mean z-index close to 0, std close to 1, <5% outliers
    - Fair: Some deviations but acceptable (<10% outliers)
    - Poor: High outlier rate or poor normalization (>10% outliers)
    
    Note: Some outliers in glucose data are expected and may indicate
    hypoglycemic or hyperglycemic events, which are clinically significant.
    """)

def compute_temporal_target_correlation(subject_id: str = "SUB001", 
                                       subjects_dir: str = "data/processed/subjects",
                                       save_plot: str = None,
                                       show_plot: bool = True) -> pd.DataFrame:
    """
    Compute correlation matrix between temporal features and target CGM value.
    
    Temporal features include:
    - hour: Hour of day (0-23)
    - day_of_week: Day of week (0-6)
    - is_weekend: Weekend indicator (0/1)
    - hour_sin: Sine transformation of hour for cyclical encoding
    - hour_cos: Cosine transformation of hour for cyclical encoding
    
    Args:
        subject_id: Subject ID to analyze (default: "SUB001" for subject 1)
        subjects_dir: Directory containing processed subject files
        save_plot: Optional path to save correlation heatmap plot
        show_plot: Whether to display the plot (default: True)
    
    Returns:
        DataFrame containing correlation matrix
    """
    import os
    
    # Load subject data
    filepath = os.path.join(subjects_dir, f"Subject_{subject_id}.xlsx")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Subject file not found: {filepath}")
    
    # Load data
    df = pd.read_excel(filepath, sheet_name='processed_data', index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Extract temporal features and target
    temporal_features = ['hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos']
    
    # Check which features exist
    available_temporal = [f for f in temporal_features if f in df.columns]
    if 'target' not in df.columns:
        raise ValueError("'target' column not found in data")
    
    # Select features for correlation
    features_to_analyze = available_temporal + ['target']
    
    # Create subset dataframe
    df_subset = df[features_to_analyze].copy()
    
    # Remove any rows with NaN values
    df_subset = df_subset.dropna()
    
    if len(df_subset) == 0:
        raise ValueError("No valid data after removing NaN values")
    
    # Compute correlation matrix
    corr_matrix = df_subset.corr()
    
    # Print correlation statistics
    print("=" * 80)
    print(f"CORRELATION MATRIX: Temporal Features vs Target CGM (Subject: {subject_id})")
    print("=" * 80)
    print(f"\nDataset Info:")
    print(f"  - Total Samples: {len(df_subset):,}")
    print(f"  - Temporal Features Analyzed: {', '.join(available_temporal)}")
    
    print(f"\nCorrelation with Target:")
    print("-" * 80)
    target_corr = corr_matrix['target'].sort_values(key=abs, ascending=False)
    for feature, corr_value in target_corr.items():
        if feature != 'target':
            strength = "Strong" if abs(corr_value) > 0.5 else "Moderate" if abs(corr_value) > 0.3 else "Weak"
            direction = "Positive" if corr_value > 0 else "Negative"
            print(f"  {feature:15s}: {corr_value:7.4f}  ({strength} {direction})")
    
    # Visualize correlation matrix
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        mask = None
        # Optionally mask upper triangle for cleaner visualization
        # mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   mask=mask,
                   vmin=-1, 
                   vmax=1)
        
        plt.title(f'Temporal Features vs Target CGM Correlation\nSubject: {subject_id}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(save_plot, dpi=300, bbox_inches='tight')
            print(f"\n✓ Correlation plot saved to: {save_plot}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
    except ImportError:
        print("\nWarning: matplotlib/seaborn not available. Skipping visualization.")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("=" * 80)
    print("""
    Correlation Coefficient Interpretation:
    - |r| > 0.7: Strong correlation
    - 0.3 < |r| < 0.7: Moderate correlation  
    - |r| < 0.3: Weak correlation
    
    Temporal Feature Insights:
    - hour: Direct hour correlation (0-23)
    - hour_sin/hour_cos: Cyclical encoding captures circadian patterns better
    - day_of_week: Weekly patterns (0=Monday to 6=Sunday)
    - is_weekend: Weekend/weekday differences in glucose patterns
    
    Clinical Significance:
    - Strong temporal correlations indicate predictable circadian patterns
    - This suggests the model can benefit from time-based features
    - Cyclical encodings (sin/cos) often capture patterns better than raw hour
    """)
    
    return corr_matrix

def comprehensive_feature_analysis(subject_id: str = "SUB001",
                                  subjects_dir: str = "data/processed/subjects",
                                  correlation_threshold: float = 0.9,
                                  target_corr_threshold: float = 0.05,
                                  save_report: str = None,
                                  save_plots: bool = True,
                                  output_dir: str = "experiments/cgm_fl_benchmark/feature_visualizations") -> Dict[str, Any]:
    """
    Comprehensive feature analysis to identify useful features for CGM prediction.
    
    This analysis includes:
    1. Correlation analysis with target
    2. Feature redundancy detection (high inter-feature correlations)
    3. Variance analysis
    4. Feature importance ranking
    5. Recommendations for feature selection
    
    Args:
        subject_id: Subject ID to analyze (default: "SUB001")
        subjects_dir: Directory containing processed subject files
        correlation_threshold: Threshold for identifying redundant features (default: 0.9)
        target_corr_threshold: Minimum correlation with target to consider useful (default: 0.05)
        save_report: Optional path to save detailed report (JSON)
        save_plots: Whether to save visualization plots
        output_dir: Directory to save plots
    
    Returns:
        Dictionary containing comprehensive feature analysis results
    """
    import os
    from scipy.stats import pearsonr
    from sklearn.feature_selection import mutual_info_regression
    
    # Load subject data
    filepath = os.path.join(subjects_dir, f"Subject_{subject_id}.xlsx")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Subject file not found: {filepath}")
    
    df = pd.read_excel(filepath, sheet_name='processed_data', index_col=0)
    df.index = pd.to_datetime(df.index)
    
    # Extract features (exclude target and subject_id)
    feature_cols = [col for col in df.columns if col not in ['target', 'subject_id']]
    df_features = df[feature_cols].copy()
    df_target = df['target'].dropna()
    
    # Align features and target, remove NaN
    common_idx = df_features.index.intersection(df_target.index)
    df_features = df_features.loc[common_idx]
    df_target = df_target.loc[common_idx]
    
    # Remove features with constant values or too many NaNs
    valid_features = []
    for col in feature_cols:
        if col in df_features.columns:
            col_data = df_features[col].dropna()
            if len(col_data) > 0:
                if col_data.nunique() > 1 and col_data.std() > 1e-10:
                    valid_features.append(col)
    
    df_features = df_features[valid_features].dropna()
    
    # Re-align after dropping NaNs
    common_idx = df_features.index.intersection(df_target.index)
    df_features = df_features.loc[common_idx]
    df_target = df_target.loc[common_idx]
    
    if len(df_features) == 0 or len(df_target) == 0:
        raise ValueError("No valid data after preprocessing")
    
    print("=" * 80)
    print(f"COMPREHENSIVE FEATURE ANALYSIS - Subject: {subject_id}")
    print("=" * 80)
    print(f"\nDataset Info:")
    print(f"  - Total Samples: {len(df_features):,}")
    print(f"  - Valid Features: {len(valid_features)}")
    
    # 1. CORRELATION WITH TARGET
    print(f"\n{'=' * 80}")
    print("1. CORRELATION WITH TARGET")
    print("=" * 80)
    
    target_correlations = {}
    for col in valid_features:
        feature_data = df_features[col].dropna()
        target_aligned = df_target.loc[feature_data.index]
        if len(feature_data) > 1 and feature_data.nunique() > 1:
            corr, p_value = pearsonr(feature_data, target_aligned)
            target_correlations[col] = {
                'correlation': float(corr),
                'abs_correlation': float(abs(corr)),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
    
    # Sort by absolute correlation
    sorted_correlations = sorted(target_correlations.items(), 
                               key=lambda x: x[1]['abs_correlation'], 
                               reverse=True)
    
    print(f"\nTop Features by Target Correlation:")
    print("-" * 80)
    print(f"{'Feature':<30} {'Correlation':<12} {'Abs Corr':<12} {'P-value':<12} {'Significant':<12}")
    print("-" * 80)
    
    useful_features = []
    for feature, stats in sorted_correlations[:20]:  # Top 20
        sig_mark = "✓" if stats['significant'] else ""
        print(f"{feature:<30} {stats['correlation']:>11.4f}  {stats['abs_correlation']:>11.4f}  "
              f"{stats['p_value']:>11.4e}  {sig_mark:>11}")
        if stats['abs_correlation'] >= target_corr_threshold:
            useful_features.append(feature)
    
    # 2. FEATURE REDUNDANCY ANALYSIS
    print(f"\n{'=' * 80}")
    print("2. FEATURE REDUNDANCY ANALYSIS")
    print("=" * 80)
    
    # Compute inter-feature correlation matrix
    feature_corr_matrix = df_features.corr().abs()
    
    # Find highly correlated feature pairs
    redundant_pairs = []
    for i, feat1 in enumerate(feature_corr_matrix.columns):
        for j, feat2 in enumerate(feature_corr_matrix.columns):
            if i < j:
                corr_val = feature_corr_matrix.loc[feat1, feat2]
                if corr_val >= correlation_threshold:
                    # Determine which to keep (keep one with higher target correlation)
                    corr1 = target_correlations.get(feat1, {}).get('abs_correlation', 0)
                    corr2 = target_correlations.get(feat2, {}).get('abs_correlation', 0)
                    redundant_pairs.append({
                        'feature1': feat1,
                        'feature2': feat2,
                        'correlation': float(corr_val),
                        'keep': feat1 if corr1 >= corr2 else feat2,
                        'remove': feat2 if corr1 >= corr2 else feat1
                    })
    
    if redundant_pairs:
        print(f"\nFound {len(redundant_pairs)} highly correlated feature pairs (corr >= {correlation_threshold}):")
        print("-" * 80)
        print(f"{'Feature 1':<25} {'Feature 2':<25} {'Correlation':<12} {'Recommendation':<20}")
        print("-" * 80)
        for pair in sorted(redundant_pairs, key=lambda x: x['correlation'], reverse=True)[:15]:
            print(f"{pair['feature1']:<25} {pair['feature2']:<25} {pair['correlation']:>11.4f}  "
                  f"Keep: {pair['keep']:<20}")
        
        features_to_remove = list(set([p['remove'] for p in redundant_pairs]))
        print(f"\nRecommendation: Consider removing {len(features_to_remove)} redundant features")
    else:
        print(f"\nNo highly redundant features found (correlation >= {correlation_threshold})")
    
    # 3. VARIANCE ANALYSIS
    print(f"\n{'=' * 80}")
    print("3. VARIANCE ANALYSIS")
    print("=" * 80)
    
    variance_analysis = {}
    for col in valid_features:
        feature_data = df_features[col].dropna()
        if len(feature_data) > 0:
            variance_analysis[col] = {
                'variance': float(feature_data.var()),
                'std': float(feature_data.std()),
                'range': float(feature_data.max() - feature_data.min()),
                'coefficient_variation': float(feature_data.std() / feature_data.mean()) if feature_data.mean() != 0 else 0
            }
    
    # Sort by variance
    sorted_variance = sorted(variance_analysis.items(), 
                            key=lambda x: x[1]['variance'], 
                            reverse=True)
    
    print(f"\nTop Features by Variance:")
    print("-" * 80)
    print(f"{'Feature':<30} {'Variance':<15} {'Std Dev':<15} {'Range':<15}")
    print("-" * 80)
    for feature, stats in sorted_variance[:15]:
        print(f"{feature:<30} {stats['variance']:>14.4f}  {stats['std']:>14.4f}  {stats['range']:>14.4f}")
    
    # 4. MUTUAL INFORMATION (if available)
    print(f"\n{'=' * 80}")
    print("4. MUTUAL INFORMATION ANALYSIS")
    print("=" * 80)
    
    mi_scores = {}
    try:
        # Prepare data for MI calculation
        X_mi = df_features.values
        y_mi = df_target.values
        
        # Calculate mutual information
        mi = mutual_info_regression(X_mi, y_mi, random_state=42)
        
        for i, feature in enumerate(valid_features):
            if i < len(mi):
                mi_scores[feature] = float(mi[i])
        
        sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop Features by Mutual Information:")
        print("-" * 80)
        print(f"{'Feature':<30} {'MI Score':<15}")
        print("-" * 80)
        for feature, score in sorted_mi[:20]:
            print(f"{feature:<30} {score:>14.6f}")
            
    except Exception as e:
        print(f"\nMutual information calculation failed: {e}")
        print("Skipping MI analysis...")
    
    # 5. COMPREHENSIVE FEATURE RANKING
    print(f"\n{'=' * 80}")
    print("5. COMPREHENSIVE FEATURE RANKING")
    print("=" * 80)
    
    # Combine all metrics into a composite score
    feature_scores = {}
    max_target_corr = max([s['abs_correlation'] for s in target_correlations.values()]) if target_correlations else 1
    max_variance = max([s['variance'] for s in variance_analysis.values()]) if variance_analysis else 1
    max_mi = max(mi_scores.values()) if mi_scores else 1
    
    for feature in valid_features:
        target_corr = target_correlations.get(feature, {}).get('abs_correlation', 0) / max_target_corr if max_target_corr > 0 else 0
        variance = variance_analysis.get(feature, {}).get('variance', 0) / max_variance if max_variance > 0 else 0
        mi = mi_scores.get(feature, 0) / max_mi if max_mi > 0 else 0
        
        # Composite score (weighted combination)
        composite_score = (0.5 * target_corr + 0.3 * mi + 0.2 * variance)
        
        feature_scores[feature] = {
            'composite_score': float(composite_score),
            'target_correlation': float(target_correlations.get(feature, {}).get('abs_correlation', 0)),
            'mutual_information': float(mi_scores.get(feature, 0)),
            'variance': float(variance_analysis.get(feature, {}).get('variance', 0)),
            'is_redundant': feature in [p['remove'] for p in redundant_pairs]
        }
    
    sorted_features = sorted(feature_scores.items(), 
                           key=lambda x: x[1]['composite_score'], 
                           reverse=True)
    
    print(f"\nTop Features by Composite Score (Target Corr: 50%, MI: 30%, Variance: 20%):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature':<30} {'Score':<12} {'Target Corr':<12} {'MI':<12} {'Redundant':<12}")
    print("-" * 80)
    
    recommended_features = []
    for rank, (feature, scores) in enumerate(sorted_features, 1):
        redundant_mark = "Yes" if scores['is_redundant'] else "No"
        print(f"{rank:<6} {feature:<30} {scores['composite_score']:>11.4f}  "
              f"{scores['target_correlation']:>11.4f}  {scores['mutual_information']:>11.6f}  {redundant_mark:>12}")
        
        if not scores['is_redundant'] and scores['composite_score'] > 0.1:
            recommended_features.append(feature)
    
    # 6. RECOMMENDATIONS
    print(f"\n{'=' * 80}")
    print("6. FEATURE SELECTION RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\n✓ Recommended to KEEP ({len(recommended_features[:30])} features):")
    print("-" * 80)
    for i, feat in enumerate(recommended_features[:30], 1):
        print(f"  {i:2d}. {feat}")
    
    if redundant_pairs:
        print(f"\n✗ Recommended to REMOVE ({len(set([p['remove'] for p in redundant_pairs]))} redundant features):")
        print("-" * 80)
        for feat in sorted(set([p['remove'] for p in redundant_pairs]))[:20]:
            print(f"  - {feat}")
    
    # Visualizations
    if save_plots:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot 1: Top correlations with target
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Top 20 features by correlation
            top_features = sorted_correlations[:20]
            features_list = [f[0] for f in top_features]
            corr_values = [f[1]['correlation'] for f in top_features]
            
            axes[0].barh(range(len(features_list)), corr_values, color='steelblue')
            axes[0].set_yticks(range(len(features_list)))
            axes[0].set_yticklabels(features_list)
            axes[0].set_xlabel('Correlation with Target')
            axes[0].set_title('Top 20 Features by Target Correlation')
            axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.3)
            axes[0].grid(axis='x', alpha=0.3)
            
            # Composite scores
            top_by_score = sorted_features[:20]
            feat_names = [f[0] for f in top_by_score]
            scores = [f[1]['composite_score'] for f in top_by_score]
            
            axes[1].barh(range(len(feat_names)), scores, color='darkgreen')
            axes[1].set_yticks(range(len(feat_names)))
            axes[1].set_yticklabels(feat_names)
            axes[1].set_xlabel('Composite Score')
            axes[1].set_title('Top 20 Features by Composite Score')
            axes[1].grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'feature_analysis_{subject_id}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\n✓ Visualization saved to: {plot_path}")
            
        except ImportError:
            print("\nWarning: matplotlib/seaborn not available. Skipping visualizations.")
    
    # Save report
    if save_report:
        report = {
            'subject_id': subject_id,
            'total_samples': len(df_features),
            'total_features': len(valid_features),
            'target_correlations': {k: {'correlation': v['correlation'], 
                                       'p_value': v['p_value'],
                                       'significant': v['significant']} 
                                   for k, v in target_correlations.items()},
            'redundant_pairs': redundant_pairs,
            'mutual_information': mi_scores,
            'feature_scores': feature_scores,
            'recommended_features': recommended_features[:30],
            'features_to_remove': list(set([p['remove'] for p in redundant_pairs])) if redundant_pairs else []
        }
        
        with open(save_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"✓ Report saved to: {save_report}")
    
    # Summary statistics
    summary = {
        'subject_id': subject_id,
        'total_features': len(valid_features),
        'useful_features': len(useful_features),
        'recommended_features': recommended_features[:30],
        'redundant_features': list(set([p['remove'] for p in redundant_pairs])) if redundant_pairs else [],
        'top_correlated_features': [f[0] for f in sorted_correlations[:10]],
        'target_correlations': target_correlations,
        'feature_scores': feature_scores,
        'redundant_pairs': redundant_pairs,
        'top_features_composite': [f[0] for f in sorted_features[:20]]
    }
    
    return summary

def multi_subject_feature_analysis(subject_ids: List[str] = None,
                                   subjects_dir: str = "data/processed/subjects",
                                   save_aggregated_report: str = None,
                                   output_dir: str = "experiments/cgm_fl_benchmark/feature_visualizations") -> Dict[str, Any]:
    """
    Run comprehensive feature analysis across multiple subjects and aggregate results.
    
    Identifies features that are consistently important across subjects and explains
    how features were identified as relevant.
    
    Args:
        subject_ids: List of subject IDs to analyze. If None, analyzes all available subjects.
        subjects_dir: Directory containing processed subject files
        save_aggregated_report: Optional path to save aggregated analysis report
        output_dir: Directory to save individual and aggregated reports
    
    Returns:
        Dictionary containing aggregated feature analysis across subjects
    """
    import os
    import glob
    
    # Get all available subjects if not provided
    if subject_ids is None:
        subject_files = glob.glob(os.path.join(subjects_dir, "Subject_*.xlsx"))
        subject_ids = sorted([os.path.basename(f).replace('Subject_', '').replace('.xlsx', '') 
                             for f in subject_files])
    
    print("=" * 80)
    print(f"MULTI-SUBJECT FEATURE ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing {len(subject_ids)} subjects: {', '.join(subject_ids[:10])}{'...' if len(subject_ids) > 10 else ''}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis for each subject
    all_subject_results = {}
    failed_subjects = []
    
    for i, subject_id in enumerate(subject_ids, 1):
        print(f"\n{'=' * 80}")
        print(f"Analyzing Subject {i}/{len(subject_ids)}: {subject_id}")
        print("=" * 80)
        
        try:
            subject_report_path = os.path.join(output_dir, f'feature_analysis_report_{subject_id}.json')
            result = comprehensive_feature_analysis(
                subject_id=subject_id,
                subjects_dir=subjects_dir,
                save_plots=False,
                save_report=subject_report_path,
                output_dir=output_dir
            )
            all_subject_results[subject_id] = result
            print(f"✓ Completed analysis for {subject_id}")
        except Exception as e:
            print(f"✗ Failed to analyze {subject_id}: {e}")
            failed_subjects.append(subject_id)
            continue
    
    if not all_subject_results:
        raise ValueError("No subjects were successfully analyzed")
    
    print(f"\n{'=' * 80}")
    print("AGGREGATING RESULTS ACROSS SUBJECTS")
    print("=" * 80)
    
    # Aggregate results
    # 1. Feature frequency (how many subjects have this feature in top recommendations)
    feature_frequency = {}
    feature_correlation_scores = {}
    feature_composite_scores = {}
    all_redundant_features = set()
    
    for subject_id, result in all_subject_results.items():
        # Count recommended features
        for feat in result.get('recommended_features', []):
            feature_frequency[feat] = feature_frequency.get(feat, 0) + 1
        
        # Aggregate correlation scores
        target_corrs = result.get('target_correlations', {})
        for feat, stats in target_corrs.items():
            if feat not in feature_correlation_scores:
                feature_correlation_scores[feat] = []
            feature_correlation_scores[feat].append(stats.get('abs_correlation', 0))
        
        # Aggregate composite scores
        feature_scores = result.get('feature_scores', {})
        for feat, scores in feature_scores.items():
            if feat not in feature_composite_scores:
                feature_composite_scores[feat] = []
            feature_composite_scores[feat].append(scores.get('composite_score', 0))
        
        # Collect redundant features
        all_redundant_features.update(result.get('redundant_features', []))
    
    # Calculate statistics
    feature_stats = {}
    for feat in set(list(feature_frequency.keys()) + list(feature_correlation_scores.keys())):
        freq = feature_frequency.get(feat, 0)
        avg_corr = np.mean(feature_correlation_scores.get(feat, [0]))
        avg_composite = np.mean(feature_composite_scores.get(feat, [0]))
        consistency = freq / len(all_subject_results)  # How consistent across subjects
        
        feature_stats[feat] = {
            'frequency': freq,
            'consistency': consistency,
            'avg_correlation': avg_corr,
            'avg_composite_score': avg_composite,
            'is_consistently_important': consistency >= 0.7  # Important in 70%+ of subjects
        }
    
    # Sort by consistency and average composite score
    sorted_aggregated = sorted(feature_stats.items(), 
                               key=lambda x: (x[1]['consistency'], x[1]['avg_composite_score']), 
                               reverse=True)
    
    # Identify consistently important features
    consistently_important = [feat for feat, stats in sorted_aggregated 
                            if stats['is_consistently_important']]
    
    print(f"\n{'=' * 80}")
    print("AGGREGATED FEATURE ANALYSIS RESULTS")
    print("=" * 80)
    
    print(f"\nSuccessfully analyzed: {len(all_subject_results)} subjects")
    print(f"Failed: {len(failed_subjects)} subjects")
    if failed_subjects:
        print(f"Failed subjects: {', '.join(failed_subjects)}")
    
    print(f"\n{'=' * 80}")
    print("TOP FEATURES BY CONSISTENCY ACROSS SUBJECTS")
    print("=" * 80)
    print(f"{'Rank':<6} {'Feature':<30} {'Frequency':<12} {'Consistency':<12} {'Avg Corr':<12} {'Avg Score':<12}")
    print("-" * 80)
    
    for rank, (feat, stats) in enumerate(sorted_aggregated[:30], 1):
        print(f"{rank:<6} {feat:<30} {stats['frequency']:<12}/{len(all_subject_results)} "
              f"{stats['consistency']:>11.2%}  {stats['avg_correlation']:>11.4f}  "
              f"{stats['avg_composite_score']:>11.4f}")
    
    print(f"\n{'=' * 80}")
    print("CONSISTENTLY IMPORTANT FEATURES (Important in ≥70% of subjects)")
    print("=" * 80)
    
    if consistently_important:
        print(f"\nFound {len(consistently_important)} consistently important features:")
        print("-" * 80)
        for i, feat in enumerate(consistently_important, 1):
            stats = feature_stats[feat]
            print(f"  {i:2d}. {feat:<30} (Appears in {stats['frequency']}/{len(all_subject_results)} subjects, "
                  f"{stats['consistency']:.1%} consistency, avg correlation: {stats['avg_correlation']:.4f})")
    else:
        print("\nNo features were consistently important across ≥70% of subjects.")
        print("Top features by frequency:")
        for i, (feat, stats) in enumerate(sorted_aggregated[:10], 1):
            print(f"  {i:2d}. {feat:<30} ({stats['frequency']}/{len(all_subject_results)} subjects, "
                  f"{stats['consistency']:.1%} consistency)")
    
    # Explain how features were identified
    print(f"\n{'=' * 80}")
    print("HOW FEATURES WERE IDENTIFIED AS RELEVANT")
    print("=" * 80)
    print("""
    Feature Relevance Identification Process:
    
    1. TARGET CORRELATION ANALYSIS:
       - Computed Pearson correlation coefficient between each feature and target CGM value
       - Identified statistically significant correlations (p-value < 0.05)
       - Features with |correlation| > 0.05 were considered useful
    
    2. MUTUAL INFORMATION SCORING:
       - Calculated mutual information (non-linear dependency measure) between features and target
       - Identified features with high mutual information scores
       - Captures non-linear relationships that correlation might miss
    
    3. VARIANCE ANALYSIS:
       - Analyzed feature variance to identify informative features
       - Low variance features (constant/near-constant) were flagged
       - High variance features contribute more to model learning
    
    4. REDUNDANCY DETECTION:
       - Computed inter-feature correlation matrix
       - Identified highly correlated feature pairs (correlation >= 0.9)
       - Recommended removing redundant features to reduce multicollinearity
       - Kept feature with higher target correlation in each pair
    
    5. COMPOSITE SCORING:
       - Combined metrics: Target Correlation (50%), Mutual Information (30%), Variance (20%)
       - Ranked features by composite score
       - Features with composite score > 0.1 and not redundant were recommended
    
    6. CROSS-SUBJECT CONSISTENCY:
       - Analyzed feature importance across multiple subjects
       - Features appearing in top recommendations for ≥70% of subjects are "consistently important"
       - Provides robustness and generalizability across different glucose patterns
    
    CONSISTENTLY IMPORTANT FEATURES are those that:
    - Appear in top recommended features for ≥70% of analyzed subjects
    - Have high average correlation with target across subjects
    - Demonstrate high mutual information scores
    - Are not redundant with other features
    """)
    
    # Save aggregated report
    aggregated_report = {
        'analysis_date': str(pd.Timestamp.now()),
        'total_subjects_analyzed': len(all_subject_results),
        'failed_subjects': failed_subjects,
        'subject_ids': list(all_subject_results.keys()),
        'feature_statistics': {k: {
            'frequency': v['frequency'],
            'consistency': float(v['consistency']),
            'avg_correlation': float(v['avg_correlation']),
            'avg_composite_score': float(v['avg_composite_score']),
            'is_consistently_important': v['is_consistently_important']
        } for k, v in feature_stats.items()},
        'consistently_important_features': consistently_important,
        'common_redundant_features': list(all_redundant_features),
        'top_features_by_consistency': [f[0] for f in sorted_aggregated[:30]],
        'individual_subject_results': {sid: {
            'recommended_features': res.get('recommended_features', []),
            'redundant_features': res.get('redundant_features', []),
            'top_correlated_features': res.get('top_correlated_features', [])
        } for sid, res in all_subject_results.items()}
    }
    
    if save_aggregated_report:
        report_path = save_aggregated_report
    else:
        report_path = os.path.join(output_dir, 'aggregated_feature_analysis.json')
    
    with open(report_path, 'w') as f:
        json.dump(aggregated_report, f, indent=2, default=str)
    
    print(f"\n✓ Aggregated report saved to: {report_path}")
    
    return aggregated_report

def outlier_analysis(subject_ids: List[str] = None,
                     subjects_dir: str = "data/processed/subjects",
                     num_subjects: int = 10,
                     save_report: str = None,
                     output_dir: str = "experiments/cgm_fl_benchmark/feature_visualizations") -> Dict[str, Any]:
    """
    Conduct outlier analysis on CGM data for specified subjects.
    
    Analyzes glucose values and features for outliers using:
    - Z-score method (|z| > 3)
    - IQR method (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    - Isolation Forest
    - Statistical summary
    
    Args:
        subject_ids: List of subject IDs to analyze. If None, randomly selects num_subjects.
        subjects_dir: Directory containing processed subject files
        num_subjects: Number of subjects to analyze if subject_ids is None
        save_report: Optional path to save outlier analysis report
        output_dir: Directory to save reports
    
    Returns:
        Dictionary containing outlier analysis results
    """
    import os
    import glob
    import random
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    
    # Get subjects to analyze
    if subject_ids is None:
        all_subjects = sorted([os.path.basename(f).replace('Subject_', '').replace('.xlsx', '') 
                              for f in glob.glob(os.path.join(subjects_dir, "Subject_*.xlsx"))])
        subject_ids = random.sample(all_subjects, min(num_subjects, len(all_subjects)))
    
    print("=" * 80)
    print(f"OUTLIER ANALYSIS - {len(subject_ids)} Subjects")
    print("=" * 80)
    print(f"\nSubjects selected: {', '.join(subject_ids)}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each subject
    all_results = {}
    
    for subject_id in subject_ids:
        print(f"\n{'=' * 80}")
        print(f"Analyzing Subject: {subject_id}")
        print("=" * 80)
        
        try:
            filepath = os.path.join(subjects_dir, f"Subject_{subject_id}.xlsx")
            if not os.path.exists(filepath):
                print(f"✗ File not found: {filepath}")
                continue
            
            # Load data
            df = pd.read_excel(filepath, sheet_name='processed_data', index_col=0)
            df.index = pd.to_datetime(df.index)
            
            # Focus on glucose-related features
            glucose_cols = [col for col in df.columns if 'glucose' in col.lower() or col in ['target', 'current_glucose']]
            
            if not glucose_cols:
                print(f"✗ No glucose columns found for {subject_id}")
                continue
            
            subject_results = {
                'subject_id': subject_id,
                'total_samples': len(df),
                'features_analyzed': [],
                'outlier_summary': {}
            }
            
            # Analyze each glucose-related feature
            for col in glucose_cols[:3]:  # Analyze top 3 glucose features
                if col not in df.columns:
                    continue
                
                data = df[col].dropna()
                if len(data) == 0:
                    continue
                
                print(f"\n  Analyzing feature: {col}")
                print(f"    Samples: {len(data):,}")
                
                feature_results = {
                    'feature': col,
                    'n_samples': len(data),
                    'statistics': {},
                    'outliers': {}
                }
                
                # Statistical summary
                feature_results['statistics'] = {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'iqr': float(data.quantile(0.75) - data.quantile(0.25))
                }
                
                stats_dict = feature_results['statistics']
                print(f"    Mean: {stats_dict['mean']:.2f}, Std: {stats_dict['std']:.2f}")
                print(f"    Range: [{stats_dict['min']:.2f}, {stats_dict['max']:.2f}]")
                
                # Method 1: Z-score method (|z| > 3)
                z_scores = np.abs((data - stats_dict['mean']) / stats_dict['std'])
                z_outliers = data[z_scores > 3]
                z_outlier_count = len(z_outliers)
                z_outlier_pct = (z_outlier_count / len(data)) * 100
                
                feature_results['outliers']['z_score'] = {
                    'count': z_outlier_count,
                    'percentage': z_outlier_pct,
                    'threshold': 3.0,
                    'outlier_indices': z_outliers.index.tolist()[:50]  # Limit to first 50
                }
                
                print(f"    Z-score outliers (|z| > 3): {z_outlier_count} ({z_outlier_pct:.2f}%)")
                
                # Method 2: IQR method
                lower_bound = stats_dict['q25'] - 1.5 * stats_dict['iqr']
                upper_bound = stats_dict['q75'] + 1.5 * stats_dict['iqr']
                iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
                iqr_outlier_count = len(iqr_outliers)
                iqr_outlier_pct = (iqr_outlier_count / len(data)) * 100
                
                feature_results['outliers']['iqr'] = {
                    'count': iqr_outlier_count,
                    'percentage': iqr_outlier_pct,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'outlier_indices': iqr_outliers.index.tolist()[:50]
                }
                
                print(f"    IQR outliers: {iqr_outlier_count} ({iqr_outlier_pct:.2f}%)")
                print(f"    IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
                
                # Method 3: Isolation Forest (if enough samples)
                if len(data) > 100:
                    try:
                        # Reshape for Isolation Forest
                        X = data.values.reshape(-1, 1)
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outliers_pred = iso_forest.fit_predict(X)
                        iso_outliers = data[outliers_pred == -1]
                        iso_outlier_count = len(iso_outliers)
                        iso_outlier_pct = (iso_outlier_count / len(data)) * 100
                        
                        feature_results['outliers']['isolation_forest'] = {
                            'count': iso_outlier_count,
                            'percentage': iso_outlier_pct,
                            'contamination': 0.1,
                            'outlier_indices': iso_outliers.index.tolist()[:50]
                        }
                        
                        print(f"    Isolation Forest outliers: {iso_outlier_count} ({iso_outlier_pct:.2f}%)")
                    except Exception as e:
                        print(f"    Isolation Forest failed: {e}")
                        feature_results['outliers']['isolation_forest'] = None
                else:
                    feature_results['outliers']['isolation_forest'] = None
                    print(f"    Isolation Forest skipped (insufficient samples)")
                
                # Determine overall outlier status
                outlier_methods = [m for m in ['z_score', 'iqr', 'isolation_forest'] 
                                 if feature_results['outliers'].get(m) is not None]
                avg_outlier_pct = np.mean([feature_results['outliers'][m]['percentage'] 
                                          for m in outlier_methods])
                
                if avg_outlier_pct > 10:
                    status = "High outlier rate (>10%)"
                elif avg_outlier_pct > 5:
                    status = "Moderate outlier rate (5-10%)"
                else:
                    status = "Normal outlier rate (<5%)"
                
                feature_results['overall_status'] = status
                feature_results['average_outlier_percentage'] = float(avg_outlier_pct)
                
                print(f"    Overall status: {status}")
                
                subject_results['features_analyzed'].append(feature_results)
            
            # Summary for subject
            if subject_results['features_analyzed']:
                avg_outlier_pct = np.mean([f['average_outlier_percentage'] 
                                          for f in subject_results['features_analyzed']])
                subject_results['outlier_summary'] = {
                    'average_outlier_percentage': float(avg_outlier_pct),
                    'features_with_outliers': len([f for f in subject_results['features_analyzed'] 
                                                  if f['average_outlier_percentage'] > 5])
                }
            
            all_results[subject_id] = subject_results
            print(f"✓ Completed analysis for {subject_id}")
            
        except Exception as e:
            print(f"✗ Error analyzing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Aggregate results
    print(f"\n{'=' * 80}")
    print("OUTLIER ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nSuccessfully analyzed: {len(all_results)} subjects")
    
    # Overall statistics
    all_outlier_pcts = []
    for subject_id, result in all_results.items():
        if result.get('outlier_summary'):
            all_outlier_pcts.append(result['outlier_summary']['average_outlier_percentage'])
    
    if all_outlier_pcts:
        print(f"\nOverall Statistics:")
        print(f"  - Average outlier percentage: {np.mean(all_outlier_pcts):.2f}%")
        print(f"  - Min outlier percentage: {np.min(all_outlier_pcts):.2f}%")
        print(f"  - Max outlier percentage: {np.max(all_outlier_pcts):.2f}%")
        print(f"  - Median outlier percentage: {np.median(all_outlier_pcts):.2f}%")
    
    # Subject-level summary
    print(f"\nSubject-Level Summary:")
    print(f"{'Subject':<20} {'Avg Outlier %':<15} {'Features w/ Outliers':<20} {'Status':<20}")
    print("-" * 80)
    
    for subject_id, result in sorted(all_results.items()):
        summary = result.get('outlier_summary', {})
        avg_pct = summary.get('average_outlier_percentage', 0)
        features_with = summary.get('features_with_outliers', 0)
        
        if avg_pct > 10:
            status = "High Risk"
        elif avg_pct > 5:
            status = "Moderate Risk"
        else:
            status = "Normal"
        
        print(f"{subject_id:<20} {avg_pct:>13.2f}%  {features_with:>19}  {status:<20}")
    
    # Save report
    report_data = {
        'analysis_date': str(pd.Timestamp.now()),
        'subjects_analyzed': len(all_results),
        'subject_ids': list(all_results.keys()),
        'overall_statistics': {
            'average_outlier_percentage': float(np.mean(all_outlier_pcts)) if all_outlier_pcts else 0,
            'min_outlier_percentage': float(np.min(all_outlier_pcts)) if all_outlier_pcts else 0,
            'max_outlier_percentage': float(np.max(all_outlier_pcts)) if all_outlier_pcts else 0,
            'median_outlier_percentage': float(np.median(all_outlier_pcts)) if all_outlier_pcts else 0
        },
        'subject_results': all_results
    }
    
    if save_report:
        report_path = save_report
    else:
        report_path = os.path.join(output_dir, 'outlier_analysis_report.json')
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"\n✓ Outlier analysis report saved to: {report_path}")
    
    return report_data

