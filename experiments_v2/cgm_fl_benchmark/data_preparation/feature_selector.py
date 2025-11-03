"""
Feature Selection Module for V2 Experiments
Based on comprehensive feature analysis results
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class FeatureSelector:
    """Feature selector based on analysis results"""
    
    # Core 24 features (consistently important ≥70% of subjects)
    CORE_24_FEATURES = [
        "hour",
        "hour_cos",
        "hourly_baseline",
        "hour_sin",
        "roc_60min",
        "current_glucose",
        "roc_30min",
        "min_6h",
        "max_6h",
        "mean_24h",
        "max_12h",
        "std_3h",
        "mean_12h",
        "std_12h",
        "min_12h",
        "min_24h",
        "max_3h",
        "std_6h",
        "max_24h",
        "std_1h",
        "std_24h",
        "in_tight_range",
        "is_hyperglycemic",
        "deviation_from_recent_trend"
    ]
    
    # Top 10 features (highest composite scores)
    TOP_10_FEATURES = [
        "current_glucose",
        "hour",
        "roc_60min",
        "hourly_baseline",
        "in_tight_range",
        "deviation_from_recent_trend",
        "is_hyperglycemic",
        "max_6h",
        "min_6h",
        "roc_30min"
    ]
    
    # Temporal-focused features (10 features)
    TEMPORAL_FOCUSED = [
        "hour",
        "hour_cos",
        "hourly_baseline",
        "hour_sin",
        "current_glucose",
        "roc_60min",
        "roc_30min",
        "deviation_from_recent_trend",
        "in_tight_range",
        "is_hyperglycemic"
    ]
    
    # Redundant features to remove
    REDUNDANT_FEATURES = [
        "deviation_from_daily_mean",  # redundant with current_glucose (r=0.9846)
        "in_target_range",             # redundant with is_hyperglycemic (r=0.9995)
    ]
    
    def __init__(self, feature_set: str = "core_24"):
        """
        Initialize feature selector
        
        Args:
            feature_set: One of 'core_24', 'top_10', 'temporal_focused', 'all'
        """
        self.feature_set_name = feature_set
        self.feature_list = self._get_feature_list(feature_set)
    
    def _get_feature_list(self, feature_set: str) -> List[str]:
        """Get feature list based on strategy"""
        if feature_set == "core_24":
            return self.CORE_24_FEATURES.copy()
        elif feature_set == "top_10":
            return self.TOP_10_FEATURES.copy()
        elif feature_set == "temporal_focused":
            return self.TEMPORAL_FOCUSED.copy()
        elif feature_set == "all":
            return None  # Use all features
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")
    
    def select_features(self, df: pd.DataFrame, remove_redundant: bool = True) -> pd.DataFrame:
        """
        Select features from dataframe
        
        Args:
            df: Input dataframe
            remove_redundant: Whether to remove redundant features
        
        Returns:
            Dataframe with selected features + target
        """
        if self.feature_list is None:
            # Use all features
            selected_df = df.copy()
        else:
            # Check which features are available
            available_features = [f for f in self.feature_list if f in df.columns]
            missing_features = [f for f in self.feature_list if f not in df.columns]
            
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
            
            # Select available features
            feature_cols = available_features.copy()
            
            # Always include target if present
            if 'target' in df.columns:
                feature_cols.append('target')
            
            selected_df = df[feature_cols].copy()
        
        # Remove redundant features if requested
        if remove_redundant:
            for red_feat in self.REDUNDANT_FEATURES:
                if red_feat in selected_df.columns and red_feat not in self.feature_list:
                    selected_df = selected_df.drop(columns=[red_feat])
        
        return selected_df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature interactions based on analysis findings
        
        Args:
            df: Input dataframe with selected features
        
        Returns:
            Dataframe with interaction features added
        """
        result_df = df.copy()
        
        # Temporal × Glucose interactions
        if 'hour' in df.columns and 'current_glucose' in df.columns:
            result_df['hour_x_glucose'] = df['hour'] * df['current_glucose']
        
        if 'hour_cos' in df.columns and 'current_glucose' in df.columns:
            result_df['hour_cos_x_glucose'] = df['hour_cos'] * df['current_glucose']
        
        # Rate of change × Current state
        if 'roc_60min' in df.columns and 'current_glucose' in df.columns:
            result_df['roc60_x_current'] = df['roc_60min'] * df['current_glucose']
        
        # Medical × Trends
        if 'is_hyperglycemic' in df.columns and 'deviation_from_recent_trend' in df.columns:
            result_df['hyper_x_deviation'] = df['is_hyperglycemic'] * df['deviation_from_recent_trend']
        
        return result_df
    
    def get_feature_count(self) -> int:
        """Get number of features in current selection"""
        return len(self.feature_list) if self.feature_list else None
    
    def get_feature_list(self) -> List[str]:
        """Get current feature list"""
        return self.feature_list.copy() if self.feature_list else None

