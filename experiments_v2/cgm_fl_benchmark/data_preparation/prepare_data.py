#!/usr/bin/env python3
"""
Data Preparation Script for V2 Experiments
Implements feature selection and preprocessing based on analysis results
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from feature_selector import FeatureSelector
from sklearn.preprocessing import RobustScaler

class DataPreprocessor:
    """Data preprocessor for V2 experiments"""
    
    # Valid subjects (47 subjects, excluding 10 failed)
    VALID_SUBJECTS = [
        "SUB001", "SUB002", "SUB003", "SUB004", "SUB005",
        "Subject10", "Subject11", "Subject12", "Subject16", "Subject17",
        "Subject19", "Subject20", "Subject22", "Subject23", "Subject24",
        "Subject26", "Subject27", "Subject28", "Subject29", "Subject3",
        "Subject30", "Subject31", "Subject33", "Subject34", "Subject35",
        "Subject36", "Subject37", "Subject38", "Subject39", "Subject4",
        "Subject40", "Subject41", "Subject42", "Subject43", "Subject45",
        "Subject46", "Subject47", "Subject48", "Subject49", "Subject50",
        "Subject51", "Subject52", "Subject53", "Subject54", "Subject6",
        "Subject8", "Subject9"
    ]
    
    def __init__(self, 
                 feature_set: str = "core_24",
                 use_robust_scaling: bool = True,
                 create_interactions: bool = False):
        """
        Initialize preprocessor
        
        Args:
            feature_set: Feature set to use ('core_24', 'top_10', 'temporal_focused', 'all')
            use_robust_scaling: Use robust scaling (median/IQR) instead of standard scaling
            create_interactions: Create interaction features
        """
        self.feature_selector = FeatureSelector(feature_set)
        self.use_robust_scaling = use_robust_scaling
        self.create_interactions = create_interactions
        self.scaler = RobustScaler() if use_robust_scaling else None
    
    def load_subject_data(self, subject_id: str, 
                         data_dir: str = "data/processed/subjects") -> pd.DataFrame:
        """
        Load subject data
        
        Args:
            subject_id: Subject ID
            data_dir: Directory containing subject files
        
        Returns:
            Loaded dataframe
        """
        filepath = Path(data_dir) / f"Subject_{subject_id}.xlsx"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Subject file not found: {filepath}")
        
        df = pd.read_excel(filepath, sheet_name='processed_data', index_col=0)
        df.index = pd.to_datetime(df.index)
        
        return df
    
    def preprocess_subject(self, subject_id: str, 
                          data_dir: str = "data/processed/subjects") -> pd.DataFrame:
        """
        Preprocess single subject data
        
        Args:
            subject_id: Subject ID
            data_dir: Directory containing subject files
        
        Returns:
            Preprocessed dataframe
        """
        # Load data
        df = self.load_subject_data(subject_id, data_dir)
        
        # Select features
        df_selected = self.feature_selector.select_features(df, remove_redundant=True)
        
        # Create interaction features if requested
        if self.create_interactions:
            df_selected = self.feature_selector.create_interaction_features(df_selected)
        
        # Separate features and target
        if 'target' in df_selected.columns:
            feature_cols = [col for col in df_selected.columns if col != 'target']
            X = df_selected[feature_cols]
            y = df_selected[['target']]
        else:
            X = df_selected
            y = None
        
        # Apply scaling
        if self.scaler is not None:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                index=X.index,
                columns=X.columns
            )
        else:
            X_scaled = X.copy()
        
        # Combine back
        if y is not None:
            result = pd.concat([X_scaled, y], axis=1)
        else:
            result = X_scaled
        
        # Remove NaN
        result = result.dropna()
        
        return result
    
    def prepare_all_subjects(self, 
                            output_dir: str = None,
                            data_dir: str = "data/processed/subjects") -> Dict[str, pd.DataFrame]:
        """
        Prepare data for all valid subjects
        
        Args:
            output_dir: Directory to save processed data (optional)
            data_dir: Directory containing subject files
        
        Returns:
            Dictionary of {subject_id: preprocessed_dataframe}
        """
        all_data = {}
        
        for subject_id in self.VALID_SUBJECTS:
            try:
                print(f"Processing {subject_id}...")
                processed_data = self.preprocess_subject(subject_id, data_dir)
                all_data[subject_id] = processed_data
                
                if output_dir:
                    output_path = Path(output_dir) / f"{subject_id}_preprocessed.csv"
                    processed_data.to_csv(output_path)
                    print(f"  Saved to {output_path}")
                
            except Exception as e:
                print(f"  Failed to process {subject_id}: {e}")
                continue
        
        print(f"\nSuccessfully processed {len(all_data)}/{len(self.VALID_SUBJECTS)} subjects")
        
        return all_data

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare data for V2 experiments")
    parser.add_argument("--feature_set", type=str, default="core_24",
                       choices=["core_24", "top_10", "temporal_focused", "all"],
                       help="Feature set to use")
    parser.add_argument("--robust_scaling", action="store_true", default=True,
                       help="Use robust scaling")
    parser.add_argument("--interactions", action="store_true",
                       help="Create interaction features")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for processed data")
    parser.add_argument("--data_dir", type=str, default="data/processed/subjects",
                       help="Input data directory")
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = DataPreprocessor(
        feature_set=args.feature_set,
        use_robust_scaling=args.robust_scaling,
        create_interactions=args.interactions
    )
    
    # Prepare all subjects
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    all_data = preprocessor.prepare_all_subjects(
        output_dir=args.output_dir,
        data_dir=args.data_dir
    )
    
    print(f"\n✓ Data preparation complete!")
    print(f"  Feature set: {args.feature_set}")
    print(f"  Features per subject: {preprocessor.feature_selector.get_feature_count()}")
    print(f"  Subjects processed: {len(all_data)}")

if __name__ == "__main__":
    main()

