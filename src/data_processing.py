import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

class DataConfig:
    """Configuration for data processing"""
    def __init__(self):
        self.raw_data_dir = "data/raw"
        self.processed_data_dir = "data/processed"
        self.prediction_horizon = 12  # 60 minutes
        self.history_days = 15  # Use last 15 days of data
        self.min_samples_per_subject = 500  # Reduced since we're using only 15 days
        self.glucose_column = "glucose"
        self.data_frequency = "5T"
        
    def setup_directories(self):
        """Create directory structure for data processing"""
        dirs = {
            'raw': self.raw_data_dir,
            'processed': self.processed_data_dir,
            'subjects': f"{self.processed_data_dir}/subjects",
            'metadata': f"{self.processed_data_dir}/metadata"
        }
        
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        return dirs

class CGMFeatureEngineer:
    """Feature engineering for CGM data focusing on last 15 days"""
    
    def __init__(self, prediction_horizon: int = 12, history_days: int = 15):
        self.prediction_horizon = prediction_horizon
        self.history_days = history_days
        self.samples_per_hour = 12
        self.samples_per_day = 288
        self.total_samples_needed = history_days * self.samples_per_day
        
    def prepare_subject_data(self, subject_id: str, df: pd.DataFrame, 
                           glucose_col: str = 'glucose') -> pd.DataFrame:
        """Prepare dataset using only the last 15 days of data"""
        try:
            print(f"Processing subject {subject_id}...")
            
            # Validate input data
            if df.empty:
                raise ValueError(f"Empty DataFrame for subject {subject_id}")
                
            if glucose_col not in df.columns:
                raise ValueError(f"Glucose column '{glucose_col}' not found")
            
            # Ensure datetime index and sort
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("DataFrame index must be datetime")
            
            df = df.sort_index()
            # Coerce glucose to numeric and drop invalid rows
            df[glucose_col] = pd.to_numeric(df[glucose_col], errors='coerce')
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=[glucose_col])
            
            # Extract only the last 15 days of data
            if len(df) > 0:
                end_date = df.index.max()
                start_date = end_date - timedelta(days=self.history_days)
                df = df.loc[start_date:end_date]
                
                print(f"  Using data from {start_date.date()} to {end_date.date()} "
                      f"({len(df)} samples)")
            
            # Create target
            df_processed = df.copy()
            df_processed['target'] = df_processed[glucose_col].shift(-self.prediction_horizon)
            
            # Create features
            features = self._create_15day_features(df_processed, glucose_col)
            features['target'] = df_processed['target']
            
            # Remove rows with NaN targets
            final_data = features.dropna(subset=['target'])
            # Enforce numeric-only features and clean any residual invalids
            final_data = final_data.apply(pd.to_numeric, errors='coerce')
            final_data = final_data.replace([np.inf, -np.inf], np.nan).dropna()
            
            if final_data.empty:
                raise ValueError(f"No valid samples after preprocessing")
            
            # Add subject metadata
            final_data['subject_id'] = subject_id
            
            # Normalize glucose values to [0, 1] range
            glucose_min = 40  # mg/dL
            glucose_max = 400  # mg/dL
            final_data['target'] = (final_data['target'] - glucose_min) / (glucose_max - glucose_min)
            
            # Also normalize current_glucose feature if present
            if 'current_glucose' in final_data.columns:
                final_data['current_glucose'] = (final_data['current_glucose'] - glucose_min) / (glucose_max - glucose_min)
            
            print(f"  Final dataset: {len(final_data)} samples with {features.shape[1]-1} features (target normalized)")
            return final_data
            
        except Exception as e:
            print(f"Error processing subject {subject_id}: {e}")
            return pd.DataFrame()
    
    def _create_15day_features(self, df: pd.DataFrame, glucose_col: str) -> pd.DataFrame:
        """Create features optimized for 15-day CGM data"""
        features = pd.DataFrame(index=df.index)
        
        # 1. TEMPORAL FEATURES (5 features)
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # 2. RECENT VALUES AND TRENDS (6 features)
        features['current_glucose'] = df[glucose_col]
        features['prev_15min'] = df[glucose_col].shift(3).fillna(method='bfill')
        features['prev_30min'] = df[glucose_col].shift(6).fillna(method='bfill')
        features['prev_60min'] = df[glucose_col].shift(12).fillna(method='bfill')
        features['roc_30min'] = (df[glucose_col] - features['prev_30min']) / 6
        features['roc_60min'] = (df[glucose_col] - features['prev_60min']) / 12
        
        # 3. ROLLING STATISTICS - Optimized for 15-day window (8 features)
        windows = {
            '1h': self.samples_per_hour,      # 12 samples
            '3h': 3 * self.samples_per_hour,  # 36 samples
            '6h': 6 * self.samples_per_hour,  # 72 samples
            '12h': 12 * self.samples_per_hour, # 144 samples
            '24h': self.samples_per_day,      # 288 samples
        }
        
        for window_name, window_size in windows.items():
            # Only create essential statistics for each window
            features[f'mean_{window_name}'] = df[glucose_col].rolling(
                window_size, min_periods=1).mean()
            features[f'std_{window_name}'] = df[glucose_col].rolling(
                window_size, min_periods=1).std()
            
            # For longer windows, add min/max
            if window_size >= 36:  # 3h or more
                features[f'min_{window_name}'] = df[glucose_col].rolling(
                    window_size, min_periods=1).min()
                features[f'max_{window_name}'] = df[glucose_col].rolling(
                    window_size, min_periods=1).max()
        
        # 4. MEDICAL STATUS FEATURES (6 features)
        features['is_hypoglycemic'] = (df[glucose_col] < 70).astype(int)
        features['is_hyperglycemic'] = (df[glucose_col] > 180).astype(int)
        features['in_target_range'] = ((df[glucose_col] >= 70) & (df[glucose_col] <= 180)).astype(int)
        features['in_tight_range'] = ((df[glucose_col] >= 80) & (df[glucose_col] <= 140)).astype(int)
        
        # Recent extreme events (6h window)
        features['recent_hypo_6h'] = features['is_hypoglycemic'].rolling(
            windows['6h'], min_periods=1).sum()
        features['recent_hyper_6h'] = features['is_hyperglycemic'].rolling(
            windows['6h'], min_periods=1).sum()
        
        # 5. PATTERN AND DEVIATION FEATURES (4 features)
        features['deviation_from_daily_mean'] = df[glucose_col] - features['mean_24h']
        features['deviation_from_recent_trend'] = df[glucose_col] - features['mean_6h']
        features['glucose_variability'] = features['std_6h'] / features['mean_6h']
        features['trend_strength'] = self._calculate_trend_strength(df[glucose_col])
        
        # 6. DAILY PATTERN FEATURES (3 features)
        features = self._add_daily_patterns(features, df[glucose_col])
        
        # Handle any remaining NaN values
        features = features.fillna(method='bfill').fillna(method='ffill').fillna(0)
        # Coerce to numeric and clean any non-finite values
        features = features.apply(pd.to_numeric, errors='coerce')
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Normalize features to [0, 1] range for better training
        for col in features.columns:
            if features[col].dtype in [np.float64, np.float32]:
                col_min = features[col].min()
                col_max = features[col].max()
                if col_max > col_min:
                    features[col] = (features[col] - col_min) / (col_max - col_min)
        
        print(f"  Created {features.shape[1]} features total (normalized)")
        return features
    
    def _calculate_trend_strength(self, series: pd.Series, window: int = 12) -> pd.Series:
        """Calculate the strength of glucose trends over recent window"""
        def linear_trend(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]  # Slope
        
        return series.rolling(window, min_periods=2).apply(linear_trend, raw=True)
    
    def _add_daily_patterns(self, features: pd.DataFrame, glucose_series: pd.Series) -> pd.DataFrame:
        """Add daily pattern features based on 15-day history"""
        # Calculate hourly patterns from the available data
        hourly_pattern = glucose_series.groupby(glucose_series.index.hour).mean()
        
        # Fill any missing hours with overall mean
        all_hours = pd.Series(index=range(24), data=glucose_series.mean())
        hourly_pattern = hourly_pattern.reindex(all_hours.index).fillna(all_hours)
        
        # Add pattern features
        features['hourly_baseline'] = features.index.hour.map(hourly_pattern)
        features['deviation_from_pattern'] = glucose_series - features['hourly_baseline']
        features['pattern_consistency'] = self._calculate_daily_consistency(glucose_series)
        
        return features
    
    def _calculate_daily_consistency(self, series: pd.Series) -> pd.Series:
        """Calculate how consistent daily patterns are across the 15-day window"""
        if len(series) < 576:  # Need at least 2 days for comparison
            return pd.Series(0.5, index=series.index)  # Neutral consistency
        
        consistency_scores = []
        for i in range(len(series)):
            current_time = series.index[i]
            current_hour = current_time.hour
            current_minute = current_time.minute
            
            # Compare with same time on previous days
            similar_times = []
            for days_back in range(1, min(8, len(series) // 288)):  # Check up to 7 previous days
                prev_time = current_time - timedelta(days=days_back)
                if prev_time in series.index:
                    similar_times.append(series[prev_time])
            
            if similar_times:
                current_val = series.iloc[i]
                avg_prev = np.mean(similar_times)
                # Consistency score: 1 - normalized difference
                consistency = 1 - abs(current_val - avg_prev) / max(current_val, avg_prev, 1)
                consistency_scores.append(max(0, consistency))
            else:
                consistency_scores.append(0.5)
        
        return pd.Series(consistency_scores, index=series.index)

class CGMDataProcessor:
    """Main data processing pipeline focusing on last 15 days"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.dirs = config.setup_directories()
        self.feature_engineer = CGMFeatureEngineer(
            prediction_horizon=config.prediction_horizon,
            history_days=config.history_days
        )
        
    def process_raw_data(self, data_loader_function: Optional[Callable] = None):
        """Main data processing pipeline using last 15 days"""
        print("=== Starting CGM Data Processing (Last 15 Days) ===")
        
        # Load raw data
        if data_loader_function:
            raw_subjects_data = data_loader_function()
        else:
            raw_subjects_data = self._load_sample_data()
        
        if not raw_subjects_data:
            raise ValueError("No raw data loaded")
        
        print(f"Loaded data for {len(raw_subjects_data)} subjects")
        print(f"Using last {self.config.history_days} days for each subject")
        
        # Process each subject
        processed_subjects = {}
        subject_metadata = {}
        skipped_subjects = []
        
        for subject_id, subject_data in raw_subjects_data.items():
            print(f"\n--- Processing {subject_id} ---")
            
            # Check if we have enough historical data
            if len(subject_data) < self.config.min_samples_per_subject:
                print(f"  Skipping {subject_id}: insufficient data ({len(subject_data)} samples)")
                skipped_subjects.append(subject_id)
                continue
            
            processed_data = self.feature_engineer.prepare_subject_data(subject_id, subject_data)
            
            if not processed_data.empty and len(processed_data) >= self.config.min_samples_per_subject:
                # Save individual subject file
                filename = f"Subject_{subject_id}.xlsx"
                filepath = f"{self.dirs['subjects']}/{filename}"
                
                # Save as Excel with multiple sheets
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    # Main data sheet
                    processed_data.to_excel(writer, sheet_name='processed_data', index=True)
                    
                    # Statistics sheet
                    stats = self._calculate_subject_statistics(processed_data, subject_id)
                    stats_df = pd.DataFrame([stats])
                    stats_df.to_excel(writer, sheet_name='statistics', index=False)
                    
                    # Feature description sheet
                    feature_desc = self._create_feature_description(processed_data)
                    feature_desc.to_excel(writer, sheet_name='feature_descriptions', index=False)
                
                processed_subjects[subject_id] = processed_data
                subject_metadata[subject_id] = stats
                
                print(f"✓ Saved {filename}")
            else:
                print(f"✗ Skipped {subject_id}: insufficient processed data")
                skipped_subjects.append(subject_id)
        
        # Save global metadata
        self._save_global_metadata(processed_subjects, subject_metadata, skipped_subjects)
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed: {len(processed_subjects)} subjects")
        print(f"Skipped: {len(skipped_subjects)} subjects")
        print(f"Output directory: {self.dirs['subjects']}")
        
        return processed_subjects, subject_metadata
    
    def _calculate_subject_statistics(self, data: pd.DataFrame, subject_id: str) -> Dict:
        """Calculate comprehensive statistics for a subject's 15-day data"""
        glucose_data = data['current_glucose'] if 'current_glucose' in data.columns else data.iloc[:, 0]
        
        stats = {
            'subject_id': subject_id,
            'total_samples': len(data),
            'date_range_start': data.index.min().strftime('%Y-%m-%d'),
            'date_range_end': data.index.max().strftime('%Y-%m-%d'),
            'duration_days': (data.index.max() - data.index.min()).days,
            'samples_per_day': len(data) / max(1, (data.index.max() - data.index.min()).days),
            
            # Glucose statistics
            'mean_glucose': float(glucose_data.mean()),
            'median_glucose': float(glucose_data.median()),
            'std_glucose': float(glucose_data.std()),
            'min_glucose': float(glucose_data.min()),
            'max_glucose': float(glucose_data.max()),
            'coefficient_variation': float(glucose_data.std() / glucose_data.mean()),
            
            # Time in ranges (if available)
            'time_in_range_pct': float(data['in_target_range'].mean() * 100) if 'in_target_range' in data.columns else 0,
            'time_hypo_pct': float(data['is_hypoglycemic'].mean() * 100) if 'is_hypoglycemic' in data.columns else 0,
            'time_hyper_pct': float(data['is_hyperglycemic'].mean() * 100) if 'is_hyperglycemic' in data.columns else 0,
            
            # Data quality
            'missing_data_pct': float(data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100),
            'features_generated': len([col for col in data.columns if col not in ['target', 'subject_id']])
        }
        
        return stats
    
    def _create_feature_description(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a description of all features"""
        feature_descriptions = []
        feature_categories = {
            'Temporal': ['hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos'],
            'Recent Values': ['current_glucose', 'prev_15min', 'prev_30min', 'prev_60min', 'roc_30min', 'roc_60min'],
            'Rolling Statistics': [col for col in data.columns if any(x in col for x in ['mean_', 'std_', 'min_', 'max_'])],
            'Medical Status': ['is_hypoglycemic', 'is_hyperglycemic', 'in_target_range', 'in_tight_range', 'recent_hypo_6h', 'recent_hyper_6h'],
            'Pattern Analysis': ['deviation_from_daily_mean', 'deviation_from_recent_trend', 'glucose_variability', 'trend_strength'],
            'Daily Patterns': ['hourly_baseline', 'deviation_from_pattern', 'pattern_consistency']
        }
        
        for category, features in feature_categories.items():
            for feature in features:
                if feature in data.columns and feature not in ['target', 'subject_id']:
                    feature_descriptions.append({
                        'feature_name': feature,
                        'category': category,
                        'description': self._get_feature_description(feature)
                    })
        
        return pd.DataFrame(feature_descriptions)
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get description for a feature"""
        descriptions = {
            'current_glucose': 'Current glucose value (mg/dL)',
            'prev_15min': 'Glucose value 15 minutes ago',
            'prev_30min': 'Glucose value 30 minutes ago',
            'prev_60min': 'Glucose value 60 minutes ago',
            'roc_30min': 'Rate of change over 30 minutes (mg/dL per sample)',
            'roc_60min': 'Rate of change over 60 minutes (mg/dL per sample)',
            'hour': 'Hour of day (0-23)',
            'day_of_week': 'Day of week (0-6, where 0=Monday)',
            'is_weekend': 'Whether current day is weekend (1) or weekday (0)',
            'hour_sin': 'Sine transformation of hour for cyclical encoding',
            'hour_cos': 'Cosine transformation of hour for cyclical encoding',
            'is_hypoglycemic': 'Glucose < 70 mg/dL (1) or not (0)',
            'is_hyperglycemic': 'Glucose > 180 mg/dL (1) or not (0)',
            'in_target_range': 'Glucose between 70-180 mg/dL (1) or not (0)',
            'in_tight_range': 'Glucose between 80-140 mg/dL (1) or not (0)',
            'deviation_from_daily_mean': 'Difference from 24-hour rolling mean',
            'deviation_from_recent_trend': 'Difference from 6-hour rolling mean',
            'glucose_variability': 'Coefficient of variation (std/mean) over 6 hours',
            'trend_strength': 'Slope of glucose trend over recent values',
            'hourly_baseline': 'Typical glucose value for this hour of day',
            'deviation_from_pattern': 'Difference from typical hourly pattern',
            'pattern_consistency': 'Consistency with previous days patterns (0-1)'
        }
        
        # Handle rolling statistics
        if 'mean_' in feature_name:
            window = feature_name.replace('mean_', '')
            return f'Rolling mean over {window} window'
        elif 'std_' in feature_name:
            window = feature_name.replace('std_', '')
            return f'Rolling standard deviation over {window} window'
        elif 'min_' in feature_name:
            window = feature_name.replace('min_', '')
            return f'Rolling minimum over {window} window'
        elif 'max_' in feature_name:
            window = feature_name.replace('max_', '')
            return f'Rolling maximum over {window} window'
        elif 'recent_hypo_6h' in feature_name:
            return 'Number of hypoglycemic events in last 6 hours'
        elif 'recent_hyper_6h' in feature_name:
            return 'Number of hyperglycemic events in last 6 hours'
        
        return descriptions.get(feature_name, 'No description available')
    
    def _save_global_metadata(self, processed_subjects: Dict, subject_metadata: Dict, skipped_subjects: List):
        """Save global metadata and summary"""
        # Overall summary
        summary = {
            'processing_date': datetime.now().isoformat(),
            'total_subjects_processed': len(processed_subjects),
            'total_subjects_skipped': len(skipped_subjects),
            'skipped_subjects': skipped_subjects,
            'prediction_horizon': self.config.prediction_horizon,
            'history_days': self.config.history_days,
            'min_samples_per_subject': self.config.min_samples_per_subject,
            'subjects_processed': list(processed_subjects.keys()),
            'total_samples': sum([len(data) for data in processed_subjects.values()]),
            'average_samples_per_subject': np.mean([len(data) for data in processed_subjects.values()]) if processed_subjects else 0
        }
        
        with open(f"{self.dirs['metadata']}/processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Subject details
        if subject_metadata:
            subjects_df = pd.DataFrame.from_dict(subject_metadata, orient='index')
            subjects_df.to_excel(f"{self.dirs['metadata']}/subjects_metadata.xlsx", index=False)
            subjects_df.to_csv(f"{self.dirs['metadata']}/subjects_metadata.csv", index=False)
        
        # Feature list
        if processed_subjects:
            sample_subject = list(processed_subjects.values())[0]
            features = [col for col in sample_subject.columns if col not in ['target', 'subject_id']]
            
            feature_info = {
                'total_features': len(features),
                'features': features,
                'feature_categories': {
                    'temporal': [f for f in features if any(x in f for x in ['hour', 'day', 'weekend', 'sin', 'cos'])],
                    'recent_values': [f for f in features if any(x in f for x in ['prev_', 'roc_'])],
                    'statistical': [f for f in features if any(x in f for x in ['mean_', 'std_', 'min_', 'max_'])],
                    'medical': [f for f in features if any(x in f for x in ['hypo', 'hyper', 'target_range', 'tight_range'])],
                    'pattern': [f for f in features if any(x in f for x in ['deviation', 'variability', 'trend', 'pattern', 'consistency'])]
                }
            }
            
            with open(f"{self.dirs['metadata']}/features_info.json", 'w') as f:
                json.dump(feature_info, f, indent=2)
        
        print(f"\nMetadata saved in: {self.dirs['metadata']}")
    
    def _load_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Load actual raw data from data/raw directory"""
        import os
        raw_dir = self.config.raw_data_dir
        subjects_data = {}
        
        # Get all subject files
        subject_files = [f for f in os.listdir(raw_dir) 
                        if f.startswith('Subject') and f.endswith('.xlsx') 
                        and 'Demographics' not in f]
        
        # Sort files to ensure consistent ordering
        subject_files.sort(key=lambda x: int(x.replace('Subject', '').replace('.xlsx', '')))
        
        print(f"Loading {len(subject_files)} raw subject files from {raw_dir}...")
        
        for filename in subject_files:
            try:
                filepath = os.path.join(raw_dir, filename)
                subject_id = filename.replace('.xlsx', '')
                
                # Read Excel file - first sheet only
                df = pd.read_excel(filepath, sheet_name=0)
                
                # Debug: print column names for first file
                if len(subjects_data) == 0:
                    print(f"    Columns in {filename}: {list(df.columns)}")
                    print(f"    First few rows:\n{df.head()}")
                
                # Find glucose column (could be 'CGM', 'Glucose', 'BG', 'mg/dl', etc.)
                glucose_col = None
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'glucose' in col_lower or 'cgm' in col_lower or 'bg' in col_lower or 'mg/dl' in col_lower or 'mg_dl' in col_lower:
                        glucose_col = col
                        df = df.rename(columns={col: 'glucose'})
                        break
                
                if glucose_col is None:
                    # Use first numeric column as glucose
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        glucose_col = numeric_cols[0]
                        df = df.rename(columns={glucose_col: 'glucose'})
                
                # Convert to datetime index
                # Check if index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    # Try to find datetime column
                    for col in df.columns:
                        if 'time' in str(col).lower() or 'date' in str(col).lower():
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            df = df.set_index(col)
                            break
                    
                    # If still no datetime index, create one with 5-min intervals
                    if not isinstance(df.index, pd.DatetimeIndex):
                        start_date = datetime(2024, 1, 1)
                        df.index = pd.date_range(start=start_date, periods=len(df), freq='5T')
                
                # Clean up: only keep glucose column with valid datetime index
                df = df[['glucose']].copy()
                df.index.name = 'timestamp'
                
                subjects_data[subject_id] = df
                print(f"  Loaded {filename}: {len(df)} rows")
                
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        
        return subjects_data

def create_sample_data_loader(num_subjects: int = 10, days_per_subject: int = 365):
    """Create sample data loader function that generates multi-year CGM data"""
    def load_sample_data() -> Dict[str, pd.DataFrame]:
        """Generate sample CGM data for multiple subjects with realistic patterns"""
        subjects_data = {}
        
        # Subject patterns representing different glycemic control levels
        subject_profiles = [
            # Well-controlled (HbA1c ~6.5%)
            {'base': 115, 'variability': 18, 'meal_effect': 35, 'trend_noise': 0.1, 'hypo_risk': 0.02, 'hyper_risk': 0.05},
            # Moderately controlled (HbA1c ~7.5%)
            {'base': 135, 'variability': 25, 'meal_effect': 45, 'trend_noise': 0.15, 'hypo_risk': 0.03, 'hyper_risk': 0.15},
            # Poorly controlled (HbA1c ~8.5%)
            {'base': 155, 'variability': 35, 'meal_effect': 60, 'trend_noise': 0.25, 'hypo_risk': 0.05, 'hyper_risk': 0.25},
            # Very unstable (HbA1c ~9.0+%)
            {'base': 170, 'variability': 45, 'meal_effect': 75, 'trend_noise': 0.35, 'hypo_risk': 0.08, 'hyper_risk': 0.35},
            # Conservative treatment (frequent mild hypos)
            {'base': 100, 'variability': 22, 'meal_effect': 30, 'trend_noise': 0.12, 'hypo_risk': 0.10, 'hyper_risk': 0.02},
        ]
        
        for i in range(num_subjects):
            subject_id = f"SUB{i+1:03d}"
            profile = subject_profiles[i % len(subject_profiles)]
            
            # Set unique random seed for each subject to ensure different data patterns
            np.random.seed(i * 42)  # Each subject gets unique seed
            
            # Create date range with 5-minute intervals for multiple years
            start_date = datetime(2022, 1, 1)  # Start 2 years ago
            dates = pd.date_range(start_date, periods=days_per_subject*288, freq='5T')
            
            # Generate realistic glucose patterns
            glucose = generate_realistic_glucose_pattern(dates, profile)
            
            subjects_data[subject_id] = pd.DataFrame(
                {'glucose': glucose}, 
                index=dates
            )
            
            print(f"Generated {days_per_subject} days of data for {subject_id} "
                  f"(profile: {profile['base']}±{profile['variability']} mg/dL)")
        
        return subjects_data
    
    return load_sample_data

def generate_realistic_glucose_pattern(dates: pd.DatetimeIndex, profile: Dict) -> np.ndarray:
    """Generate realistic glucose patterns with daily rhythms and meal effects"""
    base_glucose = profile['base']
    variability = profile['variability']
    meal_effect = profile['meal_effect']
    trend_noise = profile['trend_noise']
    hypo_risk = profile['hypo_risk']
    hyper_risk = profile['hyper_risk']
    
    # Base pattern with circadian rhythm
    hour_of_day = dates.hour + dates.minute/60
    circadian = 15 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Peak around 2 PM
    
    # Meal effects at typical meal times
    meal_times = [7, 12, 18, 22]  # Breakfast, Lunch, Dinner, Snack
    meal_effects = np.zeros(len(dates))
    
    for i, date in enumerate(dates):
        hour = date.hour + date.minute/60
        for meal_time in meal_times:
            time_diff = abs(hour - meal_time)
            if time_diff < 4:  # 4-hour meal effect window
                # Gaussian meal response
                meal_effects[i] += meal_effect * np.exp(-(time_diff**2) / 2)
    
    # Long-term trends and seasonal variations (subtle)
    days_from_start = (dates - dates[0]).days
    seasonal = 5 * np.sin(2 * np.pi * days_from_start / 365)  # Annual cycle
    trend = np.cumsum(np.random.normal(0, trend_noise, len(dates)))  # Random walk
    
    # Random noise (correlated - glucose doesn't change abruptly)
    noise = np.random.normal(0, variability, len(dates))
    # Add some correlation to noise (simple moving average)
    for i in range(2, len(noise)):
        noise[i] = 0.7 * noise[i] + 0.3 * noise[i-1]
    
    # Combine all components
    glucose = base_glucose + circadian + meal_effects + seasonal + trend + noise
    glucose = np.array(glucose)  # Ensure it's a numpy array
    
    # Add occasional hypo/hyper events based on risk profile
    for i in range(len(glucose)):
        if np.random.random() < hypo_risk and glucose[i] < 90:
            glucose[i] = np.random.uniform(40, 65)  # Hypo event
        elif np.random.random() < hyper_risk and glucose[i] > 150:
            glucose[i] = np.random.uniform(200, 350)  # Hyper event
    
    # Ensure realistic bounds
    glucose = np.clip(glucose, 40, 400)
    
    return glucose

# Test the data processing就是在
if __name__ == "__main__":
    config = DataConfig()
    processor = CGMDataProcessor(config)
    
    # Load from actual raw files (no loader function needed)
    processed_data, metadata = processor.process_raw_data()
    
    if processed_data:
        print(f"\nSuccessfully processed {len(processed_data)} subjects")
        for subject_id, data in processed_data.items():
            print(f"  {subject_id}: {len(data)} samples, {data.shape[1]-2} features")
