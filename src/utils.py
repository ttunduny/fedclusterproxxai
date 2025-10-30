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
