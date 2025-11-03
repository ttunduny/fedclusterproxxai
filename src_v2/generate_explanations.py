"""
Generate counterfactual explanations for FedClusterProxXAI
and LIME/SHAP explanations for other strategies
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import json
from typing import Dict, List
import tensorflow as tf

from fl_benchmark import BenchmarkRunner, ExperimentConfig
from strategies import create_cgm_model


def generate_counterfactuals_fedclusterproxxai(strategy_results: Dict, model, client_data: pd.DataFrame):
    """Generate counterfactual explanations for FedClusterProxXAI"""
    print("\n" + "="*80)
    print("GENERATING COUNTERFACTUAL EXPLANATIONS FOR FedClusterProxXAI")
    print("="*80)
    
    x_val = client_data.drop('target', axis=1).values.astype(np.float32)
    y_val = client_data['target'].values.astype(np.float32)
    
    counterfactuals = []
    np.random.seed(42)
    
    # Sample 5 instances
    indices = np.random.choice(len(x_val), min(5, len(x_val)), replace=False)
    
    for idx in indices:
        original_input = x_val[idx:idx+1]
        original_pred = model(original_input, training=False).numpy()[0][0]
        
        # Calculate feature importance
        with tf.GradientTape() as tape:
            inputs = tf.Variable(original_input, dtype=tf.float32)
            tape.watch(inputs)
            pred = model(inputs)
        
        gradients = tape.gradient(pred, inputs)
        feature_importance = tf.abs(gradients[0]).numpy()
        
        top_indices = np.argsort(feature_importance)[-3:]
        
        cf_examples = []
        for feature_idx in top_indices:
            # Reduce feature
            cf_input_reduce = original_input.copy()
            cf_input_reduce[0, feature_idx] = max(0, original_input[0, feature_idx] - 0.2)
            cf_pred_reduce = model(tf.constant(cf_input_reduce, dtype=tf.float32), training=False).numpy()[0][0]
            
            # Increase feature
            cf_input_increase = original_input.copy()
            cf_input_increase[0, feature_idx] = min(1, original_input[0, feature_idx] + 0.2)
            cf_pred_increase = model(tf.constant(cf_input_increase, dtype=tf.float32), training=False).numpy()[0][0]
            
            cf_examples.append({
                'feature_idx': int(feature_idx),
                'feature_name': list(client_data.columns)[feature_idx],
                'feature_importance': float(feature_importance[feature_idx]),
                'original_value': float(original_input[0, feature_idx]),
                'scenario': 'reduce',
                'new_value': float(cf_input_reduce[0, feature_idx]),
                'original_prediction': float(original_pred),
                'counterfactual_prediction': float(cf_pred_reduce),
                'change': float(original_pred - cf_pred_reduce)
            })
            
            cf_examples.append({
                'feature_idx': int(feature_idx),
                'feature_name': list(client_data.columns)[feature_idx],
                'feature_importance': float(feature_importance[feature_idx]),
                'original_value': float(original_input[0, feature_idx]),
                'scenario': 'increase',
                'new_value': float(cf_input_increase[0, feature_idx]),
                'original_prediction': float(original_pred),
                'counterfactual_prediction': float(cf_pred_increase),
                'change': float(cf_pred_increase - original_pred)
            })
        
        counterfactuals.append({
            'instance': int(idx),
            'original_prediction': float(original_pred),
            'actual_value': float(y_val[idx]),
            'examples': cf_examples
        })
    
    return counterfactuals


def generate_lime_shap_explanations(strategy_name: str, model, client_data: pd.DataFrame):
    """Generate LIME and SHAP explanations"""
    print(f"\n" + "="*80)
    print(f"GENERATING LIME/SHAP EXPLANATIONS FOR {strategy_name}")
    print("="*80)
    
    try:
        import lime
        import lime.lime_tabular
        import shap
        
        x_val = client_data.drop('target', axis=1).values
        y_val = client_data['target'].values
        
        # LIME explanation
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            x_val,
            feature_names=list(client_data.columns[:-1]),
            mode='regression'
        )
        
        instance_idx = 0
        exp = explainer_lime.explain_instance(
            x_val[instance_idx],
            model.predict,
            num_features=10
        )
        
        lime_features = []
        for feature, importance in exp.as_list():
            lime_features.append({
                'feature': feature,
                'importance': float(importance)
            })
        
        # SHAP explanation
        def model_wrapper(x):
            return model(tf.constant(x, dtype=tf.float32), training=False).numpy()
        
        explainer_shap = shap.Explainer(model_wrapper, x_val[:100])
        shap_values = explainer_shap(x_val[:1])
        
        shap_features = []
        for i, importance in enumerate(shap_values.values[0]):
            shap_features.append({
                'feature_idx': int(i),
                'feature_name': list(client_data.columns)[i],
                'importance': float(importance)
            })
        
        return {
            'lime': lime_features,
            'shap': shap_features
        }
    
    except ImportError:
        print("LIME/SHAP not installed. Install with: pip install lime shap")
        return None


def main():
    """Main function to generate all explanations"""
    config = ExperimentConfig()
    
    # Load benchmark results
    results_path = 'experiments/cgm_fl_benchmark/fedclusterproxxai/results/results.json'
    with open(results_path, 'r') as f:
        fedcluster_results = json.load(f)
    
    # Load sample client data
    import pandas as pd
    client_data = pd.read_excel('data/processed/subjects/Subject_Subject3.xlsx', sheet_name='processed_data')
    
    # Create model
    input_dim = len(client_data.columns) - 1
    model = create_cgm_model(input_dim=input_dim, horizon=1)
    
    # Set model parameters (in real scenario, load from saved model)
    # For demo, use a fresh model
    
    # Generate counterfactuals for FedClusterProxXAI
    counterfactuals = generate_counterfactuals_fedclusterproxxai(
        fedcluster_results, model, client_data
    )
    
    # Save counterfactuals
    output_path = 'experiments/cgm_fl_benchmark/fedclusterproxxai/results/counterfactuals.json'
    with open(output_path, 'w') as f:
        json.dump({'counterfactuals': counterfactuals}, f, indent=2)
    
    print(f"\n✓ Counterfactuals saved to {output_path}")
    
    # Generate LIME/SHAP for other strategies
    for strategy in ['fedavg', 'fedprox', 'fedsgd']:
        explanations = generate_lime_shap_explanations(strategy, model, client_data)
        if explanations:
            output_path = f'experiments/cgm_fl_benchmark/{strategy}/results/explanations.json'
            with open(output_path, 'w') as f:
                json.dump(explanations, f, indent=2)
            print(f"✓ Explanations saved to {output_path}")


if __name__ == "__main__":
    main()


