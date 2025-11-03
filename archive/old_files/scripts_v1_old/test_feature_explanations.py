#!/usr/bin/env python3
"""
Test script to visualize feature importance and XAI metrics for FedClusterProxXAI
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from strategies import BaseCGMClient, create_cgm_model
from fl_benchmark import SubjectDataLoader, BenchmarkRunner, ExperimentConfig


def load_trained_model_and_data(client_id: str, strategy_key: str = "fedclusterproxxai", quick_train: bool = True):
    """Load a client model and its data for visualization
    
    Args:
        client_id: Client/subject ID
        strategy_key: Strategy name
        quick_train: If True, train model for a few epochs to get meaningful weights
    """
    
    # Load processed data
    subject_loader = SubjectDataLoader('data/processed')
    subjects_data = subject_loader.load_all_subjects()
    
    if client_id not in subjects_data:
        raise ValueError(f"Client {client_id} not found. Available: {list(subjects_data.keys())[:5]}")
    
    data = subjects_data[client_id].copy()
    
    # Remove subject_id if present
    if 'subject_id' in data.columns:
        data = data.drop('subject_id', axis=1)
    
    # Split data
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Get feature names (exclude target)
    feature_names = [col for col in train_data.columns if col != 'target']
    
    # Create model with same architecture
    input_dim = len(feature_names)
    model_id = "fedcluster" if strategy_key == "fedclusterproxxai" else "base"
    model = create_cgm_model(input_dim=input_dim, horizon=1, model_id=model_id)
    
    # Create client
    client = BaseCGMClient(
        client_id=client_id,
        train_data=train_data,
        val_data=val_data,
        model=model,
        strategy_type=strategy_key
    )
    
    # Quick training to get meaningful weights (optional)
    if quick_train:
        print(f"\nQuick training model for {client_id} (5 epochs for demo)...")
        client.model.fit(
            client.x_train, client.y_train,
            epochs=5,
            batch_size=32,
            verbose=1,
            validation_data=(client.x_val[:100], client.y_val[:100])
        )
        print("✓ Model trained")
    else:
        # Use initialized parameters
        global_params = client.get_parameters()
        client.set_parameters(global_params)
    
    return client, feature_names, val_data


def visualize_feature_importance(client: BaseCGMClient, feature_names: list, num_samples: int = 10):
    """Calculate and visualize feature importance"""
    
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Sample validation data
    x_val = client.x_val[:num_samples]
    
    # Calculate feature importance using gradients
    feature_importance_per_sample = []
    
    for i in range(num_samples):
        sample = x_val[i:i+1]
        
        with tf.GradientTape() as tape:
            inputs = tf.Variable(sample, dtype=tf.float32)
            tape.watch(inputs)
            prediction = client.model(inputs)
        
        gradients = tape.gradient(prediction, inputs)
        importance = tf.abs(gradients[0]).numpy()
        feature_importance_per_sample.append(importance)
    
    # Average across samples
    avg_importance = np.mean(feature_importance_per_sample, axis=0)
    std_importance = np.std(feature_importance_per_sample, axis=0)
    
    # Sort by importance
    sorted_indices = np.argsort(avg_importance)[::-1]
    
    print(f"\nTop 15 Most Important Features (averaged over {num_samples} samples):\n")
    print(f"{'Rank':<6} {'Feature Name':<35} {'Importance':<15} {'Std Dev':<15}")
    print("-" * 80)
    
    for rank, idx in enumerate(sorted_indices[:15], 1):
        feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
        importance = avg_importance[idx]
        std = std_importance[idx]
        print(f"{rank:<6} {feature_name:<35} {importance:<15.6f} {std:<15.6f}")
    
    # Create visualization
    top_n = 15
    top_indices = sorted_indices[:top_n]
    top_features = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" 
                    for i in top_indices]
    top_importance = [avg_importance[i] for i in top_indices]
    top_std = [std_importance[i] for i in top_indices]
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Bar plot
    y_pos = np.arange(len(top_features))
    bars = ax1.barh(y_pos, top_importance, xerr=top_std, capsize=3, alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_features)
    ax1.invert_yaxis()
    ax1.set_xlabel('Average Feature Importance (Gradient Magnitude)', fontsize=12)
    ax1.set_title(f'Top {top_n} Feature Importance\n(FedClusterProxXAI)', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Color bars by importance
    max_imp = max(top_importance)
    for i, (bar, imp) in enumerate(zip(bars, top_importance)):
        intensity = imp / max_imp
        bar.set_color(plt.cm.viridis(intensity))
    
    # Category breakdown
    categories = {
        'Temporal': ['hour', 'day', 'weekend', 'sin', 'cos'],
        'Recent Values': ['prev_', 'roc_'],
        'Statistics': ['mean_', 'std_', 'min_', 'max_'],
        'Medical': ['hypo', 'hyper', 'target_range', 'tight_range'],
        'Patterns': ['deviation', 'variability', 'trend', 'pattern', 'consistency']
    }
    
    category_importance = {cat: [] for cat in categories.keys()}
    
    for idx, feature in enumerate(feature_names):
        importance_val = avg_importance[idx]
        for cat, keywords in categories.items():
            if any(keyword in feature.lower() for keyword in keywords):
                category_importance[cat].append(importance_val)
                break
    
    category_means = {cat: np.mean(vals) if vals else 0 for cat, vals in category_importance.items()}
    
    # Pie chart
    categories_list = list(category_means.keys())
    importance_list = list(category_means.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories_list)))
    
    ax2.pie(importance_list, labels=categories_list, autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Feature Importance by Category', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('experiments/cgm_fl_benchmark/feature_visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_dir / 'feature_importance.png'}")
    
    plt.show()
    
    return avg_importance, feature_names


def visualize_counterfactuals(client: BaseCGMClient, feature_names: list, num_examples: int = 3):
    """Generate and visualize counterfactual explanations"""
    
    print("\n" + "="*80)
    print("COUNTERFACTUAL EXPLANATIONS")
    print("="*80)
    
    # Generate counterfactuals
    cf_results = client.generate_counterfactuals(sample_size=num_examples)
    
    for i, cf_instance in enumerate(cf_results['counterfactuals'][:num_examples]):
        print(f"\n--- Instance {i+1} ---")
        instance_idx = cf_instance['instance_idx']
        original_pred = cf_instance['original_prediction']
        
        print(f"Instance Index: {instance_idx}")
        print(f"Original Prediction: {original_pred:.4f} (glucose in 60 min)")
        print(f"\nCounterfactual Scenarios:")
        print(f"{'Feature':<35} {'Change':<15} {'Original':<12} {'New Value':<12} {'New Pred':<12} {'Δ Pred':<12}")
        print("-" * 100)
        
        for j, cf_ex in enumerate(cf_instance['counterfactual_examples'][:6]):  # Top 6
            feature_idx = cf_ex['feature_idx']
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature_{feature_idx}"
            change_type = cf_ex['change_type']
            original_val = cf_ex['original_value']
            new_val = cf_ex['new_value']
            original_pred_cf = cf_ex['original_prediction']
            cf_pred = cf_ex['counterfactual_prediction']
            pred_change = cf_ex['prediction_change']
            
            change_str = f"{change_type} by {abs(new_val - original_val):.3f}"
            
            print(f"{feature_name:<35} {change_str:<15} {original_val:<12.4f} {new_val:<12.4f} "
                  f"{cf_pred:<12.4f} {pred_change:<12.4f}")
    
    return cf_results


def show_xai_metrics(client: BaseCGMClient, global_params):
    """Calculate and display all XAI metrics"""
    
    print("\n" + "="*80)
    print("XAI METRICS SUMMARY")
    print("="*80)
    
    # Set model parameters
    client.set_parameters(global_params)
    
    # Evaluate
    predictions = client.model(client.x_val[:100], training=False)
    
    # Calculate metrics
    xai_metrics = client._calculate_xai_metrics(predictions)
    
    print("\nExplainability Metrics:")
    print(f"  Feature Importance Mean: {xai_metrics['feature_importance_mean']:.6f}")
    print(f"  Feature Importance Max:  {xai_metrics['feature_importance_max']:.6f}")
    print(f"  Feature Importance Std: {xai_metrics['feature_importance_std']:.6f}")
    print(f"\nStability Metrics:")
    print(f"  Prediction Stability:    {xai_metrics['prediction_stability']:.6f}")
    print(f"  Local Stability:        {xai_metrics['local_stability']:.6f}")
    print(f"  Prediction Consistency: {xai_metrics['prediction_consistency']:.6f}")
    print(f"\nFaithfulness & Monotonicity:")
    print(f"  Faithfulness:           {xai_metrics['faithfulness']:.6f}")
    print(f"  Monotonicity:           {xai_metrics['monotonicity']:.6f}")
    print(f"\nNormalized Scores:")
    print(f"  Importance Score:       {xai_metrics['importance_score']:.6f}")
    print(f"  Stability Score:        {xai_metrics['stability_score']:.6f}")
    print(f"  Faithfulness Score:     {xai_metrics['faithfulness_score']:.6f}")
    print(f"  Overall XAI Score:      {xai_metrics['xai_score']:.6f}")
    
    return xai_metrics


def main():
    """Main function to test feature explanations"""
    
    print("="*80)
    print("FEDCLUSTERPROXXAI - FEATURE EXPLANATION TEST")
    print("="*80)
    
    # Load results to see available clients
    results_path = Path('experiments/cgm_fl_benchmark/fedclusterproxxai/results/results.json')
    
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        available_clients = results.get('subjects_used', [])
        print(f"\nAvailable clients from results: {available_clients[:10]}")
        
        # Use first client
        test_client_id = available_clients[0] if available_clients else None
    else:
        # Fallback: load from processed data
        subject_loader = SubjectDataLoader('data/processed')
        subjects_data = subject_loader.load_all_subjects()
        test_client_id = list(subjects_data.keys())[0]
    
    if not test_client_id:
        print("ERROR: No clients available!")
        return
    
    print(f"\nTesting with client: {test_client_id}")
    
    try:
        # Load model and data (quick_train=True trains for 5 epochs to get meaningful weights)
        client, feature_names, val_data = load_trained_model_and_data(
            test_client_id, "fedclusterproxxai", quick_train=True
        )
        
        # Get current model parameters
        global_params = client.get_parameters()
        
        # 1. Show XAI metrics
        xai_metrics = show_xai_metrics(client, global_params)
        
        # 2. Visualize feature importance
        avg_importance, feature_names_verified = visualize_feature_importance(client, feature_names, num_samples=50)
        
        # 3. Show counterfactuals
        counterfactuals = visualize_counterfactuals(client, feature_names, num_examples=3)
        
        print("\n" + "="*80)
        print("✅ FEATURE EXPLANATION TEST COMPLETE!")
        print("="*80)
        print(f"\nVisualizations saved in: experiments/cgm_fl_benchmark/feature_visualizations/")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

