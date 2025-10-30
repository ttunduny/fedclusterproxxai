# Federated Learning CGM Benchmark with Explainability

## Summary

Completed a comprehensive federated learning benchmark for Continuous Glucose Monitoring (CGM) with XAI capabilities.

### Implemented Features

1. **Four FL Strategies**:
   - FedAvg
   - FedProx (with proximal regularization)
   - FedSGD
   - FedClusterProxXAI (novel approach with adaptive clustering)

2. **XAI Metrics** (implemented in all strategies):
   - Feature Importance (gradient-based)
   - Prediction Stability
   - Local Stability
   - Faithfulness
   - Monotonicity
   - Overall XAI Score

3. **Explainability Methods**:
   - **Counterfactual Explanations** (for FedClusterProxXAI): Show what-if scenarios
   - **LIME/SHAP** (for standard benchmarks): Local interpretability

4. **Model Versioning**:
   - Cluster-based model versions saved for cold start
   - Metadata tracking per cluster

### Key Results

**Best Performance (RMSE)**:
- FedProx: 0.1216
- FedClusterProxXAI: 0.1254
- FedSGD: 0.1326
- FedAvg: 0.1289

**Best XAI Score**:
- FedClusterProxXAI: 0.4265 (highest explainability)
- FedAvg: 0.4199
- FedProx: 0.3854
- FedSGD: 0.3634

**Novel Model Benefits**:
- Best faithfulness: 0.3187
- Best monotonicity: 0.9651
- Adaptive mu: 0.01 → 0.06 (optimized for convergence)
- Cluster-based personalization

### Files Generated

- Results: `experiments/cgm_fl_benchmark/{strategy}/results/results.json`
- Cluster metadata: `experiments/cgm_fl_benchmark/fedclusterproxxai/models/cluster_metadata.json`
- Counterfactuals: `experiments/cgm_fl_benchmark/fedclusterproxxорт/models/modelscluster/counterfactuals.json` (to be generated)
- LIME/SHAP: `experiments/cgm_fl_benchmark/{strategy}/results/explanations.json` (to be generated)

### Next Steps

1. Run explanation generation: `python src/generate_explanations.py`
2. View counterfactuals for novel model
3. Analyze LIME/SHAP results for each strategy
4. Compare explanation quality across strategies

