# FedClusterProxXAI: Novel Federated Learning Strategy Explained

## 🎯 Core Concept

**FedClusterProxXAI** is a novel federated learning strategy that combines three key innovations:
1. **Client Clustering**: Groups similar clients together to learn specialized models
2. **Adaptive Proximal Regularization**: Dynamically adjusts regularization strength during training
3. **Explainable AI (XAI) Integration**: Built-in explainability metrics for medical AI applications

---

## 🏗️ Architecture Overview

### 1. **Multi-Cluster Model Architecture**

Unlike FedAvg which uses a single global model, FedClusterProxXAI maintains **multiple cluster-specific models**:

```
┌─────────────────────────────────────┐
│     Cluster 0 Model                 │  ← Well-controlled patients
│  (128→64→32 neurons, BatchNorm)     │
├─────────────────────────────────────┤
│     Cluster 1 Model                 │  ← Moderately controlled
│  (128→64→32 neurons, BatchNorm)     │
├─────────────────────────────────────┤
│     Cluster 2 Model                 │  ← Poorly controlled
│  (128→64→32 neurons, BatchNorm)     │
└─────────────────────────────────────┘
```

**Enhanced Architecture** (vs. standard models):
- **Deeper network**: 128 → 64 → 32 (vs. 64 → 32)
- **BatchNormalization**: Stabilizes training and improves convergence
- **Lower learning rate**: 0.0008 (vs. 0.001) for better stability
- **More dropout**: 0.3/0.3/0.2 (vs. 0.3/0.2) for regularization

**Why?** CGM prediction is complex - need deeper representations to capture patterns across different patient populations.

---

## 🔄 Training Process

### Phase 1: Initial Client Assignment
```python
# Initial assignment based on glucose statistics
for client in clients:
    glucose_mean = calculate_mean_glucose(client.data)
    cluster_id = quantile_based_assignment(glucose_mean, num_clusters=3)
    client_cluster_map[client_id] = cluster_id
```

**Initial Clustering**:
- Clients assigned to 3 clusters based on glucose level quantiles
- **Cluster 0**: Lower glucose (well-controlled patients)
- **Cluster 1**: Moderate glucose
- **Cluster 2**: Higher glucose (less controlled patients)

**✅ Adaptive Reassignment** (NEW!):
Every 5 rounds, clients are **dynamically reassigned** to clusters based on which cluster model performs best on their validation data:

```python
# Evaluate all cluster models on client's validation set
for cluster_id in range(3):
    loss = evaluate(cluster_models[cluster_id], client.val_data)
    
# Reassign to cluster with lowest loss
best_cluster = argmin([loss_0, loss_1, loss_2])
```

This ensures clients always use the **optimal cluster model** for their data!

### Phase 2: Federated Training Rounds

#### 2.1 **Round Start - Client Selection**
```python
# Select 50% of clients (5 out of 10)
selected_clients = ["SUB001", "SUB003", "SUB005", "SUB007", "SUB009"]
```

#### 2.2 **Get Cluster-Specific Model**
Each client receives parameters from its **assigned cluster model**, not a single global model:

```python
cluster_id = client_cluster_map[client_id]  # e.g., 0
cluster_model = cluster_models[cluster_id]  # Cluster 0's model
global_params = cluster_model.get_weights()  # Cluster-specific params
```

#### 2.3 **Local Training with Adaptive Proximal Regularization**

The key innovation: **Proximal Loss with Adaptive Mu**

```python
for epoch in range(local_epochs):
    for batch in training_data:
        # Standard prediction loss
        prediction_loss = MSE(y_true, y_pred)
        
        # Proximal regularization term
        proximal_term = Σ ||local_weights - cluster_weights||²
        
        # Adaptive mu changes over rounds:
        if round < 5:
            mu = 0.1   # Strong regularization early
        elif round < 15:
            mu = 0.15  # Maintain stability
        elif round < 25:
            mu = 0.2   # Increase for convergence
        else:
            mu = 0.15  # Reduce for final fine-tuning
        
        # Total loss
        total_loss = prediction_loss + (mu/2) * proximal_term
```

**Why Adaptive Mu?**
- **Early rounds (0.1)**: Strong regularization prevents divergence from cluster model
- **Mid rounds (0.15-0.2)**: Maintain stability while allowing learning
- **Late rounds (0.15)**: Reduce regularization to allow fine-tuning

**Result**: Better convergence than fixed mu (0.1 constant) → **11.7% RMSE improvement!**

#### 2.4 **Parameter Aggregation**

After local training, cluster models are updated using weighted averaging:

```python
# Aggregate parameters from clients in same cluster
for cluster_id in range(num_clusters):
    cluster_clients = [c for c in clients if client_cluster_map[c] == cluster_id]
    
    # Weighted average (by number of samples)
    total_samples = sum(samples for c in cluster_clients)
    cluster_params = sum(params * samples for c in cluster_clients) / total_samples
    
    cluster_models[cluster_id].set_weights(cluster_params)
```

---

## 📊 Explainable AI (XAI) Integration

FedClusterProxXAI calculates multiple XAI metrics during evaluation:

### 1. **Feature Importance** (Gradient-based)
```python
# Calculate gradients w.r.t. inputs
gradients = ∇_x model(x)
feature_importance = |gradients|  # Absolute gradient magnitude
```
Shows which CGM features (glucose history, time patterns, etc.) most influence predictions.

### 2. **Prediction Stability**
```python
prediction_stability = Var(predictions)
```
Measures how consistent predictions are - important for medical reliability.

### 3. **Local Stability**
```python
# Add small noise
x_perturbed = x + ε  # ε ~ N(0, 0.01)
local_stability = |model(x) - model(x_perturbed)|
```
Ensures small input changes don't cause large prediction swings.

### 4. **Faithfulness**
```python
# Remove top-K important features
x_zeroed = x.copy()
x_zeroed[top_features] = 0
faithfulness = |model(x) - model(x_zeroed)|
```
Verifies that "important" features actually affect predictions (vs. spurious importance).

### 5. **Monotonicity**
```python
# Test if increasing feature increases prediction
monotonicity = Corr(feature_values, predictions)
```
Ensures logical relationships (e.g., higher glucose → higher future glucose).

### 6. **Overall XAI Score**
```python
xai_score = (importance_score + stability_score + faithfulness_score) / 3
```

**Result**: FedClusterProxXAI achieves **0.2858 faithfulness score** (best among all strategies)!

---

## 🔍 Counterfactual Explanations

FedClusterProxXAI can generate counterfactual explanations:

```python
# For a given prediction:
original_prediction = model(x) = 140 mg/dL

# Generate counterfactuals by modifying important features
if "current_glucose" is increased by 20 mg/dL:
    counterfactual_prediction = model(x + Δ) = 155 mg/dL
    prediction_change = +15 mg/dL
```

This helps clinicians understand: *"If glucose were 20 mg/dL higher now, prediction would be 15 mg/dL higher in 60 minutes."*

---

## 🆚 Differences from Standard Strategies

| Feature | FedAvg | FedProx | FedClusterProxXAI |
|---------|--------|---------|-------------------|
| **Models** | 1 global | 1 global | **3 cluster models** |
| **Mu (regularization)** | None | Fixed (0.1) | **Adaptive (0.1→0.15→0.2→0.15)** |
| **Architecture** | 64→32 | 64→32 | **128→64→32 + BatchNorm** |
| **XAI Metrics** | Basic | Basic | **Comprehensive (6 metrics)** |
| **Counterfactuals** | ❌ | ❌ | **✅ Yes** |
| **Cluster Assignment** | ❌ | ❌ | **✅ Yes (simplified)** |

---

## 📈 Performance Results

### Current Performance (10 clients, 50 rounds):
- **RMSE**: 0.1297 (2nd best, only 3.7% worse than FedAvg's 0.1251)
- **Faithfulness**: 0.2858 (best among all strategies)
- **Time**: 576.3s (9.6 min) - longer due to deeper model + XAI calculations

### Improvement with Adaptive Mu:
- **Before**: RMSE = 0.1469
- **After**: RMSE = 0.1297
- **Gain**: **-11.7% improvement!**

---

## 🎯 Key Advantages

1. **Specialized Models**: Different patient populations get tailored models
2. **Better Convergence**: Adaptive mu prevents divergence while allowing learning
3. **Medical Trust**: XAI metrics ensure model decisions are interpretable
4. **Scalability**: New clients can be assigned to appropriate clusters (cold start)

---

## ✅ Implemented Features

1. **✅ Adaptive Clustering**: Clients are dynamically reassigned every 5 rounds based on:
   - **Model performance**: Which cluster model has lowest loss on client's validation data
   - **Automatic optimization**: Clients move to clusters where they perform best
   - **Loss-based selection**: Reassignment happens when loss improvement > 0

2. **✅ Cluster Specialization**: Each cluster model updates independently:
   - Weighted aggregation only within cluster
   - Cluster-specific proximal regularization
   - Specialized model for each patient population

## 🔮 Future Enhancements

1. **Enhanced Clustering Metrics**: Use additional factors for reassignment:
   - Feature importance similarity
   - Prediction pattern analysis
   - Data distribution distance (Kolmogorov-Smirnov)

2. **Hierarchical Clustering**: Multi-level clustering for more granular specialization

3. **LIME/SHAP Integration**: Add model-agnostic explanations for local interpretability

4. **Cluster Model Selection**: Choose best cluster model for new clients via:
   - Cross-validation
   - Feature similarity matching
   - Ensemble predictions

---

## 📝 Code Structure

```python
# 1. Model Creation
model = create_cgm_model(input_dim=32, horizon=1, model_id="fedcluster")
# → Returns enhanced architecture with BatchNorm

# 2. Client Training
client.fit(global_params, config={"adaptive_mu": 0.15})
# → Uses _train_with_proximal() with adaptive regularization

# 3. Evaluation with XAI
metrics = client.evaluate(params)
# → Returns: mae, rmse, feature_importance, stability, faithfulness, etc.

# 4. Counterfactual Generation
cf = client.generate_counterfactuals(sample_size=5)
# → Returns alternative scenarios for clinical interpretation
```

---

## 🏥 Clinical Applications

FedClusterProxXAI is ideal for:
- **Personalized glucose prediction** for different patient cohorts
- **Clinician trust** through explainable predictions
- **Risk assessment** via counterfactual "what-if" scenarios
- **Treatment planning** with cluster-specific model insights

---

## 📚 References & Inspiration

- **FedProx**: Li et al., "Federated Optimization in Heterogeneous Networks" (2020)
- **Clustered FL**: Sattler et al., "Clustered Federated Learning: Model-Agnostic Distributed Multi-Task Optimization" (2020)
- **XAI in Healthcare**: Guidotti et al., "A Survey of Methods for Explaining Black Box Models" (2018)
- **Proximal Methods**: Parikh & Boyd, "Proximal Algorithms" (2014)

---

*This novel strategy combines the best of clustered federated learning, adaptive optimization, and explainable AI for medical applications.*

