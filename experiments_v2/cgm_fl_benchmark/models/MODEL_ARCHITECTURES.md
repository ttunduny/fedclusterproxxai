# Model Architectures - V2
**Feature-Optimized Architectures**

---

## Architecture Overview

V2 experiments use optimized architectures for different feature sets:

### Core 24 Architecture (Primary)

**LSTM Architecture:**
```
Input: 24 features
LSTM Layer 1: 128 units (capture temporal patterns)
  Dropout: 0.3
LSTM Layer 2: 64 units (capture short-term dependencies)
  Dropout: 0.3
Dense Layer: 32 units
Output: 1 (predicted glucose)
```

**Rationale:**
- Reduced input from 42 to 24 features
- Optimized for temporal patterns (high consistency of temporal features)
- Dropout prevents overfitting

### Top 10 Architecture

**LSTM Architecture:**
```
Input: 10 features
LSTM Layer 1: 64 units
  Dropout: 0.3
Dense Layer: 32 units
  Dropout: 0.3
Output: 1
```

**Rationale:**
- Smaller architecture for minimal feature set
- Suitable for computational efficiency

### Temporal-Focused Architecture

**LSTM Architecture:**
```
Input: 10 features (temporal-focused)
LSTM Layer 1: 96 units (emphasize temporal)
  Dropout: 0.3
LSTM Layer 2: 48 units
  Dropout: 0.3
Output: 1
```

**Rationale:**
- Optimized for temporal pattern capture
- Balanced size for 10 features

### Baseline Architecture (42 features)

**LSTM Architecture:**
```
Input: 42 features (original)
LSTM Layer 1: 128 units
  Dropout: 0.3
LSTM Layer 2: 64 units
  Dropout: 0.3
Dense Layer: 32 units
Output: 1
```

**Rationale:**
- Baseline for comparison
- Original architecture with all features

---

## Transformer-Enhanced (Experimental)

**Architecture:**
```
Input: 24 features
Positional Encoding: Hour-based
Transformer Encoder: 2 layers, 4 heads, 64 d_model
  Dropout: 0.3
Dense Layers: [64, 32]
Output: 1
```

**Rationale:**
- Attention mechanism for feature interactions
- Better captures long-range dependencies

---

## Training Parameters

**Common Parameters:**
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Batch Size: 32
- Epochs: 5 (local training)
- Regularization: L2 (0.0001), Dropout (0.3)
- Early Stopping: Patience=10, min_delta=0.001

---

*See configs/experiment_config_v2.json for detailed configurations*

