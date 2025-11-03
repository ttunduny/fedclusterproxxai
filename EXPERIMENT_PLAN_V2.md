# Experiment Plan V2: Feature-Optimized CGM Prediction
**Based on Comprehensive Feature Analysis Results**

**Generated:** 2025-11-02  
**Status:** Ready for Implementation  
**Previous Analysis:** Feature analysis completed on 47 subjects

---

## Executive Summary

This experiment plan incorporates findings from comprehensive feature analysis conducted on 47 subjects, identifying 24 consistently important features (≥70% consistency) for CGM prediction. The plan focuses on data-driven feature selection, model optimization, and validation strategies informed by statistical analysis.

### Key Findings from Feature Analysis

- **24 Consistently Important Features** identified across 47 subjects
- **Average Outlier Rate:** 3.80% (normal, <5% threshold)
- **Top Predictive Features:**
  - `current_glucose` (0.6974 correlation, 97.9% consistency)
  - `deviation_from_recent_trend` (0.5455 correlation, 93.6% consistency)
  - `is_hyperglycemic` (0.5481 correlation, 93.6% consistency)
  - `roc_60min` (0.3768 correlation, 100% consistency)
  - `hourly_baseline` (0.3128 correlation, 97.9% consistency)

---

## Phase 1: Data Preparation & Feature Engineering

### 1.1 Core Feature Set Implementation

**Primary Feature Set (24 Consistently Important Features):**

#### Temporal Features (4 features)
- `hour` (100% consistency) - Raw hour of day
- `hourly_baseline` (97.9% consistency) - Circadian baseline
- `hour_sin` (87.2% consistency) - Sine transformation
- `hour_cos` (80.9% consistency) - Cosine transformation

#### Recent Values (3 features)
- `roc_60min` (100% consistency) - Rate of change over 60 min
- `current_glucose` (97.9% consistency) - Current glucose value
- `roc_30min` (93.6% consistency) - Rate of change over 30 min

#### Rolling Statistics (14 features)
- `min_6h`, `max_6h`, `mean_24h`, `max_12h`, `std_3h` (≥95.7% consistency)
- `mean_12h`, `std_12h`, `min_12h` (≥91.5% consistency)
- `min_24h`, `max_3h`, `std_6h` (≥85.1% consistency)
- `max_24h`, `std_1h`, `std_24h` (≥76.6% consistency)

#### Medical Indicators (2 features)
- `in_tight_range` (97.9% consistency)
- `is_hyperglycemic` (93.6% consistency)

#### Pattern Analysis (1 feature)
- `deviation_from_recent_trend` (93.6% consistency)

**Action Items:**
1. ✅ Update feature engineering pipeline to prioritize these 24 features
2. ✅ Create feature selection function that filters to core features
3. ✅ Remove or deprioritize redundant features (30 identified)
4. ✅ Implement feature importance-based weighting

### 1.2 Feature Redundancy Handling

**Redundant Features to Remove/Combine:**
- Remove: `deviation_from_daily_mean` (highly correlated with `current_glucose`, r=0.9846)
- Remove: `hour_sin` (redundant with `mean_12h` in some subjects)
- Consider: Combining rolling statistics from same time windows (e.g., `mean_3h` + `std_3h` + `min_3h` + `max_3h`)

**Implementation:**
```python
# Feature selection based on analysis
CORE_FEATURES = [
    'hour', 'hour_cos', 'hourly_baseline', 'hour_sin',
    'roc_60min', 'current_glucose', 'roc_30min',
    'min_6h', 'max_6h', 'mean_24h', 'max_12h', 'std_3h',
    'mean_12h', 'std_12h', 'min_12h',
    'min_24h', 'max_3h', 'std_6h', 'max_24h', 'std_1h', 'std_24h',
    'in_tight_range', 'is_hyperglycemic',
    'deviation_from_recent_trend'
]

# Redundant features to remove
REDUNDANT_FEATURES = [
    'deviation_from_daily_mean',
    'in_target_range',  # redundant with is_hyperglycemic
    # Additional redundant features as identified per subject
]
```

### 1.3 Outlier Handling Strategy

**Findings:**
- Average outlier rate: 3.80% (normal, acceptable)
- Range: 3.33% - 4.87% across subjects
- All subjects show normal outlier rates (<5%)

**Strategy:**
1. **Keep outliers** - They represent clinically significant events (hypo/hyperglycemia)
2. **No aggressive removal** - Outliers are informative for prediction
3. **Robust scaling** - Use robust scalers (median/IQR) instead of mean/std
4. **Model regularization** - Use dropout and L2 regularization to handle outliers gracefully

### 1.4 Data Quality Assurance

**Failed Subjects (10):** Subject13, Subject14, Subject15, Subject18, Subject21, Subject25, Subject32, Subject44, Subject5, Subject7

**Action:**
- Exclude these subjects from training/evaluation
- Document exclusion reasons in experiment logs
- Use remaining 47 subjects for experiments

---

## Phase 2: Model Architecture Updates

### 2.1 Input Feature Configuration

**Current Configuration:**
- Input features: 20 (outdated)
- Features used: Mixed set without feature analysis

**New Configuration:**
- **Primary Model:** 24 core features (consistently important)
- **Extended Model:** 24 core + selected subject-specific features (up to 30 total)
- **Baseline Model:** Top 10 features only (for comparison)

**Feature Sets:**
```json
{
  "feature_sets": {
    "core_24": [
      "hour", "hour_cos", "hourly_baseline", "hour_sin",
      "roc_60min", "current_glucose", "roc_30min",
      "min_6h", "max_6h", "mean_24h", "max_12h", "std_3h",
      "mean_12h", "std_12h", "min_12h",
      "min_24h", "max_3h", "std_6h", "max_24h", "std_1h", "std_24h",
      "in_tight_range", "is_hyperglycemic",
      "deviation_from_recent_trend"
    ],
    "top_10": [
      "current_glucose", "hour", "roc_60min", "hourly_baseline",
      "in_tight_range", "deviation_from_recent_trend", "is_hyperglycemic",
      "max_6h", "min_6h", "roc_30min"
    ],
    "temporal_focused": [
      "hour", "hour_cos", "hourly_baseline", "hour_sin",
      "current_glucose", "roc_60min", "roc_30min",
      "deviation_from_recent_trend", "in_tight_range", "is_hyperglycemic"
    ]
  }
}
```

### 2.2 Model Architecture Modifications

**Insights from Feature Analysis:**
1. **Temporal patterns are critical** - Need models that capture circadian rhythms
2. **Short-term dependencies** - Recent values (15-60 min) highly predictive
3. **Rate of change matters** - Velocity features (`roc_30min`, `roc_60min`) are universally important

**Architecture Recommendations:**

#### LSTM Architecture (Primary)
```
Input Layer: 24 features
LSTM Layer 1: 128 units (capture temporal patterns)
Dropout: 0.3
LSTM Layer 2: 64 units (capture short-term dependencies)
Dropout: 0.3
Dense Layer: 32 units
Output Layer: 1 (predicted glucose)
```

**Rationale:**
- LSTM well-suited for temporal sequences
- Two layers capture both long-term (circadian) and short-term patterns
- Dropout prevents overfitting with reduced feature set

#### Transformer-Enhanced Architecture (Experimental)
```
Input Layer: 24 features
Positional Encoding: Hour-based encoding
Transformer Encoder: 2 layers, 4 heads, 64 d_model
Dense Layers: [64, 32]
Output Layer: 1
```

**Rationale:**
- Attention mechanism can focus on important temporal windows
- Better captures interactions between features

### 2.3 Feature Interaction Engineering

Based on findings:
- Temporal × Glucose interactions (hour × current_glucose)
- Rate of change × Current state (roc_60min × current_glucose)
- Medical indicators × Trends (is_hyperglycemic × deviation_from_recent_trend)

**Implementation:**
```python
def create_interaction_features(df):
    """Create feature interactions based on analysis findings"""
    # Temporal × Glucose
    df['hour_x_glucose'] = df['hour'] * df['current_glucose']
    df['hour_cos_x_glucose'] = df['hour_cos'] * df['current_glucose']
    
    # Rate of change × Current state
    df['roc60_x_current'] = df['roc_60min'] * df['current_glucose']
    
    # Medical × Trends
    df['hyper_x_deviation'] = df['is_hyperglycemic'] * df['deviation_from_recent_trend']
    
    return df
```

---

## Phase 3: Training Strategy Updates

### 3.1 Federated Learning Configuration

**Updated Configuration:**
```json
{
  "federated_settings": {
    "num_clients": 47,
    "clients_per_round": 10,
    "min_fit_clients": 8,
    "min_evaluate_clients": 8,
    "client_selection": "random",
    "stratify_by_pattern": false
  },
  "training_parameters": {
    "num_rounds": 60,
    "local_epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss_function": "mse",
    "early_stopping_patience": 10,
    "min_delta": 0.001
  }
}
```

**Rationale:**
- 47 available subjects (excluding 10 failed subjects)
- Higher rounds (60) to account for reduced feature complexity
- Maintain local epochs at 5 to prevent overfitting

### 3.2 Cross-Validation Strategy

**Recommended Approach:**
1. **Subject-level cross-validation** - Leave-one-subject-out (LOO) or K-fold
2. **Temporal validation** - Train on past, validate on future (respect time order)
3. **Stratified sampling** - Ensure balance of glucose patterns across folds

**Implementation:**
```python
# Subject-level K-fold (K=5)
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
# Group by subject_id to ensure no data leakage
```

### 3.3 Handling Data Heterogeneity

**Findings:**
- Feature importance consistent across subjects (≥70% consistency)
- Some features show subject-specific patterns

**Strategy:**
1. **Personalized models** - FedClusterProxXAI strategy groups similar subjects
2. **Feature importance weighting** - Weight features by consistency score
3. **Adaptive feature selection** - Allow cluster-specific feature selection

---

## Phase 4: Evaluation & Metrics

### 4.1 Primary Metrics

Based on feature analysis insights:

1. **RMSE** - Overall prediction accuracy
2. **MAE** - Average prediction error
3. **Time-in-Range (TIR)** - Clinical relevance (70-180 mg/dL)
4. **Hypoglycemia Detection** - Sensitivity for glucose < 70 mg/dL
5. **Hyperglycemia Detection** - Sensitivity for glucose > 180 mg/dL

### 4.2 Feature Importance Validation

**Post-training Analysis:**
1. Compute SHAP values for trained models
2. Compare learned importance vs. statistical importance
3. Validate that core features receive high importance scores
4. Identify any discrepancies for investigation

### 4.3 Model Comparison Strategy

**Baseline Comparisons:**
1. **Full Feature Set (42 features)** vs. **Core 24 Features**
2. **Top 10 Features** vs. **Core 24 Features**
3. **Temporal-Focused (10 features)** vs. **Core 24 Features**
4. **With Redundant Features** vs. **Without Redundant Features**

**Expected Outcomes:**
- Core 24 features should match or exceed full feature set performance
- Reduced feature set should improve generalization
- Model training time should decrease

---

## Phase 5: Experiment Execution Plan

### 5.1 Experiment Phases

#### Phase 5.1.1: Baseline Establishment (Week 1)
- [ ] Implement core 24 feature selection
- [ ] Train baseline models with all feature sets (full, core24, top10)
- [ ] Establish performance baselines
- [ ] Document baseline metrics

#### Phase 5.1.2: Feature Set Optimization (Week 2)
- [ ] Test interaction features
- [ ] Evaluate redundancy removal impact
- [ ] Optimize feature subsets per cluster
- [ ] Compare temporal-focused vs. full feature sets

#### Phase 5.1.3: Model Architecture Tuning (Week 3)
- [ ] Optimize LSTM architecture (layers, units, dropout)
- [ ] Test Transformer-Enhanced architecture
- [ ] Hyperparameter tuning (learning rate, batch size, epochs)
- [ ] Regularization optimization

#### Phase 5.1.4: Federated Learning Optimization (Week 4)
- [ ] Test different FL strategies (FedAvg, FedProx, FedClusterProxXAI)
- [ ] Optimize client selection strategies
- [ ] Tune federated learning hyperparameters
- [ ] Evaluate convergence rates

#### Phase 5.1.5: Comprehensive Evaluation (Week 5)
- [ ] Cross-validation across all subjects
- [ ] Temporal validation
- [ ] Feature importance analysis (SHAP/LIME)
- [ ] Clinical metric evaluation
- [ ] Statistical significance testing

### 5.2 Experimental Conditions

**Experiment 1: Feature Set Comparison**
- Condition A: All 42 features (baseline)
- Condition B: Core 24 features (recommended)
- Condition C: Top 10 features (minimal)
- Condition D: Temporal-focused 10 features
- **Hypothesis:** Core 24 features will match or exceed all-features performance

**Experiment 2: Redundancy Impact**
- Condition A: With redundant features
- Condition B: Without redundant features
- **Hypothesis:** Removing redundant features improves generalization

**Experiment 3: Architecture Comparison**
- Condition A: LSTM (128, 64) with 24 features
- Condition B: Transformer-Enhanced with 24 features
- Condition C: LSTM (original architecture) with 24 features
- **Hypothesis:** Feature-optimized architectures will outperform original

**Experiment 4: FL Strategy Comparison**
- Condition A: FedAvg with core features
- Condition B: FedProx with core features
- Condition C: FedClusterProxXAI with core features
- **Hypothesis:** FedClusterProxXAI will benefit from feature-optimized input

---

## Phase 6: Expected Outcomes & Success Criteria

### 6.1 Success Metrics

**Primary Goals:**
1. **RMSE Reduction:** ≤ 15 mg/dL (vs. baseline)
2. **Feature Efficiency:** Match baseline performance with 24 features (vs. 42)
3. **Training Time:** Reduce by ≥20% with smaller feature set
4. **Generalization:** Improved performance on unseen subjects

**Secondary Goals:**
1. **TIR Prediction:** >80% accuracy for time-in-range prediction
2. **Hypo/Hyper Detection:** >85% sensitivity for extreme events
3. **Model Interpretability:** Clear feature importance alignment with analysis

### 6.2 Key Performance Indicators (KPIs)

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| RMSE (mg/dL) | TBD | ≤ 20 | ≤ 15 |
| MAE (mg/dL) | TBD | ≤ 15 | ≤ 12 |
| TIR Accuracy | TBD | ≥ 80% | ≥ 85% |
| Training Time | TBD | -20% | -30% |
| Feature Count | 42 | 24 | 20 |

### 6.3 Deliverables

1. **Feature-Optimized Models:**
   - Trained models with core 24 features
   - Comparison models with different feature sets
   - Performance evaluation reports

2. **Feature Analysis Report:**
   - Feature importance validation (SHAP/LIME)
   - Learned vs. statistical importance comparison
   - Feature redundancy impact analysis

3. **Comprehensive Results:**
   - Performance metrics across all experiments
   - Statistical significance tests
   - Visualization of results

4. **Recommendations:**
   - Optimal feature set for production
   - Model architecture recommendations
   - Deployment considerations

---

## Phase 7: Risk Mitigation & Contingencies

### 7.1 Potential Risks

**Risk 1: Reduced Performance with Fewer Features**
- **Probability:** Medium
- **Impact:** High
- **Mitigation:** Maintain baseline with all features, compare systematically
- **Contingency:** Hybrid approach (core + selective additional features)

**Risk 2: Subject-Specific Feature Needs**
- **Probability:** Low
- **Impact:** Medium
- **Mitigation:** Use FedClusterProxXAI for personalized feature selection
- **Contingency:** Cluster-specific feature sets

**Risk 3: Outlier Impact on Model**
- **Probability:** Low
- **Impact:** Medium
- **Mitigation:** Robust scaling, regularization, outlier-aware loss functions
- **Contingency:** Keep outliers (they're clinically significant)

### 7.2 Validation Strategies

1. **Holdout Validation:** Reserve 20% of subjects for final testing
2. **Temporal Validation:** Ensure no future data leakage
3. **Cross-Validation:** 5-fold subject-level CV for robust estimates
4. **Statistical Testing:** Friedman test, Wilcoxon signed-rank test

---

## Phase 8: Implementation Checklist

### 8.1 Code Updates Required

- [ ] **Feature Selection Module:**
  - [ ] Implement `select_core_features()` function
  - [ ] Create feature importance weighting
  - [ ] Implement redundancy removal logic

- [ ] **Model Architecture:**
  - [ ] Update input dimension to 24 (from 20/42)
  - [ ] Optimize LSTM layers for reduced features
  - [ ] Implement Transformer-Enhanced variant (optional)

- [ ] **Data Pipeline:**
  - [ ] Filter to 47 valid subjects
  - [ ] Implement robust scaling
  - [ ] Add feature interaction engineering

- [ ] **Experiment Configuration:**
  - [ ] Update `experiment_config.json` with core features
  - [ ] Add feature set comparison experiments
  - [ ] Update evaluation metrics

### 8.2 Configuration Files

**New Config File:** `configs/feature_optimized_config.json`
```json
{
  "feature_selection": {
    "strategy": "core_24",
    "core_features": [...24 features...],
    "remove_redundant": true,
    "interaction_features": true
  },
  "data": {
    "valid_subjects": [...47 subjects...],
    "excluded_subjects": [...10 subjects...],
    "robust_scaling": true
  },
  "model": {
    "input_features": 24,
    "architecture": "lstm_optimized"
  }
}
```

---

## Phase 9: Timeline & Resources

### Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Data Preparation | 3 days | Pending |
| Phase 2: Model Updates | 5 days | Pending |
| Phase 3: Training Updates | 3 days | Pending |
| Phase 4: Evaluation Setup | 2 days | Pending |
| Phase 5: Experiment Execution | 3-4 weeks | Pending |
| Phase 6: Analysis & Reporting | 1 week | Pending |

**Total Estimated Time:** 5-6 weeks

### Resource Requirements

- **Compute:** GPU resources for model training
- **Storage:** Space for experiment results and models
- **Data:** Access to 47 processed subject files
- **Tools:** Feature analysis scripts, SHAP/LIME for interpretability

---

## Phase 10: References & Documentation

### Key Documents

1. **Feature Analysis Report:**
   - `experiments/cgm_fl_benchmark/feature_visualizations/FEATURE_ANALYSIS_REPORT.md`
   - Contains detailed feature statistics and recommendations

2. **Aggregated Analysis:**
   - `experiments/cgm_fl_benchmark/feature_visualizations/aggregated_feature_analysis_all_subjects.json`
   - Complete statistical analysis across 47 subjects

3. **Outlier Analysis:**
   - `experiments/cgm_fl_benchmark/feature_visualizations/outlier_analysis_report.json`
   - Outlier detection results (3.80% average, normal)

### Methodology References

- **Feature Selection:** Correlation (50%) + Mutual Information (30%) + Variance (20%)
- **Outlier Detection:** Z-score (|z|>3), IQR method, Isolation Forest
- **Consistency Threshold:** ≥70% across subjects

---

## Appendix: Feature Importance Summary

### Top 10 Features by Composite Score

1. `current_glucose` - 0.7138 (97.9% consistency, 0.6974 correlation)
2. `hour` - 0.4865 (100% consistency, 0.1583 correlation)
3. `max_3h` - 0.5273 (85.1% consistency, 0.4447 correlation)
4. `deviation_from_recent_trend` - 0.5401 (93.6% consistency, 0.5455 correlation)
5. `is_hyperglycemic` - 0.4659 (93.6% consistency, 0.5481 correlation)
6. `hourly_baseline` - 0.4003 (97.9% consistency, 0.3128 correlation)
7. `min_6h` - 0.4017 (97.9% consistency, 0.2451 correlation)
8. `max_6h` - 0.4343 (95.7% consistency, 0.3231 correlation)
9. `roc_60min` - 0.3722 (100% consistency, 0.3768 correlation)
10. `in_tight_range` - 0.3610 (97.9% consistency, 0.4277 correlation)

---

**Document Status:** Ready for Implementation  
**Next Steps:** Begin Phase 1 - Data Preparation & Feature Engineering  
**Contact:** Review feature analysis report for detailed statistics

---

*Last Updated: 2025-11-02*

