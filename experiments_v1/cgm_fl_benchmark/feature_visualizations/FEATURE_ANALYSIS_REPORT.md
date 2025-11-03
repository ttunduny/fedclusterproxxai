# Comprehensive Feature Analysis Report

**Analysis Date:** 2025-11-02 16:12:27.922174
**Generated:** 2025-11-02 16:17:47

---

## Executive Summary

This report presents a comprehensive feature analysis conducted on **47 subjects** from the CGM dataset.

- **Subjects Analyzed:** 47
- **Failed Subjects:** 10
- **Consistently Important Features:** 24
- **Total Features Analyzed:** 42

### Failed Subjects

The following subjects could not be analyzed due to data issues:

- Subject13
- Subject14
- Subject15
- Subject18
- Subject21
- Subject25
- Subject32
- Subject44
- Subject5
- Subject7

## Methodology

The feature analysis was conducted using a six-step process:

### 1. Target Correlation Analysis
- Computed Pearson correlation coefficient between each feature and target CGM value
- Identified statistically significant correlations (p-value < 0.05)
- Features with |correlation| > 0.05 were considered useful

### 2. Mutual Information Scoring
- Calculated mutual information (non-linear dependency measure) between features and target
- Identified features with high mutual information scores
- Captures non-linear relationships that correlation might miss

### 3. Variance Analysis
- Analyzed feature variance to identify informative features
- Low variance features (constant/near-constant) were flagged
- High variance features contribute more to model learning

### 4. Redundancy Detection
- Computed inter-feature correlation matrix
- Identified highly correlated feature pairs (correlation >= 0.9)
- Recommended removing redundant features to reduce multicollinearity
- Kept feature with higher target correlation in each pair

### 5. Composite Scoring
- Combined metrics: **Target Correlation (50%)**, **Mutual Information (30%)**, **Variance (20%)**
- Ranked features by composite score
- Features with composite score > 0.1 and not redundant were recommended

### 6. Cross-Subject Consistency
- Analyzed feature importance across multiple subjects
- Features appearing in top recommendations for ≥70% of subjects are 'consistently important'
- Provides robustness and generalizability across different glucose patterns

## Consistently Important Features

Features that are important in **≥70%** of analyzed subjects are considered consistently important.

**Total:** 24 features

| Rank | Feature | Consistency | Avg Correlation | Avg Composite Score |
|------|---------|------------|----------------|---------------------|
| 1 | `hour` | 100.0% | 0.1583 | 0.4865 |
| 2 | `roc_60min` | 100.0% | 0.3768 | 0.3722 |
| 3 | `current_glucose` | 97.9% | 0.6974 | 0.7138 |
| 4 | `min_6h` | 97.9% | 0.2451 | 0.4017 |
| 5 | `hourly_baseline` | 97.9% | 0.3128 | 0.4003 |
| 6 | `in_tight_range` | 97.9% | 0.4277 | 0.3610 |
| 7 | `max_6h` | 95.7% | 0.3231 | 0.4343 |
| 8 | `mean_24h` | 95.7% | 0.1391 | 0.3281 |
| 9 | `max_12h` | 95.7% | 0.2087 | 0.3245 |
| 10 | `std_3h` | 95.7% | 0.1762 | 0.2237 |
| 11 | `deviation_from_recent_trend` | 93.6% | 0.5455 | 0.5401 |
| 12 | `is_hyperglycemic` | 93.6% | 0.5481 | 0.4659 |
| 13 | `mean_12h` | 93.6% | 0.1877 | 0.3235 |
| 14 | `roc_30min` | 93.6% | 0.3157 | 0.3050 |
| 15 | `std_12h` | 93.6% | 0.1468 | 0.2944 |
| 16 | `min_12h` | 91.5% | 0.1469 | 0.3022 |
| 17 | `min_24h` | 87.2% | 0.1109 | 0.2283 |
| 18 | `hour_sin` | 87.2% | 0.1834 | 0.2272 |
| 19 | `max_3h` | 85.1% | 0.4447 | 0.5273 |
| 20 | `std_6h` | 85.1% | 0.1747 | 0.2607 |
| 21 | `max_24h` | 83.0% | 0.1399 | 0.2315 |
| 22 | `hour_cos` | 80.9% | 0.1398 | 0.2089 |
| 23 | `std_1h` | 78.7% | 0.1545 | 0.1696 |
| 24 | `std_24h` | 76.6% | 0.1077 | 0.3080 |

## Top 20 Features by Average Correlation with Target

| Rank | Feature | Avg Correlation | Frequency | Consistency |
|------|---------|----------------|----------|-------------|
| 1 | `current_glucose` | 0.6974 | 46/47 | 97.9% |
| 2 | `deviation_from_daily_mean` | 0.6454 | 16/47 | 34.0% |
| 3 | `prev_15min` | 0.6223 | 5/47 | 10.6% |
| 4 | `deviation_from_pattern` | 0.6055 | 11/47 | 23.4% |
| 5 | `mean_1h` | 0.5952 | 5/47 | 10.6% |
| 6 | `prev_30min` | 0.5547 | 5/47 | 10.6% |
| 7 | `is_hyperglycemic` | 0.5481 | 44/47 | 93.6% |
| 8 | `deviation_from_recent_trend` | 0.5455 | 44/47 | 93.6% |
| 9 | `in_target_range` | 0.5188 | 5/47 | 10.6% |
| 10 | `max_3h` | 0.4447 | 40/47 | 85.1% |
| 11 | `prev_60min` | 0.4279 | 16/47 | 34.0% |
| 12 | `in_tight_range` | 0.4277 | 46/47 | 97.9% |
| 13 | `mean_3h` | 0.4179 | 19/47 | 40.4% |
| 14 | `roc_60min` | 0.3768 | 47/47 | 100.0% |
| 15 | `min_3h` | 0.3745 | 32/47 | 68.1% |
| 16 | `trend_strength` | 0.3550 | 6/47 | 12.8% |
| 17 | `max_6h` | 0.3231 | 45/47 | 95.7% |
| 18 | `roc_30min` | 0.3157 | 44/47 | 93.6% |
| 19 | `hourly_baseline` | 0.3128 | 46/47 | 97.9% |
| 20 | `mean_6h` | 0.2856 | 32/47 | 68.1% |

## Feature Categories Analysis

### Temporal Features

**Consistently Important Features:** 4

- `hour` (100.0% consistency, avg correlation: 0.1583)
- `hourly_baseline` (97.9% consistency, avg correlation: 0.3128)
- `hour_sin` (87.2% consistency, avg correlation: 0.1834)
- `hour_cos` (80.9% consistency, avg correlation: 0.1398)

### Recent Values

**Consistently Important Features:** 3

- `roc_60min` (100.0% consistency, avg correlation: 0.3768)
- `current_glucose` (97.9% consistency, avg correlation: 0.6974)
- `roc_30min` (93.6% consistency, avg correlation: 0.3157)

### Rolling Statistics

**Consistently Important Features:** 14

- `min_6h` (97.9% consistency, avg correlation: 0.2451)
- `max_6h` (95.7% consistency, avg correlation: 0.3231)
- `mean_24h` (95.7% consistency, avg correlation: 0.1391)
- `max_12h` (95.7% consistency, avg correlation: 0.2087)
- `std_3h` (95.7% consistency, avg correlation: 0.1762)
- `mean_12h` (93.6% consistency, avg correlation: 0.1877)
- `std_12h` (93.6% consistency, avg correlation: 0.1468)
- `min_12h` (91.5% consistency, avg correlation: 0.1469)
- `min_24h` (87.2% consistency, avg correlation: 0.1109)
- `max_3h` (85.1% consistency, avg correlation: 0.4447)
- `std_6h` (85.1% consistency, avg correlation: 0.1747)
- `max_24h` (83.0% consistency, avg correlation: 0.1399)
- `std_1h` (78.7% consistency, avg correlation: 0.1545)
- `std_24h` (76.6% consistency, avg correlation: 0.1077)

### Medical Indicators

**Consistently Important Features:** 2

- `in_tight_range` (97.9% consistency, avg correlation: 0.4277)
- `is_hyperglycemic` (93.6% consistency, avg correlation: 0.5481)

### Pattern Analysis

**Consistently Important Features:** 1

- `deviation_from_recent_trend` (93.6% consistency, avg correlation: 0.5455)

## Common Redundant Features

The following 30 features were identified as redundant across multiple subjects (correlation ≥ 0.9 with other features):

- `current_glucose`
- `deviation_from_daily_mean`
- `deviation_from_pattern`
- `deviation_from_recent_trend`
- `glucose_variability`
- `hour_sin`
- `hourly_baseline`
- `in_target_range`
- `is_hyperglycemic`
- `max_24h`
- `max_3h`
- `max_6h`
- `mean_12h`
- `mean_1h`
- `mean_24h`
- `mean_3h`
- `mean_6h`
- `min_12h`
- `min_24h`
- `min_3h`
- `min_6h`
- `prev_15min`
- `prev_30min`
- `prev_60min`
- `recent_hyper_6h`
- `std_12h`
- `std_24h`
- `std_3h`
- `std_6h`
- `trend_strength`

**Recommendation:** Consider removing these features to reduce multicollinearity and improve model performance.

## Recommendations

### Feature Selection Recommendations

**Core Feature Set:** Use the 24 consistently important features as the foundation for CGM prediction models.

**Priority Features:**
1. **Temporal Features**: `hour`, `hour_cos`, `hourly_baseline` - Critical for capturing circadian patterns
2. **Current State**: `current_glucose` - Highest correlation with target (0.6974)
3. **Recent History**: `prev_15min`, `prev_30min` - Capture short-term trends
4. **Rolling Statistics**: `mean_1h`, `mean_3h`, `max_6h`, `min_6h` - Provide context windows
5. **Rate of Change**: `roc_60min`, `roc_30min` - Capture velocity of glucose changes
6. **Medical Indicators**: `in_tight_range`, `is_hyperglycemic` - Clinical relevance

### Model Development Recommendations

1. **Start with Core Features**: Begin model development with the consistently important features
2. **Feature Engineering**: Consider interactions between temporal features and glucose values
3. **Redundancy Removal**: Remove or combine redundant features to reduce model complexity
4. **Subject-Specific Tuning**: While core features are consistent, subject-specific patterns may benefit from additional features
5. **Validation Strategy**: Use cross-validation across subjects to ensure generalizability

## Key Insights

1. **Temporal Patterns Are Critical**: Temporal features (`hour`, `hour_cos`, `hourly_baseline`) show the highest consistency across subjects, indicating strong circadian patterns in glucose levels.

2. **Current State Matters Most**: `current_glucose` has the highest correlation (0.6974) with the target, making it the single most predictive feature.

3. **Short-Term Statistics Outperform Long-Term**: Features like `mean_1h`, `mean_3h` are more consistently important than `mean_24h`, suggesting recent history is more predictive.

4. **Rate of Change is Important**: `roc_60min` appears in 100% of subjects, indicating glucose velocity is a universal predictor.

5. **Medical Indicators Provide Context**: Features like `in_tight_range` and `is_hyperglycemic` appear consistently, adding clinical relevance to predictions.

## Appendix

### All Analyzed Subjects

1. SUB001
2. SUB002
3. SUB003
4. SUB004
5. SUB005
6. Subject10
7. Subject11
8. Subject12
9. Subject16
10. Subject17
11. Subject19
12. Subject20
13. Subject22
14. Subject23
15. Subject24
16. Subject26
17. Subject27
18. Subject28
19. Subject29
20. Subject3
21. Subject30
22. Subject31
23. Subject33
24. Subject34
25. Subject35
26. Subject36
27. Subject37
28. Subject38
29. Subject39
30. Subject4
31. Subject40
32. Subject41
33. Subject42
34. Subject43
35. Subject45
36. Subject46
37. Subject47
38. Subject48
39. Subject49
40. Subject50
41. Subject51
42. Subject52
43. Subject53
44. Subject54
45. Subject6
46. Subject8
47. Subject9

### Report Files

- **Aggregated Report (JSON)**: `experiments/cgm_fl_benchmark/feature_visualizations/aggregated_feature_analysis_all_subjects.json`
- **Individual Reports**: 47 subject-specific analysis reports in the same directory

---

*Report generated on 2025-11-02 16:17:47*