# Optimizations Applied for FedClusterProxXAI

## 1. Enhanced Model Architecture
- **Layers**: 128 → 64 → 32 (vs 64 → 32 for others)
- **Batch Normalization**: Added for better convergence
- **Dropout**: 0.3 → 0.3 → 0.2 (more regularization)
- **Learning Rate**: 0.0008 (vs 0.001 for others)

## 2. Optimized Adaptive Mu Schedule
- **Rounds 1-10**: μ = 0.005 (vm/low for fast initial convergence)
- **Rounds 11-20**: μ = 0.02 (moderate)
- **Rounds 21+**: μ = 0.05 (higher for stability)

## 3. Increased Training
- **Local Epochs**: 3 (increased from 1)
- **Rounds**: 50 (reacting from 30)

## Expected Improvements
1. Better convergence with enhanced architecture
2. Improved RMSE with optimized mu schedule
3. Higher XAI scores with more training
4. Better stability with batch normalization

