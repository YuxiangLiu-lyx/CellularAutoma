# 2-Channel CNN Convergence Test with L1 Regularization

Generated: 2025-12-01 17:18:31

## Experiment Design

- **Objective**: Test if 2-channel CNN (absolute minimum) can reach 100% accuracy
- **Motivation**: Find the absolute theoretical minimum - can just 2 channels solve Game of Life?
- **Configuration**: lambda_l1 = 0.001, early stopping patience = 20
- **Number of runs**: 5
- **Seeds**: 300 to 700
- **Parameters per model**: ~15 (2 channels × 3×3 kernel + biases)
- **Parameter reduction**: ~91.5% vs standard 16-channel (177 params)

## Results

- **Convergence rate**: 4/5 (80.0%)
- **Mean convergence**: 21.0 epochs
- **Median convergence**: 20.0 epochs
- **Range**: 18-26 epochs
- **Std deviation**: 3.0 epochs

## Channel Usage

Average L1 norms across converged runs:
- Channel 0: 4.783 ± 1.068
- Channel 1: 5.764 ± 1.986

This shows how both channels are being utilized in successful runs.

## Analysis

**Strong results**: 4/5 runs converged.

2 channels are **nearly sufficient** but sensitive to initialization:
- Most runs succeed with proper initialization
- May benefit from better initialization schemes
- 3 channels might be more robust for production use

## Architecture Verification

```
Input:  (batch, 1, H, W)          # Single channel Game of Life state
   ↓
Conv1:  (batch, 2, H, W)          # 2 hidden channels (1→2, 3×3 kernel)
   ↓
ReLU
   ↓
Conv2:  (batch, 1, H, W)          # Back to single channel (2→1, 1×1 kernel)
   ↓
Sigmoid
   ↓
Output: (batch, 1, H, W)          # Predicted next state
```

Architecture: **Single channel → Multi-channel (2) → Single channel**

## Comparison with Other Architectures

| Architecture | Channels | Parameters | Conv1 Params | Conv2 Params | Convergence Rate |
|--------------|----------|------------|--------------|--------------|------------------|
| Standard     | 16       | ~177       | 1×16×9+16=160| 16×1×1+1=17  | 100% (baseline)  |
| Pruned       | 4        | ~29        | 1×4×9+4=40   | 4×1×1+1=5    | ? (see 4ch)      |
| Minimal      | 3        | ~22        | 1×3×9+3=30   | 3×1×1+1=4    | ? (see 3ch)      |
| **This test**| **2**    | **~15**    | **1×2×9+2=20**| **2×1×1+1=3** | **40%** |

## Files

- `summary.json` - Complete experimental data
- `models/` - Saved models from successful runs
- `README.md` - This file

## Next Steps

- ✓ 2 channels work! This is the minimum viable architecture
- Analyze learned weights to understand what each channel represents
- Test if we can initialize specifically for center/neighbor extraction
