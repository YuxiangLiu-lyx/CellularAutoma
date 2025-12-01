# 6-Channel CNN Convergence Test

Generated: 2025-11-30 09:59:54

## Experiment Design

- **Objective**: Test if 6-channel CNN can reach 100% accuracy
- **Motivation**: Pruning tests showed some models can work with only 6 channels
- **Configuration**: lambda_l1 = 0.001
- **Number of runs**: 10
- **Seeds**: 100 to 1000

## Results

- **Convergence rate**: 10/10 (100.0%)
- **Mean convergence**: 17.2 epochs
- **Median convergence**: 13.5 epochs
- **Range**: 9-42 epochs
- **Std deviation**: 9.2 epochs

## Analysis

**✓ SUCCESS**: All 10 runs converged to 100% accuracy!

6 channels are **sufficient** for this task. This validates:
- Pruning test results showing 6-channel minimum
- Can train 6-channel models directly instead of 16→6 pruning
- Significant parameter reduction: 6ch has ~43 params vs 16ch with 177 params

## Files

- `summary.json` - Complete experimental data
- `models/` - Saved models from successful runs
- `README.md` - This file
