# Convergence Stability Test

Generated: 2025-11-30 00:00:56

## Experiment Design

- **Objective**: Test convergence stability of 16-channel CNN with L1 regularization
- **Configuration**: lambda_l1 = 0.001
- **Number of runs**: 30
- **Seeds**: 42 to 2942

## Results

- **Convergence rate**: 30/30 (100.0%)
- **Mean convergence**: 10.8 epochs
- **Median convergence**: 10.0 epochs
- **Range**: 6-22 epochs
- **Std deviation**: 3.6 epochs

## Analysis

**Highly stable convergence** (std < 5 epochs)

L1 regularization provides consistent convergence across different initializations.

## Files

- `summary.json` - Complete experimental data
- `models/` - Saved models from each run
- `README.md` - This file
