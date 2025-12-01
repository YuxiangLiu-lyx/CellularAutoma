# 4-Channel CNN Convergence Test (No L1 Regularization)

Generated: 2025-12-01 05:07:16

## Experiment Design

- **Objective**: Test if 4-channel CNN can reach 100% accuracy WITHOUT L1 regularization
- **Hypothesis**: Maybe L1 is not necessary, just need enough training epochs
- **Configuration**: No L1 regularization, only BCE loss
- **Number of runs**: 30
- **Seeds**: 42 to 2942

## Results

- **Convergence rate**: 25/30 (83.3%)
- **Mean convergence**: 20.9 epochs
- **Median convergence**: 18.0 epochs
- **Range**: 11-52 epochs
- **Std deviation**: 8.7 epochs

## Analysis

**Mostly successful**: 25/30 runs converged.

This suggests:
- **L1 may help but is not critical**
- Most initializations can reach 100% without L1
- Some unlucky initializations may benefit from L1

## Comparison with L1 Regularization

To compare, run the test with L1 (lambda=0.001):
```bash
python scripts/test_4ch_convergence.py
```

## Files

- `summary.json` - Complete experimental data
- `models/` - Saved models from successful runs
- `README.md` - This file
