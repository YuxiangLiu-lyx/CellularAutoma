# Experiment Summary: Game of Life CNN

## Overview

This project explores the minimum network capacity required to learn Conway's Game of Life rules using CNNs with various regularization techniques.

## Experiments Conducted

### 1. Convergence Stability Test (16-channel baseline)
**Location**: `experiments/convergence_stability/`

- **Objective**: Establish baseline convergence behavior for 16-channel CNN with L1 regularization
- **Configuration**: 16 channels, L1 lambda=0.001, 30 runs
- **Results**:
  - Convergence rate: 100% (30/30)
  - Mean convergence: 10.8 epochs
  - Std: 3.6 epochs
  - Range: 6-22 epochs
- **Conclusion**: L1 regularization provides stable, consistent convergence

### 2. Direct Training (10-channel)
**Location**: `experiments/direct_training/`

- **Objective**: Verify that 10 channels (found via L1 pruning) can be trained directly
- **Configuration**: 10 channels, no pruning
- **Results**:
  - Converged: Yes (epoch 24)
  - Accuracy: 100%
  - Parameters: 111
- **Conclusion**: 10-channel architecture is sufficient without pruning

### 3. 4-Channel Convergence Test
**Location**: `experiments/4ch_convergence/`

- **Objective**: Test if 4 channels can solve Game of Life
- **Configuration**: 4 channels, L1 lambda=0.001, 10 runs
- **Results**:
  - Convergence rate: 100% (10/10)
  - Mean convergence: 16.8 epochs
  - Range: 13-29 epochs
- **Conclusion**: 4 channels are sufficient with L1 regularization

### 4. 2-Channel Convergence Test (L1)
**Location**: `experiments/2ch_convergence/`

- **Objective**: Find theoretical minimum channel count
- **Configuration**: 2 channels, L1 lambda=0.001, 5 runs
- **Results**:
  - Convergence rate: 80% (4/5)
  - Mean convergence: 21.0 epochs
- **Conclusion**: 2 channels work but are sensitive to initialization

### 5. Comprehensive 2-Channel Regularization Comparison
**Location**: `experiments/2ch_comprehensive/`

- **Objective**: Compare L1, L2, and no regularization on 2-channel models
- **Configuration**: 2 channels, 30 runs per regularization type, 90 total runs
- **Results**:

| Regularization | Converged | Rate   | Mean Epoch |
|---------------|-----------|--------|------------|
| L2            | 14/30     | 46.7%  | 19.6       |
| L1            | 11/30     | 36.7%  | 19.4       |
| None          | 5/30      | 16.7%  | 17.6       |

- **Conclusion**: L2 regularization provides the best convergence rate for minimal architectures

### 6. Grokking Experiment (Modular Arithmetic)
**Location**: `experiments/grokking/`

- **Objective**: Test if L1/L2 can induce grokking in bilinear networks on modular arithmetic
- **Configuration**: P=97, 10000 epochs, bilinear model
- **Results**:
  - L1: No grokking, weights collapsed to near-zero
  - L2: No grokking, weights stabilized at non-sparse values
- **Conclusion**: Strong regularization prevents grokking in this setup

### 7. Logic Gates Experiment
**Location**: `experiments/logic_gates/`

- **Objective**: Test minimal network on random 5-variable boolean function
- **Configuration**: 32-2-1 MLP, 200 epochs
- **Results**:
  - Converged: No
  - Best accuracy: 84.8%
- **Conclusion**: 2 hidden units insufficient for arbitrary 5-variable logic

## Key Findings

1. **Minimum Architecture**: 2 channels can solve Game of Life (~15 parameters), but 4 channels provide more reliable convergence

2. **Regularization Effect**: 
   - L1 promotes sparsity, useful for pruning
   - L2 slightly better for training minimal architectures
   - Both significantly better than no regularization

3. **Parameter Efficiency**:
   | Architecture | Parameters | Convergence |
   |-------------|------------|-------------|
   | 16-channel  | 177        | 100%        |
   | 10-channel  | 111        | 100%        |
   | 4-channel   | 45         | 100%        |
   | 2-channel   | 21         | 17-47%      |

4. **Training Stability**: Standard deviation of convergence epochs correlates with channel count - more channels = more stable

## File Structure

```
experiments/
├── 2ch_comprehensive/    # L1 vs L2 vs None comparison
├── 2ch_convergence/      # 2-channel with L1
├── 2ch_l2/               # 2-channel with L2
├── 2ch_no_l1/            # 2-channel without regularization
├── 4ch_convergence/      # 4-channel test
├── convergence_stability/ # 16-channel baseline
├── direct_training/      # 10-channel direct training
├── grokking/             # Modular arithmetic experiment
└── logic_gates/          # Boolean function experiment
```

## Scripts

- `test_2ch_comprehensive.py` - Run full L1/L2/None comparison
- `test_2ch_convergence.py` - Test 2-channel with L1
- `test_2ch_l2.py` - Test 2-channel with L2
- `test_2ch_no_l1.py` - Test 2-channel without regularization
- `convergence_stability_test.py` - 16-channel baseline test
- `train_with_l1.py` - Train with various L1 strengths
