# 10-Channel CNN Direct Training

Generated: 2025-11-29 13:59:29

## Summary

- **Architecture**: GameOfLifeCNN with 10 channels
- **Training Method**: Direct training (no pruning)
- **Parameters**: 111
- **Best Accuracy**: 100.00%
- **Converged**: Yes
- **Training Epochs**: 24

## Motivation

Based on L1 regularization experiments, we found that only 10 out of 16 channels are needed.
This experiment tests if training a 10-channel model directly (without pruning) can achieve 100% accuracy.

## Results

✓ **SUCCESS**: 10 channels are sufficient!

- Achieved 100.00% accuracy
- Validates L1 pruning findings
- Can train 10-channel model directly without pruning

## Comparison

| Approach | Channels | Parameters | Accuracy |
|----------|----------|------------|----------|
| Original 16-ch | 16 | 177 | 100.00% |
| L1 Pruned (16→10) | 10 active | ~111 | 100.00% |
| Direct 10-ch | 10 | 111 | 100.00% |

## Files

- `cnn_10ch.pth` - Trained model parameters
- `training_history.json` - Epoch-by-epoch training log
- `README.md` - This file

## Usage

```python
import torch
from src.models.cnn import GameOfLifeCNN

checkpoint = torch.load('experiments/direct_training/cnn_10ch.pth')
model = GameOfLifeCNN(hidden_channels=10, padding_mode='circular')
model.load_state_dict(checkpoint['model_state_dict'])
```
