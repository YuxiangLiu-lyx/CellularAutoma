# L1 Regularization Pruned Model

Generated: 2025-11-29 13:39:21

## Summary

- **Method**: L1 Regularization (lambda=0.001)
- **Original Channels**: 16
- **Active Channels**: 10
- **Pruned Channels**: 6
- **Accuracy**: 100.00%
- **Parameter Reduction**: 37.3%

## Channel Analysis

### Active Channels (L1 norm >= 0.5)
```
Channel  0: L1 norm = 3.4199
Channel  2: L1 norm = 2.6781
Channel  3: L1 norm = 0.9772
Channel  5: L1 norm = 0.5097
Channel  8: L1 norm = 3.2657
Channel  9: L1 norm = 2.9815
Channel 10: L1 norm = 2.3862
Channel 12: L1 norm = 2.5894
Channel 14: L1 norm = 2.1970
Channel 15: L1 norm = 1.5692
```

### Pruned Channels (L1 norm < 0.5)
```
Channel  1: L1 norm = 0.3586
Channel  4: L1 norm = 0.4261
Channel  6: L1 norm = 0.0770
Channel  7: L1 norm = 0.0009
Channel 11: L1 norm = 0.0014
Channel 13: L1 norm = 0.0007
```

## Key Findings

- L1 regularization (lambda=0.001) automatically identified 6 redundant channels
- Only 10 channels are needed to achieve 100% accuracy
- Parameter reduction: 37.3%
- Pruned channels have L1 norms < 0.5, indicating minimal contribution
- Active channels: [0, 2, 3, 5, 8, 9, 10, 12, 14, 15]
- Pruned channels: [1, 4, 6, 7, 11, 13]

## Files

- `model.pth` - Pruned model parameters
- `pruning_report.json` - Detailed pruning statistics and metadata
- `README.md` - This file

## Usage

```python
import torch
from src.models.cnn import GameOfLifeCNN

# Load pruned model
checkpoint = torch.load('models/pruned_l1/model.pth')
model = GameOfLifeCNN(hidden_channels=16, padding_mode='circular')
model.load_state_dict(checkpoint['model_state_dict'])

# Check which channels are active
print(f"Active channels: {checkpoint['active_channels']}")
print(f"Pruned channels: {checkpoint['pruned_channels']}")
```
