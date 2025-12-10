# Game of Life CNN Project

Neural network models for learning Conway's Game of Life rules using convolutional architectures.

## Experiment Results Summary

### Completed Experiments

| Experiment | Location | Configuration | Results |
|------------|----------|---------------|---------|
| 16-ch convergence stability | `experiments/convergence_stability/` | 16ch, L1=0.001, 30 runs | 100% convergence, avg 10.8 epochs |
| 10-ch direct training | `experiments/direct_training/` | 10ch, no pruning | 100% accuracy, 24 epochs |
| 4-ch convergence | `experiments/4ch_convergence/` | 4ch, L1=0.001, 10 runs | 100% convergence, avg 16.8 epochs |
| 2-ch convergence (L1) | `experiments/2ch_convergence/` | 2ch, L1=0.001, 5 runs | 80% convergence, avg 21 epochs; 1 model 100% on all 512 patterns |
| 2-ch comprehensive | `experiments/2ch_comprehensive/` | 2ch, L1/L2/None, 300 runs | L2 best (47.0%), L1 (44.0%), None (28.0%) |
| Maze 2-ch comprehensive | `experiments/maze_comprehensive/` | Maze rule, 2ch, L1/L2/None, 300 runs | 25–39% convergence depending on regularization |
| Maze transfer from GoL | `experiments/maze_transfer/` | 2ch, fine-tune from 36 converged GoL models | 36/36 convergence, avg 3.1 epochs |

### Key Findings

1. **Minimum Architecture**: 2 channels can solve Game of Life (~23 parameters) – a 2‑channel CNN reached 100% accuracy on all 512 possible 3×3 patterns, but 4 channels provide more reliable convergence.

2. **Lottery Ticket & Regularization Effect**: For 2‑channel models on GoL (now 300 runs in `experiments/2ch_comprehensive/`), convergence rates are 28/100 (28.0%) with no regularization, 44/100 (44.0%) with L1, and 47/100 (47.0%) with L2, showing that good initializations are still relatively rare without regularization and that L2 continues to work best when we want channels to behave uniformly (neighbors play symmetric roles).

3. **Rule Continuity vs Capacity**: GoL (B3/S23) and Maze (B3/S12345) both have survival on continuous neighbor intervals, and 2‑channel CNNs can learn them (many runs reach 100% for both rules). In contrast, the HighLife rule (B36/S23) has a disjoint birth set {3, 6}; theoretically a 2‑channel model that operates on “center + neighbor count” cannot carve out such a disconnected region cleanly. In our HighLife 2‑channel runs (30 attempts, no converged checkpoints saved in `experiments/highlife_comprehensive/`), the observed convergence rate was effectively 0/30.

4. **Transfer Learning Across Rules**: When we fine‑tune Maze from 36 converged 2‑channel GoL models (`experiments/maze_transfer/`), all 36 runs converge (100% success) with a mean convergence epoch of ≈3.1. This indicates that the model has already learned the shared spatial structure (counting neighbors) on GoL and can adapt to the closely related Maze rule almost immediately.

5. **Parameter Efficiency**:

| Architecture | Parameters | Convergence Rate |
|-------------|------------|------------------|
| 16-channel  | 177        | 100%             |
| 10-channel  | 111        | 100%             |
| 4-channel   | 45         | 100%             |
| 2-channel   | 23         | 17-80%           |

6. **Training Stability**: Standard deviation of convergence epochs correlates with channel count - more channels = more stable

## Project Structure

```
project/
├── src/              Source code
├── scripts/          Training and evaluation scripts
├── data/             Datasets
├── experiments/      Experimental results
├── models/           Saved model checkpoints
├── figures/          Visualizations
└── doc/              Documentation (gitignored)
```

## Key Directories

### experiments/

Contains results from various training runs and ablation studies.

- `2ch_convergence/` - Two-channel network convergence tests
  - `models/` - Saved checkpoints from successful runs
  - `weight_analysis/` - Extracted weights and analysis
  - `exhaustive_test/` - 512-pattern exhaustive validation results
  
- `3ch_convergence/` - Three-channel network tests
  - `models/` - Saved model checkpoints
  
- `4ch_convergence/` - Four-channel network with L1 regularization
  - `summary.json` - Aggregated statistics across runs
  - `README.md` - Experiment description and findings
  - `models/` - Saved checkpoints

- `4ch_no_l1/` - Four-channel without L1, comparison baseline
  - Same structure as 4ch_convergence

- `6ch_convergence/` - Six-channel experiments
  
- `maze_comprehensive/` - 2-channel Maze rule training (L1/L2/None)
  - `summary.json` - Aggregated statistics across runs
  - `models/` - Saved checkpoints
  
- `maze_transfer/` - Transfer learning from converged GoL 2-channel models to Maze
  - `transfer_results.json` - Fine-tuning statistics
  - `models/` - Fine-tuned checkpoints
  
- `convergence_stability/` - Multi-run stability tests
  - `pruning_limits.json` - Analysis of minimum viable channels
  - `exhaustive_test/` - Pattern-level validation
  
- `direct_training/` - Direct training on minimal architectures
  - `training_history.json` - Loss and accuracy curves

### models/

Saved model checkpoints and metadata.

- `pruned/` - Models after channel pruning
  - `channel_importance.png` - Visualization of channel importance scores
  - `*.pth` - Checkpoint files with model state dicts

- `pruned_l1/` - Models trained with L1 and pruned
  - `pruning_report.json` - Channel-by-channel pruning decisions
  - `README.md` - Methodology and results

### figures/

Generated visualizations and animations.

- `samples/` - Example Game of Life patterns
  - `*_animation.gif` - Evolution over time
  - `*_trajectory.png` - State space trajectories
  - `*_initial.png` - Initial conditions

- `evaluation/` - Model predictions vs ground truth
  - `*_comparison.gif` - Side-by-side comparisons

- `4ch_patterns/` - Four-channel model evaluations
  - Pattern-specific prediction comparisons

### data/

Training and test datasets.

- `processed/` - Preprocessed HDF5 datasets
  - `train.h5` - Training set
  - `val.h5` - Validation set
  - Note: Large files, gitignored

- `holdout_patterns.json` - Held-out patterns for OOD testing

## Key Files

### Experiment Summaries

Each experiment directory typically contains:

- `summary.json` - Quantitative results
  - Configuration parameters
  - Convergence statistics
  - Per-run metrics
  
- `README.md` - Qualitative analysis
  - Experiment motivation
  - Key findings
  - Recommendations

### Model Checkpoints

Model files (`.pth`) contain:

```python
{
  'model_state_dict': ...,    # Trained weights
  'hidden_channels': int,     # Architecture parameter
  'convergence_epoch': int,   # Training duration
  'val_accuracy': float,      # Final validation accuracy
  'lambda_l1': float,         # L1 regularization strength
  ...
}
```

### Exhaustive Tests

`exhaustive_test/*.json` files contain validation on all 512 possible 3x3 patterns:

```python
{
  'total_patterns': 512,
  'correct': int,
  'errors': int,
  'accuracy': float,
  'error_patterns': [...]     # Detailed error cases
}
```

## Running Experiments

Training scripts are in `scripts/`. Key examples:

```bash
# Two-channel convergence test
python3 scripts/test_2ch_convergence.py

# Exhaustive pattern validation
python3 scripts/test_all_patterns.py --model path/to/model.pth

# Evaluate trained model
python3 scripts/evaluate_model.py --model path/to/model.pth
```

## Data Format

HDF5 datasets contain:

- `states_t` - Current state (N, 1, H, W)
- `states_t1` - Next state (N, 1, H, W)

Binary arrays with 0 (dead) and 1 (alive).
