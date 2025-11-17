# Cellular Automata Rule Learning

A deep learning project for learning Conway's Game of Life rules using CNNs. This project explores whether neural networks can generalize cellular automata rules from incomplete training data.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Generation](#data-generation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Overview

This project investigates whether deep learning models can learn the abstract rules of cellular automata (specifically Conway's Game of Life) and generalize to unseen patterns. We implement:

- A synthetic data generation pipeline for Game of Life on 32×32 grids
- Lightweight CNN architectures with spatial inductive bias
- Holdout pattern evaluation for testing true out-of-distribution generalization
- Comprehensive metrics for single-step and multi-step autoregressive evaluation

### Key Findings

- **Perfect In-Distribution Performance**: CNN trained on random data achieves 100% accuracy
- **Perfect OOD Generalization**: Model trained on incomplete data (with holdout patterns) successfully infers the complete Game of Life algorithm
- **Inductive Bias Matters**: A 3×3 convolutional kernel matches the local neighborhood structure of the rules

## Project Structure

```
CellularAutoma/
├── data/                           # Data storage
│   ├── processed/                  # Full random dataset
│   │   ├── train.h5               # 10K training samples
│   │   ├── val.h5                 # 2K validation samples
│   │   ├── test_random.h5         # Random test samples
│   │   └── test_patterns.h5       # Classic patterns for evaluation
│   └── processed_holdout/          # Holdout dataset (incomplete rules)
│       ├── train.h5               # Training with holdout patterns excluded
│       └── val.h5                 # Validation with holdout patterns excluded
├── src/                            # Source code
│   ├── models/                     # Neural network models
│   │   ├── cnn.py                 # CNN architectures
│   │   └── train.py               # Training script
│   ├── utils/                      # Utilities
│   │   ├── game_of_life.py        # Simulator
│   │   ├── patterns.py            # Classic patterns (glider, blinker, etc.)
│   │   ├── data_loader.py         # Data loading utilities
│   │   └── visualization.py       # Visualization tools
│   └── evaluation/                 # Evaluation metrics
│       └── metrics.py             # Accuracy and preservation metrics
├── scripts/                        # Executable scripts
│   ├── generate_dataset.py        # Generate full random dataset
│   ├── generate_dataset_holdout.py # Generate holdout dataset
│   ├── generate_samples.py        # Generate pattern samples for visualization
│   ├── select_holdout_patterns.py # Select patterns to hold out
│   ├── evaluate_model.py          # Multi-step autoregressive evaluation
│   ├── evaluate_holdout.py        # Test on holdout patterns
│   └── analyze_training_coverage.py # Analyze which rules were covered
├── experiments/                    # Training experiments and checkpoints
├── figures/                        # Generated visualizations
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CellularAutoma
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `numpy` - Numerical computations
- `torch` - Deep learning framework
- `h5py` - HDF5 data storage
- `matplotlib` - Visualization
- `tqdm` - Progress bars
- `scikit-learn` - Additional utilities

## Data Generation

### 1. Generate Full Random Dataset

This creates a complete dataset with 30% random cell density covering all 512 possible 3×3 local rules.

```bash
python scripts/generate_dataset.py
```

**Output:**
- `data/processed/train.h5` - 10,000 training samples
- `data/processed/val.h5` - 2,000 validation samples
- `data/processed/test_random.h5` - 2,000 test samples
- `data/processed/test_patterns.h5` - Classic patterns (gliders, oscillators, etc.)

**Purpose:** Train a model that can potentially "memorize" all rules and achieve perfect performance.

### 2. Generate Holdout Dataset (Out-of-Distribution Test)

This creates a dataset that explicitly excludes specific 3×3 patterns to test true generalization.

**Step 1:** Select holdout patterns
```bash
python scripts/select_holdout_patterns.py
```

This randomly selects a subset of the 512 possible 3×3 neighborhood patterns and saves them to `data/holdout_patterns.json`.

**Step 2:** Generate holdout training data
```bash
python scripts/generate_dataset_holdout.py
```

**Output:**
- `data/processed_holdout/train.h5` - 10,000 samples excluding holdout patterns
- `data/processed_holdout/val.h5` - 2,000 validation samples excluding holdout patterns

**Purpose:** Train a model on incomplete data to test if it can infer the complete algorithm.

### 3. Generate Sample Visualizations

Generate visualizations of classic Game of Life patterns:

```bash
python scripts/generate_samples.py
```

**Output:** Saved to `figures/samples/`
- Initial states, trajectories, and animated GIFs for 10 classic patterns
- Still lifes: block, beehive, boat, loaf
- Oscillators: blinker, toad, beacon, pulsar
- Spaceships: glider, LWSS (lightweight spaceship)

## Training

### Train on Full Random Dataset

```bash
python src/models/train.py --model simple --hidden 16 --epochs 20 --batch-size 32 --lr 0.001 --save experiments/cnn/model.pt
```

### Train on Holdout Dataset

```bash
python src/models/train.py --model simple --hidden 16 --epochs 20 --batch-size 32 --lr 0.001 --data-dir data/processed_holdout --save experiments/cnn/model_holdout.pt
```

### Training Arguments

- `--model` - Model type: `simple` (single 3×3 conv) or `deep` (multi-layer)
- `--hidden` - Number of hidden channels (default: 16)
- `--layers` - Number of layers for deep model (default: 2)
- `--epochs` - Number of training epochs (default: 20)
- `--batch-size` - Batch size (default: 32)
- `--lr` - Learning rate (default: 0.001)
- `--data-dir` - Data directory (default: `data/processed`)
- `--save` - Path to save the trained model

### Model Architecture

**GameOfLifeCNN (Simple):**
- 3×3 Convolutional layer (16 channels) + ReLU
- 1×1 Convolutional layer + Sigmoid
- Uses `padding_mode='circular'` to match periodic boundary conditions
- ~300 parameters

**Rationale:** The 3×3 kernel perfectly matches the Game of Life's local neighborhood dependency.

## Evaluation

### 1. Single-Step Pixel Accuracy

Evaluate pixel-wise accuracy on test sets with different densities:

```bash
python scripts/evaluate_model.py experiments/cnn/model.pt
```

**Tests performed:**
- Random test set (30% density)
- Various densities (2%, 3%, 5%, 10%, 20%, 30%, 50%, 70%, 90%)
- Classic patterns (glider, blinker, toad, etc.)

**Output:** Generates side-by-side comparison GIFs in `figures/evaluation/`

### 2. Holdout Pattern Evaluation

Test the holdout-trained model on previously unseen patterns:

```bash
python scripts/evaluate_holdout.py experiments/cnn/model_holdout.pt
```

**Tests:**
- Validation set with holdout patterns included
- Analysis of whether unseen rules are correctly predicted

### 3. Multi-Step Autoregressive Evaluation

The evaluation script automatically generates 50-step autoregressive predictions:
- Model predicts next state
- Prediction is fed back as input
- Process repeats for 50 steps
- Compare with ground truth trajectory

**Metrics:**
- Visual comparison via GIFs
- Pattern preservation score
- Trajectory divergence analysis

### 4. Analyze Training Coverage

Check which 3×3 patterns were covered in training:

```bash
python scripts/analyze_training_coverage.py
```

## Results

### Key Experimental Findings

#### 1. Full Random Dataset Performance
- **Training Accuracy:** 100%
- **Validation Accuracy:** 100%
- **Test Accuracy (all densities):** 100%
- **Multi-step autoregressive:** Perfect for 50+ steps

#### 2. Holdout Dataset Performance
- **Model trained on incomplete data (with holdout patterns excluded)**
- **Test on holdout patterns:** 100% accuracy
- **Conclusion:** The model successfully inferred the complete Game of Life algorithm from incomplete training data

### Analysis

1. **The 3×3 Kernel is Critical:** The convolutional kernel size matches the Game of Life's local rule structure, providing a strong inductive bias.

2. **Perfect Generalization:** Unlike some prior work suggesting difficulty in learning CA rules, our simple CNN achieves perfect generalization. This may be due to:
   - Proper architectural choices (3×3 kernel, circular padding)
   - Sufficient training data coverage
   - Appropriate training procedure

3. **Not Just Memorization:** The holdout experiment proves the model learns the algorithm, not a lookup table of 512 rules.

## Next Steps

1. **Symbolic Regression:** Extract interpretable rules from the trained network
2. **More Complex Rules:** Test on non-standard cellular automata (different neighbor counts, states)
3. **Failure Cases:** Reproduce "failing" CNN scenarios from literature to understand critical success factors
4. **Network Analysis:** Investigate what the convolutional filters have learned

