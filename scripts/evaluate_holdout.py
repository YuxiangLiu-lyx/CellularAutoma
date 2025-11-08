"""
Evaluate model trained without holdout patterns.
Test on both data WITH and WITHOUT holdout patterns to measure generalization.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import json
import h5py
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN
from src.utils.game_of_life import GameOfLife


def load_holdout_patterns(holdout_file):
    """Load holdout patterns from JSON file."""
    with open(holdout_file, 'r') as f:
        data = json.load(f)
    return set(data['holdout_patterns'])


def extract_3x3_pattern_at(state, i, j):
    """Extract 3x3 pattern at position (i,j) with periodic boundaries."""
    h, w = state.shape
    neighborhood = np.zeros((3, 3), dtype=np.uint8)
    
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ni = (i + di) % h
            nj = (j + dj) % w
            neighborhood[di + 1, dj + 1] = state[ni, nj]
    
    return ''.join(neighborhood.flatten().astype(str))


def classify_predictions(states, predictions, ground_truth, holdout_patterns):
    """
    Classify each cell prediction as on holdout or non-holdout pattern.
    
    Args:
        states: Input states (N, H, W)
        predictions: Model predictions (N, H, W)
        ground_truth: True next states (N, H, W)
        holdout_patterns: Set of holdout pattern strings
        
    Returns:
        Dictionary with accuracy on holdout vs non-holdout patterns
    """
    n, h, w = states.shape
    
    holdout_correct = 0
    holdout_total = 0
    non_holdout_correct = 0
    non_holdout_total = 0
    
    for idx in range(n):
        state = states[idx]
        pred = predictions[idx]
        truth = ground_truth[idx]
        
        for i in range(h):
            for j in range(w):
                pattern = extract_3x3_pattern_at(state, i, j)
                is_correct = (pred[i, j] == truth[i, j])
                
                if pattern in holdout_patterns:
                    holdout_total += 1
                    if is_correct:
                        holdout_correct += 1
                else:
                    non_holdout_total += 1
                    if is_correct:
                        non_holdout_correct += 1
    
    return {
        'holdout_accuracy': holdout_correct / holdout_total if holdout_total > 0 else 0,
        'holdout_total': holdout_total,
        'non_holdout_accuracy': non_holdout_correct / non_holdout_total if non_holdout_total > 0 else 0,
        'non_holdout_total': non_holdout_total,
        'overall_accuracy': (holdout_correct + non_holdout_correct) / (holdout_total + non_holdout_total)
    }


def generate_test_data_with_holdout(num_samples, grid_size, density, 
                                     holdout_patterns, gol, seed=None):
    """Generate test data that contains holdout patterns."""
    if seed is not None:
        np.random.seed(seed)
    
    h, w = grid_size
    states = []
    
    # Generate random grids (will naturally contain some holdout patterns)
    for _ in range(num_samples):
        state = (np.random.random((h, w)) < density).astype(np.uint8)
        states.append(state)
    
    states = np.array(states)
    
    # Compute next states
    next_states = np.zeros_like(states)
    for i in range(len(states)):
        next_states[i] = gol.step(states[i])
    
    return states, next_states


def generate_test_data_without_holdout(num_samples, grid_size, density,
                                        holdout_patterns, gol, seed=None):
    """Generate test data that does NOT contain holdout patterns."""
    # Reuse the holdout dataset generation logic
    from scripts.generate_dataset_holdout import generate_random_states_holdout
    
    states = generate_random_states_holdout(
        num_samples, grid_size, density, holdout_patterns, seed=seed
    )
    
    # Compute next states
    next_states = np.zeros_like(states)
    for i in range(len(states)):
        next_states[i] = gol.step(states[i])
    
    return states, next_states


def evaluate_model(model, states, ground_truth, device):
    """Evaluate model on given states."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(len(states)):
            state_tensor = torch.FloatTensor(states[i:i+1]).unsqueeze(1).to(device)
            pred = model(state_tensor)
            pred_binary = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            predictions.append(pred_binary)
    
    predictions = np.array(predictions)
    
    # Compute pixel-wise accuracy
    accuracy = np.mean(predictions == ground_truth)
    
    return predictions, accuracy


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("Holdout Pattern Evaluation")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # Load holdout patterns
    holdout_file = project_root / "data" / "holdout_patterns.json"
    if not holdout_file.exists():
        print("\nError: Holdout patterns file not found!")
        return
    
    holdout_patterns = load_holdout_patterns(holdout_file)
    print(f"\nHoldout patterns: {len(holdout_patterns)}")
    
    # Load model trained on holdout data
    model_path = project_root / "experiments" / "models" / "game_of_life_cnn_holdout_best.pth"
    
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first:")
        print("  python src/models/train.py --data_dir data/processed_holdout --output_dir experiments_holdout")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GameOfLifeCNN(hidden_channels=16, padding_mode='circular').to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded model from: {model_path}")
        print(f"  Training epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Validation accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
    else:
        model.load_state_dict(checkpoint)
        print(f"\nLoaded model from: {model_path}")
    
    # Configuration
    grid_size = (32, 32)
    density = 0.3
    num_test = 1000
    gol = GameOfLife(grid_size)
    
    # Test 1: Data WITHOUT holdout patterns (should be high accuracy)
    print("\n" + "=" * 60)
    print("Test 1: Data WITHOUT Holdout Patterns")
    print("=" * 60)
    print("Generating test data (excluding holdout patterns)...")
    
    test_states_no_holdout, test_truth_no_holdout = generate_test_data_without_holdout(
        num_test, grid_size, density, holdout_patterns, gol, seed=100
    )
    
    print("Evaluating model...")
    preds_no_holdout, acc_no_holdout = evaluate_model(
        model, test_states_no_holdout, test_truth_no_holdout, device
    )
    
    print(f"\nOverall Accuracy: {acc_no_holdout*100:.2f}%")
    
    # Detailed analysis
    results_no_holdout = classify_predictions(
        test_states_no_holdout, preds_no_holdout, test_truth_no_holdout, holdout_patterns
    )
    
    print(f"  Non-holdout patterns: {results_no_holdout['non_holdout_accuracy']*100:.2f}% "
          f"({results_no_holdout['non_holdout_total']:,} predictions)")
    print(f"  Holdout patterns: {results_no_holdout['holdout_accuracy']*100:.2f}% "
          f"({results_no_holdout['holdout_total']:,} predictions)")
    print(f"  Note: Holdout patterns should be 0 or very few due to repair process")
    
    # Test 2: Data WITH holdout patterns (should show lower accuracy on holdout)
    print("\n" + "=" * 60)
    print("Test 2: Data WITH Holdout Patterns")
    print("=" * 60)
    print("Generating test data (including holdout patterns)...")
    
    test_states_with_holdout, test_truth_with_holdout = generate_test_data_with_holdout(
        num_test, grid_size, density, holdout_patterns, gol, seed=101
    )
    
    print("Evaluating model...")
    preds_with_holdout, acc_with_holdout = evaluate_model(
        model, test_states_with_holdout, test_truth_with_holdout, device
    )
    
    print(f"\nOverall Accuracy: {acc_with_holdout*100:.2f}%")
    
    # Detailed analysis
    results_with_holdout = classify_predictions(
        test_states_with_holdout, preds_with_holdout, test_truth_with_holdout, holdout_patterns
    )
    
    print(f"  Non-holdout patterns: {results_with_holdout['non_holdout_accuracy']*100:.2f}% "
          f"({results_with_holdout['non_holdout_total']:,} predictions)")
    print(f"  Holdout patterns: {results_with_holdout['holdout_accuracy']*100:.2f}% "
          f"({results_with_holdout['holdout_total']:,} predictions)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print(f"\nTest on data WITHOUT holdout patterns:")
    print(f"  Overall: {acc_no_holdout*100:.2f}%")
    
    print(f"\nTest on data WITH holdout patterns:")
    print(f"  Overall: {acc_with_holdout*100:.2f}%")
    print(f"  Non-holdout: {results_with_holdout['non_holdout_accuracy']*100:.2f}%")
    print(f"  Holdout: {results_with_holdout['holdout_accuracy']*100:.2f}%")
    
    holdout_percentage = results_with_holdout['holdout_total'] / (results_with_holdout['holdout_total'] + results_with_holdout['non_holdout_total']) * 100
    print(f"  Holdout coverage: {holdout_percentage:.1f}% of predictions")
    
    print("\n" + "=" * 60)
    print("Generalization Analysis")
    print("=" * 60)
    
    gap = results_with_holdout['non_holdout_accuracy'] - results_with_holdout['holdout_accuracy']
    
    print(f"  Accuracy gap: {gap*100:.2f}%")
    print(f"  Non-holdout accuracy: {results_with_holdout['non_holdout_accuracy']*100:.2f}%")
    print(f"  Holdout accuracy: {results_with_holdout['holdout_accuracy']*100:.2f}%")
    
    if gap < 0.05:
        print("\n  STRONG GENERALIZATION")
        print("  Model performs similarly on seen and unseen patterns")
    elif gap < 0.15:
        print("\n  MODERATE GENERALIZATION")
        print("  Model shows some ability to generalize")
    else:
        print("\n  WEAK GENERALIZATION")
        print("  Model performs worse on unseen patterns")


if __name__ == "__main__":
    main()

