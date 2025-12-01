"""
Test CNN model on ALL possible 3x3 patterns (exhaustive test).
Since Game of Life only depends on local 3x3 neighborhoods,
we can enumerate all 2^9 = 512 possible patterns and verify accuracy.
"""
import sys
from pathlib import Path
import torch
import numpy as np
from itertools import product

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN


def game_of_life_rule(pattern):
    """
    Apply Game of Life rule to center cell of a 3x3 pattern.
    
    Args:
        pattern: 3x3 numpy array (0 or 1)
        
    Returns:
        Next state of center cell (0 or 1)
    """
    center = pattern[1, 1]
    # Count alive neighbors (excluding center)
    alive_neighbors = np.sum(pattern) - center
    
    # Game of Life rules
    if center == 1:  # Cell is alive
        if alive_neighbors in [2, 3]:
            return 1  # Survive
        else:
            return 0  # Die
    else:  # Cell is dead
        if alive_neighbors == 3:
            return 1  # Birth
        else:
            return 0  # Stay dead


def generate_all_patterns():
    """
    Generate all possible 3x3 binary patterns.
    
    Returns:
        patterns: numpy array of shape (512, 3, 3)
        labels: numpy array of shape (512,) - expected center cell next state
    """
    patterns = []
    labels = []
    
    # Generate all 2^9 = 512 combinations
    for bits in product([0, 1], repeat=9):
        pattern = np.array(bits).reshape(3, 3).astype(np.float32)
        label = game_of_life_rule(pattern)
        
        patterns.append(pattern)
        labels.append(label)
    
    return np.array(patterns), np.array(labels)


def test_model_exhaustive(model, device):
    """
    Test model on all 512 possible 3x3 patterns.
    
    Args:
        model: CNN model
        device: torch device
        
    Returns:
        accuracy: float
        errors: list of (pattern, predicted, expected) tuples for errors
    """
    model.eval()
    
    # Generate all patterns
    patterns, labels = generate_all_patterns()
    
    print(f"Generated {len(patterns)} unique 3x3 patterns")
    print(f"Label distribution: {np.sum(labels)} alive, {len(labels) - np.sum(labels)} dead")
    
    # Convert to torch tensors
    # Add batch and channel dimensions: (512, 3, 3) -> (512, 1, 3, 3)
    patterns_tensor = torch.from_numpy(patterns[:, np.newaxis, :, :]).to(device)
    labels_tensor = torch.from_numpy(labels).float().to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(patterns_tensor)  # (512, 1, 3, 3)
        
        # Extract center cell predictions
        center_outputs = outputs[:, 0, 1, 1]  # (512,)
        
        # Apply threshold
        predictions = (center_outputs > 0.5).float()
        
        # Compute accuracy
        correct = (predictions == labels_tensor).sum().item()
        total = len(labels)
        accuracy = correct / total
        
        # Find errors
        errors = []
        for i in range(len(patterns)):
            if predictions[i] != labels_tensor[i]:
                errors.append({
                    'pattern': patterns[i],
                    'predicted': int(predictions[i].item()),
                    'expected': int(labels[i]),
                    'output_value': float(center_outputs[i].item()),
                    'pattern_id': i
                })
    
    return accuracy, errors


def analyze_errors(errors):
    """Analyze and display error patterns."""
    if not errors:
        print("\n✓✓✓ NO ERRORS - Perfect on all 512 patterns!")
        return
    
    print(f"\n✗ Found {len(errors)} errors out of 512 patterns")
    print(f"  Error rate: {len(errors)/512*100:.2f}%")
    
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    
    # Group errors by type
    false_positives = [e for e in errors if e['predicted'] == 1 and e['expected'] == 0]
    false_negatives = [e for e in errors if e['predicted'] == 0 and e['expected'] == 1]
    
    print(f"\nError types:")
    print(f"  False Positives (predicted alive, should be dead): {len(false_positives)}")
    print(f"  False Negatives (predicted dead, should be alive): {len(false_negatives)}")
    
    print("\n" + "-"*70)
    print("Detailed Errors:")
    print("-"*70)
    
    for i, error in enumerate(errors[:20], 1):  # Show first 20 errors
        pattern = error['pattern']
        center = pattern[1, 1]
        alive_neighbors = int(np.sum(pattern) - center)
        
        print(f"\nError #{i} (Pattern ID: {error['pattern_id']}):")
        print(f"  Pattern:")
        for row in pattern:
            print(f"    {int(row[0])} {int(row[1])} {int(row[2])}")
        print(f"  Center: {int(center)}, Alive neighbors: {alive_neighbors}")
        print(f"  Predicted: {error['predicted']}, Expected: {error['expected']}")
        print(f"  Model output: {error['output_value']:.6f} (threshold: 0.5)")
        
        # Explain the rule
        if center == 1:
            print(f"  Rule: Alive cell with {alive_neighbors} neighbors → ", end="")
            if alive_neighbors in [2, 3]:
                print("survives")
            else:
                print("dies")
        else:
            print(f"  Rule: Dead cell with {alive_neighbors} neighbors → ", end="")
            if alive_neighbors == 3:
                print("born")
            else:
                print("stays dead")
    
    if len(errors) > 20:
        print(f"\n... and {len(errors) - 20} more errors")


def test_specific_cases(model, device):
    """Test specific important Game of Life patterns."""
    print("\n" + "="*70)
    print("TESTING SPECIFIC GAME OF LIFE CASES")
    print("="*70)
    
    test_cases = [
        {
            'name': 'Dead cell, 3 neighbors (birth)',
            'pattern': np.array([[1, 1, 0],
                                [1, 0, 0],
                                [0, 0, 0]], dtype=np.float32),
            'expected': 1
        },
        {
            'name': 'Alive cell, 2 neighbors (survive)',
            'pattern': np.array([[1, 1, 0],
                                [0, 1, 0],
                                [0, 0, 0]], dtype=np.float32),
            'expected': 1
        },
        {
            'name': 'Alive cell, 3 neighbors (survive)',
            'pattern': np.array([[1, 1, 0],
                                [1, 1, 0],
                                [0, 0, 0]], dtype=np.float32),
            'expected': 1
        },
        {
            'name': 'Alive cell, 1 neighbor (die - underpopulation)',
            'pattern': np.array([[0, 1, 0],
                                [0, 1, 0],
                                [0, 0, 0]], dtype=np.float32),
            'expected': 0
        },
        {
            'name': 'Alive cell, 4 neighbors (die - overpopulation)',
            'pattern': np.array([[1, 1, 0],
                                [1, 1, 1],
                                [0, 0, 0]], dtype=np.float32),
            'expected': 0
        },
        {
            'name': 'Dead cell, 2 neighbors (stay dead)',
            'pattern': np.array([[1, 1, 0],
                                [0, 0, 0],
                                [0, 0, 0]], dtype=np.float32),
            'expected': 0
        },
        {
            'name': 'All alive (8 neighbors, center dies)',
            'pattern': np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]], dtype=np.float32),
            'expected': 0
        },
        {
            'name': 'All dead (0 neighbors, stays dead)',
            'pattern': np.array([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]], dtype=np.float32),
            'expected': 0
        },
    ]
    
    model.eval()
    all_correct = True
    
    for case in test_cases:
        pattern = case['pattern']
        expected = case['expected']
        
        # Add batch and channel dimensions
        input_tensor = torch.from_numpy(pattern[np.newaxis, np.newaxis, :, :]).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            center_output = output[0, 0, 1, 1].item()
            prediction = 1 if center_output > 0.5 else 0
        
        correct = (prediction == expected)
        status = "✓" if correct else "✗"
        all_correct = all_correct and correct
        
        print(f"\n{status} {case['name']}")
        print(f"  Pattern:")
        for row in pattern:
            print(f"    {int(row[0])} {int(row[1])} {int(row[2])}")
        print(f"  Expected: {expected}, Predicted: {prediction}")
        print(f"  Model output: {center_output:.6f}")
        
        if not correct:
            print(f"  ✗ ERROR!")
    
    return all_correct


def main():
    """Main testing function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CNN on all possible 3x3 patterns')
    parser.add_argument('--model', type=str, 
                       default='experiments/2ch_convergence/models/run_01_seed_300.pth',
                       help='Path to model checkpoint')
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    model_path = project_root / args.model
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("EXHAUSTIVE 3x3 PATTERN TEST")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {model_path.name}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(f"\nModel Information:")
    print(f"  Hidden Channels: {checkpoint['hidden_channels']}")
    print(f"  Convergence Epoch: {checkpoint['convergence_epoch']}")
    print(f"  Validation Accuracy: {checkpoint['val_accuracy']:.6f}")
    
    model = GameOfLifeCNN(
        hidden_channels=checkpoint['hidden_channels'],
        padding_mode='circular'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"\n{'='*70}")
    print("EXHAUSTIVE TEST: All 512 possible 3x3 patterns")
    print('='*70)
    
    # Test all patterns
    accuracy, errors = test_model_exhaustive(model, device)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print('='*70)
    print(f"Total patterns: 512")
    print(f"Correct: {512 - len(errors)}")
    print(f"Errors: {len(errors)}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    # Analyze errors
    analyze_errors(errors)
    
    # Test specific cases
    specific_correct = test_specific_cases(model, device)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    if accuracy == 1.0:
        print("✓✓✓ PERFECT: Model correctly predicts ALL 512 patterns!")
        print(f"    The {checkpoint['hidden_channels']}-channel CNN has fully learned Game of Life rules!")
    elif accuracy >= 0.99:
        print(f"✓✓ Excellent: {accuracy*100:.2f}% accuracy")
        print(f"   Only {len(errors)} errors out of 512 patterns")
    elif accuracy >= 0.95:
        print(f"✓ Good: {accuracy*100:.2f}% accuracy")
        print(f"   {len(errors)} errors need investigation")
    else:
        print(f"✗ Needs improvement: {accuracy*100:.2f}% accuracy")
        print(f"   {len(errors)} errors - model hasn't fully learned the rules")
    
    print("\n" + "="*70)
    
    # Save results
    output_dir = model_path.parent.parent / "exhaustive_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model': str(model_path.name),
        'hidden_channels': int(checkpoint['hidden_channels']),
        'total_patterns': 512,
        'correct': 512 - len(errors),
        'errors': len(errors),
        'accuracy': float(accuracy),
        'error_patterns': errors
    }
    
    import json
    results_path = output_dir / f"{checkpoint['hidden_channels']}ch_exhaustive_test.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
