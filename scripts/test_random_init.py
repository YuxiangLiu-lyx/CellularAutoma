"""
Test randomly initialized models (no training) on all 512 patterns.
This shows baseline performance before learning.
"""
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN
from test_all_patterns import test_model_exhaustive, analyze_errors, test_specific_cases


def main():
    """Test random initialized models."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("TESTING RANDOMLY INITIALIZED MODELS (No Training)")
    print("="*70)
    print(f"Device: {device}\n")
    
    channels_to_test = [2, 3, 4, 8, 16]
    
    for hidden_channels in channels_to_test:
        print("\n" + "="*70)
        print(f"Testing {hidden_channels}-channel CNN (Random Initialization)")
        print("="*70)
        
        # Create random model
        torch.manual_seed(42)  # For reproducibility
        model = GameOfLifeCNN(hidden_channels=hidden_channels, padding_mode='circular')
        model = model.to(device)
        
        print(f"Model: {hidden_channels}-channel CNN")
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Status: UNTRAINED (random initialization)")
        
        # Test on all patterns
        print(f"\nTesting on all 512 patterns...")
        accuracy, errors = test_model_exhaustive(model, device)
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print('='*70)
        print(f"Total patterns: 512")
        print(f"Correct: {512 - len(errors)}")
        print(f"Errors: {len(errors)}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        
        # Brief error summary
        if errors:
            false_positives = [e for e in errors if e['predicted'] == 1 and e['expected'] == 0]
            false_negatives = [e for e in errors if e['predicted'] == 0 and e['expected'] == 1]
            
            print(f"\nError types:")
            print(f"  False Positives: {len(false_positives)}")
            print(f"  False Negatives: {len(false_negatives)}")
        else:
            print("\n✓ PERFECT - No errors!")
        
        # Test specific cases (brief)
        print(f"\nKey test cases:")
        test_cases_brief = [
            ("Birth (3 neighbors)", [[1,1,0],[1,0,0],[0,0,0]], 1),
            ("Survive (2 neighbors)", [[1,1,0],[0,1,0],[0,0,0]], 1),
            ("Die (1 neighbor)", [[0,1,0],[0,1,0],[0,0,0]], 0),
            ("Die (4 neighbors)", [[1,1,0],[1,1,1],[0,0,0]], 0),
        ]
        
        model.eval()
        for name, pattern, expected in test_cases_brief:
            import numpy as np
            pattern_tensor = torch.from_numpy(
                np.array(pattern, dtype=np.float32)[np.newaxis, np.newaxis, :, :]
            ).to(device)
            
            with torch.no_grad():
                output = model(pattern_tensor)
                center_output = output[0, 0, 1, 1].item()
                prediction = 1 if center_output > 0.5 else 0
            
            status = "✓" if prediction == expected else "✗"
            print(f"  {status} {name}: pred={prediction}, expected={expected}, output={center_output:.3f}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Random initialization (before training) shows baseline performance.")
    print("Compare this with trained models to see learning progress!")
    print("="*70)


if __name__ == "__main__":
    main()
