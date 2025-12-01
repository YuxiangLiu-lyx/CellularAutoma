"""
Quick test to visualize different CNN architectures.
"""
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import (GameOfLifeCNN, MinimalGameOfLifeCNN, 
                            UltraMinimalGameOfLifeCNN, count_parameters)


def print_model_structure(name, model):
    """Print detailed model structure."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print('='*70)
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    test_input = torch.randn(1, 1, 32, 32)
    output = model(test_input)
    print(f"Input shape:  {tuple(test_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")


def main():
    print("="*70)
    print("GAME OF LIFE CNN ARCHITECTURES COMPARISON")
    print("="*70)
    
    models = [
        ("1. YOUR OLD ARCHITECTURE - Standard (h=16)", 
         GameOfLifeCNN(hidden_channels=16)),
        
        ("2. NEW ULTRA-MINIMAL (2 channels only)", 
         UltraMinimalGameOfLifeCNN()),
        
        ("3. NEW MINIMAL with hidden layer (h=8)", 
         MinimalGameOfLifeCNN(hidden_channels=8)),
        
        ("4. Standard with fewer channels (h=8)", 
         GameOfLifeCNN(hidden_channels=8)),
    ]
    
    for name, model in models:
        print_model_structure(name, model)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nOld architecture (what you trained before):")
    print("  GameOfLifeCNN(h=16): 177 parameters")
    print("  Structure: 1→16(3x3) → 16→1(1x1)")
    
    print("\nNew architectures (based on your idea):")
    print("  UltraMinimal: 23 parameters (2 channels)")
    print("    Structure: 1→2(3x3) → 2→1(1x1)")
    print("    Idea: Channel 0=center, Channel 1=neighbor count")
    
    print("\n  Minimal(h=8): 53 parameters")
    print("    Structure: 1→2(3x3) → 2→8(1x1) → 8→1(1x1)")
    print("    Idea: Extract center+neighbors, then combine with hidden layer")
    
    print("\n  Standard(h=8): 89 parameters")
    print("    Structure: 1→8(3x3) → 8→1(1x1)")
    print("    Idea: Let network learn 8 different features freely")


if __name__ == "__main__":
    main()

