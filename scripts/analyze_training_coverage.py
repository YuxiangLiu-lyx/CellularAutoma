"""
Analyze how many unique 3x3 patterns are covered in the training data
"""
import sys
from pathlib import Path
import numpy as np
import h5py
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))


def extract_3x3_patterns(state: np.ndarray) -> list:
    """
    Extract all 3x3 patterns from a grid state (with periodic boundaries).
    
    Args:
        state: 2D grid (H, W)
        
    Returns:
        List of pattern strings (each 3x3 pattern as 9-bit binary string)
    """
    h, w = state.shape
    patterns = []
    
    for i in range(h):
        for j in range(w):
            # Extract 3x3 neighborhood with periodic boundaries
            neighborhood = np.zeros((3, 3), dtype=np.uint8)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni = (i + di) % h
                    nj = (j + dj) % w
                    neighborhood[di + 1, dj + 1] = state[ni, nj]
            
            # Convert to binary string
            pattern_str = ''.join(neighborhood.flatten().astype(str))
            patterns.append(pattern_str)
    
    return patterns


def main():
    """Analyze training data pattern coverage."""
    project_root = Path(__file__).parent.parent
    train_file = project_root / "data" / "processed" / "train.h5"
    
    print("=" * 60)
    print("Training Data Pattern Coverage Analysis")
    print("=" * 60)
    
    # Load training data
    print("\nLoading training data...")
    with h5py.File(train_file, 'r') as f:
        states_t = f['states_t'][:]
        print(f"  Training samples: {len(states_t)}")
        print(f"  Grid size: {states_t.shape[1:3]}")
    
    # Extract all patterns
    print("\nExtracting 3x3 patterns from all training samples...")
    all_patterns = []
    for state in states_t:
        patterns = extract_3x3_patterns(state)
        all_patterns.extend(patterns)
    
    print(f"  Total pattern instances: {len(all_patterns):,}")
    
    # Count unique patterns
    unique_patterns = set(all_patterns)
    print(f"  Unique patterns: {len(unique_patterns)}")
    print(f"  Theoretical maximum: 512 (2^9)")
    print(f"  Coverage: {len(unique_patterns)/512*100:.2f}%")
    
    # Analyze pattern distribution
    pattern_counts = Counter(all_patterns)
    most_common = pattern_counts.most_common(10)
    least_common = pattern_counts.most_common()[-10:]
    
    print("\n" + "=" * 60)
    print("Pattern Distribution")
    print("=" * 60)
    print(f"\nMost common patterns:")
    for i, (pattern, count) in enumerate(most_common, 1):
        alive_cells = pattern.count('1')
        print(f"  {i}. Pattern '{pattern}' ({alive_cells}/9 alive): {count:,} times")
    
    print(f"\nLeast common patterns:")
    for i, (pattern, count) in enumerate(least_common, 1):
        alive_cells = pattern.count('1')
        print(f"  {i}. Pattern '{pattern}' ({alive_cells}/9 alive): {count} times")
    
    # Check for missing patterns
    all_possible = set(format(i, '09b') for i in range(512))
    missing_patterns = all_possible - unique_patterns
    
    print("\n" + "=" * 60)
    print("Missing Patterns Analysis")
    print("=" * 60)
    print(f"Missing patterns: {len(missing_patterns)}")
    
    if len(missing_patterns) > 0:
        print(f"\nExamples of missing patterns:")
        for i, pattern in enumerate(list(missing_patterns)[:20], 1):
            alive_cells = pattern.count('1')
            print(f"  {i}. Pattern '{pattern}' ({alive_cells}/9 alive)")
        
        # Analyze missing patterns by density
        missing_by_density = {}
        for pattern in missing_patterns:
            density = pattern.count('1')
            missing_by_density[density] = missing_by_density.get(density, 0) + 1
        
        print(f"\nMissing patterns by alive cell count:")
        for density in sorted(missing_by_density.keys()):
            print(f"  {density}/9 alive: {missing_by_density[density]} patterns missing")
    
    print("\n" + "=" * 60)
    print("Conclusion")
    print("=" * 60)
    
    if len(unique_patterns) == 512:
        print("FULL COVERAGE DETECTED")
        print("  - Training data covers ALL 512 possible 3x3 patterns")
        print("  - Model only needs to memorize these patterns")
        print("  - 100% accuracy is expected without true generalization")
    else:
        coverage = len(unique_patterns) / 512 * 100
        print(f"Partial coverage: {coverage:.1f}%")
        print(f"  - {len(missing_patterns)} patterns not seen during training")
        print(f"  - These can be used for OOD testing")


if __name__ == "__main__":
    main()

