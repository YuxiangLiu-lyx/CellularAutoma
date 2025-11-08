"""
Analyze if test data is truly out-of-distribution from training data.
Check if local 3×3 patterns in test data appear in training data.
"""
import sys
from pathlib import Path
import numpy as np
from collections import Counter
import h5py

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.game_of_life import GameOfLife


def extract_3x3_patterns(state):
    """
    Extract all 3×3 local patterns from a state.
    
    Returns:
        List of 3×3 patterns as tuples (hashable)
    """
    h, w = state.shape
    patterns = []
    
    for i in range(h - 2):
        for j in range(w - 2):
            pattern = state[i:i+3, j:j+3]
            # Convert to tuple for hashing
            patterns.append(tuple(pattern.flatten()))
    
    return patterns


def analyze_density_distribution(density, num_samples=1000):
    """Analyze 3×3 pattern distribution for a given density."""
    gol = GameOfLife(grid_size=(32, 32))
    
    all_patterns = []
    
    for i in range(num_samples):
        np.random.seed(10000 + i)
        state = (np.random.random((32, 32)) < density).astype(np.uint8)
        patterns = extract_3x3_patterns(state)
        all_patterns.extend(patterns)
    
    return Counter(all_patterns)


def main():
    """Analyze if test patterns overlap with training patterns."""
    
    print("="*70)
    print("Pattern Distribution Analysis")
    print("="*70)
    print("Question: Are test data truly OOD from training data?")
    print("Method: Compare 3×3 local pattern distributions")
    print("="*70)
    
    # Analyze training density (30%)
    print("\nAnalyzing training density (30%)...")
    train_patterns = analyze_density_distribution(0.30, num_samples=1000)
    train_unique = len(train_patterns)
    train_total = sum(train_patterns.values())
    
    print(f"  Total 3×3 patterns: {train_total:,}")
    print(f"  Unique patterns: {train_unique:,}")
    print(f"  Most common patterns:")
    for pattern, count in train_patterns.most_common(5):
        prob = count / train_total
        alive_count = sum(pattern)
        print(f"    Pattern with {alive_count} alive cells: {count:,} ({prob*100:.2f}%)")
    
    # Analyze test densities
    test_densities = [0.02, 0.05, 0.10, 0.50, 0.70, 0.90]
    
    print("\n" + "="*70)
    print("Overlap Analysis")
    print("="*70)
    
    results = []
    
    for density in test_densities:
        print(f"\nTest density: {density:.0%}")
        test_patterns = analyze_density_distribution(density, num_samples=1000)
        test_unique = len(test_patterns)
        test_total = sum(test_patterns.values())
        
        # Calculate overlap
        overlap_patterns = set(test_patterns.keys()) & set(train_patterns.keys())
        overlap_count = sum(test_patterns[p] for p in overlap_patterns)
        overlap_ratio = overlap_count / test_total
        
        # Calculate pattern coverage
        unique_to_test = set(test_patterns.keys()) - set(train_patterns.keys())
        coverage = len(overlap_patterns) / test_unique if test_unique > 0 else 0
        
        print(f"  Unique patterns: {test_unique:,}")
        print(f"  Overlap with training: {len(overlap_patterns):,} patterns ({coverage*100:.1f}%)")
        print(f"  Overlap coverage: {overlap_ratio*100:.1f}% of test instances")
        print(f"  New patterns not in training: {len(unique_to_test):,}")
        
        results.append({
            'density': density,
            'unique': test_unique,
            'overlap_patterns': len(overlap_patterns),
            'overlap_ratio': overlap_ratio,
            'new_patterns': len(unique_to_test)
        })
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"{'Density':<12} {'Unique':<10} {'Overlap%':<12} {'Coverage%':<12} {'New':<10}")
    print('-' * 70)
    
    for r in results:
        coverage = r['overlap_patterns'] / r['unique'] * 100 if r['unique'] > 0 else 0
        print(f"{r['density']:<12.2f} {r['unique']:<10} "
              f"{r['overlap_ratio']*100:<12.1f} {coverage:<12.1f} {r['new_patterns']:<10}")
    
    print("\n" + "="*70)
    print("Interpretation")
    print("="*70)
    
    high_overlap = all(r['overlap_ratio'] > 0.95 for r in results)
    
    if high_overlap:
        print("HIGH OVERLAP DETECTED")
        print("  - >95% of test patterns appear in training data")
        print("  - Test data is NOT truly out-of-distribution")
        print("  - Model may be memorizing local 3x3 patterns")
    else:
        print("Patterns are sufficiently different")
        print("  - Test data contains many new patterns")
    
    print("="*70)


if __name__ == "__main__":
    main()

