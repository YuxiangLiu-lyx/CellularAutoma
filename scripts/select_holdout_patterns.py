"""
Select and save patterns to hold out from training.
Mix of specific patterns (from known structures) and random patterns.
"""
import sys
from pathlib import Path
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.game_of_life import GameOfLife
from src.utils.patterns import GLIDER, BLINKER, TOAD, LWSS, BLOCK, BEEHIVE


def extract_3x3_from_pattern(pattern, gol, num_steps=10):
    """
    Extract all unique 3x3 patterns from a pattern's evolution.
    
    Args:
        pattern: 2D array of the pattern
        gol: GameOfLife simulator
        num_steps: Number of evolution steps to simulate
        
    Returns:
        Set of 3x3 pattern strings
    """
    # Place pattern in center of a larger grid
    grid_size = (20, 20)
    h, w = pattern.shape
    grid = np.zeros(grid_size, dtype=np.uint8)
    start_h = (grid_size[0] - h) // 2
    start_w = (grid_size[1] - w) // 2
    grid[start_h:start_h+h, start_w:start_w+w] = pattern
    
    patterns_3x3 = set()
    
    # Simulate and extract patterns
    current = grid.copy()
    for step in range(num_steps):
        # Extract all 3x3 patterns
        for i in range(1, grid_size[0]-1):
            for j in range(1, grid_size[1]-1):
                window = current[i-1:i+2, j-1:j+2]
                pattern_str = ''.join(window.flatten().astype(str))
                patterns_3x3.add(pattern_str)
        
        current = gol.step(current)
    
    return patterns_3x3


def select_specific_patterns():
    """Select patterns from known Game of Life structures."""
    print("\n" + "="*60)
    print("1. Extracting patterns from known structures")
    print("="*60)
    
    gol = GameOfLife((20, 20))
    
    structures = {
        'glider': GLIDER,
        'blinker': BLINKER,
        'toad': TOAD,
        'lwss': LWSS,
        'beehive': BEEHIVE,
    }
    
    all_specific = set()
    structure_patterns = {}
    
    for name, pattern in structures.items():
        patterns = extract_3x3_from_pattern(pattern, gol, num_steps=20)
        structure_patterns[name] = patterns
        all_specific.update(patterns)
        print(f"  {name}: {len(patterns)} unique 3x3 patterns")
    
    # Select representative patterns from each structure
    selected_specific = set()
    
    # From glider: select 4 patterns
    glider_patterns = list(structure_patterns['glider'])
    selected_specific.update(glider_patterns[:4])
    
    # From blinker: select 2 patterns
    blinker_patterns = list(structure_patterns['blinker'])
    selected_specific.update(blinker_patterns[:2])
    
    # From toad: select 2 patterns
    toad_patterns = list(structure_patterns['toad'])
    selected_specific.update(toad_patterns[:2])
    
    # From lwss: select 3 patterns
    lwss_patterns = list(structure_patterns['lwss'])
    selected_specific.update(lwss_patterns[:3])
    
    print(f"\n  Selected {len(selected_specific)} specific patterns")
    
    return selected_specific, structure_patterns


def select_random_patterns(num_random=15, seed=42):
    """Select random patterns from the 512 possible patterns."""
    print("\n" + "="*60)
    print("2. Selecting random patterns")
    print("="*60)
    
    np.random.seed(seed)
    
    # Generate all 512 patterns
    all_patterns = [format(i, '09b') for i in range(512)]
    
    # Exclude trivial patterns (all 0s or all 1s)
    all_patterns.remove('000000000')
    all_patterns.remove('111111111')
    
    # Randomly select
    selected_random = set(np.random.choice(all_patterns, num_random, replace=False))
    
    print(f"  Selected {len(selected_random)} random patterns")
    
    # Show examples
    print("\n  Examples:")
    for i, pattern in enumerate(list(selected_random)[:5], 1):
        alive = pattern.count('1')
        print(f"    {i}. '{pattern}' ({alive}/9 alive)")
    
    return selected_random


def analyze_holdout_patterns(holdout_patterns):
    """Analyze the selected holdout patterns."""
    print("\n" + "="*60)
    print("3. Analyzing holdout patterns")
    print("="*60)
    
    print(f"  Total holdout patterns: {len(holdout_patterns)}")
    print(f"  Percentage of 512: {len(holdout_patterns)/512*100:.1f}%")
    
    # Analyze by alive cell count
    by_density = {}
    for pattern in holdout_patterns:
        alive = pattern.count('1')
        by_density[alive] = by_density.get(alive, 0) + 1
    
    print("\n  Distribution by alive cells:")
    for alive in sorted(by_density.keys()):
        count = by_density[alive]
        print(f"    {alive}/9 alive: {count} patterns")


def save_holdout_patterns(holdout_patterns, output_file):
    """Save holdout patterns to file."""
    # Convert set to sorted list
    patterns_list = sorted(list(holdout_patterns))
    
    # Save as JSON
    data = {
        'holdout_patterns': patterns_list,
        'num_patterns': len(patterns_list),
        'percentage': len(patterns_list) / 512 * 100,
        'description': 'Patterns held out from training to test generalization'
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n  Saved to: {output_file}")


def main():
    """Main function to select and save holdout patterns."""
    print("="*60)
    print("Holdout Pattern Selection")
    print("="*60)
    
    # Select specific patterns from known structures
    specific_patterns, structure_patterns = select_specific_patterns()
    
    # Select random patterns
    random_patterns = select_random_patterns(num_random=15, seed=42)
    
    # Combine
    holdout_patterns = specific_patterns.union(random_patterns)
    
    # Remove overlap
    print(f"\n  Overlap between specific and random: {len(specific_patterns.intersection(random_patterns))}")
    
    # Analyze
    analyze_holdout_patterns(holdout_patterns)
    
    # Save
    project_root = Path(__file__).parent.parent
    output_file = project_root / "data" / "holdout_patterns.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_holdout_patterns(holdout_patterns, output_file)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"  Total holdout patterns: {len(holdout_patterns)}")
    print(f"  - From specific structures: {len(specific_patterns)}")
    print(f"  - Random: {len(random_patterns)}")
    print(f"  - Overlap: {len(specific_patterns.intersection(random_patterns))}")
    print(f"\n  Training will use: {512 - len(holdout_patterns)} patterns")
    print(f"  Holdout for testing: {len(holdout_patterns)} patterns")


if __name__ == "__main__":
    main()

