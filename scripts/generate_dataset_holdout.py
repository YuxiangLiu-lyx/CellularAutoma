"""
Generate training dataset with holdout patterns excluded.
Uses repair-based rejection sampling for efficiency.
"""
import sys
from pathlib import Path
import numpy as np
import h5py
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.game_of_life import GameOfLife


def load_holdout_patterns(holdout_file):
    """Load holdout patterns from JSON file."""
    with open(holdout_file, 'r') as f:
        data = json.load(f)
    return set(data['holdout_patterns'])


def extract_3x3_patterns_with_positions(state):
    """
    Extract all 3x3 patterns with their positions (with periodic boundaries).
    
    Args:
        state: 2D grid (H, W)
        
    Returns:
        List of (i, j, pattern_str) tuples
    """
    h, w = state.shape
    patterns_with_pos = []
    
    for i in range(h):
        for j in range(w):
            # Extract 3x3 neighborhood with periodic boundaries
            neighborhood = np.zeros((3, 3), dtype=np.uint8)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni = (i + di) % h
                    nj = (j + dj) % w
                    neighborhood[di + 1, dj + 1] = state[ni, nj]
            
            pattern_str = ''.join(neighborhood.flatten().astype(str))
            patterns_with_pos.append((i, j, pattern_str))
    
    return patterns_with_pos


def check_violations(state, holdout_patterns):
    """
    Check which positions violate the holdout constraint.
    
    Args:
        state: 2D grid
        holdout_patterns: Set of forbidden pattern strings
        
    Returns:
        List of (i, j) positions that violate
    """
    patterns_with_pos = extract_3x3_patterns_with_positions(state)
    violations = []
    
    for i, j, pattern in patterns_with_pos:
        if pattern in holdout_patterns:
            violations.append((i, j))
    
    return violations


def repair_grid(state, holdout_patterns, max_attempts=100):
    """
    Repair a grid to remove holdout patterns by flipping cells.
    
    Args:
        state: 2D grid to repair
        holdout_patterns: Set of forbidden patterns
        max_attempts: Maximum repair attempts
        
    Returns:
        Repaired grid, or None if repair failed
    """
    state = state.copy()
    
    for attempt in range(max_attempts):
        violations = check_violations(state, holdout_patterns)
        
        if len(violations) == 0:
            return state  # Success!
        
        # Randomly select one violation to fix
        i, j = violations[np.random.randint(len(violations))]
        
        # Flip the center cell
        state[i, j] = 1 - state[i, j]
    
    return None  # Failed to repair


def generate_random_states_holdout(num_samples, grid_size, density, 
                                   holdout_patterns, seed=None, max_retries=10):
    """
    Generate random states without holdout patterns.
    
    Args:
        num_samples: Number of states to generate
        grid_size: Grid dimensions
        density: Probability of alive cell
        holdout_patterns: Set of forbidden patterns
        seed: Random seed
        max_retries: Max retries per sample
        
    Returns:
        Array of states (num_samples, H, W)
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = grid_size
    states = []
    
    with tqdm(total=num_samples, desc="Generating samples") as pbar:
        while len(states) < num_samples:
            # Generate random grid
            state = (np.random.random((h, w)) < density).astype(np.uint8)
            
            # Try to repair
            repaired = repair_grid(state, holdout_patterns, max_attempts=100)
            
            if repaired is not None:
                states.append(repaired)
                pbar.update(1)
            else:
                # Retry with new random state
                continue
    
    return np.array(states)


def generate_single_step_dataset(initial_states, gol):
    """Generate (state_t, state_t+1) pairs."""
    num_samples = len(initial_states)
    states_t = initial_states
    states_t1 = np.zeros_like(states_t)
    
    for i in tqdm(range(num_samples), desc="Computing next states"):
        states_t1[i] = gol.step(states_t[i])
    
    return states_t, states_t1


def save_dataset(file_path, states_t, states_t1, metadata=None):
    """Save dataset to HDF5 file."""
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('states_t', data=states_t, compression='gzip')
        f.create_dataset('states_t1', data=states_t1, compression='gzip')
        
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value
    
    print(f"  Saved: {file_path}")
    print(f"  Samples: {len(states_t)}")
    print(f"  Size: {Path(file_path).stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Generate holdout datasets."""
    print("=" * 60)
    print("Holdout Dataset Generation")
    print("=" * 60)
    
    # Load holdout patterns
    project_root = Path(__file__).parent.parent
    holdout_file = project_root / "data" / "holdout_patterns.json"
    
    if not holdout_file.exists():
        print(f"\nError: Holdout patterns file not found!")
        print(f"Please run: python scripts/select_holdout_patterns.py")
        return
    
    holdout_patterns = load_holdout_patterns(holdout_file)
    print(f"\nLoaded {len(holdout_patterns)} holdout patterns")
    print(f"Training will use {512 - len(holdout_patterns)} patterns")
    
    # Configuration
    grid_size = (32, 32)
    density = 0.3
    num_train = 10000
    num_val = 2000
    
    output_dir = project_root / "data" / "processed_holdout"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gol = GameOfLife(grid_size)
    
    print(f"\nConfiguration:")
    print(f"  Grid size: {grid_size}")
    print(f"  Density: {density}")
    print(f"  Training samples: {num_train}")
    print(f"  Validation samples: {num_val}")
    
    # Generate training set
    print("\n" + "=" * 60)
    print("1. Generating training set (excluding holdout patterns)...")
    print("=" * 60)
    train_states = generate_random_states_holdout(
        num_train, grid_size, density, holdout_patterns, seed=42
    )
    train_t, train_t1 = generate_single_step_dataset(train_states, gol)
    
    save_dataset(
        output_dir / "train.h5",
        train_t, train_t1,
        metadata={
            'num_samples': num_train,
            'grid_size': grid_size,
            'density': density,
            'holdout_patterns': len(holdout_patterns),
            'description': 'Random states excluding holdout patterns'
        }
    )
    
    # Generate validation set
    print("\n" + "=" * 60)
    print("2. Generating validation set (excluding holdout patterns)...")
    print("=" * 60)
    val_states = generate_random_states_holdout(
        num_val, grid_size, density, holdout_patterns, seed=43
    )
    val_t, val_t1 = generate_single_step_dataset(val_states, gol)
    
    save_dataset(
        output_dir / "val.h5",
        val_t, val_t1,
        metadata={
            'num_samples': num_val,
            'grid_size': grid_size,
            'density': density,
            'holdout_patterns': len(holdout_patterns),
            'description': 'Random states excluding holdout patterns'
        }
    )
    
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"\nGenerated files in {output_dir.absolute()}:")
    print(f"  - train.h5: {num_train} samples (no holdout patterns)")
    print(f"  - val.h5: {num_val} samples (no holdout patterns)")


if __name__ == "__main__":
    main()

