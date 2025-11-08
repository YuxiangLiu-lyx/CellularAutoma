"""
Generate training and test datasets for Game of Life prediction
"""
import sys
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.game_of_life import GameOfLife, place_pattern
from src.utils.patterns import get_pattern, PATTERN_CATEGORIES


def generate_random_states(num_samples, grid_size=(32, 32), density=0.3, seed=None):
    """
    Generate random initial states.
    
    Args:
        num_samples: Number of states to generate
        grid_size: Grid dimensions
        density: Probability of alive cell
        seed: Random seed
        
    Returns:
        Array of random states (num_samples, H, W)
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = grid_size
    states = (np.random.random((num_samples, h, w)) < density).astype(np.uint8)
    return states


def generate_single_step_dataset(initial_states, gol):
    """
    Generate (state_t, state_t+1) pairs from initial states.
    
    Args:
        initial_states: Array of initial states (N, H, W)
        gol: GameOfLife simulator
        
    Returns:
        states_t, states_t1: Current and next states
    """
    num_samples = len(initial_states)
    states_t = initial_states
    states_t1 = np.zeros_like(states_t)
    
    for i in tqdm(range(num_samples), desc="Evolving states"):
        states_t1[i] = gol.step(states_t[i])
    
    return states_t, states_t1


def generate_pattern_test_set(gol, grid_size=(32, 32), num_steps=50):
    """
    Generate test set with specific patterns for multi-step evaluation.
    
    Args:
        gol: GameOfLife simulator
        grid_size: Grid dimensions
        num_steps: Number of evolution steps
        
    Returns:
        Dictionary with pattern trajectories
    """
    pattern_data = {}
    
    patterns_to_test = {
        'still_lifes': ['block', 'beehive', 'boat'],
        'oscillators_p2': ['blinker', 'toad', 'beacon'],
        'spaceships': ['glider', 'lwss']
    }
    
    num_samples_per_pattern = {
        'still_lifes': 50,
        'oscillators_p2': 50,
        'spaceships': 100
    }
    
    for category, pattern_names in patterns_to_test.items():
        for pattern_name in pattern_names:
            print(f"  Generating {pattern_name}...")
            pattern = get_pattern(pattern_name)
            
            num_samples = num_samples_per_pattern[category]
            trajectories = []
            
            for _ in range(num_samples):
                # Random position
                ph, pw = pattern.shape
                max_h = grid_size[0] - ph - 1
                max_w = grid_size[1] - pw - 1
                
                if max_h > 0 and max_w > 0:
                    pos = (np.random.randint(1, max_h), 
                           np.random.randint(1, max_w))
                else:
                    pos = None
                
                initial_state = place_pattern(grid_size, pattern, position=pos)
                trajectory = gol.simulate(initial_state, num_steps)
                trajectories.append(trajectory)
            
            pattern_data[pattern_name] = {
                'trajectories': np.array(trajectories),
                'category': category
            }
    
    return pattern_data


def save_dataset(file_path, states_t, states_t1, metadata=None):
    """
    Save dataset to HDF5 file.
    
    Args:
        file_path: Output file path
        states_t: Current states (N, H, W)
        states_t1: Next states (N, H, W)
        metadata: Optional metadata dictionary
    """
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('states_t', data=states_t, compression='gzip')
        f.create_dataset('states_t1', data=states_t1, compression='gzip')
        
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value
    
    print(f"Saved to {file_path}")
    print(f"  Shape: {states_t.shape}")
    print(f"  Size: {Path(file_path).stat().st_size / 1024 / 1024:.2f} MB")


def save_pattern_dataset(file_path, pattern_data):
    """
    Save pattern test set to HDF5 file.
    
    Args:
        file_path: Output file path
        pattern_data: Dictionary of pattern trajectories
    """
    with h5py.File(file_path, 'w') as f:
        for pattern_name, data in pattern_data.items():
            grp = f.create_group(pattern_name)
            grp.create_dataset('trajectories', 
                             data=data['trajectories'], 
                             compression='gzip')
            grp.attrs['category'] = data['category']
    
    print(f"Saved to {file_path}")
    print(f"  Patterns: {list(pattern_data.keys())}")
    print(f"  Size: {Path(file_path).stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Generate all datasets."""
    
    print("=" * 60)
    print("Game of Life Dataset Generation")
    print("=" * 60)
    
    # Configuration
    grid_size = (32, 32)
    density = 0.3
    num_train = 10000
    num_val = 2000
    num_test = 2000
    
    # Use absolute path to project root
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gol = GameOfLife(grid_size)
    
    print(f"\nConfiguration:")
    print(f"  Grid size: {grid_size}")
    print(f"  Density: {density}")
    print(f"  Training samples: {num_train}")
    print(f"  Validation samples: {num_val}")
    print(f"  Test samples: {num_test}")
    
    # Generate training set
    print("\n" + "=" * 60)
    print("1. Generating training set (random states)...")
    print("=" * 60)
    train_states = generate_random_states(num_train, grid_size, density, seed=42)
    train_t, train_t1 = generate_single_step_dataset(train_states, gol)
    
    save_dataset(
        output_dir / "train.h5",
        train_t, train_t1,
        metadata={
            'num_samples': num_train,
            'grid_size': grid_size,
            'density': density,
            'description': 'Random initial states for training'
        }
    )
    
    # Generate validation set
    print("\n" + "=" * 60)
    print("2. Generating validation set (random states)...")
    print("=" * 60)
    val_states = generate_random_states(num_val, grid_size, density, seed=43)
    val_t, val_t1 = generate_single_step_dataset(val_states, gol)
    
    save_dataset(
        output_dir / "val.h5",
        val_t, val_t1,
        metadata={
            'num_samples': num_val,
            'grid_size': grid_size,
            'density': density,
            'description': 'Random initial states for validation'
        }
    )
    
    # Generate test set (random)
    print("\n" + "=" * 60)
    print("3. Generating test set (random states)...")
    print("=" * 60)
    test_states = generate_random_states(num_test, grid_size, density, seed=44)
    test_t, test_t1 = generate_single_step_dataset(test_states, gol)
    
    save_dataset(
        output_dir / "test_random.h5",
        test_t, test_t1,
        metadata={
            'num_samples': num_test,
            'grid_size': grid_size,
            'density': density,
            'description': 'Random initial states for testing'
        }
    )
    
    # Generate test set (patterns)
    print("\n" + "=" * 60)
    print("4. Generating test set (specific patterns)...")
    print("=" * 60)
    pattern_data = generate_pattern_test_set(gol, grid_size, num_steps=50)
    save_pattern_dataset(output_dir / "test_patterns.h5", pattern_data)
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print("=" * 60)
    print(f"\nGenerated files in {output_dir.absolute()}:")
    print(f"  - train.h5: {num_train} random samples")
    print(f"  - val.h5: {num_val} random samples")
    print(f"  - test_random.h5: {num_test} random samples")
    print(f"  - test_patterns.h5: Specific patterns with 50-step trajectories")


if __name__ == "__main__":
    main()

