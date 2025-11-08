"""
True Out-of-Distribution test: structured patterns not in training.
"""
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN
from src.utils.game_of_life import GameOfLife
from src.evaluation.metrics import pixel_accuracy


def create_checkerboard(size=32):
    """Create checkerboard pattern."""
    indices = np.indices((size, size))
    return ((indices[0] + indices[1]) % 2).astype(np.uint8)


def create_stripes_horizontal(size=32, width=2):
    """Create horizontal stripes."""
    return ((np.arange(size) // width) % 2).reshape(size, 1).repeat(size, axis=1).astype(np.uint8)


def create_stripes_vertical(size=32, width=2):
    """Create vertical stripes."""
    return ((np.arange(size) // width) % 2).reshape(1, size).repeat(size, axis=0).astype(np.uint8)


def create_diagonal(size=32):
    """Create diagonal pattern."""
    return np.eye(size, dtype=np.uint8)


def create_border(size=32, thickness=1):
    """Create border pattern (only edges alive)."""
    state = np.zeros((size, size), dtype=np.uint8)
    state[:thickness, :] = 1
    state[-thickness:, :] = 1
    state[:, :thickness] = 1
    state[:, -thickness:] = 1
    return state


def create_clusters(size=32, num_clusters=4, cluster_size=5):
    """Create clustered pattern (dense islands in empty space)."""
    state = np.zeros((size, size), dtype=np.uint8)
    np.random.seed(42)
    
    for _ in range(num_clusters):
        # Random center
        ci = np.random.randint(cluster_size, size - cluster_size)
        cj = np.random.randint(cluster_size, size - cluster_size)
        
        # Dense cluster
        state[ci-cluster_size//2:ci+cluster_size//2+1,
              cj-cluster_size//2:cj+cluster_size//2+1] = 1
    
    return state


def test_structured_pattern(model, device, pattern_name, initial_state, num_steps=10):
    """Test model on structured pattern."""
    gol = GameOfLife(grid_size=(32, 32))
    
    # Generate ground truth
    true_traj = gol.simulate(initial_state, num_steps=num_steps)
    
    # Generate predictions (autoregressive)
    pred_traj = [initial_state]
    current_state = initial_state.copy()
    
    for t in range(num_steps):
        with torch.no_grad():
            state_tensor = torch.from_numpy(current_state).float().unsqueeze(0).unsqueeze(0)
            state_tensor = state_tensor.to(device)
            output = model(state_tensor)
            pred = (output > 0.5).float()
            next_state = pred.squeeze().cpu().numpy().astype(np.uint8)
        
        pred_traj.append(next_state)
        current_state = next_state
    
    # Calculate accuracies
    accuracies = []
    for t in range(1, len(true_traj)):
        acc = pixel_accuracy(true_traj[t], pred_traj[t])
        accuracies.append(acc)
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'min_accuracy': np.min(accuracies),
        'final_accuracy': accuracies[-1],
        'step_accuracies': accuracies
    }


def main():
    """Test model on truly OOD structured patterns."""
    
    project_root = Path(__file__).parent.parent
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GameOfLifeCNN(hidden_channels=16)
    
    model_path = project_root / "experiments" / "cnn" / "model.pt"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("="*70)
    print("TRUE OUT-OF-DISTRIBUTION TEST")
    print("="*70)
    print("Testing on STRUCTURED patterns never seen in training")
    print("Training data: Random i.i.d. points with 30% density")
    print("Test data: Highly structured patterns (checkerboard, stripes, etc.)")
    print("="*70)
    
    # Define structured patterns
    test_patterns = {
        'Checkerboard': create_checkerboard(32),
        'Horizontal Stripes (2px)': create_stripes_horizontal(32, width=2),
        'Horizontal Stripes (4px)': create_stripes_horizontal(32, width=4),
        'Vertical Stripes (2px)': create_stripes_vertical(32, width=2),
        'Diagonal': create_diagonal(32),
        'Border (1px)': create_border(32, thickness=1),
        'Border (2px)': create_border(32, thickness=2),
        'Clusters': create_clusters(32, num_clusters=4, cluster_size=5),
    }
    
    results = {}
    
    for pattern_name, initial_state in test_patterns.items():
        print(f"\n{'='*70}")
        print(f"Testing: {pattern_name}")
        print(f"Initial alive cells: {np.sum(initial_state)}/{32*32} ({np.sum(initial_state)/(32*32)*100:.1f}%)")
        print('='*70)
        
        result = test_structured_pattern(model, device, pattern_name, initial_state, num_steps=10)
        results[pattern_name] = result
        
        print(f"Results:")
        print(f"  Mean accuracy:  {result['mean_accuracy']:.4f} ({result['mean_accuracy']*100:.2f}%)")
        print(f"  Min accuracy:   {result['min_accuracy']:.4f} ({result['min_accuracy']*100:.2f}%)")
        print(f"  Final accuracy: {result['final_accuracy']:.4f} ({result['final_accuracy']*100:.2f}%)")
        
        print(f"\nStep-by-step:")
        for t, acc in enumerate(result['step_accuracies'], 1):
            status = "OK" if acc > 0.95 else "FAIL"
            print(f"  Step {t:2d}: {acc:.4f} {status}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: True OOD Performance")
    print("="*70)
    print(f"{'Pattern':<30} {'Mean Acc':<12} {'Min Acc':<12} {'Final Acc':<12}")
    print('-' * 70)
    
    for pattern_name, result in results.items():
        print(f"{pattern_name:<30} {result['mean_accuracy']:<12.4f} "
              f"{result['min_accuracy']:<12.4f} {result['final_accuracy']:<12.4f}")
    
    avg_mean = np.mean([r['mean_accuracy'] for r in results.values()])
    avg_min = np.mean([r['min_accuracy'] for r in results.values()])
    
    print('-' * 70)
    print(f"{'AVERAGE':<30} {avg_mean:<12.4f} {avg_min:<12.4f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if avg_mean > 0.99:
        print("Excellent performance on structured patterns")
        print("  - Average accuracy > 99%")
        print("  - Model likely learned the true Game of Life rules")
    elif avg_mean > 0.90:
        print("Good but not perfect performance")
        print(f"  - Average accuracy: {avg_mean*100:.1f}%")
        print("  - Some degradation on structured patterns")
    else:
        print("Poor performance on structured patterns")
        print(f"  - Average accuracy: {avg_mean*100:.1f}%")
        print("  - Significant performance drop")
        print("  - Evidence of pattern memorization, not rule learning")
    
    print("="*70)


if __name__ == "__main__":
    main()

