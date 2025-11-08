"""
Comprehensive model evaluation on different densities and classic patterns.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN
from src.utils.game_of_life import GameOfLife
from src.utils.patterns import GLIDER, BLINKER, TOAD, LWSS, BLOCK
from src.evaluation.metrics import pixel_accuracy, pattern_preservation_score


def generate_prediction_trajectory(model, initial_state, num_steps, device):
    """Generate trajectory using autoregressive model predictions."""
    trajectory = np.zeros((num_steps + 1, *initial_state.shape), dtype=np.uint8)
    trajectory[0] = initial_state
    
    current_state = initial_state.copy()
    
    for t in range(num_steps):
        with torch.no_grad():
            state_tensor = torch.from_numpy(current_state).float().unsqueeze(0).unsqueeze(0)
            state_tensor = state_tensor.to(device)
            output = model(state_tensor)
            pred = (output > 0.5).float()
            next_state = pred.squeeze().cpu().numpy().astype(np.uint8)
        
        trajectory[t + 1] = next_state
        current_state = next_state
    
    return trajectory


def create_comparison_gif(true_traj, pred_traj, title, save_path, fps=10):
    """Create side-by-side GIF comparing true and predicted trajectories."""
    num_frames = min(len(true_traj), len(pred_traj))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    im1 = ax1.imshow(true_traj[0], cmap='binary', interpolation='nearest', animated=True)
    ax1.set_title('Ground Truth', fontsize=14)
    ax1.axis('off')
    
    im2 = ax2.imshow(pred_traj[0], cmap='binary', interpolation='nearest', animated=True)
    ax2.set_title('Model Prediction', fontsize=14)
    ax2.axis('off')
    
    suptitle = fig.suptitle(f'{title} - Step 0', fontsize=16)
    
    def update(frame):
        im1.set_array(true_traj[frame])
        im2.set_array(pred_traj[frame])
        
        true_alive = np.sum(true_traj[frame])
        pred_alive = np.sum(pred_traj[frame])
        acc = np.mean(true_traj[frame] == pred_traj[frame])
        
        suptitle.set_text(f'{title} - Step {frame}\n'
                         f'Alive: True={true_alive}, Pred={pred_alive} | Acc: {acc:.1%}')
        
        return [im1, im2, suptitle]
    
    anim = FuncAnimation(fig, update, frames=num_frames, 
                        interval=1000//fps, blit=True, repeat=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    plt.close()


def place_pattern_center(grid_size, pattern):
    """Place pattern in center of grid."""
    state = np.zeros(grid_size, dtype=np.uint8)
    ph, pw = pattern.shape
    h, w = grid_size
    
    start_i = (h - ph) // 2
    start_j = (w - pw) // 2
    
    state[start_i:start_i+ph, start_j:start_j+pw] = pattern
    return state


def evaluate_density(model, device, density, num_samples=100, num_steps=50):
    """Evaluate model on specific density."""
    gol = GameOfLife(grid_size=(32, 32))
    
    accuracies = []
    pattern_scores = []
    
    for i in range(num_samples):
        np.random.seed(1000 + i)
        initial_state = (np.random.random((32, 32)) < density).astype(np.uint8)
        
        # Ground truth (1 step)
        true_next = gol.step(initial_state)
        
        # Prediction
        with torch.no_grad():
            state_tensor = torch.from_numpy(initial_state).float().unsqueeze(0).unsqueeze(0)
            state_tensor = state_tensor.to(device)
            output = model(state_tensor)
            pred = (output > 0.5).float()
            pred_next = pred.squeeze().cpu().numpy().astype(np.uint8)
        
        acc = pixel_accuracy(true_next, pred_next)
        ps = pattern_preservation_score(true_next, pred_next)
        
        accuracies.append(acc)
        pattern_scores.append(ps)
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'mean_pattern_score': np.mean(pattern_scores),
        'std_accuracy': np.std(accuracies),
        'min_accuracy': np.min(accuracies)
    }


def evaluate_density_with_trajectory(model, device, density, num_steps=50, seed=42):
    """Evaluate model on specific density and return full trajectory for visualization."""
    gol = GameOfLife(grid_size=(32, 32))
    
    # Generate one example for visualization
    np.random.seed(seed)
    initial_state = (np.random.random((32, 32)) < density).astype(np.uint8)
    
    # Generate trajectories
    true_traj = gol.simulate(initial_state, num_steps=num_steps)
    pred_traj = generate_prediction_trajectory(model, initial_state, num_steps, device)
    
    # Calculate accuracies over time
    accuracies = []
    for t in range(1, len(true_traj)):
        acc = pixel_accuracy(true_traj[t], pred_traj[t])
        accuracies.append(acc)
    
    return {
        'true_traj': true_traj,
        'pred_traj': pred_traj,
        'mean_accuracy': np.mean(accuracies),
        'final_accuracy': accuracies[-1],
        'min_accuracy': np.min(accuracies)
    }


def evaluate_pattern(model, device, pattern_name, pattern, num_steps=50):
    """Evaluate model on specific pattern."""
    gol = GameOfLife(grid_size=(32, 32))
    
    # Place pattern
    initial_state = place_pattern_center((32, 32), pattern)
    
    # Generate trajectories
    true_traj = gol.simulate(initial_state, num_steps=num_steps)
    pred_traj = generate_prediction_trajectory(model, initial_state, num_steps, device)
    
    # Calculate accuracies over time
    accuracies = []
    for t in range(1, len(true_traj)):
        acc = pixel_accuracy(true_traj[t], pred_traj[t])
        accuracies.append(acc)
    
    # Find collapse step (accuracy < 95%)
    collapse_step = -1
    for t, acc in enumerate(accuracies):
        if acc < 0.95:
            collapse_step = t + 1
            break
    
    return {
        'true_traj': true_traj,
        'pred_traj': pred_traj,
        'mean_accuracy': np.mean(accuracies),
        'final_accuracy': accuracies[-1],
        'min_accuracy': np.min(accuracies),
        'collapse_step': collapse_step
    }


def main():
    """Run comprehensive evaluation."""
    
    project_root = Path(__file__).parent.parent
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GameOfLifeCNN(hidden_channels=16)
    
    model_path = project_root / "experiments" / "cnn" / "model.pt"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first: python3 src/models/train.py")
        return
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("="*70)
    print("Model Evaluation")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = project_root / "figures" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Part 1: Density Evaluation
    # ========================================================================
    print("\n" + "="*70)
    print("Part 1: Evaluation on Different Densities")
    print("="*70)
    
    test_densities = [0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    density_results = {}
    density_traj_results = {}
    
    for density in test_densities:
        print(f"\nTesting density: {density:.0%}...")
        
        density_name = f"density_{int(density*100):02d}"
        gif_path = output_dir / f"{density_name}_comparison.gif"
        
        # Check if GIF already exists
        if gif_path.exists():
            print(f"  GIF already exists: {gif_path.name}, skipping generation")
            # Still need to evaluate for statistics and trajectory accuracy
            results = evaluate_density(model, device, density, num_samples=100)
            density_results[density] = results
            
            # Load trajectory results for summary (need to regenerate for accuracy)
            traj_results = evaluate_density_with_trajectory(model, device, density, num_steps=50)
            density_traj_results[density] = traj_results
            
            print(f"  Mean accuracy: {results['mean_accuracy']:.4f} ({results['mean_accuracy']*100:.2f}%)")
            print(f"  Trajectory mean accuracy: {traj_results['mean_accuracy']:.4f}")
        else:
            # Statistical evaluation (100 samples)
            results = evaluate_density(model, device, density, num_samples=100)
            density_results[density] = results
            
            print(f"  Mean accuracy: {results['mean_accuracy']:.4f} ({results['mean_accuracy']*100:.2f}%)")
            print(f"  Mean F1 score: {results['mean_pattern_score']:.4f}")
            print(f"  Min accuracy:  {results['min_accuracy']:.4f}")
            
            # Generate trajectory for visualization
            print(f"  Generating trajectory for visualization...")
            traj_results = evaluate_density_with_trajectory(model, device, density, num_steps=50)
            density_traj_results[density] = traj_results
            
            print(f"  Trajectory mean accuracy: {traj_results['mean_accuracy']:.4f}")
            print(f"  Creating GIF...")
            
            # Create GIF
            create_comparison_gif(
                traj_results['true_traj'],
                traj_results['pred_traj'],
                f"Density {density:.0%}",
                gif_path,
                fps=10
            )
            print(f"  Saved: {gif_path.name}")
    
    # Summary table
    print("\n" + "="*70)
    print("Density Evaluation Summary")
    print("="*70)
    print(f"{'Density':<12} {'Mean Acc':<12} {'Std':<12} {'Min Acc':<12} {'Traj Acc':<12} {'Status':<10}")
    print('-' * 80)
    
    for density in test_densities:
        results = density_results[density]
        traj_acc = density_traj_results[density]['mean_accuracy']
        status = "IN-DIST" if abs(density - 0.30) < 0.01 else "OOD"
        print(f"{density:<12.2f} {results['mean_accuracy']:<12.6f} "
              f"{results['std_accuracy']:<12.6f} {results['min_accuracy']:<12.6f} "
              f"{traj_acc:<12.6f} {status:<10}")
    
    avg_ood = np.mean([r['mean_accuracy'] for d, r in density_results.items() if abs(d - 0.30) > 0.01])
    print(f"\nAverage OOD accuracy: {avg_ood:.4f} ({avg_ood*100:.2f}%)")
    
    # ========================================================================
    # Part 2: Pattern Evaluation
    # ========================================================================
    print("\n" + "="*70)
    print("Part 2: Evaluation on Classic Patterns (50 steps)")
    print("="*70)
    
    test_patterns = {
        'glider': GLIDER,
        'blinker': BLINKER,
        'toad': TOAD,
        'lwss': LWSS,
        'block': BLOCK
    }
    
    pattern_results = {}
    
    for pattern_name, pattern in test_patterns.items():
        print(f"\nTesting {pattern_name}...")
        
        gif_path = output_dir / f"{pattern_name}_comparison.gif"
        
        # Check if GIF already exists
        if gif_path.exists():
            print(f"  GIF already exists: {gif_path.name}, skipping generation")
            # Still need to evaluate for statistics
            results = evaluate_pattern(model, device, pattern_name, pattern, num_steps=50)
            pattern_results[pattern_name] = results
            
            print(f"  Mean accuracy:  {results['mean_accuracy']:.4f} ({results['mean_accuracy']*100:.2f}%)")
            print(f"  Final accuracy: {results['final_accuracy']:.4f} ({results['final_accuracy']*100:.2f}%)")
            print(f"  Collapse step:  {results['collapse_step'] if results['collapse_step'] != -1 else 'Never'}")
        else:
            # Evaluate pattern
            results = evaluate_pattern(model, device, pattern_name, pattern, num_steps=50)
            pattern_results[pattern_name] = results
            
            print(f"  Mean accuracy:  {results['mean_accuracy']:.4f} ({results['mean_accuracy']*100:.2f}%)")
            print(f"  Final accuracy: {results['final_accuracy']:.4f} ({results['final_accuracy']*100:.2f}%)")
            print(f"  Min accuracy:   {results['min_accuracy']:.4f}")
            print(f"  Collapse step:  {results['collapse_step'] if results['collapse_step'] != -1 else 'Never'}")
            
            # Create GIF
            print(f"  Creating GIF...")
            create_comparison_gif(
                results['true_traj'],
                results['pred_traj'],
                pattern_name.upper(),
                gif_path,
                fps=10
            )
            print(f"  Saved: {gif_path.name}")
    
    # Pattern summary
    print("\n" + "="*70)
    print("Pattern Evaluation Summary")
    print("="*70)
    print(f"{'Pattern':<15} {'Mean Acc':<12} {'Final Acc':<12} {'Min Acc':<12} {'Collapse@':<12}")
    print('-' * 70)
    
    for pattern_name, results in pattern_results.items():
        collapse = "Never" if results['collapse_step'] == -1 else str(results['collapse_step'])
        print(f"{pattern_name:<15} {results['mean_accuracy']:<12.6f} "
              f"{results['final_accuracy']:<12.6f} {results['min_accuracy']:<12.6f} {collapse:<12}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*70)
    print("Overall Summary")
    print("="*70)
    
    in_dist_acc = density_results[0.30]['mean_accuracy']
    avg_pattern_acc = np.mean([r['mean_accuracy'] for r in pattern_results.values()])
    never_collapse = sum(1 for r in pattern_results.values() if r['collapse_step'] == -1)
    
    print(f"In-distribution density (30%):     {in_dist_acc:.4f} ({in_dist_acc*100:.2f}%)")
    print(f"Average OOD density performance:   {avg_ood:.4f} ({avg_ood*100:.2f}%)")
    print(f"Average pattern performance:       {avg_pattern_acc:.4f} ({avg_pattern_acc*100:.2f}%)")
    print(f"Patterns never collapsed:          {never_collapse}/{len(test_patterns)}")
    
    print(f"\nGIFs saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

