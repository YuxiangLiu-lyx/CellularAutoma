"""
Evaluate 4-channel CNN on different Game of Life patterns with multi-step prediction.
Uses the same logic as evaluate_model.py but for 4-channel model.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN, count_parameters
from src.utils.game_of_life import GameOfLife
from src.utils.patterns import GLIDER, BLINKER, TOAD, BEACON, BLOCK, BEEHIVE, PULSAR, LWSS
from src.evaluation.metrics import pixel_accuracy


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
    
    start_h = (h - ph) // 2
    start_w = (w - pw) // 2
    
    state[start_h:start_h+ph, start_w:start_w+pw] = pattern
    
    return state


def evaluate_pattern(model, device, pattern_name, pattern, num_steps=50):
    """Evaluate model on specific pattern."""
    gol = GameOfLife(grid_size=(32, 32))
    
    initial_state = place_pattern_center((32, 32), pattern)
    
    true_traj = gol.simulate(initial_state, num_steps=num_steps)
    pred_traj = generate_prediction_trajectory(model, initial_state, num_steps, device)
    
    accuracies = []
    for t in range(1, len(true_traj)):
        acc = pixel_accuracy(true_traj[t], pred_traj[t])
        accuracies.append(acc)
    
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
    """Main evaluation function."""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "minimal" / "cnn_4ch.pth"
    output_dir = project_root / "figures" / "4ch_patterns"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first: python scripts/train_minimal.py")
        return
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GameOfLifeCNN(hidden_channels=4, padding_mode='circular')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("="*70)
    print("4-Channel CNN Evaluation on Patterns")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Parameters: {count_parameters(model):,}")
    
    print("\n" + "="*70)
    print("Evaluation on Classic Patterns (50 steps)")
    print("="*70)
    
    test_patterns = {
        'glider': GLIDER,
        'blinker': BLINKER,
        'toad': TOAD,
        'lwss': LWSS,
        'block': BLOCK,
        'beacon': BEACON,
        'beehive': BEEHIVE,
        'pulsar': PULSAR
    }
    
    pattern_results = {}
    
    for pattern_name, pattern in test_patterns.items():
        print(f"\nTesting {pattern_name}...")
        
        gif_path = output_dir / f"{pattern_name}_comparison.gif"
        
        results = evaluate_pattern(model, device, pattern_name, pattern, num_steps=50)
        pattern_results[pattern_name] = results
        
        print(f"  Mean accuracy:  {results['mean_accuracy']:.4f} ({results['mean_accuracy']*100:.2f}%)")
        print(f"  Final accuracy: {results['final_accuracy']:.4f} ({results['final_accuracy']*100:.2f}%)")
        print(f"  Min accuracy:   {results['min_accuracy']:.4f}")
        print(f"  Collapse step:  {results['collapse_step'] if results['collapse_step'] != -1 else 'Never'}")
        
        print(f"  Creating GIF...")
        create_comparison_gif(
            results['true_traj'],
            results['pred_traj'],
            pattern_name.upper(),
            gif_path,
            fps=10
        )
        print(f"  Saved: {gif_path.name}")
    
    print("\n" + "="*70)
    print("Pattern Evaluation Summary")
    print("="*70)
    print(f"{'Pattern':<15} {'Mean Acc':<12} {'Final Acc':<12} {'Min Acc':<12} {'Collapse@':<12}")
    print('-' * 70)
    
    for pattern_name, results in pattern_results.items():
        collapse = "Never" if results['collapse_step'] == -1 else str(results['collapse_step'])
        print(f"{pattern_name:<15} {results['mean_accuracy']:<12.6f} "
              f"{results['final_accuracy']:<12.6f} {results['min_accuracy']:<12.6f} {collapse:<12}")
    
    avg_pattern_acc = np.mean([r['mean_accuracy'] for r in pattern_results.values()])
    never_collapse = sum(1 for r in pattern_results.values() if r['collapse_step'] == -1)
    
    print("\n" + "="*70)
    print("Overall Summary")
    print("="*70)
    print(f"Average pattern performance:  {avg_pattern_acc:.4f} ({avg_pattern_acc*100:.2f}%)")
    print(f"Patterns never collapsed:     {never_collapse}/{len(test_patterns)}")
    
    print(f"\nGIFs saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

