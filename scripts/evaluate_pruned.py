"""
Evaluate pruned models on test data and patterns.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import h5py

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN, count_parameters
from src.evaluation.metrics import pixel_accuracy
from src.utils.game_of_life import GameOfLife
from src.utils.patterns import GLIDER, BLINKER, BLOCK


def count_nonzero_parameters(model):
    """Count non-zero parameters in the model."""
    total_params = 0
    nonzero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += torch.count_nonzero(param).item()
    
    return total_params, nonzero_params


def evaluate_on_test_set(model, test_h5_path, device, batch_size=64):
    """Evaluate model on test dataset."""
    with h5py.File(test_h5_path, 'r') as f:
        states_t = f['states_t'][:]
        states_t1 = f['states_t1'][:]
    
    num_samples = len(states_t)
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_t = states_t[i:i+batch_size]
            batch_t = torch.from_numpy(batch_t).float().unsqueeze(1).to(device)
            
            output = model(batch_t)
            pred = (output > 0.5).cpu().numpy().squeeze(1)
            all_predictions.append(pred)
    
    predictions = np.concatenate(all_predictions, axis=0)
    accuracy = pixel_accuracy(predictions, states_t1)
    
    return accuracy


def place_pattern_center(grid_size, pattern):
    """Place pattern in center of grid."""
    state = np.zeros(grid_size, dtype=np.uint8)
    ph, pw = pattern.shape
    h, w = grid_size
    
    start_h = (h - ph) // 2
    start_w = (w - pw) // 2
    
    state[start_h:start_h+ph, start_w:start_w+pw] = pattern
    
    return state


def evaluate_pattern_multistep(model, pattern, num_steps, device):
    """Evaluate model on a pattern for multiple steps."""
    gol = GameOfLife(grid_size=(32, 32))
    initial_state = place_pattern_center((32, 32), pattern)
    
    true_traj = gol.simulate(initial_state, num_steps=num_steps)
    
    current_state = initial_state.copy()
    accuracies = []
    
    for t in range(num_steps):
        with torch.no_grad():
            state_tensor = torch.from_numpy(current_state).float().unsqueeze(0).unsqueeze(0)
            state_tensor = state_tensor.to(device)
            output = model(state_tensor)
            pred = (output > 0.5).float()
            next_state = pred.squeeze().cpu().numpy().astype(np.uint8)
        
        acc = pixel_accuracy(next_state, true_traj[t + 1])
        accuracies.append(acc)
        current_state = next_state
    
    return np.mean(accuracies)


def main():
    """Main evaluation function."""
    project_root = Path(__file__).parent.parent
    test_path = project_root / "data" / "processed" / "test_random.h5"
    pruned_dir = project_root / "models" / "pruned"
    
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        return
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    print("="*80)
    print("Evaluating Pruned Models")
    print("="*80)
    
    pruned_models = [
        ('Original (16ch)', project_root / "experiments" / "cnn" / "model.pt", None),
        ('Pruned 10%', pruned_dir / "model_pruned_p10.pth", 10),
        ('Pruned 25%', pruned_dir / "model_pruned_p25.pth", 25),
        ('Pruned 50%', pruned_dir / "model_pruned_p50.pth", 50),
    ]
    
    results = []
    
    for model_name, model_path, percentile in pruned_models:
        if not model_path.exists():
            print(f"\nSkipping {model_name}: file not found")
            continue
        
        print(f"\n{'-'*80}")
        print(f"Model: {model_name}")
        print(f"{'-'*80}")
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = GameOfLifeCNN(hidden_channels=16, padding_mode='circular')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        total_params, nonzero_params = count_nonzero_parameters(model)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Non-zero parameters: {nonzero_params:,}")
        print(f"Sparsity: {(1 - nonzero_params/total_params)*100:.2f}%")
        
        if percentile is not None and 'pruning_stats' in checkpoint:
            stats = checkpoint['pruning_stats']
            print(f"Active channels: {stats['active_channels']}/{stats['total_channels']}")
            print(f"Active channel IDs: {stats['active_channel_ids']}")
        
        print("\nTesting on random test set...")
        test_acc = evaluate_on_test_set(model, test_path, device)
        print(f"Test accuracy: {test_acc:.6f} ({test_acc*100:.2f}%)")
        
        print("\nTesting on patterns (50 steps)...")
        glider_acc = evaluate_pattern_multistep(model, GLIDER, 50, device)
        blinker_acc = evaluate_pattern_multistep(model, BLINKER, 50, device)
        block_acc = evaluate_pattern_multistep(model, BLOCK, 50, device)
        
        print(f"  Glider:  {glider_acc:.6f}")
        print(f"  Blinker: {blinker_acc:.6f}")
        print(f"  Block:   {block_acc:.6f}")
        
        avg_pattern_acc = np.mean([glider_acc, blinker_acc, block_acc])
        
        results.append({
            'name': model_name,
            'total_params': total_params,
            'nonzero_params': nonzero_params,
            'sparsity': (1 - nonzero_params/total_params) * 100,
            'test_acc': test_acc,
            'glider_acc': glider_acc,
            'blinker_acc': blinker_acc,
            'block_acc': block_acc,
            'avg_pattern_acc': avg_pattern_acc
        })
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<20} {'Nonzero Params':<20} {'Sparsity':<12} {'Test Acc':<12} {'Avg Pattern':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<20} {r['nonzero_params']:<20,} {r['sparsity']:<12.2f}% "
              f"{r['test_acc']:<12.6f} {r['avg_pattern_acc']:<12.6f}")
    
    print("\n" + "="*80)
    print("Analysis")
    print("="*80)
    
    if len(results) > 1:
        orig_params = results[0]['nonzero_params']
        orig_test_acc = results[0]['test_acc']
        
        for r in results[1:]:
            param_reduction = (1 - r['nonzero_params'] / orig_params) * 100
            acc_drop = (orig_test_acc - r['test_acc']) * 100
            
            print(f"\n{r['name']}:")
            print(f"  Parameter reduction: {param_reduction:.1f}%")
            print(f"  Accuracy drop: {acc_drop:.2f}%")
            
            if acc_drop < 1.0:
                print(f"  Minimal impact - this pruning level is safe")
            elif acc_drop < 5.0:
                print(f"  Small impact - acceptable trade-off")
            else:
                print(f"  Significant impact - too aggressive")


if __name__ == "__main__":
    main()

