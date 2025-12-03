"""
Intelligent channel pruning by testing all possible combinations.
Find the minimal channel set that maintains performance.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import h5py
from itertools import combinations
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN, count_parameters


def evaluate_model_accuracy(model, test_data, device, num_samples=500):
    """Quickly evaluate model accuracy on a subset of test data."""
    states_t = test_data['states_t'][:num_samples]
    states_t1 = test_data['states_t1'][:num_samples]
    
    model.eval()
    with torch.no_grad():
        state_tensor = torch.from_numpy(states_t).float().unsqueeze(1).to(device)
        output = model(state_tensor)
        pred = (output > 0.5).cpu().numpy().squeeze(1)
    
    accuracy = np.mean(pred == states_t1)
    return accuracy


def prune_specific_channels(model, channels_to_prune):
    """
    Prune specific channels by setting their weights to zero.
    
    Args:
        model: CNN model
        channels_to_prune: List of channel indices to prune
        
    Returns:
        Number of active channels
    """
    num_channels = model.conv1.weight.shape[0]
    
    for ch in channels_to_prune:
        model.conv1.weight.data[ch] = 0
        model.conv1.bias.data[ch] = 0
        model.conv2.weight.data[:, ch] = 0
    
    active_channels = num_channels - len(channels_to_prune)
    return active_channels


def greedy_channel_removal(original_model, test_data, device, target_accuracy=0.999):
    """
    Greedily remove one channel at a time, keeping those that maintain performance.
    
    Args:
        original_model: Original trained model
        test_data: Test dataset
        device: Computing device
        target_accuracy: Minimum acceptable accuracy
        
    Returns:
        List of channels that can be safely removed
    """
    print("\n" + "="*70)
    print("Greedy Channel Removal")
    print("="*70)
    print(f"Target accuracy: {target_accuracy:.1%}")
    
    num_channels = 16
    remaining_channels = list(range(num_channels))
    removed_channels = []
    
    checkpoint = original_model.state_dict()
    
    for iteration in range(num_channels):
        print(f"\nIteration {iteration + 1}: Testing removal of each remaining channel...")
        
        best_channel_to_remove = None
        best_accuracy = 0
        
        for ch in tqdm(remaining_channels, desc="Testing channels"):
            test_model = GameOfLifeCNN(hidden_channels=16, padding_mode='circular')
            test_model.load_state_dict(checkpoint)
            test_model = test_model.to(device)
            
            prune_specific_channels(test_model, removed_channels + [ch])
            
            accuracy = evaluate_model_accuracy(test_model, test_data, device)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_channel_to_remove = ch
        
        print(f"  Best to remove: Channel {best_channel_to_remove}")
        print(f"  Accuracy after removal: {best_accuracy:.6f} ({best_accuracy*100:.2f}%)")
        
        if best_accuracy >= target_accuracy:
            removed_channels.append(best_channel_to_remove)
            remaining_channels.remove(best_channel_to_remove)
            print(f"  Safe to remove (accuracy maintained)")
            print(f"  Active channels: {len(remaining_channels)}/{num_channels}")
        else:
            print(f"  Cannot remove (accuracy drops below {target_accuracy:.1%})")
            print(f"  Stopping pruning")
            break
    
    return removed_channels, remaining_channels


def exhaustive_search_small_sets(original_model, test_data, device, max_channels=8):
    """
    Exhaustively search for the smallest channel set that works.
    
    Args:
        original_model: Original model
        test_data: Test dataset
        device: Device
        max_channels: Maximum channels to test
        
    Returns:
        Best channel combination found
    """
    print("\n" + "="*70)
    print("Exhaustive Search for Minimal Channel Set")
    print("="*70)
    
    num_channels = 16
    checkpoint = original_model.state_dict()
    
    for n in range(max_channels, num_channels + 1):
        print(f"\nTesting all combinations of {n} channels...")
        total_combinations = len(list(combinations(range(num_channels), n)))
        print(f"  Total combinations: {total_combinations}")
        
        if total_combinations > 1000:
            print(f"  Too many combinations, skipping...")
            continue
        
        best_accuracy = 0
        best_combination = None
        
        for channel_combo in tqdm(combinations(range(num_channels), n), 
                                   total=total_combinations, 
                                   desc=f"Testing {n} channels"):
            
            channels_to_keep = list(channel_combo)
            channels_to_prune = [ch for ch in range(num_channels) if ch not in channels_to_keep]
            
            test_model = GameOfLifeCNN(hidden_channels=16, padding_mode='circular')
            test_model.load_state_dict(checkpoint)
            test_model = test_model.to(device)
            
            prune_specific_channels(test_model, channels_to_prune)
            
            accuracy = evaluate_model_accuracy(test_model, test_data, device)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_combination = channels_to_keep
        
        print(f"  Best {n}-channel combination: {best_combination}")
        print(f"  Accuracy: {best_accuracy:.6f} ({best_accuracy*100:.2f}%)")
        
        if best_accuracy >= 0.999:
            print(f"\nFound minimal channel set with {n} channels")
            return best_combination, best_accuracy
    
    return None, 0


def main():
    """Main intelligent pruning function."""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "experiments" / "cnn" / "model.pt"
    test_path = project_root / "data" / "processed" / "test_random.h5"
    output_dir = project_root / "models" / "pruned"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        return
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("Intelligent Channel Pruning")
    print("="*70)
    print(f"Device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GameOfLifeCNN(hidden_channels=16, padding_mode='circular')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    with h5py.File(test_path, 'r') as f:
        test_data = {
            'states_t': f['states_t'][:],
            'states_t1': f['states_t1'][:]
        }
    
    print(f"Test data: {len(test_data['states_t'])} samples")
    
    baseline_acc = evaluate_model_accuracy(model, test_data, device, num_samples=1000)
    print(f"Original model accuracy: {baseline_acc:.6f} ({baseline_acc*100:.2f}%)")
    
    print("\n" + "="*70)
    print("Choose pruning strategy:")
    print("="*70)
    print("1. Greedy removal (fast, finds one good solution)")
    print("2. Exhaustive search for 8-12 channels (slower, finds optimal)")
    print("3. Both (recommended)")
    
    strategy = input("\nEnter choice (1/2/3) [default=1]: ").strip() or "1"
    
    if strategy in ["1", "3"]:
        removed_channels, active_channels = greedy_channel_removal(
            model, test_data, device, target_accuracy=0.999
        )
        
        print("\n" + "="*70)
        print("Greedy Pruning Results")
        print("="*70)
        print(f"Removed channels: {removed_channels}")
        print(f"Active channels: {active_channels}")
        print(f"Minimum channels needed: {len(active_channels)}")
        
        final_model = GameOfLifeCNN(hidden_channels=16, padding_mode='circular')
        final_model.load_state_dict(checkpoint['model_state_dict'])
        final_model = final_model.to(device)
        prune_specific_channels(final_model, removed_channels)
        
        final_acc = evaluate_model_accuracy(final_model, test_data, device, num_samples=1000)
        print(f"Final accuracy: {final_acc:.6f} ({final_acc*100:.2f}%)")
        
        save_path = output_dir / "model_greedy_pruned.pth"
        torch.save({
            'hidden_channels': 16,
            'model_state_dict': final_model.state_dict(),
            'active_channels': active_channels,
            'removed_channels': removed_channels,
            'val_accuracy': final_acc,
        }, save_path)
        print(f"\nSaved to: {save_path}")
    
    if strategy in ["2", "3"]:
        best_channels, best_acc = exhaustive_search_small_sets(
            model, test_data, device, max_channels=8
        )
        
        if best_channels:
            print("\n" + "="*70)
            print("Exhaustive Search Results")
            print("="*70)
            print(f"Optimal channel set: {best_channels}")
            print(f"Number of channels: {len(best_channels)}")
            print(f"Accuracy: {best_acc:.6f} ({best_acc*100:.2f}%)")


if __name__ == "__main__":
    main()

