"""
Batch pruning test for all 30 trained models from convergence stability test.
Find the pruning limit for each model.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import h5py
from tqdm import tqdm
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN, count_parameters


def evaluate_model_accuracy(model, test_data, device, num_samples=1000):
    """Evaluate model accuracy on test data."""
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


def greedy_channel_removal(original_model, test_data, device, target_accuracy=0.999, verbose=False):
    """
    Greedily remove one channel at a time to find pruning limit.
    
    Args:
        original_model: Original trained model
        test_data: Test dataset
        device: Computing device
        target_accuracy: Minimum acceptable accuracy
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with pruning results
    """
    num_channels = 16
    remaining_channels = list(range(num_channels))
    removed_channels = []
    
    checkpoint = original_model.state_dict()
    
    # Track accuracy at each step
    accuracy_history = []
    
    for iteration in range(num_channels):
        if verbose:
            print(f"  Iteration {iteration + 1}: Testing removal of each remaining channel...")
        
        best_channel_to_remove = None
        best_accuracy = 0
        
        # Test removing each remaining channel
        iterator = tqdm(remaining_channels, desc=f"  Testing", leave=False) if verbose else remaining_channels
        
        for ch in iterator:
            # Create test model
            test_model = GameOfLifeCNN(hidden_channels=16, padding_mode='circular')
            test_model.load_state_dict(checkpoint)
            test_model = test_model.to(device)
            
            # Prune this channel along with already removed ones
            prune_specific_channels(test_model, removed_channels + [ch])
            
            # Evaluate
            accuracy = evaluate_model_accuracy(test_model, test_data, device)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_channel_to_remove = ch
        
        # Check if we can remove this channel
        if best_accuracy >= target_accuracy:
            removed_channels.append(best_channel_to_remove)
            remaining_channels.remove(best_channel_to_remove)
            accuracy_history.append({
                'iteration': iteration + 1,
                'removed_channel': best_channel_to_remove,
                'active_channels': len(remaining_channels),
                'accuracy': float(best_accuracy)
            })
            
            if verbose:
                print(f"    Removed channel {best_channel_to_remove}, "
                      f"active: {len(remaining_channels)}/{num_channels}, "
                      f"acc: {best_accuracy:.6f}")
        else:
            if verbose:
                print(f"    Cannot remove more channels (acc would drop to {best_accuracy:.6f})")
            break
    
    return {
        'removed_channels': removed_channels,
        'active_channels': remaining_channels,
        'min_channels': len(remaining_channels),
        'max_removed': len(removed_channels),
        'final_accuracy': float(best_accuracy) if best_accuracy > 0 else None,
        'accuracy_history': accuracy_history
    }


def test_single_model(model_path, test_data, device, verbose=False):
    """
    Test pruning limit for a single model.
    
    Args:
        model_path: Path to model file
        test_data: Test dataset
        device: Computing device
        verbose: Whether to print progress
        
    Returns:
        Dictionary with test results
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GameOfLifeCNN(hidden_channels=16, padding_mode='circular')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Get baseline accuracy
    baseline_acc = evaluate_model_accuracy(model, test_data, device)
    
    if verbose:
        print(f"  Baseline accuracy: {baseline_acc:.6f}")
    
    # Perform greedy pruning
    pruning_result = greedy_channel_removal(model, test_data, device, 
                                           target_accuracy=0.999, 
                                           verbose=verbose)
    
    # Combine results
    result = {
        'run_id': checkpoint.get('run_id'),
        'seed': checkpoint.get('seed'),
        'convergence_epoch': checkpoint.get('convergence_epoch'),
        'baseline_accuracy': float(baseline_acc),
        'min_channels': pruning_result['min_channels'],
        'max_removed_channels': pruning_result['max_removed'],
        'removed_channels': pruning_result['removed_channels'],
        'active_channels': pruning_result['active_channels'],
        'final_accuracy': pruning_result['final_accuracy'],
        'accuracy_history': pruning_result['accuracy_history']
    }
    
    return result


def main():
    """Main batch pruning function."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "experiments" / "convergence_stability" / "models"
    test_path = project_root / "data" / "processed" / "val.h5"
    output_dir = project_root / "experiments" / "convergence_stability"
    
    if not models_dir.exists():
        print(f"Error: Models directory not found at {models_dir}")
        return
    
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        return
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("Batch Pruning Test - All 30 Models")
    print("="*70)
    print(f"Device: {device}")
    print(f"Models directory: {models_dir}")
    print(f"Test data: {test_path}")
    
    # Load test data
    print("\nLoading test data...")
    with h5py.File(test_path, 'r') as f:
        test_data = {
            'states_t': f['states_t'][:],
            'states_t1': f['states_t1'][:]
        }
    print(f"Test data loaded: {len(test_data['states_t'])} samples")
    
    # Find all model files
    model_files = sorted(models_dir.glob("run_*.pth"))
    print(f"\nFound {len(model_files)} model files")
    
    if len(model_files) == 0:
        print("No model files found!")
        return
    
    # Test each model
    print("\n" + "="*70)
    print("Testing Pruning Limits")
    print("="*70)
    
    all_results = []
    start_time = datetime.now()
    
    for i, model_path in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] Testing {model_path.name}")
        
        try:
            result = test_single_model(model_path, test_data, device, verbose=True)
            all_results.append(result)
            
            print(f"  Result: {result['min_channels']} channels minimum "
                  f"(removed {result['max_removed_channels']}/{16})")
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # Analyze results
    print("\n" + "="*70)
    print("PRUNING RESULTS SUMMARY")
    print("="*70)
    
    min_channels_list = [r['min_channels'] for r in all_results]
    
    if min_channels_list:
        print(f"\nMinimal Channel Statistics:")
        print(f"  Mean:   {np.mean(min_channels_list):.2f} channels")
        print(f"  Median: {np.median(min_channels_list):.1f} channels")
        print(f"  Min:    {np.min(min_channels_list)} channels")
        print(f"  Max:    {np.max(min_channels_list)} channels")
        print(f"  Std:    {np.std(min_channels_list):.2f}")
        
        print(f"\nDistribution of Minimal Channels:")
        unique_values = sorted(set(min_channels_list))
        for val in unique_values:
            count = min_channels_list.count(val)
            print(f"  {val} channels: {count} models")
        
        print(f"\nDetailed Results:")
        print(f"{'Run':<6} {'Seed':<8} {'Conv Epoch':<12} {'Min Ch':<10} {'Removed':<10} {'Final Acc':<12}")
        print("-" * 70)
        
        for r in all_results:
            print(f"{r['run_id']:<6} "
                  f"{r['seed']:<8} "
                  f"{r['convergence_epoch']:<12} "
                  f"{r['min_channels']:<10} "
                  f"{r['max_removed_channels']:<10} "
                  f"{r['final_accuracy']:.6f}")
    
    # Save results
    summary = {
        'experiment': {
            'name': 'Batch Pruning Test',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_time_seconds': total_time,
            'num_models_tested': len(all_results)
        },
        'statistics': {
            'mean_min_channels': float(np.mean(min_channels_list)) if min_channels_list else None,
            'median_min_channels': float(np.median(min_channels_list)) if min_channels_list else None,
            'min_min_channels': int(np.min(min_channels_list)) if min_channels_list else None,
            'max_min_channels': int(np.max(min_channels_list)) if min_channels_list else None,
            'std_min_channels': float(np.std(min_channels_list)) if min_channels_list else None,
        },
        'results': all_results
    }
    
    # Save to JSON
    output_path = output_dir / "pruning_limits.json"
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per model: {total_time/len(all_results):.1f} seconds")
    print("="*70)


if __name__ == "__main__":
    main()
