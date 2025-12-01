"""
Prune the 16-channel CNN model by analyzing channel importance.
Find and remove channels with very small weights.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN, count_parameters


def analyze_channel_importance(model):
    """
    Analyze the importance of each channel in the first convolutional layer.
    
    Args:
        model: Trained CNN model
        
    Returns:
        Dictionary with channel importance metrics
    """
    conv1_weight = model.conv1.weight.data.cpu().numpy()
    
    num_channels = conv1_weight.shape[0]
    channel_norms = []
    
    for i in range(num_channels):
        channel_weight = conv1_weight[i]
        l1_norm = np.sum(np.abs(channel_weight))
        l2_norm = np.sqrt(np.sum(channel_weight ** 2))
        max_abs = np.max(np.abs(channel_weight))
        
        channel_norms.append({
            'channel': i,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            'max_abs': max_abs
        })
    
    return channel_norms


def visualize_channel_importance(channel_norms, save_path):
    """Visualize channel importance."""
    channels = [c['channel'] for c in channel_norms]
    l1_norms = [c['l1_norm'] for c in channel_norms]
    l2_norms = [c['l2_norm'] for c in channel_norms]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.bar(channels, l1_norms, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Channel Index')
    ax1.set_ylabel('L1 Norm (Sum of Absolute Weights)')
    ax1.set_title('Channel Importance by L1 Norm')
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(channels, l2_norms, color='coral', alpha=0.7)
    ax2.set_xlabel('Channel Index')
    ax2.set_ylabel('L2 Norm')
    ax2.set_title('Channel Importance by L2 Norm')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Channel importance visualization saved to: {save_path}")


def prune_channels_by_threshold(model, threshold_percentile=25):
    """
    Prune channels with L1 norm below a certain percentile.
    
    Args:
        model: CNN model
        threshold_percentile: Percentile threshold (0-100)
        
    Returns:
        Pruned model and pruning statistics
    """
    conv1_weight = model.conv1.weight.data
    conv1_bias = model.conv1.bias.data
    conv2_weight = model.conv2.weight.data
    
    num_channels = conv1_weight.shape[0]
    
    channel_l1_norms = []
    for i in range(num_channels):
        l1_norm = torch.sum(torch.abs(conv1_weight[i])).item()
        channel_l1_norms.append(l1_norm)
    
    threshold = np.percentile(channel_l1_norms, threshold_percentile)
    
    active_channels = []
    pruned_channels = []
    
    for i in range(num_channels):
        if channel_l1_norms[i] >= threshold:
            active_channels.append(i)
        else:
            pruned_channels.append(i)
            conv1_weight[i] = 0
            conv1_bias[i] = 0
            conv2_weight[:, i] = 0
    
    model.conv1.weight.data = conv1_weight
    model.conv1.bias.data = conv1_bias
    model.conv2.weight.data = conv2_weight
    
    return model, {
        'total_channels': num_channels,
        'active_channels': len(active_channels),
        'pruned_channels': len(pruned_channels),
        'active_channel_ids': active_channels,
        'pruned_channel_ids': pruned_channels,
        'threshold': threshold,
        'channel_l1_norms': channel_l1_norms
    }


def count_nonzero_parameters(model):
    """Count non-zero parameters in the model."""
    total_params = 0
    nonzero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += torch.count_nonzero(param).item()
    
    return total_params, nonzero_params


def main():
    """Main pruning function."""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "experiments" / "cnn" / "model.pt"
    output_dir = project_root / "models" / "pruned"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first:")
        print("  python src/models/train.py")
        return
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("Model Pruning Analysis")
    print("="*70)
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = GameOfLifeCNN(hidden_channels=16, padding_mode='circular')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Original parameters: {count_parameters(model):,}")
    
    print("\n" + "="*70)
    print("Analyzing Channel Importance")
    print("="*70)
    
    channel_norms = analyze_channel_importance(model)
    
    print(f"\n{'Channel':<10} {'L1 Norm':<15} {'L2 Norm':<15} {'Max |Weight|':<15}")
    print("-" * 70)
    
    for c in channel_norms:
        print(f"{c['channel']:<10} {c['l1_norm']:<15.6f} {c['l2_norm']:<15.6f} {c['max_abs']:<15.6f}")
    
    l1_norms = [c['l1_norm'] for c in channel_norms]
    print(f"\nL1 Norm Statistics:")
    print(f"  Mean:   {np.mean(l1_norms):.6f}")
    print(f"  Median: {np.median(l1_norms):.6f}")
    print(f"  Min:    {np.min(l1_norms):.6f}")
    print(f"  Max:    {np.max(l1_norms):.6f}")
    print(f"  Std:    {np.std(l1_norms):.6f}")
    
    fig_path = output_dir / "channel_importance.png"
    visualize_channel_importance(channel_norms, fig_path)
    
    print("\n" + "="*70)
    print("Pruning Channels")
    print("="*70)
    
    percentiles = [10, 25, 50]
    
    for percentile in percentiles:
        print(f"\nTesting pruning threshold: {percentile}th percentile")
        
        model_test = GameOfLifeCNN(hidden_channels=16, padding_mode='circular')
        model_test.load_state_dict(checkpoint['model_state_dict'])
        model_test = model_test.to(device)
        
        pruned_model, stats = prune_channels_by_threshold(model_test, percentile)
        
        total_params, nonzero_params = count_nonzero_parameters(pruned_model)
        
        print(f"  Threshold (L1 norm): {stats['threshold']:.6f}")
        print(f"  Active channels: {stats['active_channels']}/{stats['total_channels']}")
        print(f"  Active channel IDs: {stats['active_channel_ids']}")
        print(f"  Pruned channel IDs: {stats['pruned_channel_ids']}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-zero parameters: {nonzero_params:,}")
        print(f"  Sparsity: {(1 - nonzero_params/total_params)*100:.2f}%")
        
        save_path = output_dir / f"model_pruned_p{percentile}.pth"
        torch.save({
            'hidden_channels': 16,
            'model_state_dict': pruned_model.state_dict(),
            'pruning_stats': stats,
            'val_accuracy': checkpoint.get('val_accuracy', None),
        }, save_path)
        print(f"  Saved to: {save_path}")
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("Pruned models saved to models/pruned/")
    print("  - model_pruned_p10.pth: Prune bottom 10% channels")
    print("  - model_pruned_p25.pth: Prune bottom 25% channels")
    print("  - model_pruned_p50.pth: Prune bottom 50% channels")
    print("\nVisualization saved to models/pruned/channel_importance.png")
    print("\nUse evaluate_pruned.py to test pruned models")


if __name__ == "__main__":
    main()

