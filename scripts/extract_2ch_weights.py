"""
Extract and save 2-channel model weights to text files for easy analysis.
"""
import sys
from pathlib import Path
import torch
import numpy as np
import json

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN


def main():
    """Extract weights from 2-channel model."""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "experiments" / "2ch_convergence" / "models" / "run_01_seed_300.pth"
    output_dir = project_root / "experiments" / "2ch_convergence" / "weight_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Extracting 2-Channel CNN Weights")
    print("="*70)
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = GameOfLifeCNN(hidden_channels=2, padding_mode='circular')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract weights
    conv1_weight = model.conv1.weight.data.cpu().numpy()  # shape: (2, 1, 3, 3)
    conv1_bias = model.conv1.bias.data.cpu().numpy()      # shape: (2,)
    conv2_weight = model.conv2.weight.data.cpu().numpy()  # shape: (1, 2, 1, 1)
    conv2_bias = model.conv2.bias.data.cpu().numpy()      # shape: (1,)
    
    # Save as numpy arrays
    np.save(output_dir / "conv1_weight.npy", conv1_weight)
    np.save(output_dir / "conv1_bias.npy", conv1_bias)
    np.save(output_dir / "conv2_weight.npy", conv2_weight)
    np.save(output_dir / "conv2_bias.npy", conv2_bias)
    
    # Save as readable text
    with open(output_dir / "weights.txt", 'w') as f:
        f.write("2-CHANNEL CNN WEIGHTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("LAYER 1: Conv2d(1 → 2, 3x3 kernel)\n")
        f.write("-"*70 + "\n\n")
        
        for ch in range(2):
            f.write(f"Channel {ch}:\n")
            kernel = conv1_weight[ch, 0]  # (3, 3)
            
            f.write("  Kernel (3x3):\n")
            f.write("    NW       N        NE\n")
            f.write(f"    {kernel[0,0]:8.5f} {kernel[0,1]:8.5f} {kernel[0,2]:8.5f}\n")
            f.write("    W        C        E\n")
            f.write(f"    {kernel[1,0]:8.5f} {kernel[1,1]:8.5f} {kernel[1,2]:8.5f}\n")
            f.write("    SW       S        SE\n")
            f.write(f"    {kernel[2,0]:8.5f} {kernel[2,1]:8.5f} {kernel[2,2]:8.5f}\n")
            
            f.write(f"  Bias: {conv1_bias[ch]:.8f}\n")
            
            # Analysis
            center = kernel[1, 1]
            neighbors = [kernel[i,j] for i in range(3) for j in range(3) if not (i==1 and j==1)]
            
            f.write(f"  Center weight: {center:.6f}\n")
            f.write(f"  Neighbor sum: {sum(neighbors):.6f}\n")
            f.write(f"  Neighbor mean: {np.mean(neighbors):.6f}\n")
            f.write(f"  L1 norm: {np.sum(np.abs(kernel)):.6f}\n")
            f.write("\n")
        
        f.write("\nLAYER 2: Conv2d(2 → 1, 1x1 kernel)\n")
        f.write("-"*70 + "\n\n")
        f.write(f"Channel 0 weight: {conv2_weight[0, 0, 0, 0]:.8f}\n")
        f.write(f"Channel 1 weight: {conv2_weight[0, 1, 0, 0]:.8f}\n")
        f.write(f"Output bias: {conv2_bias[0]:.8f}\n")
    
    # Save as JSON for easy loading
    weights_dict = {
        'conv1': {
            'channel_0': {
                'kernel': conv1_weight[0, 0].tolist(),
                'bias': float(conv1_bias[0])
            },
            'channel_1': {
                'kernel': conv1_weight[1, 0].tolist(),
                'bias': float(conv1_bias[1])
            }
        },
        'conv2': {
            'channel_0_weight': float(conv2_weight[0, 0, 0, 0]),
            'channel_1_weight': float(conv2_weight[0, 1, 0, 0]),
            'bias': float(conv2_bias[0])
        },
        'metadata': {
            'run_id': int(checkpoint['run_id']),
            'seed': int(checkpoint['seed']),
            'convergence_epoch': int(checkpoint['convergence_epoch']),
            'val_accuracy': float(checkpoint['val_accuracy'])
        }
    }
    
    with open(output_dir / "weights.json", 'w') as f:
        json.dump(weights_dict, f, indent=2)
    
    # Create Python code to reconstruct
    with open(output_dir / "weights.py", 'w') as f:
        f.write("# 2-Channel CNN learned weights\n")
        f.write("# Copy-paste into Python to reconstruct the model\n\n")
        f.write("import numpy as np\n\n")
        
        f.write("# Layer 1: Conv2d(1 → 2, 3x3)\n")
        f.write("conv1_channel_0 = np.array([\n")
        for i in range(3):
            f.write(f"    [{conv1_weight[0, 0, i, 0]:.10f}, {conv1_weight[0, 0, i, 1]:.10f}, {conv1_weight[0, 0, i, 2]:.10f}],\n")
        f.write("])\n\n")
        
        f.write("conv1_channel_1 = np.array([\n")
        for i in range(3):
            f.write(f"    [{conv1_weight[1, 0, i, 0]:.10f}, {conv1_weight[1, 0, i, 1]:.10f}, {conv1_weight[1, 0, i, 2]:.10f}],\n")
        f.write("])\n\n")
        
        f.write(f"conv1_bias = np.array([{conv1_bias[0]:.10f}, {conv1_bias[1]:.10f}])\n\n")
        
        f.write("# Layer 2: Conv2d(2 → 1, 1x1)\n")
        f.write(f"conv2_weights = np.array([{conv2_weight[0, 0, 0, 0]:.10f}, {conv2_weight[0, 1, 0, 0]:.10f}])\n")
        f.write(f"conv2_bias = {conv2_bias[0]:.10f}\n\n")
        
        f.write("# Total parameters: 23\n")
        f.write("# conv1: 2 * (3*3 + 1) = 20\n")
        f.write("# conv2: 2 * 1 + 1 = 3\n")
    
    print(f"\nWeights extracted and saved to: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - weights.txt     (human-readable)")
    print(f"  - weights.json    (JSON format)")
    print(f"  - weights.py      (Python code)")
    print(f"  - conv1_weight.npy (numpy array)")
    print(f"  - conv1_bias.npy   (numpy array)")
    print(f"  - conv2_weight.npy (numpy array)")
    print(f"  - conv2_bias.npy   (numpy array)")
    
    print("\n" + "="*70)
    print("Quick Preview")
    print("="*70)
    
    for ch in range(2):
        print(f"\nChannel {ch} kernel:")
        kernel = conv1_weight[ch, 0]
        print(f"  {kernel[0,0]:7.4f} {kernel[0,1]:7.4f} {kernel[0,2]:7.4f}")
        print(f"  {kernel[1,0]:7.4f} {kernel[1,1]:7.4f} {kernel[1,2]:7.4f}")
        print(f"  {kernel[2,0]:7.4f} {kernel[2,1]:7.4f} {kernel[2,2]:7.4f}")
        print(f"  Bias: {conv1_bias[ch]:.6f}")
    
    print(f"\nCombination layer:")
    print(f"  Ch0 weight: {conv2_weight[0, 0, 0, 0]:.6f}")
    print(f"  Ch1 weight: {conv2_weight[0, 1, 0, 0]:.6f}")
    print(f"  Bias: {conv2_bias[0]:.6f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
