"""
Evaluate the 2-channel CNN model and analyze its learned weights.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN, count_parameters
from src.utils.data_loader import create_dataloader


def evaluate_model(model, loader, device):
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for state_t, state_t1 in tqdm(loader, desc="Evaluating"):
            state_t = state_t.to(device)
            state_t1 = state_t1.to(device)
            
            output = model(state_t)
            loss = criterion(output, state_t1)
            
            total_loss += loss.item()
            
            pred = (output > 0.5).float()
            correct += (pred == state_t1).sum().item()
            total += state_t1.numel()
    
    return total_loss / len(loader), correct / total


def analyze_weights(model):
    """Analyze and display model weights."""
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    print(f"Total Parameters: {count_parameters(model)}")
    
    print("\n" + "="*70)
    print("LAYER 1: Conv2d(1 → 2, kernel_size=3x3)")
    print("="*70)
    
    conv1_weight = model.conv1.weight.data.cpu().numpy()
    conv1_bias = model.conv1.bias.data.cpu().numpy()
    
    print(f"Weight shape: {conv1_weight.shape}  (out_channels, in_channels, H, W)")
    print(f"Bias shape: {conv1_bias.shape}")
    
    for ch in range(2):
        l1_norm = np.sum(np.abs(conv1_weight[ch]))
        l2_norm = np.sqrt(np.sum(conv1_weight[ch]**2))
        
        print(f"\n--- Channel {ch} ---")
        print(f"L1 norm: {l1_norm:.4f}")
        print(f"L2 norm: {l2_norm:.4f}")
        print(f"Bias: {conv1_bias[ch]:.6f}")
        print(f"\n3x3 Kernel:")
        kernel = conv1_weight[ch, 0]  # shape (3, 3)
        
        # Print kernel in a nice format
        print("     Col 0      Col 1      Col 2")
        for i in range(3):
            print(f"Row {i}: ", end="")
            for j in range(3):
                val = kernel[i, j]
                print(f"{val:10.6f} ", end="")
            print()
        
        # Print semantic meaning
        print("\nKernel layout (Game of Life perspective):")
        print("  NW   N   NE")
        print("  W    C   E")
        print("  SW   S   SE")
        print("\nValues:")
        print(f"  {kernel[0,0]:6.3f} {kernel[0,1]:6.3f} {kernel[0,2]:6.3f}")
        print(f"  {kernel[1,0]:6.3f} {kernel[1,1]:6.3f} {kernel[1,2]:6.3f}")
        print(f"  {kernel[2,0]:6.3f} {kernel[2,1]:6.3f} {kernel[2,2]:6.3f}")
        
        # Analyze center vs neighbors
        center_weight = kernel[1, 1]
        neighbor_weights = [kernel[i,j] for i in range(3) for j in range(3) if not (i==1 and j==1)]
        neighbor_sum = sum(neighbor_weights)
        neighbor_mean = np.mean(neighbor_weights)
        
        print(f"\nAnalysis:")
        print(f"  Center weight: {center_weight:.6f}")
        print(f"  Neighbor sum: {neighbor_sum:.6f}")
        print(f"  Neighbor mean: {neighbor_mean:.6f}")
        print(f"  Neighbor std: {np.std(neighbor_weights):.6f}")
    
    print("\n" + "="*70)
    print("LAYER 2: Conv2d(2 → 1, kernel_size=1x1)")
    print("="*70)
    
    conv2_weight = model.conv2.weight.data.cpu().numpy()
    conv2_bias = model.conv2.bias.data.cpu().numpy()
    
    print(f"Weight shape: {conv2_weight.shape}  (out_channels, in_channels, 1, 1)")
    print(f"Bias shape: {conv2_bias.shape}")
    
    print(f"\nCombination weights:")
    print(f"  Channel 0 → Output: {conv2_weight[0, 0, 0, 0]:.6f}")
    print(f"  Channel 1 → Output: {conv2_weight[0, 1, 0, 0]:.6f}")
    print(f"  Output bias: {conv2_bias[0]:.6f}")
    
    print(f"\nInterpretation:")
    w0 = conv2_weight[0, 0, 0, 0]
    w1 = conv2_weight[0, 1, 0, 0]
    if abs(w0) > abs(w1):
        print(f"  Channel 0 is MORE important (weight: {abs(w0):.3f} vs {abs(w1):.3f})")
    else:
        print(f"  Channel 1 is MORE important (weight: {abs(w1):.3f} vs {abs(w0):.3f})")
    
    print("\n" + "="*70)
    print("COMPLETE WEIGHT EXPORT (for manual analysis)")
    print("="*70)
    
    print("\n# Conv1 weights (copy-paste friendly)")
    print("conv1_ch0_kernel = np.array([")
    for i in range(3):
        print(f"    [{conv1_weight[0, 0, i, 0]:.8f}, {conv1_weight[0, 0, i, 1]:.8f}, {conv1_weight[0, 0, i, 2]:.8f}],")
    print("])")
    
    print("\nconv1_ch1_kernel = np.array([")
    for i in range(3):
        print(f"    [{conv1_weight[1, 0, i, 0]:.8f}, {conv1_weight[1, 0, i, 1]:.8f}, {conv1_weight[1, 0, i, 2]:.8f}],")
    print("])")
    
    print(f"\nconv1_bias = np.array([{conv1_bias[0]:.8f}, {conv1_bias[1]:.8f}])")
    print(f"conv2_weight = np.array([{conv2_weight[0, 0, 0, 0]:.8f}, {conv2_weight[0, 1, 0, 0]:.8f}])")
    print(f"conv2_bias = {conv2_bias[0]:.8f}")


def main():
    """Main evaluation function."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    model_path = project_root / "experiments" / "2ch_convergence" / "models" / "run_01_seed_300.pth"
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("="*70)
    print("2-Channel CNN Model Evaluation")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {model_path.name}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(f"\nModel Information:")
    print(f"  Run ID: {checkpoint['run_id']}")
    print(f"  Seed: {checkpoint['seed']}")
    print(f"  Hidden Channels: {checkpoint['hidden_channels']}")
    print(f"  L1 Lambda: {checkpoint['lambda_l1']}")
    print(f"  Convergence Epoch: {checkpoint['convergence_epoch']}")
    print(f"  Validation Accuracy: {checkpoint['val_accuracy']:.6f} ({checkpoint['val_accuracy']*100:.2f}%)")
    
    # Load model
    model = GameOfLifeCNN(hidden_channels=2, padding_mode='circular')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Analyze weights first
    analyze_weights(model)
    
    # Evaluate on all datasets
    print("\n" + "="*70)
    print("PERFORMANCE EVALUATION")
    print("="*70)
    
    # Training set
    print("\nEvaluating on training set...")
    train_loader = create_dataloader(
        data_dir / "train.h5",
        batch_size=64,
        shuffle=False
    )
    train_loss, train_acc = evaluate_model(model, train_loader, device)
    print(f"Training Set:")
    print(f"  Loss: {train_loss:.6f}")
    print(f"  Accuracy: {train_acc:.6f} ({train_acc*100:.2f}%)")
    
    # Validation set
    print("\nEvaluating on validation set...")
    val_loader = create_dataloader(
        data_dir / "val.h5",
        batch_size=64,
        shuffle=False
    )
    val_loss, val_acc = evaluate_model(model, val_loader, device)
    print(f"Validation Set:")
    print(f"  Loss: {val_loss:.6f}")
    print(f"  Accuracy: {val_acc:.6f} ({val_acc*100:.2f}%)")
    
    # Test set (if exists)
    test_path = data_dir / "test.h5"
    if test_path.exists():
        print("\nEvaluating on test set...")
        test_loader = create_dataloader(
            test_path,
            batch_size=64,
            shuffle=False
        )
        test_loss, test_acc = evaluate_model(model, test_loader, device)
        print(f"Test Set:")
        print(f"  Loss: {test_loss:.6f}")
        print(f"  Accuracy: {test_acc:.6f} ({test_acc*100:.2f}%)")
    else:
        print("\nTest set not found, skipping...")
        test_acc = None
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Model: 2-channel CNN (23 parameters)")
    print(f"Converged at epoch: {checkpoint['convergence_epoch']}")
    print(f"\nFinal Accuracy:")
    print(f"  Train: {train_acc*100:.2f}%")
    print(f"  Val:   {val_acc*100:.2f}%")
    if test_acc is not None:
        print(f"  Test:  {test_acc*100:.2f}%")
    
    if train_acc >= 1.0 and val_acc >= 1.0:
        print(f"\n✓✓✓ PERFECT ACCURACY on both train and validation sets!")
        print(f"    2 channels are SUFFICIENT to solve Game of Life!")
        print(f"    This is the theoretical minimum architecture!")
    elif test_acc is not None:
        if test_acc >= 1.0:
            print(f"\n✓✓✓ PERFECT ACCURACY on test set!")
            print(f"    2 channels are SUFFICIENT to solve Game of Life!")
        elif test_acc >= 0.99:
            print(f"\n✓✓ Near-perfect accuracy!")
            print(f"   2 channels achieve {test_acc*100:.2f}% on unseen data")
        else:
            print(f"\n⚠ Some errors on test set ({(1-test_acc)*100:.2f}% error rate)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
