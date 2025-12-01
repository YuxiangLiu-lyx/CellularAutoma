"""
Train CNN with L1 regularization for automatic channel pruning.
L1 regularization promotes sparsity, making weights go to zero.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models.cnn import GameOfLifeCNN, count_parameters
from src.utils.data_loader import create_dataloader


def compute_l1_loss(model, lambda_l1):
    """
    Compute L1 regularization loss on first convolutional layer.
    
    Args:
        model: CNN model
        lambda_l1: L1 regularization strength
        
    Returns:
        L1 loss value
    """
    l1_loss = 0
    for param in model.conv1.parameters():
        l1_loss += torch.sum(torch.abs(param))
    
    return lambda_l1 * l1_loss


def train_epoch(model, loader, criterion, optimizer, device, lambda_l1=0.0):
    """Train for one epoch with L1 regularization."""
    model.train()
    total_loss = 0
    total_bce_loss = 0
    total_l1_loss = 0
    correct = 0
    total = 0
    
    for state_t, state_t1 in tqdm(loader, desc="Training"):
        state_t = state_t.to(device)
        state_t1 = state_t1.to(device)
        
        optimizer.zero_grad()
        output = model(state_t)
        
        bce_loss = criterion(output, state_t1)
        l1_loss = compute_l1_loss(model, lambda_l1)
        
        loss = bce_loss + l1_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_bce_loss += bce_loss.item()
        total_l1_loss += l1_loss.item()
        
        pred = (output > 0.5).float()
        correct += (pred == state_t1).sum().item()
        total += state_t1.numel()
    
    return {
        'total_loss': total_loss / len(loader),
        'bce_loss': total_bce_loss / len(loader),
        'l1_loss': total_l1_loss / len(loader),
        'accuracy': correct / total
    }


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for state_t, state_t1 in loader:
            state_t = state_t.to(device)
            state_t1 = state_t1.to(device)
            
            output = model(state_t)
            loss = criterion(output, state_t1)
            
            total_loss += loss.item()
            
            pred = (output > 0.5).float()
            correct += (pred == state_t1).sum().item()
            total += state_t1.numel()
    
    return total_loss / len(loader), correct / total


def analyze_sparsity(model, threshold=1e-3):
    """
    Analyze sparsity of the model.
    
    Args:
        model: CNN model
        threshold: Threshold for considering weight as zero
        
    Returns:
        Dictionary with sparsity statistics
    """
    conv1_weights = model.conv1.weight.data.cpu().numpy()
    
    channel_l1_norms = []
    channel_sparsity = []
    
    for i in range(16):
        channel_weight = conv1_weights[i]
        l1_norm = np.sum(np.abs(channel_weight))
        near_zero = np.sum(np.abs(channel_weight) < threshold)
        sparsity = near_zero / channel_weight.size
        
        channel_l1_norms.append(l1_norm)
        channel_sparsity.append(sparsity)
    
    total_params = sum(p.numel() for p in model.parameters())
    near_zero_params = sum(
        torch.sum(torch.abs(p) < threshold).item() 
        for p in model.parameters()
    )
    
    return {
        'channel_l1_norms': channel_l1_norms,
        'channel_sparsity': channel_sparsity,
        'total_sparsity': near_zero_params / total_params,
        'near_zero_channels': [i for i, norm in enumerate(channel_l1_norms) if norm < threshold]
    }


def main():
    """Main training function."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    lambda_l1_values = [0.0, 0.0001, 0.001, 0.01]
    
    results_summary = []
    
    for lambda_l1 in lambda_l1_values:
        print("\n" + "="*70)
        print(f"Training with L1 regularization (lambda={lambda_l1})")
        print("="*70)
        
        model = GameOfLifeCNN(hidden_channels=16, padding_mode='circular')
        model = model.to(device)
        
        print(f"Parameters: {count_parameters(model):,}")
        
        train_loader = create_dataloader(
            data_dir / "train.h5",
            batch_size=64,
            shuffle=True
        )
        
        val_loader = create_dataloader(
            data_dir / "val.h5",
            batch_size=64,
            shuffle=False
        )
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        num_epochs = 20
        best_val_acc = 0
        
        if lambda_l1 == 0.0:
            save_name = "cnn_16ch_no_reg.pth"
        else:
            save_name = f"cnn_16ch_l1_{lambda_l1}.pth"
        
        save_path = project_root / "experiments" / "l1_regularization" / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTraining for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_results = train_epoch(model, train_loader, criterion, optimizer, device, lambda_l1)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Train - Total Loss: {train_results['total_loss']:.4f}, "
                      f"BCE: {train_results['bce_loss']:.4f}, "
                      f"L1: {train_results['l1_loss']:.4f}, "
                      f"Acc: {train_results['accuracy']:.4f}")
                print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'hidden_channels': 16,
                    'lambda_l1': lambda_l1,
                    'model_state_dict': model.state_dict(),
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                }, save_path)
        
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")
        print(f"Saved to: {save_path}")
        
        sparsity_stats = analyze_sparsity(model, threshold=0.01)
        
        print(f"\nSparsity Analysis (lambda_l1={lambda_l1}):")
        print(f"  Overall sparsity: {sparsity_stats['total_sparsity']*100:.2f}% weights near zero")
        print(f"  Near-zero channels: {sparsity_stats['near_zero_channels']}")
        
        print(f"\nChannel L1 norms:")
        for i, norm in enumerate(sparsity_stats['channel_l1_norms']):
            status = "PRUNABLE" if norm < 0.5 else ""
            print(f"  Channel {i:2d}: {norm:8.4f}  {status}")
        
        print(f"\nStatistics:")
        print(f"  Mean: {np.mean(sparsity_stats['channel_l1_norms']):.4f}")
        print(f"  Std:  {np.std(sparsity_stats['channel_l1_norms']):.4f}")
        print(f"  Min:  {np.min(sparsity_stats['channel_l1_norms']):.4f}")
        print(f"  Max:  {np.max(sparsity_stats['channel_l1_norms']):.4f}")
        
        weakest_channels = sorted(range(16), key=lambda i: sparsity_stats['channel_l1_norms'][i])[:4]
        print(f"  Weakest 4 channels: {weakest_channels}")
        
        results_summary.append({
            'lambda_l1': lambda_l1,
            'best_val_acc': best_val_acc,
            'sparsity': sparsity_stats['total_sparsity'],
            'weakest_channels': weakest_channels,
            'near_zero_channels': sparsity_stats['near_zero_channels']
        })
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Lambda':<12} {'Val Acc':<12} {'Sparsity':<12} {'Weakest Channels':<30}")
    print("-" * 70)
    
    for r in results_summary:
        print(f"{r['lambda_l1']:<12} {r['best_val_acc']:<12.4f} "
              f"{r['sparsity']*100:<11.2f}% {str(r['weakest_channels']):<30}")
    
    print("\n" + "="*70)
    print("Recommendation")
    print("="*70)
    
    best_config = max(
        [r for r in results_summary if r['val_acc'] >= 0.999],
        key=lambda x: x['sparsity'],
        default=None
    )
    
    if best_config:
        print(f"Best L1 regularization: lambda={best_config['lambda_l1']}")
        print(f"  Accuracy: {best_config['best_val_acc']:.4f}")
        print(f"  Sparsity: {best_config['sparsity']*100:.2f}%")
        print(f"  Channels to prune: {best_config['near_zero_channels']}")
    else:
        print("L1 regularization too strong - accuracy dropped below 99.9%")
        print("Consider using smaller lambda values")
    
    print("\nModels saved to: experiments/l1_regularization/")


if __name__ == "__main__":
    main()

