"""
Train single CNN with optimal L1 regularization.
Based on previous experiments, lambda=0.001 is a good starting point.
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
    """Compute L1 regularization loss on first convolutional layer."""
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
    
    # Single L1 regularization strength
    lambda_l1 = 0.001
    
    print("\n" + "="*70)
    print(f"Training 16-channel CNN with L1 regularization (lambda={lambda_l1})")
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
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    max_epochs = 100
    target_accuracy = 1.0
    best_val_acc = 0
    
    save_path = project_root / "experiments" / "l1_regularization" / "cnn_16ch_l1_optimal.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("Training")
    print("="*70)
    print(f"Target: {target_accuracy*100:.2f}% accuracy or {max_epochs} epochs")
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        
        train_results = train_epoch(model, train_loader, criterion, optimizer, device, lambda_l1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
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
                'epoch': epoch + 1,
            }, save_path)
            print(f"  Saved best model (acc: {best_val_acc:.6f})")
        
        if val_acc >= target_accuracy:
            print(f"\n{'='*70}")
            print(f"Target accuracy {target_accuracy*100:.2f}% reached!")
            print(f"Stopping at epoch {epoch+1}")
            print(f"{'='*70}")
            break
    
    print("\n" + "="*70)
    print("Training Complete")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.6f} ({best_val_acc*100:.2f}%)")
    print(f"Model saved to: {save_path}")
    
    if best_val_acc >= target_accuracy:
        print(f"\nSuccess: Reached {target_accuracy*100:.2f}% accuracy!")
    else:
        print(f"\nNote: Did not reach {target_accuracy*100:.2f}% accuracy within {max_epochs} epochs")
        print(f"Best achieved: {best_val_acc*100:.2f}%")
    
    # Analyze sparsity
    conv1_weights = model.conv1.weight.data.cpu().numpy()
    
    print("\n" + "="*70)
    print("Channel Analysis (L1-induced sparsity)")
    print("="*70)
    
    channel_l1_norms = []
    for i in range(16):
        l1_norm = np.sum(np.abs(conv1_weights[i]))
        channel_l1_norms.append(l1_norm)
        status = "← PRUNABLE" if l1_norm < 0.5 else ""
        print(f"  Channel {i:2d}: L1 norm = {l1_norm:8.4f}  {status}")
    
    print(f"\nStatistics:")
    print(f"  Mean: {np.mean(channel_l1_norms):.4f}")
    print(f"  Std:  {np.std(channel_l1_norms):.4f}")
    print(f"  Min:  {np.min(channel_l1_norms):.4f}")
    print(f"  Max:  {np.max(channel_l1_norms):.4f}")
    
    prunable_channels = [i for i, norm in enumerate(channel_l1_norms) if norm < 0.5]
    print(f"\nPrunable channels (L1 norm < 0.5): {prunable_channels}")
    print(f"Active channels needed: {16 - len(prunable_channels)}")


if __name__ == "__main__":
    main()

